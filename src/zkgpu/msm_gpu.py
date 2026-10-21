"""
zkgpu/msm_gpu.py — Pippenger Multi-Scalar Multiplication over BN254 G1 on the GPU.

What runs where
---------------
msm_pippenger_gpu computes Σ sᵢ·Pᵢ with Pippenger's bucket method:
  1. CPU: reduce every scalar mod r (the order of BN254 G1, so sᵢ·P is
     unchanged), validate every point, and upload both.
  2. GPU: one kernel extracts every w-bit window digit of every scalar.
  3. GPU: drop the digit-0 entries (they add nothing) and sort the rest by
     (window, digit) key, so each bucket's points are contiguous. Then sum
     every bucket with mixed additions (zkgpu.curve_gpu's formulas). This is
     the bulk of the work: about one point addition per point and window.
  4. GPU: reduce each window's buckets to Σ b·B_b. A single running sum per
     window would leave one thread doing 2^(w+1) sequential additions, so
     each window is split into segments of _SEGMENT_LEN buckets, each
     handled by its own thread, and the segment results are then summed.
  5. CPU: combine the window sums with Horner's rule (about 254 doublings)
     and convert the result to an affine Point with a single inverse.

Without a usable GPU every step runs on the CPU with Python integers (with
a GpuFallbackWarning): the same algorithm, so the two are directly
comparable. The GPU kernels are built with gpu_backend.compile_module and
checked against a CPU MSM when first loaded.

Load balance: buckets can be very uneven, even for random scalars. Scalars
below r < 2^254 leave the top window only 254 mod w bits, so at w = 9 two
of its buckets each collect about a third of all points; and boolean
witness values, common in real circuits, put half of all points into a
single bucket. Adding a bucket's points on one thread would make such a
bucket a long sequential chain (at 100,000 points that made the GPU part
5.5x slower at w = 9 and 8.6x slower for boolean scalars). Instead,
every sum here (buckets, and a window's segments) is chunked: one thread
adds at most _CHUNK_LEN points or partial sums, and the chunk results are
summed the same way, level by level.
"""

from __future__ import annotations

import operator
from collections.abc import Iterable
from typing import Any, SupportsIndex

import numpy as np

from zkgpu import curve_gpu
from zkgpu.curve import (
    JACOBIAN_IDENTITY,
    G,
    Jacobian,
    Point,
    from_jacobian,
    jacobian_add,
    jacobian_add_affine,
    jacobian_double,
)
from zkgpu.field import FIELD_MODULUS as _GROUP_ORDER  # BN254's r: order of G1
from zkgpu.field256_gpu import BN254_FQ, LIMBS as _FIELD_LIMBS
from zkgpu.gpu_backend import GPU_AVAILABLE as _gpu_available, compile_module, cp, warn_cpu_fallback
from zkgpu.msm import msm_double_and_add

# Reduced scalars are < r < 2^254, so 254 bits and four 64-bit limbs suffice
_SCALAR_BITS = _GROUP_ORDER.bit_length()
_LIMBS = 4

# 2^16 buckets per window is far past the useful range for either the CPU
# or the GPU bucket accumulation; larger windows only burn memory and time.
_MAX_WINDOW_SIZE = 16

# Buckets per reduction thread: each thread does 2 additions per bucket plus
# a small scalar multiplication, so this trades thread count against work
# per thread. 32 keeps windows of up to 2^16 buckets at 2048 threads each.
_SEGMENT_LEN = 32

# Additions per thread in each level of a chunked sum (see _segment_sums).
# Uniformly random scalars put about n / 2^w points in a bucket, so at the
# cost model's window sizes most buckets fit in one chunk, while a bucket of
# m points takes about log_32(m) levels instead of m additions on one thread.
_CHUNK_LEN = 32
_THREADS_PER_BLOCK = 256


# ------------------------------------------------------------------
# GPU kernels
# ------------------------------------------------------------------

if _gpu_available:  # gpu-only
    # One thread per (window, point) pair: the digit is bits
    # [win·w, win·w + w) of the scalar, read from its little-endian 64-bit
    # limbs. A window may straddle two limbs; its high bits then come from
    # the next limb. offset + w > 64 implies offset > 0, so the left shift by
    # 64 - offset is always in range.
    _window_digits_kernel = cp.ElementwiseKernel(
        "raw uint64 limbs, int64 n_points, int32 w",
        "int64 digit",
        """
        const long long win = i / n_points;
        const long long idx = i - win * n_points;
        const int shift = (int)(win * w);
        const int limb = shift >> 6;    // shift / 64
        const int offset = shift & 63;  // shift % 64
        unsigned long long v = limbs[idx * 4 + limb] >> offset;
        if (offset + w > 64 && limb + 1 < 4) {
            v |= limbs[idx * 4 + limb + 1] << (64 - offset);
        }
        digit = (long long)(v & ((1ULL << w) - 1ULL));
        """,
        "zk_pippenger_window_digits",
    )

    # For _chunk_ranges: chunks per segment, ceil(length / chunk_len) but at
    # least one, so an empty segment still produces its (identity) sum.
    _chunk_counts_kernel = cp.ElementwiseKernel(
        "raw int64 bounds, int64 chunk_len",
        "int64 chunks",
        """
        const long long len = bounds[i + 1] - bounds[i];
        chunks = len > chunk_len ? (len + chunk_len - 1) / chunk_len : 1;
        """,
        "zk_msm_chunk_counts",
    )

    # Chunk i's item range. first[s] is segment s's first chunk, so i belongs
    # to the segment with first[s] <= i < first[s + 1], found by binary search
    # (first strictly increases, since every segment has a chunk).
    _chunk_bounds_kernel = cp.ElementwiseKernel(
        "raw int64 bounds, raw int64 first, int64 n_segments, int64 chunk_len",
        "int64 lo, int64 hi",
        """
        long long a = 0, b = n_segments;  // first[a] <= i < first[b]
        while (b - a > 1) {
            const long long mid = (a + b) / 2;
            if (first[mid] <= i) a = mid; else b = mid;
        }
        lo = bounds[a] + (i - first[a]) * chunk_len;
        hi = min(lo + chunk_len, bounds[a + 1]);
        """,
        "zk_msm_chunk_bounds",
    )

# Appended to curve_gpu.cuda_source(). Buckets are numbered
# key = window * 2^w + digit; bucket (window, digit) collects the points whose
# digit in that window is `digit`. Digit 0 buckets contribute nothing.
_MSM_SOURCE = r"""
// ---- Pippenger MSM: bucket accumulation and reduction ----

// One thread per chunk: the sum of the affine points sorted_idx[lo[c]], ...,
// sorted_idx[hi[c] - 1], at most one chunk length of them (an empty chunk sums
// to the identity). Their order does not matter, since addition is commutative.
extern "C" __global__ void zk_msm_sum_points(const u64* points, const long long* sorted_idx,
                                             const long long* lo, const long long* hi,
                                             long long n_chunks, u64* out) {
    long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_chunks) return;
    u64 acc[12];
    zk_set_identity(acc);
    for (long long k = lo[c]; k < hi[c]; ++k) {
        const u64* src = points + 8 * sorted_idx[k];
        u64 q[8];
        #pragma unroll
        for (int t = 0; t < 8; ++t) q[t] = src[t];
        zk_jac_add_affine(acc, q, acc);
    }
    #pragma unroll
    for (int t = 0; t < 12; ++t) out[12 * c + t] = acc[t];
}

// The same for Jacobian rows lo[c] .. hi[c] - 1: one level of a segmented sum.
extern "C" __global__ void zk_msm_sum_rows(const u64* rows, const long long* lo, const long long* hi,
                                           long long n_chunks, u64* out) {
    long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_chunks) return;
    u64 acc[12], row[12];
    zk_set_identity(acc);
    for (long long k = lo[c]; k < hi[c]; ++k) {
        #pragma unroll
        for (int t = 0; t < 12; ++t) row[t] = rows[12 * k + t];
        zk_jac_add(acc, row, acc);
    }
    #pragma unroll
    for (int t = 0; t < 12; ++t) out[12 * c + t] = acc[t];
}

// k p for a small k, by double-and-add from the top bit (k < 2^16 here).
__device__ void zk_jac_mul_small(const u64 p[12], unsigned int k, u64 r[12]) {
    u64 acc[12];
    zk_set_identity(acc);
    for (int bit = 31; bit >= 0; --bit) {
        zk_jac_double(acc, acc);
        if ((k >> bit) & 1u) zk_jac_add(acc, p, acc);
    }
    #pragma unroll
    for (int t = 0; t < 12; ++t) r[t] = acc[t];
}

// One thread per segment: seg_len consecutive buckets [lo, lo + seg_len) of
// one window. Running sums from the top of the segment down to
// first = max(lo, 1) give S = sum B_b and T = sum (b - first + 1) B_b, so the
// segment's share of the window total sum b B_b is T + (first - 1) S.
// Starting at 1 skips bucket 0, which must not count.
extern "C" __global__ void zk_msm_segments(const u64* buckets, int w, int seg_len,
                                           long long n_segments, u64* out) {
    long long s = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_segments) return;
    long long segs_per_window = (1LL << w) / seg_len;
    long long win = s / segs_per_window;
    long long lo = (s % segs_per_window) * seg_len;
    long long first = lo > 0 ? lo : 1;
    const u64* window = buckets + 12 * (win << w);
    u64 running[12], total[12], b[12];
    zk_set_identity(running);
    zk_set_identity(total);
    for (long long j = lo + seg_len - 1; j >= first; --j) {
        #pragma unroll
        for (int t = 0; t < 12; ++t) b[t] = window[12 * j + t];
        zk_jac_add(running, b, running);
        zk_jac_add(total, running, total);
    }
    if (first > 1) {
        u64 shifted[12];
        zk_jac_mul_small(running, (unsigned int)(first - 1), shifted);
        zk_jac_add(total, shifted, total);
    }
    #pragma unroll
    for (int t = 0; t < 12; ++t) out[12 * s + t] = total[t];
}
"""
_KERNEL_NAMES = ("zk_msm_sum_points", "zk_msm_sum_rows", "zk_msm_segments")
_compiled: dict[str, Any] = {}  # filled on first use, emptied again if the self-test fails


def _kernels() -> dict[str, Any]:  # gpu-only
    if not _compiled:
        module = compile_module(curve_gpu.cuda_source() + _MSM_SOURCE, "zkgpu_msm")
        _compiled.update({name: module.get_function(name) for name in _KERNEL_NAMES})
        try:
            _self_test()
        except Exception:
            _compiled.clear()  # never keep kernels that failed their self-test
            raise
    return _compiled


def _self_test() -> None:  # gpu-only
    """
    A small MSM on the freshly compiled kernels against the CPU, once per process.

    The inputs reach every branch: several points per bucket, P and -P in one
    bucket (they cancel), P twice in one bucket (doubling), digit-0 points, and
    w = 6, so each window has two reduction segments. Chunks of 2 split the
    three-point bucket across two threads, so its chunk sums take a second
    level.
    """
    points = [G, G * 2, G * 3, -G, G, G * 5, G * 6, G * 7]
    scalars = [5, 1, 2, 5, 5, _GROUP_ORDER - 1, 12345, 2**200 + 7]
    w = 6
    sums = _window_sums_on_gpu([p.affine() for p in points], scalars, w, -(-_SCALAR_BITS // w), chunk_len=2)
    if from_jacobian(_combine_windows(sums, w)) != msm_double_and_add(points, scalars):
        raise RuntimeError(
            "msm_gpu: the compiled MSM kernels failed their self-test, so the CUDA compiler "
            "produced wrong machine code for this GPU. Please report it with the output of "
            "`python scripts/check_gpu.py`."
        )


def compile_kernels() -> None:  # gpu-only
    """
    Compile, load and self-test the MSM kernels now instead of on first use.

    Raises RuntimeError if they fail their self-test.
    """
    _kernels()


def _run(kernel: Any, threads: int, *args: Any) -> None:  # gpu-only
    kernel(((threads + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK,), (_THREADS_PER_BLOCK,), args)


def _device_window_digits(scalars: list[int], w: int, num_windows: int) -> Any:  # gpu-only
    """(num_windows, n) int64 device array: bits [win·w, win·w + w) of each scalar (all in [0, r))."""
    n = len(scalars)
    # to_bytes packs each scalar as little-endian bytes, which read back as
    # little-endian uint64 limbs: limb 0 holds the least significant 64 bits.
    packed = b"".join(s.to_bytes(8 * _LIMBS, "little") for s in scalars)
    limbs = np.frombuffer(packed, dtype="<u8").astype(np.uint64).reshape(n, _LIMBS)
    digits = cp.empty((num_windows, n), dtype=cp.int64)
    _window_digits_kernel(cp.asarray(limbs), n, w, digits)
    return digits


def _chunk_ranges(bounds: Any, chunk_len: int) -> tuple[Any, Any, Any]:  # gpu-only
    """
    Split every segment [bounds[s], bounds[s + 1]) into chunks of at most chunk_len items.

    Every segment gets at least one chunk (an empty one if the segment is
    empty), numbered segment by segment, so segment s's chunk results land in
    rows new_bounds[s] .. new_bounds[s + 1] - 1 of the next level.

    Returns (lo, hi, new_bounds): chunk c covers items [lo[c], hi[c]).
    """
    # Two small kernels instead of a dozen CuPy array operations: at small n,
    # the per-operation launch cost is what this function costs
    n_segments = bounds.shape[0] - 1
    chunks = cp.empty(n_segments, dtype=cp.int64)
    _chunk_counts_kernel(bounds, np.int64(chunk_len), chunks)
    new_bounds = cp.zeros(n_segments + 1, dtype=cp.int64)
    cp.cumsum(chunks, out=new_bounds[1:])
    n_chunks = int(new_bounds[-1])
    lo, hi = cp.empty(n_chunks, dtype=cp.int64), cp.empty(n_chunks, dtype=cp.int64)
    _chunk_bounds_kernel(bounds, new_bounds, np.int64(n_segments), np.int64(chunk_len), lo, hi)
    return lo, hi, new_bounds


def _segment_sums(rows: Any, bounds: Any, chunk_len: int) -> Any:  # gpu-only
    """
    One Jacobian row per segment: the sum of rows[bounds[s]:bounds[s + 1]] for each s.

    Every segment must be non-empty, and chunk_len at least 2. Each level adds
    up chunks of at most chunk_len rows, one thread per chunk, so a segment of
    m rows takes about log_chunk_len(m) levels instead of m sequential
    additions on one thread.
    """
    n_segments = bounds.shape[0] - 1
    # Each level divides every segment's row count by about chunk_len, so 64
    # levels suffice for any int64 count. Needing more means the chunk ranges
    # are wrong (a miscompiled kernel, say): raise instead of looping forever.
    for _ in range(64):
        if rows.shape[0] == n_segments:  # with no empty segment: one row each
            return rows
        lo, hi, bounds = _chunk_ranges(bounds, chunk_len)
        out = cp.empty((lo.shape[0], 3, _FIELD_LIMBS), dtype=cp.uint64)
        _run(_kernels()["zk_msm_sum_rows"], lo.shape[0], rows, lo, hi, np.int64(lo.shape[0]), out)
        rows = out
    raise RuntimeError(
        "msm_gpu: the chunked sums did not converge, so the chunk kernels computed wrong ranges. "
        "Please report it with the output of `python scripts/check_gpu.py`."
    )


def _window_sums_on_gpu(  # gpu-only
    coords: list[tuple[int, int]],
    scalars: list[int],
    w: int,
    num_windows: int,
    *,
    chunk_len: int,
) -> list[Jacobian]:
    """Σ digitᵢ·Pᵢ for every window, computed on the GPU (steps 2-4 of the module docstring)."""
    k = _kernels()
    n = len(coords)
    w_arg = np.int32(w)
    points = BN254_FQ.from_ints(c for xy in coords for c in xy).reshape(n, 2, _FIELD_LIMBS)

    # Bucket key = window * 2^w + digit. Entries with digit 0 add nothing, so
    # they are dropped; sorting the remaining keys groups each bucket's points,
    # and a binary search finds where every bucket starts (empty buckets,
    # digit 0's included, get empty ranges)
    digits = _device_window_digits(scalars, w, num_windows).ravel()
    entry = cp.flatnonzero(digits)  # entry = window * n + point index
    keys = digits[entry] + ((entry // n) << w)
    order = cp.argsort(keys)
    sorted_idx = entry[order] % n
    n_buckets = num_windows << w
    bounds = cp.searchsorted(keys[order], cp.arange(n_buckets + 1)).astype(cp.int64)

    # Each bucket's points in chunks of at most chunk_len, one thread per
    # chunk, then the chunk sums level by level down to one row per bucket
    lo, hi, chunk_bounds = _chunk_ranges(bounds, chunk_len)
    partial = cp.empty((lo.shape[0], 3, _FIELD_LIMBS), dtype=cp.uint64)
    _run(k["zk_msm_sum_points"], lo.shape[0], points, sorted_idx, lo, hi, np.int64(lo.shape[0]), partial)
    buckets = _segment_sums(partial, chunk_bounds, chunk_len)

    seg_len = min(1 << w, _SEGMENT_LEN)
    segs_per_window = (1 << w) // seg_len
    n_segments = num_windows * segs_per_window
    segments = cp.empty((n_segments, 3, _FIELD_LIMBS), dtype=cp.uint64)
    _run(k["zk_msm_segments"], n_segments, buckets, w_arg, np.int32(seg_len), np.int64(n_segments), segments)

    # Each window's sum is the sum of its segments' shares
    window_bounds = cp.arange(0, n_segments + 1, segs_per_window, dtype=cp.int64)
    return curve_gpu.jacobian_ints(_segment_sums(segments, window_bounds, chunk_len))


# ------------------------------------------------------------------
# CPU path (fallback, and the same-algorithm reference for the benchmark)
# ------------------------------------------------------------------


def _cpu_window_digits(scalars: list[int], w: int, num_windows: int) -> list[list[int]]:
    """digits[win][i] = bits [win·w, win·w + w) of scalars[i] (all scalars in [0, r))."""
    mask = (1 << w) - 1
    return [[(s >> (win * w)) & mask for s in scalars] for win in range(num_windows)]


def _window_sum(coords: list[tuple[int, int]], digits: list[int], bucket_count: int) -> Jacobian:
    """Σ digitᵢ·Pᵢ for one window, using 2^w buckets instead of N scalar multiplications."""
    # Buckets are Jacobian and inputs affine, so filling them uses mixed additions
    buckets = [JACOBIAN_IDENTITY] * bucket_count  # bucket 0 contributes nothing
    for (x, y), digit in zip(coords, digits, strict=True):
        if digit:
            buckets[digit] = jacobian_add_affine(buckets[digit], x, y)

    # Running-sum trick: Σ k·B_k = Σ_{j≥1} (B_j + B_{j+1} + ...), so two
    # additions per bucket replace a scalar multiplication per bucket.
    running = JACOBIAN_IDENTITY
    total = JACOBIAN_IDENTITY
    for k in range(bucket_count - 1, 0, -1):
        running = jacobian_add(running, buckets[k])
        total = jacobian_add(total, running)
    return total


def _combine_windows(window_sums: list[Jacobian], w: int) -> Jacobian:
    """
    Σ 2^(w·win) · window_sums[win], by Horner's rule from the most significant window.

    Shifting the running result by w bits between windows costs
    w·(windows - 1) ≈ 254 doublings in total, instead of shifting each window
    sum separately.
    """
    acc = window_sums[-1]
    for win in range(len(window_sums) - 2, -1, -1):
        for _ in range(w):
            acc = jacobian_double(acc)
        acc = jacobian_add(acc, window_sums[win])
    return acc


def _as_ints(values: Iterable[SupportsIndex], name: str) -> list[int]:
    """Plain Python ints; operator.index rejects floats instead of truncating them."""
    try:
        return [operator.index(v) for v in values]
    except TypeError as exc:
        raise TypeError(f"{name} must contain integers: {exc}") from None


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def msm_pippenger_gpu(
    points_x: Iterable[int],
    points_y: Iterable[int],
    scalars: Iterable[int],
    *,
    window_size: int = 8,
) -> Point:
    """
    Pippenger MSM over BN254 G1: digits, buckets and bucket reduction on the GPU.

    Pippenger needs O(N + 2^w) point additions per window instead of the
    O(N · bits) of msm_naive, plus w doublings per window to combine them.

    Parameters
    ----------
    points_x, points_y : affine coordinates of the points (Python ints or an
                         object-dtype array; BN254 coordinates do not fit int64)
    scalars            : any integers, including negative and >= 2^256; they
                         are reduced mod r, which leaves every sᵢ·Pᵢ unchanged
    window_size        : bits per window (w), 1..16; each window has 2^w buckets

    Returns
    -------
    Point : Σ sᵢ·Pᵢ, which is Point.identity() for the point at infinity

    Raises
    ------
    ValueError : if the three inputs differ in length or window_size is out of range
    TypeError  : if a coordinate or scalar is not an integer
    """
    window_size = operator.index(window_size)
    if not 1 <= window_size <= _MAX_WINDOW_SIZE:
        raise ValueError(f"window_size must be in 1..{_MAX_WINDOW_SIZE}, got {window_size}")

    xs, ys = _as_ints(points_x, "points_x"), _as_ints(points_y, "points_y")
    reduced = [s % _GROUP_ORDER for s in _as_ints(scalars, "scalars")]
    if not len(xs) == len(ys) == len(reduced):
        raise ValueError(
            f"Length mismatch: {len(xs)} x-coordinates, {len(ys)} y-coordinates, {len(reduced)} scalars."
        )
    if not reduced:
        return Point.identity()

    # Point() validates every caller-supplied pair is on the curve before use
    coords = [Point(x, y).affine() for x, y in zip(xs, ys, strict=True)]
    num_windows = -(-_SCALAR_BITS // window_size)  # ceil(254 / w)
    if _gpu_available:
        window_sums = _window_sums_on_gpu(  # gpu-only
            coords, reduced, window_size, num_windows, chunk_len=_CHUNK_LEN
        )
    else:
        warn_cpu_fallback("msm_pippenger_gpu", "Pippenger on the CPU (Python integers)")
        digits = _cpu_window_digits(reduced, window_size, num_windows)
        window_sums = [_window_sum(coords, d, 1 << window_size) for d in digits]
    return from_jacobian(_combine_windows(window_sums, window_size))
