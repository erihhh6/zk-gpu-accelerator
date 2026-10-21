"""
zkgpu/msm_gpu.py — Pippenger Multi-Scalar Multiplication over BN254 G1 on the GPU.

What runs where
---------------
msm_pippenger_gpu computes Σ sᵢ·Pᵢ with Pippenger's bucket method:
  1. CPU: reduce every scalar mod r (the order of BN254 G1, so sᵢ·P is
     unchanged), validate every point, and upload both.
  2. GPU: one kernel extracts every w-bit window digit of every scalar.
  3. GPU: sort the (window, digit) keys, so each bucket's points are
     contiguous, then one thread per bucket adds its points (mixed
     additions, zkgpu.curve_gpu's formulas). This is the bulk of the work:
     about one point addition per point and window.
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

Load balance: a bucket's thread adds all of that bucket's points in turn.
Uniformly random scalars, as in real proofs, spread the points evenly; a
skewed input (every scalar equal, say) puts all points of a window into one
bucket and serialises that window's accumulation.
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

# Appended to curve_gpu.cuda_source(). Buckets are numbered
# key = window * 2^w + digit; bucket (window, digit) collects the points whose
# digit in that window is `digit`. Digit 0 buckets contribute nothing.
_MSM_SOURCE = r"""
// ---- Pippenger MSM: bucket accumulation and reduction ----

// One thread per bucket. sorted_idx lists point indices ordered by key, and
// bucket key's points sit at positions [bounds[key], bounds[key + 1]); their
// order within a bucket does not matter, since addition is commutative.
extern "C" __global__ void zk_msm_buckets(const u64* points, const long long* sorted_idx,
                                          const long long* bounds, int w, long long n_buckets,
                                          u64* buckets) {
    long long key = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (key >= n_buckets) return;
    u64 acc[12];
    zk_set_identity(acc);
    if (key & ((1LL << w) - 1)) {  // digit 0: those points add nothing
        for (long long k = bounds[key]; k < bounds[key + 1]; ++k) {
            const u64* src = points + 8 * sorted_idx[k];
            u64 q[8];
            #pragma unroll
            for (int t = 0; t < 8; ++t) q[t] = src[t];
            zk_jac_add_affine(acc, q, acc);
        }
    }
    #pragma unroll
    for (int t = 0; t < 12; ++t) buckets[12 * key + t] = acc[t];
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

// One thread per window: the sum of its segments' shares.
extern "C" __global__ void zk_msm_windows(const u64* segments, long long segs_per_window,
                                          long long n_windows, u64* out) {
    long long win = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (win >= n_windows) return;
    u64 acc[12], c[12];
    zk_set_identity(acc);
    for (long long s = 0; s < segs_per_window; ++s) {
        #pragma unroll
        for (int t = 0; t < 12; ++t) c[t] = segments[12 * (win * segs_per_window + s) + t];
        zk_jac_add(acc, c, acc);
    }
    #pragma unroll
    for (int t = 0; t < 12; ++t) out[12 * win + t] = acc[t];
}
"""
_KERNEL_NAMES = ("zk_msm_buckets", "zk_msm_segments", "zk_msm_windows")
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
    w = 6, so each window has two reduction segments.
    """
    points = [G, G * 2, G * 3, -G, G, G * 5, G * 6, G * 7]
    scalars = [5, 1, 2, 5, 5, _GROUP_ORDER - 1, 12345, 2**200 + 7]
    w = 6
    sums = _window_sums_on_gpu([p.affine() for p in points], scalars, w, -(-_SCALAR_BITS // w))
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


def _window_sums_on_gpu(  # gpu-only
    coords: list[tuple[int, int]], scalars: list[int], w: int, num_windows: int
) -> list[Jacobian]:
    """Σ digitᵢ·Pᵢ for every window, computed on the GPU (steps 2-4 of the module docstring)."""
    k = _kernels()
    n = len(coords)
    w_arg = np.int32(w)
    points = BN254_FQ.from_ints(c for xy in coords for c in xy).reshape(n, 2, _FIELD_LIMBS)

    # Bucket key = window * 2^w + digit; sorting the keys groups each bucket's
    # points, and a binary search finds where every bucket starts
    keys = (_device_window_digits(scalars, w, num_windows) + (cp.arange(num_windows)[:, None] << w)).ravel()
    order = cp.argsort(keys)
    n_buckets = num_windows << w
    bounds = cp.searchsorted(keys[order], cp.arange(n_buckets + 1)).astype(cp.int64)
    buckets = cp.empty((n_buckets, 3, _FIELD_LIMBS), dtype=cp.uint64)
    sorted_idx = (order % n).astype(cp.int64)
    _run(k["zk_msm_buckets"], n_buckets, points, sorted_idx, bounds, w_arg, np.int64(n_buckets), buckets)

    seg_len = min(1 << w, _SEGMENT_LEN)
    segs_per_window = (1 << w) // seg_len
    n_segments = num_windows * segs_per_window
    segments = cp.empty((n_segments, 3, _FIELD_LIMBS), dtype=cp.uint64)
    _run(k["zk_msm_segments"], n_segments, buckets, w_arg, np.int32(seg_len), np.int64(n_segments), segments)

    sums = cp.empty((num_windows, 3, _FIELD_LIMBS), dtype=cp.uint64)
    _run(k["zk_msm_windows"], num_windows, segments, np.int64(segs_per_window), np.int64(num_windows), sums)
    return curve_gpu.jacobian_ints(sums)


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
        window_sums = _window_sums_on_gpu(coords, reduced, window_size, num_windows)  # gpu-only
    else:
        warn_cpu_fallback("msm_pippenger_gpu", "Pippenger on the CPU (Python integers)")
        digits = _cpu_window_digits(reduced, window_size, num_windows)
        window_sums = [_window_sum(coords, d, 1 << window_size) for d in digits]
    return from_jacobian(_combine_windows(window_sums, window_size))
