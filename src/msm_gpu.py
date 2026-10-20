"""
src/msm_gpu.py — Pippenger Multi-Scalar Multiplication with GPU digit extraction.

What runs where
---------------
msm_pippenger_gpu computes Σ sᵢ·Pᵢ with Pippenger's bucket method:
  1. CPU: reduce every scalar mod r (the order of BN254 G1, so sᵢ·P is
     unchanged) and split it into four little-endian 64-bit limbs.
  2. GPU (CuPy): one kernel extracts every w-bit window digit of every scalar
     at once, then a single device-to-host copy brings them back.
  3. CPU: per window, add each point into the bucket named by its digit and
     combine the buckets with the running-sum trick. All BN254 point
     arithmetic uses Python's arbitrary-precision ints; there is no CUDA
     curve arithmetic yet.
  4. CPU: combine the window sums with Horner's rule.
The speedup over msm_naive comes from Pippenger doing far fewer point
additions, not from the GPU step; the CPU fallback path is no slower.

Falls back to CPU digit extraction when CUDA is unavailable, with a
GpuFallbackWarning.
"""

from __future__ import annotations

import operator
from collections.abc import Iterable

import numpy as np

from src.curve import Point
from src.field import FIELD_MODULUS as _GROUP_ORDER  # BN254's r: order of G1
from src.gpu_backend import GPU_AVAILABLE as _gpu_available, cp, warn_cpu_fallback

# Reduced scalars are < r < 2^254, so 254 bits and four 64-bit limbs suffice
_SCALAR_BITS = _GROUP_ORDER.bit_length()
_LIMBS = 4

# 2^16 buckets per window is already far past the useful range for the
# Python-int bucket accumulation; larger windows only burn memory and time.
_MAX_WINDOW_SIZE = 16


# ------------------------------------------------------------------
# GPU digit extraction
# ------------------------------------------------------------------

if _gpu_available:
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


def _window_digits(scalars: list[int], window_size: int, num_windows: int) -> list[list[int]]:
    """digits[win][i] = bits [win·w, win·w + w) of scalars[i] (all scalars in [0, r))."""
    if not _gpu_available:
        mask = (1 << window_size) - 1
        return [[(s >> (win * window_size)) & mask for s in scalars] for win in range(num_windows)]

    n = len(scalars)
    # to_bytes packs each scalar as little-endian bytes, which read back as
    # little-endian uint64 limbs: limb 0 holds the least significant 64 bits.
    packed = b"".join(s.to_bytes(8 * _LIMBS, "little") for s in scalars)
    limbs = np.frombuffer(packed, dtype="<u8").astype(np.uint64).reshape(n, _LIMBS)
    digits = cp.empty((num_windows, n), dtype=cp.int64)
    _window_digits_kernel(cp.asarray(limbs), n, window_size, digits)
    return digits.get().tolist()  # one transfer for every window


# ------------------------------------------------------------------
# CPU bucket accumulation
# ------------------------------------------------------------------


def _window_sum(points: list[Point], digits: list[int], bucket_count: int) -> Point:
    """Σ digitᵢ·Pᵢ for one window, using 2^w buckets instead of N scalar multiplications."""
    buckets = [Point.identity()] * bucket_count  # bucket 0 contributes nothing
    for point, digit in zip(points, digits):
        if digit:
            buckets[digit] = buckets[digit] + point

    # Running-sum trick: Σ k·B_k = Σ_{j≥1} (B_j + B_{j+1} + ...), so two
    # additions per bucket replace a scalar multiplication per bucket.
    running = Point.identity()
    total = Point.identity()
    for k in range(bucket_count - 1, 0, -1):
        running = running + buckets[k]
        total = total + running
    return total


def _as_ints(values: Iterable, name: str) -> list[int]:
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
    Pippenger MSM over BN254 G1; only the window-digit extraction runs on the GPU.

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
            f"Length mismatch: {len(xs)} x-coordinates, {len(ys)} y-coordinates, "
            f"{len(reduced)} scalars."
        )
    if not reduced:
        return Point.identity()

    points = [Point(x, y) for x, y in zip(xs, ys)]
    num_windows = -(-_SCALAR_BITS // window_size)  # ceil(254 / w)
    if not _gpu_available:
        warn_cpu_fallback("msm_pippenger_gpu", "CPU window-digit extraction")
    digits = _window_digits(reduced, window_size, num_windows)

    # Horner's rule over windows, most significant first: shifting the running
    # result by w bits between windows costs w·(num_windows - 1) ≈ 254
    # doublings in total, instead of shifting each window sum separately.
    bucket_count = 1 << window_size
    result = _window_sum(points, digits[-1], bucket_count)
    for win in range(num_windows - 2, -1, -1):
        for _ in range(window_size):
            result = result.double()
        result = result + _window_sum(points, digits[win], bucket_count)
    return result


# ------------------------------------------------------------------
# Smoke test / benchmark
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    import time
    import statistics
    from src.curve import G
    from src.msm import msm_naive
    import random

    print("=== msm_gpu.py benchmark ===")
    random.seed(42)

    SIZES = [100, 1000, 5000]

    for N in SIZES:
        scalars_list = [random.randint(1, 2**32) for _ in range(N)]
        points_list = [G * random.randint(1, 2**32) for _ in range(N)]

        px_np = np.array([p.x for p in points_list], dtype=object)
        py_np = np.array([p.y for p in points_list], dtype=object)
        sc_np = np.array(scalars_list, dtype=np.int64)

        cpu_times = []
        for _ in range(3):
            t0 = time.perf_counter()
            msm_naive(points_list, scalars_list)
            cpu_times.append(time.perf_counter() - t0)
        cpu_ms = statistics.median(cpu_times) * 1000

        # Warmup
        msm_pippenger_gpu(px_np[:10], py_np[:10], sc_np[:10], window_size=4)

        pip_times = []
        for _ in range(3):
            t0 = time.perf_counter()
            msm_pippenger_gpu(px_np, py_np, sc_np, window_size=8)
            pip_times.append(time.perf_counter() - t0)
        pip_ms = statistics.median(pip_times) * 1000
        speedup = cpu_ms / pip_ms if pip_ms > 0 else 0

        print(f"N={N:>6}: CPU {cpu_ms:8.1f} ms | Pippenger+GPU {pip_ms:6.1f} ms | Speedup {speedup:.1f}x")

    print("\nmsm_gpu.py benchmark complete.")
