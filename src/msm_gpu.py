"""
src/msm_gpu.py — Multi-Scalar Multiplication behind a GPU-shaped array API.

What runs where
---------------
msm_pippenger_gpu — Pippenger's bucket method:
  1. Split each scalar into windows of *window_size* bits.
  2. GPU (CuPy): extract one window's value for all N scalars at once, then
     copy the values back to the host. This is the only GPU work.
  3. CPU: add each point into its bucket and combine the buckets with the
     running-sum trick. All BN254 point arithmetic uses Python's
     arbitrary-precision ints; there is no CUDA curve arithmetic yet.
  4. CPU: shift each window's sum into place and accumulate.
  The speedup over msm_naive comes from Pippenger doing far fewer point
  additions, not from the GPU: the CPU fallback path is no slower.

msm_gpu — naive per-point MSM, entirely on the CPU (no GPU work at all);
  kept as a reference behind the same array API.

Known limitations (tracked as known bugs in tests/test_msm.py): on the GPU
path scalars must fit in int64, so real 254-bit scalars raise OverflowError;
scalars are not reduced mod r; and each window is shifted with its own
doubling loop rather than Horner's rule.
"""

from __future__ import annotations

import numpy as np

from src.curve import Point, P as CURVE_P
from src.gpu_backend import GPU_AVAILABLE as _gpu_available, cp, warn_cpu_fallback


# ------------------------------------------------------------------
# Public GPU MSM API
# ------------------------------------------------------------------


def msm_gpu(
    points_x: np.ndarray,
    points_y: np.ndarray,
    scalars: np.ndarray,
    mod: int,
) -> tuple[int, int]:
    """
    Naive MSM on the CPU, behind the same array API as msm_pippenger_gpu.

    Despite the name, no GPU work happens: each sᵢ·Pᵢ is computed by
    double-and-add with Python ints and the products are summed.

    Parameters
    ----------
    points_x, points_y : np.ndarray (dtype=object) — affine coordinates as Python ints
    scalars            : np.ndarray — scalar multipliers
    mod                : unused; arithmetic always uses BN254's P from src.curve

    Returns
    -------
    (x, y) of the result; (0, 0) denotes the point at infinity.
    """
    n = len(scalars)
    if n == 0:
        return 0, 0

    result = Point.identity()
    for x, y, s in zip(points_x.tolist(), points_y.tolist(), scalars.tolist()):
        result = result + Point(int(x), int(y)) * int(s)

    rx = result.x if result.x is not None else 0
    ry = result.y if result.y is not None else 0
    return int(rx), int(ry)


def msm_pippenger_gpu(
    points_x: np.ndarray,
    points_y: np.ndarray,
    scalars: np.ndarray,
    mod: int,
    window_size: int = 8,
) -> tuple[int, int]:
    """
    Pippenger MSM; only the scalar window extraction runs on the GPU.

    For each window position:
      - GPU: extract the window value for all N scalars at once (CuPy), then
        copy the values to the host
      - CPU: add each point into its bucket and combine the buckets

    Pippenger needs O(N + 2^w) point additions per window instead of the
    O(N · bits) of msm_naive. That reduction, not the GPU step, is where the
    speedup comes from.

    Parameters
    ----------
    points_x, points_y : np.ndarray (dtype=object) — affine coordinates as Python ints
    scalars            : np.ndarray — scalars; must fit in int64 on the GPU path
    mod                : unused; arithmetic always uses BN254's P from src.curve
    window_size        : bits per window (w); each window has 2^w buckets

    Returns
    -------
    (x, y) of the result; (0, 0) denotes the point at infinity.
    """
    n = len(scalars)
    if n == 0:
        return 0, 0

    num_windows = (256 + window_size - 1) // window_size
    bucket_count = 1 << window_size  # 2^window_size
    mask = bucket_count - 1

    # Scalars as Python ints for arbitrary-precision windowing
    scalars_int = [int(s) for s in scalars.tolist()]
    points = [Point(int(x), int(y)) for x, y in zip(points_x.tolist(), points_y.tolist())]

    if _gpu_available:
        # GPU-accelerated scalar window extraction
        sc_gpu = cp.array(scalars_int, dtype=cp.int64)
    else:
        warn_cpu_fallback("msm_pippenger_gpu", "CPU scalar window extraction")

    final_result = Point.identity()

    for win in range(num_windows):
        shift = win * window_size

        if _gpu_available:
            # Extract this window's bits for all scalars at once on GPU
            win_vals_gpu = (sc_gpu >> shift) & mask
            win_vals = win_vals_gpu.get().tolist()  # back to CPU as list
        else:
            win_vals = [(s >> shift) & mask for s in scalars_int]

        # CPU bucket accumulation
        buckets: list[Point] = [Point.identity()] * bucket_count
        for i, v in enumerate(win_vals):
            if v == 0:
                continue
            buckets[v] = buckets[v] + points[i]

        # Pippenger bucket sum: suffix-sum trick
        running = Point.identity()
        accumulated = Point.identity()
        for k in range(bucket_count - 1, 0, -1):
            running = running + buckets[k]
            accumulated = accumulated + running

        # Shift accumulated result left by `shift` bits (= shift doublings)
        shifted = accumulated
        for _ in range(shift):
            shifted = shifted.double()

        final_result = final_result + shifted

    rx = final_result.x if final_result.x is not None else 0
    ry = final_result.y if final_result.y is not None else 0
    return int(rx), int(ry)


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
        msm_pippenger_gpu(px_np[:10], py_np[:10], sc_np[:10], CURVE_P, window_size=4)

        pip_times = []
        for _ in range(3):
            t0 = time.perf_counter()
            msm_pippenger_gpu(px_np, py_np, sc_np, CURVE_P, window_size=8)
            pip_times.append(time.perf_counter() - t0)
        pip_ms = statistics.median(pip_times) * 1000
        speedup = cpu_ms / pip_ms if pip_ms > 0 else 0

        print(f"N={N:>6}: CPU {cpu_ms:8.1f} ms | Pippenger+GPU {pip_ms:6.1f} ms | Speedup {speedup:.1f}x")

    print("\nmsm_gpu.py benchmark complete.")
