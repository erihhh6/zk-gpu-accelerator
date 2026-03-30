"""
benchmarks/bench_msm.py — Benchmark MSM CPU vs GPU for N = 100, 500, 1000, 5000, 10000.

Measures:
  - Latency in milliseconds (median of 3 runs)
  - Memory transfer time (host→device scalar copy) separately from compute time

Results saved to results/msm_results.json.
"""

from __future__ import annotations

import json
import os
import random
import statistics
import time

import numpy as np

from src.curve import G, P as CURVE_P
from src.msm import msm_naive

SIZES = [100, 500, 1000, 5000, 10_000]
RUNS = 3

try:
    from src.msm_gpu import msm_pippenger_gpu, _gpu_available
    if _gpu_available:
        import cupy as cp
except Exception:
    _gpu_available = False
    cp = None  # type: ignore[assignment]


def _generate_inputs(n: int, seed: int = 42):
    """Generate N random points on BN254 and N random scalars."""
    rng = random.Random(seed)
    scalars = [rng.randint(1, 2**32) for _ in range(n)]
    points = [G * rng.randint(1, 2**32) for _ in range(n)]
    return points, scalars


def benchmark_size(n: int) -> dict:
    points, scalars = _generate_inputs(n)

    # CPU (median of RUNS)
    cpu_times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        msm_naive(points, scalars)
        cpu_times.append(time.perf_counter() - t0)
    cpu_ms = statistics.median(cpu_times) * 1000

    gpu_ms: float | None = None
    gpu_compute_ms: float | None = None
    mem_transfer_ms: float | None = None
    speedup: float | None = None

    if _gpu_available:
        px_np = np.array([p.x for p in points], dtype=object)
        py_np = np.array([p.y for p in points], dtype=object)
        sc_np = np.array(scalars, dtype=np.int64)

        # Warmup (compiles CuPy kernels)
        msm_pippenger_gpu(px_np[:10], py_np[:10], sc_np[:10], CURVE_P)

        total_times = []
        transfer_times = []
        compute_times = []

        for _ in range(RUNS):
            # Scalar host→device transfer time
            t_xfer_start = time.perf_counter()
            sc_gpu = cp.array(sc_np)
            cp.cuda.Stream.null.synchronize()
            t_xfer_end = time.perf_counter()

            t_run_start = time.perf_counter()
            msm_pippenger_gpu(px_np, py_np, sc_np, CURVE_P)
            t_run_end = time.perf_counter()

            transfer_times.append((t_xfer_end - t_xfer_start) * 1000)
            total_times.append((t_run_end - t_run_start) * 1000)
            compute_times.append(total_times[-1] - transfer_times[-1])

        gpu_ms = round(statistics.median(total_times), 3)
        gpu_compute_ms = round(statistics.median(compute_times), 3)
        mem_transfer_ms = round(statistics.median(transfer_times), 3)
        speedup = round(cpu_ms / gpu_ms, 2) if gpu_ms > 0 else None

    return {
        "n": n,
        "cpu_ms": round(cpu_ms, 3),
        "gpu_ms": gpu_ms,
        "gpu_compute_ms": gpu_compute_ms,
        "mem_transfer_ms": mem_transfer_ms,
        "speedup": speedup,
        "cpu_throughput_keps": round(n / (cpu_ms / 1000) / 1e3, 2),
        "gpu_throughput_keps": round(n / (gpu_ms / 1000) / 1e3, 2) if gpu_ms else None,
    }


def run_bench() -> list[dict]:
    print("=== MSM Benchmark (Shamir CPU / Pippenger+GPU) ===")
    results = []
    for n in SIZES:
        print(f"  N = {n:>6} ...", end=" ", flush=True)
        r = benchmark_size(n)
        results.append(r)
        if r["gpu_ms"]:
            print(
                f"CPU {r['cpu_ms']:.1f} ms | Pippenger+GPU {r['gpu_ms']:.1f} ms "
                f"(compute {r['gpu_compute_ms']:.1f} ms, transfer {r['mem_transfer_ms']:.1f} ms) "
                f"| Speedup {r['speedup']}x"
            )
        else:
            print(f"CPU {r['cpu_ms']:.1f} ms | GPU N/A")
    return results


if __name__ == "__main__":
    results = run_bench()
    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "msm_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
