"""
benchmarks/bench_field.py — Benchmark field arithmetic CPU vs GPU.

Measures element-wise modular multiplication and addition for increasing
array sizes. Results saved to results/field_results.json.
"""

from __future__ import annotations

import json
import os
import statistics
import time

import numpy as np

# BN254 scalar field modulus r ≈ 2^254 does not fit in int64.
# For GPU benchmarking we use the NTT-friendly prime 998244353 < 2^30,
# which ensures a*b < 2^60 — safely within int64 range.
MOD = 998_244_353
SIZES = [10_000, 100_000, 1_000_000, 10_000_000]
RUNS = 3

try:
    from src.field_gpu import gpu_batch_mul, _gpu_available
    if _gpu_available:
        import cupy as cp
except Exception:
    _gpu_available = False
    cp = None


def _cpu_batch_mul(a: np.ndarray, b: np.ndarray, mod: int) -> np.ndarray:
    return (a * b) % mod


def benchmark_size(n: int) -> dict:
    rng = np.random.default_rng(42)
    a_np = rng.integers(1, MOD, size=n, dtype=np.int64)
    b_np = rng.integers(1, MOD, size=n, dtype=np.int64)

    # CPU timing
    cpu_times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        _cpu_batch_mul(a_np, b_np, MOD)
        cpu_times.append(time.perf_counter() - t0)
    cpu_ms = statistics.median(cpu_times) * 1000

    gpu_ms: float | None = None
    speedup: float | None = None

    if _gpu_available:
        a_cp = cp.array(a_np)
        b_cp = cp.array(b_np)

        # Warmup
        gpu_batch_mul(a_cp, b_cp, MOD)

        gpu_times = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            gpu_batch_mul(a_cp, b_cp, MOD)
            cp.cuda.Stream.null.synchronize()
            gpu_times.append(time.perf_counter() - t0)
        gpu_ms = round(statistics.median(gpu_times) * 1000, 3)
        speedup = round(cpu_ms / gpu_ms, 2) if gpu_ms > 0 else None

    return {
        "n": n,
        "cpu_ms": round(cpu_ms, 3),
        "gpu_ms": gpu_ms,
        "speedup": speedup,
        "cpu_throughput_meps": round(n / (cpu_ms / 1000) / 1e6, 2),
        "gpu_throughput_meps": round(n / (gpu_ms / 1000) / 1e6, 2) if gpu_ms else None,
    }


def run_bench() -> list[dict]:
    print("=== Field Arithmetic Benchmark (modular multiplication) ===")
    results = []
    for n in SIZES:
        print(f"  N = {n:>12,} ...", end=" ", flush=True)
        r = benchmark_size(n)
        results.append(r)
        gpu_str = f"GPU {r['gpu_ms']} ms | Speedup {r['speedup']}x" if r["gpu_ms"] else "GPU N/A"
        print(f"CPU {r['cpu_ms']} ms | {gpu_str}")
    return results


if __name__ == "__main__":
    results = run_bench()
    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "field_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
