"""
benchmarks/bench_ntt.py — Benchmark NTT CPU vs GPU for N = 2^k, k in [14, 16, 18, 20].

Measures:
  - Latency in milliseconds (median of 3 runs)
  - Throughput in elements/second

Results saved to results/ntt_results.json.
"""

from __future__ import annotations

import json
import os
import statistics
import time

import numpy as np

from zkgpu.ntt import ntt as ntt_cpu, MOD

SIZES = [2**k for k in [14, 16, 18, 20]]
RUNS = 3

try:
    from zkgpu.ntt_gpu import ntt_gpu, _gpu_available
    if _gpu_available:
        import cupy as cp
except Exception:
    _gpu_available = False
    cp = None


def benchmark_size(n: int) -> dict:
    rng = np.random.default_rng(42)
    a_np = rng.integers(0, MOD, size=n, dtype=np.int64)
    a_list = a_np.tolist()

    # CPU (3 runs, median)
    cpu_times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        ntt_cpu(a_list, MOD)
        cpu_times.append(time.perf_counter() - t0)
    cpu_ms = statistics.median(cpu_times) * 1000

    gpu_ms: float | None = None
    speedup: float | None = None

    if _gpu_available:
        # Warmup (first run JIT-compiles CuPy kernels)
        _ = ntt_gpu(cp.array(a_np, dtype=cp.int64), MOD)

        gpu_times = []
        for _ in range(RUNS):
            arr = cp.array(a_np, dtype=cp.int64)
            t0 = time.perf_counter()
            ntt_gpu(arr, MOD)
            cp.cuda.Stream.null.synchronize()
            gpu_times.append(time.perf_counter() - t0)
        gpu_ms = round(statistics.median(gpu_times) * 1000, 3)
        speedup = round(cpu_ms / gpu_ms, 2) if gpu_ms > 0 else None

    return {
        "n": n,
        "cpu_ms": round(cpu_ms, 3),
        "gpu_ms": gpu_ms,
        "speedup": speedup,
        "cpu_throughput_meps": round(n / (cpu_ms / 1000) / 1e6, 4),
        "gpu_throughput_meps": round(n / (gpu_ms / 1000) / 1e6, 4) if gpu_ms else None,
    }


def run_bench() -> list[dict]:
    print("=== NTT Benchmark (Cooley-Tukey, MOD=998244353) ===")
    results = []
    for n in SIZES:
        print(f"  N = 2^{n.bit_length()-1} = {n:>8} ...", end=" ", flush=True)
        r = benchmark_size(n)
        results.append(r)
        gpu_str = f"GPU {r['gpu_ms']} ms | Speedup {r['speedup']}x" if r["gpu_ms"] else "GPU N/A"
        print(f"CPU {r['cpu_ms']:.1f} ms | {gpu_str}")
    return results


if __name__ == "__main__":
    results = run_bench()
    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "ntt_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
