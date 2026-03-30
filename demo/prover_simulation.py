"""
demo/prover_simulation.py — Simulate a Groth16 prover for N-constraint circuits.

A real Groth16 prover does:
  1. Witness computation (not modelled here)
  2. ~3 MSM calls on vectors of length ≈ N  (~65% of total time)
  3. ~2 NTT calls on vectors of length ≈ N  (~22% of total time)
  4. Miscellaneous (hashing, I/O, etc.)     (~13%)

This script measures (2) and (3) using our implementations and reports:
  - Per-operation timing
  - Total estimated prover time
  - CPU vs GPU comparison

Usage:
    python demo/prover_simulation.py
"""

from __future__ import annotations

import random
import sys
import time
import os

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.curve import G, P as CURVE_P
from src.msm import msm_naive
from src.ntt import ntt as ntt_cpu, MOD as NTT_MOD

try:
    import cupy as cp
    from src.ntt_gpu import ntt_gpu, _gpu_available as _ntt_gpu_ok
    from src.msm_gpu import msm_gpu, _gpu_available as _msm_gpu_ok
    _gpu_available = _ntt_gpu_ok and _msm_gpu_ok
except ImportError:
    _gpu_available = False


def _generate_ntt_input(n: int, seed: int = 0) -> list[int]:
    rng = random.Random(seed)
    return [rng.randint(0, NTT_MOD - 1) for _ in range(n)]


def _generate_msm_input(n: int, seed: int = 0):
    rng = random.Random(seed)
    scalars = [rng.randint(1, 2**32) for _ in range(n)]
    points = [G * rng.randint(1, 2**32) for _ in range(n)]
    return points, scalars


def simulate_prover(n_constraints: int, use_gpu: bool = False) -> dict:
    """
    Simulate a Groth16 prover for a circuit with *n_constraints* constraints.

    Returns
    -------
    dict with keys:
        n_constraints : int
        msm_time_ms   : float  — time for 3 MSM calls
        ntt_time_ms   : float  — time for 2 NTT calls
        total_time_ms : float  — sum (MSM + NTT accounts for ~87% of real prover)
        backend       : "CPU" | "GPU"
    """
    backend = "GPU" if (use_gpu and _gpu_available) else "CPU"

    # --- NTT timing (2 calls, as in PLONK/Groth16 polynomial computations) ---
    # Use next power of 2 >= n_constraints
    ntt_n = 1
    while ntt_n < n_constraints:
        ntt_n <<= 1

    ntt_input = _generate_ntt_input(ntt_n)

    ntt_total = 0.0
    for _ in range(2):
        if backend == "GPU":
            a_cp = cp.array(ntt_input, dtype=cp.int64)
            t0 = time.perf_counter()
            ntt_gpu(a_cp, NTT_MOD)
            ntt_total += time.perf_counter() - t0
        else:
            t0 = time.perf_counter()
            ntt_cpu(ntt_input, NTT_MOD)
            ntt_total += time.perf_counter() - t0

    ntt_ms = ntt_total * 1000

    # --- MSM timing (3 calls of length n_constraints each) ---
    points, scalars = _generate_msm_input(n_constraints)

    msm_total = 0.0
    for _ in range(3):
        if backend == "GPU":
            px = np.array([p.x for p in points], dtype=object)
            py = np.array([p.y for p in points], dtype=object)
            sc = np.array(scalars, dtype=np.uint64)
            t0 = time.perf_counter()
            msm_gpu(px, py, sc, CURVE_P)
            msm_total += time.perf_counter() - t0
        else:
            t0 = time.perf_counter()
            msm_naive(points, scalars)
            msm_total += time.perf_counter() - t0

    msm_ms = msm_total * 1000

    total_ms = ntt_ms + msm_ms

    return {
        "n_constraints": n_constraints,
        "msm_time_ms": round(msm_ms, 2),
        "ntt_time_ms": round(ntt_ms, 2),
        "total_time_ms": round(total_ms, 2),
        "backend": backend,
    }


def main() -> None:
    print("=" * 68)
    print("Groth16 Prover Simulation (MSM + NTT timing breakdown)")
    print("=" * 68)
    if not _gpu_available:
        print("[NOTE] GPU not detected — showing CPU-only results.\n")

    sizes = [1000, 10_000, 100_000]

    header = f"{'N':>8}  {'Backend':>6}  {'MSM ms':>10}  {'NTT ms':>10}  {'Total ms':>10}"
    print(header)
    print("-" * len(header))

    for n in sizes:
        cpu_r = simulate_prover(n, use_gpu=False)
        print(
            f"{n:>8}  {'CPU':>6}  {cpu_r['msm_time_ms']:>10.1f}  "
            f"{cpu_r['ntt_time_ms']:>10.1f}  {cpu_r['total_time_ms']:>10.1f}"
        )
        if _gpu_available:
            gpu_r = simulate_prover(n, use_gpu=True)
            speedup = cpu_r["total_time_ms"] / gpu_r["total_time_ms"]
            print(
                f"{n:>8}  {'GPU':>6}  {gpu_r['msm_time_ms']:>10.1f}  "
                f"{gpu_r['ntt_time_ms']:>10.1f}  {gpu_r['total_time_ms']:>10.1f}"
                f"   ← {speedup:.1f}x speedup"
            )
        print()

    print("=" * 68)
    print("Notes:")
    print("  MSM: 3 calls × N points (dominant cost ~65% of real prover)")
    print("  NTT: 2 calls × N elements (~22% of real prover)")
    print("  Remaining ~13% (hashing, I/O) not modelled here.")


if __name__ == "__main__":
    main()
