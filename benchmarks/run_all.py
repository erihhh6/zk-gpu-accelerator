"""
benchmarks/run_all.py — Run all benchmarks sequentially and save a combined JSON.

Usage:
    python benchmarks/run_all.py

Output:
    results/all_results.json  — combined benchmark data
    A summary table is printed to stdout.
"""

from __future__ import annotations

import json
import os
import sys
import time

# Ensure project root is on the path when invoked from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.bench_field import run_bench as bench_field
from benchmarks.bench_ntt import run_bench as bench_ntt
from benchmarks.bench_msm import run_bench as bench_msm


def _fmt(val: float | None, suffix: str = "") -> str:
    if val is None:
        return "N/A"
    return f"{val:.1f}{suffix}"


def print_summary(all_results: dict) -> None:
    print("\n" + "=" * 72)
    print("BENCHMARK SUMMARY")
    print("=" * 72)

    # Field
    print("\n[Field Arithmetic — modular multiplication]")
    print(f"{'N':>12}  {'CPU ms':>10}  {'GPU ms':>10}  {'Speedup':>8}")
    for r in all_results["field"]:
        print(
            f"{r['n']:>12,}  {_fmt(r['cpu_ms'], ' ms'):>10}  "
            f"{_fmt(r['gpu_ms'], ' ms'):>10}  {_fmt(r['speedup'], 'x'):>8}"
        )

    # NTT
    print("\n[NTT — Cooley-Tukey, MOD=998244353]")
    print(f"{'N':>10}  {'CPU ms':>10}  {'GPU ms':>10}  {'Speedup':>8}")
    for r in all_results["ntt"]:
        print(
            f"{r['n']:>10}  {_fmt(r['cpu_ms'], ' ms'):>10}  "
            f"{_fmt(r['gpu_ms'], ' ms'):>10}  {_fmt(r['speedup'], 'x'):>8}"
        )

    # MSM
    print("\n[MSM — double-and-add CPU / parallel GPU, BN254]")
    print(f"{'N':>8}  {'CPU ms':>10}  {'GPU ms':>10}  {'Speedup':>8}")
    for r in all_results["msm"]:
        print(
            f"{r['n']:>8}  {_fmt(r['cpu_ms'], ' ms'):>10}  "
            f"{_fmt(r['gpu_ms'], ' ms'):>10}  {_fmt(r['speedup'], 'x'):>8}"
        )

    print("\n" + "=" * 72)


def main() -> None:
    os.makedirs("results", exist_ok=True)
    all_results: dict = {}

    print("\n>>> Running field benchmark...")
    t0 = time.perf_counter()
    all_results["field"] = bench_field()
    print(f"    Done in {time.perf_counter() - t0:.1f}s")

    print("\n>>> Running NTT benchmark...")
    t0 = time.perf_counter()
    all_results["ntt"] = bench_ntt()
    print(f"    Done in {time.perf_counter() - t0:.1f}s")

    print("\n>>> Running MSM benchmark...")
    t0 = time.perf_counter()
    all_results["msm"] = bench_msm()
    print(f"    Done in {time.perf_counter() - t0:.1f}s")

    out_path = os.path.join("results", "all_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print_summary(all_results)
    print(f"\nAll results saved to {out_path}")


if __name__ == "__main__":
    main()
