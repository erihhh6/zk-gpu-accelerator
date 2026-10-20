"""
benchmarks/run_all.py — Run every benchmark and save one JSON file with metadata.

Usage (from the repo root, with the package installed):
    python -m benchmarks.run_all                 # full run -> results/all_results.json
    python -m benchmarks.run_all --quick         # seconds-long smoke run -> results/quick_results.json
    python -m benchmarks.run_all --output out.json

Every implementation's output is checked against a reference before its
timing is accepted, so a fast but wrong result fails the run instead of
appearing in the table. See benchmarks/_harness.py for the timing method.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from benchmarks import bench_field, bench_msm, bench_ntt
from benchmarks._harness import REPO_ROOT, collect_metadata

FULL_OUTPUT = REPO_ROOT / "results" / "all_results.json"
QUICK_OUTPUT = REPO_ROOT / "results" / "quick_results.json"  # gitignored: never mistaken for real data

# For each benchmark: the fair CPU baseline, and the GPU timing a caller actually experiences
BASELINE = {"field": "numpy", "ntt": "numpy", "msm": "naive"}


def run(*, quick: bool, output: Path) -> dict[str, Any]:
    results: dict[str, Any] = {"meta": collect_metadata(quick=quick)}
    for name, module in (("field", bench_field), ("ntt", bench_ntt), ("msm", bench_msm)):
        t0 = time.perf_counter()
        print(f"\n>>> {name}")
        results[name] = module.run_bench(quick=quick)
        print(f"    done in {time.perf_counter() - t0:.1f} s")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print_summary(results)
    print(f"\nResults saved to {output}")
    return results


def _fmt(ms: float | None) -> str:
    if ms is None:
        return "-"
    return f"{ms:.3f}" if ms < 10 else f"{ms:.1f}"


def print_summary(results: dict[str, Any]) -> None:
    meta = results["meta"]
    gpu = meta["gpu"]["name"] if meta["gpu"] else f"none ({meta['gpu_unavailable_reason']})"
    print("\n" + "=" * 78)
    print(f"SUMMARY  (CPU: {meta['cpu']}, GPU: {gpu})")
    for name in ("field", "ntt", "msm"):
        rows = results[name]
        impls = list(dict.fromkeys(k for row in rows for k in row["ms"]))
        base = BASELINE[name]
        print(f"\n[{name}]  times in ms; 'x' columns = speedup over {base}")
        print(f"{'n':>10}  " + "  ".join(f"{i:>13}" for i in impls))
        for row in rows:
            cells = []
            for impl in impls:
                ms = row["ms"].get(impl)
                ref = row["ms"].get(base)
                speedup = f" {ref / ms:5.1f}x" if ms and ref and impl != base else ""
                cells.append(f"{_fmt(ms) + speedup:>13}")
            print(f"{row['n']:>10}  " + "  ".join(cells))
    print("=" * 78)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--quick", action="store_true", help="tiny sizes and time budgets (smoke test)")
    parser.add_argument("--output", type=Path, help="where to write the JSON results")
    args = parser.parse_args(argv)
    output = args.output or (QUICK_OUTPUT if args.quick else FULL_OUTPUT)
    run(quick=args.quick, output=output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
