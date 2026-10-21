"""
benchmarks/run_all.py — Run every benchmark and save one JSON file with metadata.

Usage (from the repo root, with the package installed):
    python -m benchmarks.run_all                 # full run -> results/all_results.json
    python -m benchmarks.run_all --runs 3        # how the published results are made
    python -m benchmarks.run_all --quick         # seconds-long smoke run -> results/quick_results.json
    python -m benchmarks.run_all --output out.json

Every implementation's output is checked against a reference before its
timing is accepted, so a fast but wrong result fails the run instead of
appearing in the table. See benchmarks/_harness.py for the timing method.

With --runs N the whole suite runs N times, one complete pass after another,
and each published time is the median across the passes, with every pass's
value kept in "ms_runs". The harness re-measures a single disturbed
measurement, but a disturbance that slows every sample of one measurement
alike (seen twice: 28.7 ms for a 1.9 ms NTT, 5.5 ms for a 0.35 ms 254-bit
multiplication) looks like a steady result; with passes minutes apart, it
hits at most one of them.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from benchmarks import bench_field, bench_field256, bench_msm, bench_ntt, bench_ntt256
from benchmarks._harness import REPO_ROOT, collect_metadata

FULL_OUTPUT = REPO_ROOT / "results" / "all_results.json"
QUICK_OUTPUT = REPO_ROOT / "results" / "quick_results.json"  # gitignored: never mistaken for real data

# For each benchmark: the fair CPU baseline, and the GPU timing a caller actually experiences
BASELINE = {
    "field": "numpy",
    "field256": "python_ints",
    "ntt": "numpy",
    "ntt256": "python_ints",
    "msm": "naive",
}
BENCHMARKS = (
    ("field", bench_field),
    ("field256", bench_field256),
    ("ntt", bench_ntt),
    ("ntt256", bench_ntt256),
    ("msm", bench_msm),
)


def merge_runs(passes: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """
    One benchmark's rows from several passes -> rows of per-implementation median times.

    Every pass measures the same sizes and implementations; each pass's own
    value is kept under "ms_runs" when there is more than one.
    """
    merged = []
    for rows in zip(*passes, strict=True):
        row = dict(rows[0])
        times = {impl: [r["ms"][impl] for r in rows] for impl in rows[0]["ms"]}
        row["ms"] = {impl: statistics.median(values) for impl, values in times.items()}
        row["verified"] = all(r["verified"] for r in rows)
        if len(rows) > 1:
            row["ms_runs"] = times
        merged.append(row)
    return merged


def run(*, quick: bool, output: Path, runs: int = 1) -> dict[str, Any]:
    if runs < 1:
        raise ValueError(f"runs must be at least 1, got {runs}")
    results: dict[str, Any] = {"meta": collect_metadata(quick=quick, runs=runs)}
    passes: dict[str, list[list[dict[str, Any]]]] = {name: [] for name, _ in BENCHMARKS}
    for run_index in range(runs):
        for name, module in BENCHMARKS:
            t0 = time.perf_counter()
            print(f"\n>>> {name}" + (f"  (pass {run_index + 1} of {runs})" if runs > 1 else ""))
            passes[name].append(module.run_bench(quick=quick))
            print(f"    done in {time.perf_counter() - t0:.1f} s")
    for name, _ in BENCHMARKS:
        results[name] = merge_runs(passes[name])

    output.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n": the same bytes on every OS, matching .gitattributes (eol=lf)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8", newline="\n")
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
    for name, _ in BENCHMARKS:
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
    parser.add_argument(
        "--runs", type=int, default=1, help="complete passes; each time is the median across them (default 1)"
    )
    args = parser.parse_args(argv)
    output = args.output or (QUICK_OUTPUT if args.quick else FULL_OUTPUT)
    run(quick=args.quick, output=output, runs=args.runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
