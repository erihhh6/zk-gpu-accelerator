"""
benchmarks/plot_results.py — Generate benchmark plots from results/all_results.json.

Produces:
  results/speedup_curve.png       — GPU/CPU speedup vs N for NTT and MSM
  results/latency_comparison.png  — Bar chart of CPU vs GPU latency
  results/throughput.png          — Elements/second CPU vs GPU

Usage:
    python benchmarks/plot_results.py
    python benchmarks/plot_results.py --input results/all_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (works without display)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# ------------------------------------------------------------------
# Style helpers
# ------------------------------------------------------------------

CPU_COLOR = "#2196F3"   # blue
GPU_COLOR = "#FF5722"   # orange-red
STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "font.family": "DejaVu Sans",
    "font.size": 11,
}


def _apply_style() -> None:
    plt.rcParams.update(STYLE)


def _save(fig: plt.Figure, path: str) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ------------------------------------------------------------------
# Plot 1: Speedup curve
# ------------------------------------------------------------------


def plot_speedup(all_results: dict, out_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    _apply_style()
    fig.suptitle("GPU Speedup over CPU", fontsize=14, fontweight="bold", y=1.02)

    datasets = [
        ("ntt", "NTT (Cooley-Tukey)", axes[0]),
        ("msm", "MSM (naive vs Pippenger)", axes[1]),
    ]

    for key, title, ax in datasets:
        data = all_results.get(key, [])
        ns = [r["n"] for r in data]
        speedups = [r.get("speedup") for r in data]

        valid = [(n, s) for n, s in zip(ns, speedups) if s is not None]
        if not valid:
            ax.text(0.5, 0.5, "No GPU data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="gray")
            ax.set_title(title)
            continue

        ns_v, sp_v = zip(*valid)
        ax.plot(ns_v, sp_v, marker="o", color=GPU_COLOR, linewidth=2, markersize=8)
        ax.axhline(1.0, color="gray", linewidth=1, linestyle="--", alpha=0.6)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("N (log scale)")
        ax.set_ylabel("Speedup (GPU / CPU)")
        ax.set_title(title)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1fx"))
        for n, s in zip(ns_v, sp_v):
            ax.annotate(f"{s:.1f}x", xy=(n, s), xytext=(0, 8),
                        textcoords="offset points", ha="center", fontsize=9)

    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "speedup_curve.png"))


# ------------------------------------------------------------------
# Plot 2: Latency comparison (bar chart)
# ------------------------------------------------------------------


def plot_latency(all_results: dict, out_dir: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    _apply_style()
    fig.suptitle("CPU vs GPU Latency", fontsize=14, fontweight="bold", y=1.02)

    datasets = [
        ("field", "Field Arithmetic", "N (elements)", axes[0]),
        ("ntt",   "NTT",              "N (elements)", axes[1]),
        ("msm",   "MSM",              "N (points)",   axes[2]),
    ]

    for key, title, xlabel, ax in datasets:
        data = all_results.get(key, [])
        if not data:
            ax.set_title(title)
            continue

        ns = [r["n"] for r in data]
        cpu_ms = [r["cpu_ms"] for r in data]
        gpu_ms = [r.get("gpu_ms") for r in data]

        x = np.arange(len(ns))
        width = 0.35

        bars_cpu = ax.bar(x - width / 2, cpu_ms, width, label="CPU",
                          color=CPU_COLOR, alpha=0.85)
        has_gpu = any(v is not None for v in gpu_ms)
        if has_gpu:
            gpu_vals = [v if v is not None else 0 for v in gpu_ms]
            bars_gpu = ax.bar(x + width / 2, gpu_vals, width, label="GPU",
                              color=GPU_COLOR, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in ns], rotation=30, ha="right", fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Latency (ms)")
        ax.set_title(title)
        ax.legend()

    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "latency_comparison.png"))


# ------------------------------------------------------------------
# Plot 3: Throughput
# ------------------------------------------------------------------


def plot_throughput(all_results: dict, out_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    _apply_style()
    fig.suptitle("Throughput: Elements per Second", fontsize=14,
                 fontweight="bold", y=1.02)

    datasets_tp = [
        ("ntt", "NTT Throughput", "cpu_throughput_meps", "gpu_throughput_meps",
         "M elements/s", axes[0]),
        ("msm", "MSM Throughput", "cpu_throughput_keps", "gpu_throughput_keps",
         "K points/s",   axes[1]),
    ]

    for key, title, cpu_key, gpu_key, ylabel, ax in datasets_tp:
        data = all_results.get(key, [])
        if not data:
            ax.set_title(title)
            continue

        ns = [r["n"] for r in data]
        cpu_tp = [r.get(cpu_key, 0) or 0 for r in data]
        gpu_tp = [r.get(gpu_key) for r in data]

        x = np.arange(len(ns))
        width = 0.35

        ax.bar(x - width / 2, cpu_tp, width, label="CPU", color=CPU_COLOR, alpha=0.85)
        has_gpu = any(v is not None for v in gpu_tp)
        if has_gpu:
            gpu_vals = [v if v is not None else 0 for v in gpu_tp]
            ax.bar(x + width / 2, gpu_vals, width, label="GPU",
                   color=GPU_COLOR, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in ns], rotation=30, ha="right", fontsize=8)
        ax.set_xlabel("N")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()

    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "throughput.png"))


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------


def main(input_path: str = "results/all_results.json") -> None:
    if not os.path.exists(input_path):
        print(f"ERROR: {input_path} not found. Run benchmarks/run_all.py first.")
        sys.exit(1)

    with open(input_path) as f:
        all_results = json.load(f)

    out_dir = os.path.dirname(input_path) or "results"
    os.makedirs(out_dir, exist_ok=True)

    print("Generating plots...")
    plot_speedup(all_results, out_dir)
    plot_latency(all_results, out_dir)
    plot_throughput(all_results, out_dir)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot ZK GPU accelerator benchmark results.")
    parser.add_argument(
        "--input",
        default="results/all_results.json",
        help="Path to all_results.json (default: results/all_results.json)",
    )
    args = parser.parse_args()
    main(args.input)
