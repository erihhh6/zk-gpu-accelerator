"""
benchmarks/plot_results.py — Charts from a benchmark JSON file.

Usage:
    python benchmarks/plot_results.py                          # results/all_results.json
    python benchmarks/plot_results.py --input results/x.json   # charts go next to the input

Produces, each in a light and a "_dark" variant (for GitHub's <picture> switch):
  latency_comparison.png  time per implementation vs size: field
                          multiplication and NTT over a 64-bit prime (left)
                          and BN254's 254-bit Fr (right), and the MSM
  speedup_curve.png       what the GPU buys over the best CPU code running the
                          same algorithm (NumPy for 64-bit, Python ints for
                          254-bit, CPU Pippenger for MSM)

Design: colour follows the implementation and never its rank; every series also
has its own marker shape (a second channel for colour-vision deficiency and for
the lighter hues' low contrast on the light surface); one y-axis per panel;
solid hairline grid. The exact numbers are in the JSON and the README tables.
Palette: the dataviz reference instance, validated light and dark.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # file output only; no display needed
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.ticker import (  # noqa: E402
    FixedLocator,
    FuncFormatter,
    LogLocator,
    NullFormatter,
    NullLocator,
)

THEMES: dict[str, dict[str, Any]] = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink_secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"],
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "ink_secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"],
    },
}

# implementation -> (legend label, categorical slot, marker). Fixed, so a series
# keeps its colour whichever subset a run contains.
SERIES = {
    "gpu": ("GPU, data on device", 0, "o"),
    "numpy": ("NumPy (fair CPU baseline)", 1, "s"),
    "python_ints": ("Python ints on NumPy arrays (CPU baseline)", 1, "s"),
    "gpu_e2e": ("GPU incl. host-device copies", 2, "^"),
    "gpu_e2e_ints": ("GPU incl. Python int conversion and copies", 2, "^"),
    "python": ("Pure Python (reference only)", 3, "D"),
    "gpu_cold": ("GPU first call (builds caches)", 4, "v"),
    "pippenger_gpu": ("Pippenger, GPU", 0, "o"),
    "pippenger_cpu": ("Pippenger, CPU", 1, "s"),
    "straus": ("Straus / Shamir", 2, "^"),
    "naive": ("Naive (one scalar mult per point)", 3, "D"),
}

# (benchmark, title, size unit, grid slot). Left column: the 64-bit prime;
# right column: BN254's 254-bit fields; bottom-left stays empty.
PANELS = [
    ("field", "Field multiplication, 64-bit (998244353)", "elements", 0),
    ("field256", "Field multiplication, 254-bit (BN254 Fr)", "elements", 1),
    ("ntt", "NTT, 64-bit (998244353)", "elements", 2),
    ("ntt256", "NTT, 254-bit (BN254 Fr)", "elements", 3),
    ("msm", "MSM, BN254 G1, 254-bit scalars", "points", 5),
]
GRID = (3, 2)
FIGSIZE = (11.5, 12.6)

# (benchmark, GPU implementation, CPU implementation of the same algorithm)
SPEEDUPS = [
    ("field", "gpu", "numpy"),
    ("field", "gpu_e2e", "numpy"),
    ("field256", "gpu", "python_ints"),
    ("field256", "gpu_e2e_ints", "python_ints"),
    ("ntt", "gpu", "numpy"),
    ("ntt", "gpu_e2e", "numpy"),
    ("ntt256", "gpu", "python_ints"),
    ("ntt256", "gpu_e2e_ints", "python_ints"),
    ("msm", "pippenger_gpu", "pippenger_cpu"),
]
SPEEDUP_TITLES = {
    "field": "64-bit field multiplication: GPU vs NumPy",
    "field256": "254-bit field multiplication: GPU vs Python ints",
    "ntt": "64-bit NTT: GPU vs NumPy",
    "ntt256": "254-bit NTT: GPU vs Python ints",
    "msm": "MSM Pippenger: GPU vs CPU",
}
SPEEDUP_TICKS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000)


def _panels(axes: Any) -> Iterator[tuple[Axes, str, str, str]]:
    """(axes, benchmark, title, unit) for each panel, hiding the grid slots no panel uses."""
    used = {slot for *_, slot in PANELS}
    for i, ax in enumerate(axes.flat):
        if i not in used:
            ax.set_visible(False)
    for bench, title, unit, slot in PANELS:
        yield axes.flat[slot], bench, title, unit


def _size_label(n: int, bench: str) -> str:
    if bench.startswith("ntt"):
        return f"$2^{{{n.bit_length() - 1}}}$"
    for div, suffix in ((1_000_000, "M"), (1_000, "K")):
        if n >= div:
            return f"{n // div}{suffix}"
    return str(n)


def _style_axes(ax: Axes, theme: dict[str, Any]) -> None:
    ax.set_facecolor(theme["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["axis"])
        ax.spines[side].set_linewidth(0.8)
    ax.grid(True, which="major", color=theme["grid"], linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(colors=theme["muted"], labelsize=8.5, length=0)
    ax.xaxis.label.set_color(theme["ink_secondary"])
    ax.yaxis.label.set_color(theme["ink_secondary"])


def _plot_series(ax: Axes, theme: dict[str, Any], impl: str, xs: list[int], ys: list[float]) -> None:
    label, slot, marker = SERIES[impl]
    ax.plot(
        xs,
        ys,
        color=theme["series"][slot],
        linewidth=1.5,
        solid_capstyle="round",
        solid_joinstyle="round",
        marker=marker,
        markersize=5.5,
        markeredgecolor=theme["surface"],  # surface ring keeps crossing markers legible
        markeredgewidth=1.2,
        label=label,
    )


def _set_size_ticks(ax: Axes, sizes: list[int], bench: str) -> None:
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([_size_label(n, bench) for n in sizes])
    ax.xaxis.set_minor_locator(NullLocator())


def _series_count(ax: Axes) -> int:
    # Count data series only: reference lines carry matplotlib's "_child" labels
    return sum(not line.get_label().startswith("_") for line in ax.get_lines())


def _legend_headroom(ax: Axes, entries: int) -> None:
    """Raise a log y-axis's top so an upper-left legend of *entries* rows clears the data."""
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * (hi / lo) ** (0.06 * entries))


def _legend(ax: Axes, theme: dict[str, Any]) -> None:
    if _series_count(ax) >= 2:  # one series needs no legend: the panel title names it
        leg = ax.legend(loc="upper left", frameon=False, fontsize=8, handlelength=1.6)
        for text in leg.get_texts():
            text.set_color(theme["ink_secondary"])


def _footnote(fig: Any, meta: dict[str, Any], theme: dict[str, Any]) -> None:
    gpu = meta["gpu"]["name"] if meta.get("gpu") else "no GPU"
    commit = (meta.get("git_commit") or "unknown")[:7]
    fig.text(
        0.01,
        0.005,
        f"{gpu} · {meta.get('cpu', 'unknown CPU')} · commit {commit} · median wall-clock times",
        color=theme["muted"],
        fontsize=7.5,
        ha="left",
        va="bottom",
    )


def _no_data(ax: Axes, theme: dict[str, Any], message: str) -> None:
    style = {"ha": "center", "va": "center", "color": theme["muted"], "fontsize": 9}
    ax.text(0.5, 0.5, message, transform=ax.transAxes, **style)
    ax.set_xticks([])
    ax.set_yticks([])


def _coincide(rows: list[dict[str, Any]], a: str, b: str, tolerance: float = 0.05) -> bool:
    """True if implementations a and b are within *tolerance* of each other at every size."""
    pairs = [(row["ms"].get(a), row["ms"].get(b)) for row in rows]
    return bool(pairs) and all(x and y and abs(x - y) <= tolerance * max(x, y) for x, y in pairs)


def plot_latency(results: dict[str, Any], theme: dict[str, Any]) -> Any:
    fig, axes = plt.subplots(*GRID, figsize=FIGSIZE)
    fig.patch.set_facecolor(theme["surface"])
    fig.suptitle("Time per call, by implementation", color=theme["ink"], fontsize=13, y=0.99)
    for ax, bench, title, unit in _panels(axes):
        _style_axes(ax, theme)
        rows = results.get(bench, [])
        if not rows:  # e.g. a results file from before this benchmark existed
            ax.set_title(title, fontsize=10.5, loc="left", color=theme["ink"])
            _no_data(ax, theme, "Not measured in this run")
            continue
        sizes = [row["n"] for row in rows]
        present = {k for row in rows for k in row["ms"]}
        for impl in (s for s in SERIES if s in present):  # fixed order: legends match across panels
            pts = [(row["n"], row["ms"][impl]) for row in rows if row["ms"].get(impl)]
            if pts:
                _plot_series(ax, theme, impl, [p[0] for p in pts], [p[1] for p in pts])
        if bench == "msm" and _coincide(rows, "pippenger_cpu", "pippenger_gpu"):
            # The finding is that these two overlap; say so, since the top line hides the other
            ax.text(
                0.98,
                0.04,
                "Pippenger CPU and GPU lines coincide",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                color=theme["muted"],
            )
        if sizes:
            _set_size_ticks(ax, sizes, bench)
        ax.set_yscale("log")
        if _series_count(ax) >= 2:
            _legend_headroom(ax, _series_count(ax))
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        # Colour passed here: set_title() creates the text, so styling it earlier has no effect
        ax.set_title(title, fontsize=10.5, loc="left", color=theme["ink"])
        ax.set_xlabel(f"Input size ({unit})", fontsize=9)
        ax.set_ylabel("Milliseconds (log scale)", fontsize=9)
        _legend(ax, theme)
    _footnote(fig, results["meta"], theme)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    return fig


def plot_speedup(results: dict[str, Any], theme: dict[str, Any]) -> Any:
    fig, axes = plt.subplots(*GRID, figsize=FIGSIZE)
    fig.patch.set_facecolor(theme["surface"])
    fig.suptitle(
        "GPU speedup over CPU code running the same algorithm", color=theme["ink"], fontsize=13, y=0.99
    )
    for ax, bench, _latency_title, unit in _panels(axes):
        _style_axes(ax, theme)
        rows = results.get(bench, [])
        plotted = []
        for b, gpu_impl, cpu_impl in SPEEDUPS:
            if b != bench:
                continue
            pts = [
                (row["n"], row["ms"][cpu_impl] / row["ms"][gpu_impl])
                for row in rows
                if row["ms"].get(gpu_impl) and row["ms"].get(cpu_impl)
            ]
            if pts:
                _plot_series(ax, theme, gpu_impl, [p[0] for p in pts], [p[1] for p in pts])
                plotted.append(pts)
        ax.set_title(SPEEDUP_TITLES[bench], fontsize=10.5, loc="left", color=theme["ink"])
        if not plotted:
            _no_data(ax, theme, "No GPU timings in this run" if rows else "Not measured in this run")
            continue
        _set_size_ticks(ax, [row["n"] for row in rows], bench)
        ax.set_yscale("log")
        ys = [y for pts in plotted for _, y in pts]
        # Always span at least 0.5x..2x, so a flat ~1x series still gets readable ticks
        lo, hi = min([*ys, 0.5]) / 1.4, max([*ys, 2.0]) * 1.4
        ax.set_ylim(lo, hi)
        if len(plotted) >= 2:
            _legend_headroom(ax, len(plotted))
            hi = ax.get_ylim()[1]
        ax.yaxis.set_major_locator(FixedLocator([t for t in SPEEDUP_TICKS if lo <= t <= hi]))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}x"))
        # 1x reference: a solid hairline in the axis colour, never a dashed rule
        ax.axhline(1.0, color=theme["axis"], linewidth=1.0, zorder=1)
        ax.set_xlabel(f"Input size ({unit})", fontsize=9)
        ax.set_ylabel("Speedup (log scale)", fontsize=9)
        _direct_label_endpoints(ax, theme, plotted)
        _legend(ax, theme)
    _footnote(fig, results["meta"], theme)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    return fig


def _direct_label_endpoints(ax: Axes, theme: dict[str, Any], plotted: list[list[tuple[int, float]]]) -> None:
    """Value at each line's end, unless two ends sit too close to label without colliding."""
    ends = [pts[-1] for pts in plotted]
    logs = sorted(math.log10(y) for _, y in ends)
    if any(b - a < 0.12 for a, b in zip(logs, logs[1:], strict=False)):
        return  # converging ends: the legend carries identity instead of stacked labels
    for x, y in ends:
        ax.annotate(
            f"{y:.1f}x",
            (x, y),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=theme["ink_secondary"],
            # A ~1x end sits on the 1x reference line; the backing keeps the line out of the text
            bbox={"boxstyle": "square,pad=0.15", "facecolor": theme["surface"], "edgecolor": "none"},
        )


def main(input_path: Path) -> list[Path]:
    if not input_path.exists():
        print(f"ERROR: {input_path} not found. Run `python -m benchmarks.run_all` first.")
        sys.exit(1)
    results = json.loads(input_path.read_text(encoding="utf-8"))
    out_dir = input_path.parent
    written = []
    for theme_name, theme in THEMES.items():
        suffix = "" if theme_name == "light" else "_dark"
        for stem, make in (("latency_comparison", plot_latency), ("speedup_curve", plot_speedup)):
            fig = make(results, theme)
            path = out_dir / f"{stem}{suffix}.png"
            fig.savefig(path, dpi=150, facecolor=theme["surface"])
            plt.close(fig)
            written.append(path)
            print(f"  saved {path}")
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot benchmark results.")
    parser.add_argument("--input", type=Path, default=Path("results/all_results.json"))
    main(parser.parse_args().input)
