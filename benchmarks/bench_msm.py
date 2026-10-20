"""
benchmarks/bench_msm.py — BN254 G1 MSM algorithms on real 254-bit scalars.

Implementations (all point arithmetic is Python ints on the CPU)
------------------------------------------------------------------
naive          zkgpu.msm.msm_naive: one scalar multiplication per point
straus         zkgpu.msm.msm_double_and_add: shared accumulator (Straus/Shamir)
pippenger_cpu  zkgpu.msm_gpu.msm_pippenger_gpu with its CPU digit extraction
pippenger_gpu  the same with GPU digit extraction — the only GPU work in any MSM

The differences between rows are algorithmic; pippenger_cpu vs pippenger_gpu
isolates the GPU's contribution, which is negligible until the curve
arithmetic itself moves to CUDA.

Inputs: scalars uniform in [0, r); points P_i = P_0 + i·S for random P_0, S.
The progression is as good as independent random points for MSM cost (the
coordinates are pseudo-random, and all points are distinct) and about 50x
faster to generate at N = 10,000.
"""

from __future__ import annotations

import random
import warnings
from typing import Any

from benchmarks._harness import check_equal, forced_cpu_fallback, time_ms
from zkgpu import msm_gpu as msm_gpu_module
from zkgpu.curve import G, Point
from zkgpu.field import FIELD_MODULUS as R
from zkgpu.gpu_backend import GPU_AVAILABLE, GpuFallbackWarning
from zkgpu.msm import msm_double_and_add, msm_naive
from zkgpu.msm_gpu import msm_pippenger_gpu

SCALAR_BITS = R.bit_length()


def pippenger_window(n: int) -> int:
    """
    Window size minimising Pippenger's estimated point additions for n points.

    Per window: n additions to fill the buckets plus about 2·2^w for the
    running sums; ceil(254 / w) windows. The default w = 8 is far from optimal
    at small n, which would make Pippenger look worse than it is.
    """
    return min(range(1, 17), key=lambda w: -(-SCALAR_BITS // w) * (n + 2 ** (w + 1)))


def _inputs(n: int) -> tuple[list[Point], list[int]]:
    rng = random.Random(n)
    step = G * rng.randrange(1, R)
    points = [G * rng.randrange(1, R)]
    for _ in range(n - 1):
        points.append(points[-1] + step)
    return points, [rng.randrange(R) for _ in range(n)]


NAIVE_MAX = 1_000  # naive at 10,000 points costs over ten seconds and shows nothing new


def _measure(n: int, budget: float) -> dict[str, Any]:
    """One row: every algorithm at size n, each verified against Straus."""
    points, scalars = _inputs(n)
    xs, ys = [p.x for p in points], [p.y for p in points]
    w = pippenger_window(n)

    ms: dict[str, float] = {}
    # Pure-Python code has nothing to warm up, so skip the untimed warm-up run
    ms["straus"], expected = time_ms(
        lambda: msm_double_and_add(points, scalars), budget_s=budget, max_repeat=3, warmup=False
    )
    if n <= NAIVE_MAX:
        ms["naive"], got = time_ms(
            lambda: msm_naive(points, scalars), budget_s=budget, max_repeat=3, warmup=False
        )
        check_equal("naive", got, expected)

    with forced_cpu_fallback(msm_gpu_module), warnings.catch_warnings():
        warnings.simplefilter("ignore", GpuFallbackWarning)
        ms["pippenger_cpu"], got = time_ms(
            lambda: msm_pippenger_gpu(xs, ys, scalars, window_size=w),
            budget_s=budget,
            max_repeat=3,
            warmup=False,
        )
    check_equal("pippenger_cpu", got, expected)

    if GPU_AVAILABLE:
        msm_pippenger_gpu(xs[:2], ys[:2], scalars[:2], window_size=w)  # compile the digit kernel
        ms["pippenger_gpu"], got = time_ms(
            lambda: msm_pippenger_gpu(xs, ys, scalars, window_size=w),
            budget_s=budget,
            max_repeat=3,
            warmup=False,
            gpu=True,
        )
        check_equal("pippenger_gpu", got, expected)
    return {"n": n, "window_size": w, "verified": True, "ms": ms}


def run_bench(*, quick: bool) -> list[dict[str, Any]]:
    sizes = [8, 32] if quick else [100, 1_000, 10_000]
    rows = []
    for n in sizes:
        rows.append(_measure(n, budget=0.05 if quick else 1.0))
        timings = "  ".join(f"{k} {v:.1f} ms" for k, v in rows[-1]["ms"].items())
        print(f"  msm    n={n:>10,}  w={rows[-1]['window_size']:<2} {timings}", flush=True)
    return rows
