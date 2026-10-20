"""
benchmarks/bench_field256.py — Element-wise 254-bit modular multiplication over BN254 Fr.

Implementations
---------------
python_ints  a * b % r on NumPy object arrays of Python ints: the fastest
             CPU path in pure Python (zkgpu.field.batch_mul's list
             comprehension computes the same thing about 5% slower)
gpu          zkgpu.field256_gpu.BN254_FR.mul on device-resident rows
gpu_e2e_ints Python ints in, Python ints out: from_ints, mul, to_ints

Python ints are the only CPU implementation of 254-bit arithmetic in this
package, and an interpreted one: each product allocates new integer objects,
which native fixed-width Montgomery code never does. The python_ints ratio
therefore overstates what the GPU gains over an optimised CPU library.

gpu_e2e_ints is dominated by converting between Python ints and limbs
(about 150 ns per element each way in CPython), not by the GPU or the copies.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from benchmarks._harness import check_equal, time_ms
from zkgpu.field256_gpu import BN254_FR
from zkgpu.gpu_backend import GPU_AVAILABLE

R = BN254_FR.modulus


def _measure(n: int, budget: float) -> dict[str, Any]:
    """One row: every implementation at size n, each verified against Python ints."""
    rng = random.Random(n)
    a, b = [rng.randrange(R) for _ in range(n)], [rng.randrange(R) for _ in range(n)]
    a_obj, b_obj = np.array(a, dtype=object), np.array(b, dtype=object)

    ms: dict[str, float] = {}
    ms["python_ints"], got = time_ms(lambda: a_obj * b_obj % R, budget_s=budget, max_repeat=5)
    expected = got.tolist()

    if GPU_AVAILABLE:
        a_dev, b_dev = BN254_FR.from_ints(a), BN254_FR.from_ints(b)  # also compiles the kernels
        ms["gpu"], got = time_ms(lambda: BN254_FR.mul(a_dev, b_dev), budget_s=budget, gpu=True)
        check_equal("gpu", BN254_FR.to_ints(got), expected)
        ms["gpu_e2e_ints"], got = time_ms(
            lambda: BN254_FR.to_ints(BN254_FR.mul(BN254_FR.from_ints(a), BN254_FR.from_ints(b))),
            budget_s=budget,
            max_repeat=5,
            gpu=True,
        )
        check_equal("gpu_e2e_ints", got, expected)
    return {"n": n, "field": "BN254 Fr", "verified": True, "ms": ms}


def run_bench(*, quick: bool) -> list[dict[str, Any]]:
    sizes = [1_000, 4_000] if quick else [10_000, 100_000, 1_000_000]
    rows = []
    for n in sizes:
        rows.append(_measure(n, budget=0.05 if quick else 1.0))
        timings = "  ".join(f"{k} {v:.3f} ms" for k, v in rows[-1]["ms"].items())
        print(f"  fr256  n={n:>10,}  {timings}", flush=True)
    return rows
