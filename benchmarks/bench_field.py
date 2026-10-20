"""
benchmarks/bench_field.py — Element-wise modular multiplication: NumPy vs GPU.

Implementations
---------------
numpy    (a * b) % mod on host int64 arrays — the fair CPU baseline
gpu      zkgpu.field_gpu.gpu_batch_mul on device-resident arrays
gpu_e2e  the same, starting from host arrays and copying the result back

The modulus 998244353 keeps every product inside int64, so NumPy computes
exact results and the GPU uses its fast int64 path.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from benchmarks._harness import check_equal, time_ms
from zkgpu.field_gpu import gpu_batch_mul
from zkgpu.gpu_backend import GPU_AVAILABLE, cp

MOD = 998_244_353


def _measure(n: int, budget: float) -> dict[str, Any]:
    """One row: every implementation at size n, each verified against NumPy."""
    rng = np.random.default_rng(n)
    a = rng.integers(0, MOD, size=n, dtype=np.int64)
    b = rng.integers(0, MOD, size=n, dtype=np.int64)

    ms: dict[str, float] = {}
    ms["numpy"], expected = time_ms(lambda: a * b % MOD, budget_s=budget)

    if GPU_AVAILABLE:
        a_dev, b_dev = cp.asarray(a), cp.asarray(b)
        ms["gpu"], got = time_ms(lambda: gpu_batch_mul(a_dev, b_dev, MOD), budget_s=budget, gpu=True)
        check_equal("gpu", got.get().tolist(), expected.tolist())
        ms["gpu_e2e"], got = time_ms(
            lambda: gpu_batch_mul(cp.asarray(a), cp.asarray(b), MOD).get(), budget_s=budget, gpu=True
        )
        check_equal("gpu_e2e", got.tolist(), expected.tolist())
    return {"n": n, "mod": MOD, "verified": True, "ms": ms}


def run_bench(*, quick: bool) -> list[dict[str, Any]]:
    sizes = [1_000, 10_000] if quick else [10_000, 100_000, 1_000_000, 10_000_000]
    rows = []
    for n in sizes:
        rows.append(_measure(n, budget=0.05 if quick else 1.0))
        timings = "  ".join(f"{k} {v:.3f} ms" for k, v in rows[-1]["ms"].items())
        print(f"  field  n={n:>10,}  {timings}", flush=True)
    return rows
