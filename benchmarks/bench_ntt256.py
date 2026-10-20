"""
benchmarks/bench_ntt256.py — NTT over BN254's scalar field r (254-bit): Python ints vs GPU.

Implementations
---------------
python        zkgpu.ntt.ntt, the interpreted loop behind Radix2Domain.fft: the
              package's own CPU path for BN254, shown as the reference result
python_ints   bench_ntt.ntt_numpy on object arrays of Python ints: the same
              stage-vectorised algorithm as the 64-bit NumPy baseline, and the
              fastest pure-Python option, so it is the CPU baseline
gpu           zkgpu.field256_gpu.BN254_FR.ntt on device-resident Montgomery
              rows, twiddle table cached
gpu_cold      the first call for a size: includes building the twiddle table
gpu_e2e_ints  Python ints in, Python ints out: from_ints, ntt, to_ints

As in bench_field256, interpreted big integers are a weak baseline, so the
ratios overstate the gain over an optimised native CPU library.

The CPU columns stop at CPU_MAX (2^20 would take about 10 s per call). The
GPU-only rows above it are verified another way: a round trip through the
inverse transform, plus outputs recomputed straight from the definition
NTT(a)[k] = a(omega^k), which a round trip alone cannot replace (a wrong root
of unity still inverts correctly).
"""

from __future__ import annotations

import random
from typing import Any

from benchmarks._harness import check_equal, time_ms
from benchmarks.bench_ntt import ntt_numpy, prepare_ntt_numpy
from zkgpu import field256_gpu
from zkgpu.field256_gpu import BN254_FR
from zkgpu.gpu_backend import GPU_AVAILABLE
from zkgpu.ntt import get_ntt_root, ntt

R = BN254_FR.modulus
CPU_MAX = 2**18
SPOT_CHECKS = 3


def spot_check(coeffs: list[int], evals: list[int], indices: list[int]) -> None:
    """Fail the run unless evals[k] = sum(coeffs[i] * omega^(i k)) for each k, by Horner's rule."""
    omega = get_ntt_root(len(coeffs), R)
    for k in indices:
        x, acc = pow(omega, k, R), 0
        for c in reversed(coeffs):
            acc = (acc * x + c) % R
        check_equal(f"NTT output {k}", evals[k], acc)


def _measure(n: int, budget: float) -> dict[str, Any]:
    """One row: every implementation at size n, each verified (see the module docstring)."""
    rng = random.Random(n)
    a = [rng.randrange(R) for _ in range(n)]

    ms: dict[str, float] = {}
    expected: list[int] | None = None
    if n <= CPU_MAX:
        # Slow, single-shot reference: no warm-up, as for the 64-bit pure-Python NTT
        ms["python"], expected = time_ms(lambda: ntt(a, R), budget_s=0, max_repeat=1, warmup=False)
        prepare_ntt_numpy(n, R)  # the baseline's caches, like the GPU's, are built outside the timing
        ms["python_ints"], got = time_ms(lambda: ntt_numpy(a, R), budget_s=budget, max_repeat=3, warmup=False)
        check_equal("python_ints", got.tolist(), expected)

    if GPU_AVAILABLE:
        rows = BN254_FR.from_ints(a)
        BN254_FR.ntt(BN254_FR.from_ints([0, 0]))  # compile the kernels outside every timing

        def cold_call() -> Any:
            field256_gpu._ntt_tables.cache_clear()  # every sample pays the table build, as a first call does
            return BN254_FR.ntt(rows)

        ms["gpu_cold"], cold = time_ms(cold_call, budget_s=60, max_repeat=3, warmup=False, gpu=True)
        ms["gpu"], got = time_ms(lambda: BN254_FR.ntt(rows), budget_s=budget, gpu=True)
        evals = BN254_FR.to_ints(got)
        if expected is not None:
            check_equal("gpu", evals, expected)
        else:
            check_equal("gpu round trip", BN254_FR.to_ints(BN254_FR.ntt(got, invert=True)), a)
            spot_check(a, evals, [1, n - 1, *random.Random(-n).sample(range(n), SPOT_CHECKS - 2)])
        check_equal("gpu_cold", BN254_FR.to_ints(cold), evals)
        ms["gpu_e2e_ints"], got = time_ms(
            lambda: BN254_FR.to_ints(BN254_FR.ntt(BN254_FR.from_ints(a))),
            budget_s=budget,
            max_repeat=3,
            gpu=True,
        )
        check_equal("gpu_e2e_ints", got, evals)
    return {"n": n, "field": "BN254 Fr", "verified": True, "ms": ms}


def run_bench(*, quick: bool) -> list[dict[str, Any]]:
    sizes = [2**6, 2**8] if quick else [2**12, 2**14, 2**16, 2**18, 2**20]
    rows = []
    for n in sizes:
        rows.append(_measure(n, budget=0.05 if quick else 1.0))
        timings = "  ".join(f"{k} {v:.3f} ms" for k, v in rows[-1]["ms"].items())
        print(f"  ntt256 n=2^{n.bit_length() - 1:<7} {timings}", flush=True)
    return rows
