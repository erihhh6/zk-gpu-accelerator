"""
benchmarks/bench_ntt.py — NTT: pure Python, vectorised NumPy and GPU.

Implementations
---------------
python    zkgpu.ntt.ntt, one butterfly per interpreted loop iteration. Shown
          for scale only: it is NOT a fair CPU baseline
numpy     the same radix-2 algorithm as the GPU version, vectorised with
          NumPy (one array operation per butterfly stage): the fair CPU baseline
gpu       zkgpu.ntt_gpu.ntt_gpu on device-resident data, caches warm
gpu_e2e   the same, starting from host data and copying the result back
gpu_cold  the first call for a size: includes building the cached twiddle
          table and bit-reversal index (kernels are already compiled)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from benchmarks._harness import check_equal, time_ms
from zkgpu import ntt_gpu as ntt_gpu_module
from zkgpu.gpu_backend import GPU_AVAILABLE, cp
from zkgpu.ntt import get_ntt_root, ntt
from zkgpu.ntt_gpu import ntt_gpu

MOD = 998_244_353  # (MOD - 1)^2 < 2^63, so NumPy int64 products are exact


def ntt_numpy(a: np.ndarray, mod: int) -> np.ndarray:
    """
    Radix-2 Cooley-Tukey NTT with one vectorised NumPy operation per stage.

    Mirrors zkgpu.ntt_gpu (same roots, same twiddle layout), so the CPU/GPU
    comparison measures the hardware rather than two different algorithms.
    """
    if mod > 3_037_000_500:
        raise ValueError("int64 products overflow for this modulus")
    n = len(a)
    bits = n.bit_length() - 1
    idx, rev = np.arange(n, dtype=np.int64), np.zeros(n, dtype=np.int64)
    for _ in range(bits):
        rev = (rev << 1) | (idx & 1)
        idx >>= 1
    arr = (np.asarray(a, dtype=np.int64) % mod)[rev]

    omega = get_ntt_root(n, mod)
    table = np.ones(1, dtype=np.int64)  # [ω^0, ..., ω^(n/2 - 1)], built by doubling
    while table.size < n // 2:
        table = np.concatenate([table, table * pow(omega, table.size, mod) % mod])

    length = 2
    while length <= n:
        half = length >> 1
        wk = table[:: n // length]
        groups = arr.reshape(-1, length)
        upper, lower = groups[:, :half], groups[:, half:]
        v = lower * wk % mod
        new_upper, new_lower = (upper + v) % mod, (upper - v) % mod
        groups[:, :half], groups[:, half:] = new_upper, new_lower
        length <<= 1
    return arr


def _clear_gpu_caches() -> None:
    ntt_gpu_module._twiddle_table.cache_clear()
    ntt_gpu_module._bit_reverse_index.cache_clear()


def _measure(n: int, budget: float) -> dict[str, Any]:
    """One row: every implementation at size n, each verified against pure Python."""
    host = np.random.default_rng(n).integers(0, MOD, size=n, dtype=np.int64)
    as_list = host.tolist()

    ms: dict[str, float] = {}
    # Pure Python is the reference result; it is slow, so time it once without warm-up
    ms["python"], expected = time_ms(lambda: ntt(as_list, MOD), budget_s=0, max_repeat=1, warmup=False)
    ms["numpy"], got = time_ms(lambda: ntt_numpy(host, MOD), budget_s=budget)
    check_equal("numpy", got.tolist(), expected)

    if GPU_AVAILABLE:
        dev = cp.asarray(host)
        ntt_gpu(cp.arange(4, dtype=cp.int64), MOD)  # compile kernels outside every timing

        def cold_call() -> Any:
            _clear_gpu_caches()  # every sample pays the cache build, as a first call does
            return ntt_gpu(dev, MOD)

        # max_repeat, not the budget, bounds this: exactly three cold calls per attempt
        ms["gpu_cold"], got = time_ms(cold_call, budget_s=60, max_repeat=3, warmup=False, gpu=True)
        check_equal("gpu_cold", got.tolist(), expected)
        ms["gpu"], got = time_ms(lambda: ntt_gpu(dev, MOD), budget_s=budget, gpu=True)
        check_equal("gpu", got.tolist(), expected)
        ms["gpu_e2e"], got = time_ms(lambda: ntt_gpu(cp.asarray(host), MOD).get(), budget_s=budget, gpu=True)
        check_equal("gpu_e2e", got.tolist(), expected)
    return {"n": n, "mod": MOD, "verified": True, "ms": ms}


def run_bench(*, quick: bool) -> list[dict[str, Any]]:
    sizes = [2**6, 2**8] if quick else [2**14, 2**16, 2**18, 2**20]
    rows = []
    for n in sizes:
        rows.append(_measure(n, budget=0.05 if quick else 1.0))
        timings = "  ".join(f"{k} {v:.3f} ms" for k, v in rows[-1]["ms"].items())
        print(f"  ntt    n=2^{n.bit_length() - 1:<8} {timings}", flush=True)
    return rows
