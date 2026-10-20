"""
benchmarks/bench_ntt.py — NTT: pure Python, vectorised NumPy and GPU.

Implementations
---------------
python    zkgpu.ntt.ntt, one butterfly per interpreted loop iteration. Shown
          for scale only: it is NOT a fair CPU baseline
numpy     the same radix-2 algorithm as the GPU version, vectorised with
          NumPy (one array operation per butterfly stage): the fair CPU baseline
          (ntt_numpy; bench_ntt256 runs it over BN254's r too).
          Like the GPU version it caches its bit-reversal index and twiddle
          table per size, so warm timings on both sides cover the transform only
gpu       zkgpu.ntt_gpu.ntt_gpu on device-resident data, caches warm
gpu_e2e   the same, starting from host data and copying the result back
gpu_cold  the first call for a size: includes building the cached twiddle
          table and bit-reversal index (kernels are already compiled)
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from benchmarks._harness import check_equal, time_ms
from zkgpu import ntt_gpu as ntt_gpu_module
from zkgpu.gpu_backend import GPU_AVAILABLE, cp
from zkgpu.ntt import get_ntt_root, ntt
from zkgpu.ntt_gpu import ntt_gpu

MOD = 998_244_353  # (MOD - 1)^2 < 2^63, so NumPy int64 products are exact
_INT64_PRODUCT_SAFE_MOD = 3_037_000_500  # largest mod whose residue products fit int64


def _dtype(mod: int) -> Any:
    """int64 where residue products fit, else exact Python ints in object arrays (e.g. BN254's r)."""
    return np.int64 if mod <= _INT64_PRODUCT_SAFE_MOD else object


@lru_cache(maxsize=8)
def _bit_reverse_index(n: int) -> np.ndarray:
    bits = n.bit_length() - 1
    idx, rev = np.arange(n, dtype=np.int64), np.zeros(n, dtype=np.int64)
    for _ in range(bits):
        rev = (rev << 1) | (idx & 1)
        idx >>= 1
    rev.flags.writeable = False  # shared between calls: a stray write would corrupt later ones
    return rev


@lru_cache(maxsize=8)
def _twiddle_table(n: int, mod: int) -> np.ndarray:
    """[ω^0, ..., ω^(n/2 - 1)], built by doubling."""
    omega = get_ntt_root(n, mod)
    table = np.ones(1, dtype=_dtype(mod))
    while table.size < n // 2:
        table = np.concatenate([table, table * pow(omega, table.size, mod) % mod])
    table.flags.writeable = False
    return table


def prepare_ntt_numpy(n: int, mod: int) -> None:
    """Build ntt_numpy's per-size caches, so a timing without warm-up covers the transform only."""
    _bit_reverse_index(n)
    _twiddle_table(n, mod)


def ntt_numpy(a: np.ndarray, mod: int) -> np.ndarray:
    """
    Radix-2 Cooley-Tukey NTT with one vectorised NumPy operation per stage.

    Mirrors zkgpu.ntt_gpu (same roots, same twiddle layout, same per-size
    caches), so the CPU/GPU comparison measures the hardware rather than two
    different algorithms. Without the caches, rebuilding the index and table
    was 22-36% of every call, which the warm GPU timing never pays.

    Moduli whose products overflow int64 (BN254's r) run the same code on
    object arrays of Python ints: exact, and about 2x faster than the
    interpreted loop of zkgpu.ntt.ntt at small sizes (equal by 2^18).
    """
    n = len(a)
    arr = (np.asarray(a, dtype=_dtype(mod)) % mod)[_bit_reverse_index(n)]
    table = _twiddle_table(n, mod)

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
