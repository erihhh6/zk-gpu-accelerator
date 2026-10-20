"""
src/ntt_gpu.py — Number Theoretic Transform on GPU (CuPy).

Implements the Cooley-Tukey NTT using CuPy's vectorized array operations.
Each butterfly stage runs on the GPU using CuPy indexing and broadcasting;
only CuPy and nvrtc are needed (src.gpu_backend explains how nvrtc is found).

Algorithm (per stage)
---------------------
For stage with half = length // 2, viewing the array as (n // length, length):
  1. wk[j] = ω_length^j for j < half. Because ω_length = ω_n^(n/length), this
     is a strided view of one table [ω_n^0, ..., ω_n^(n/2 - 1)]
  2. upper = arr[:, :half], lower = arr[:, half:]   (views, no copies)
  3. v = lower · wk mod p         (wk broadcasts across all groups)
  4. arr[:, :half] = upper + v mod p,  arr[:, half:] = upper - v mod p

The twiddle table and the bit-reversal index depend only on (n, mod, invert),
so they are built once on the device and cached: repeated transforms of the
same size, the normal pattern in a prover, reuse them.

Modular products and sums go through src.field_gpu, which is exact for any
modulus below 2^63. Roots come from src.ntt.get_ntt_root, so the GPU and CPU
transforms use identical roots and agree element for element.

Limitations
-----------
- Elements are int64, so mod must be an odd prime below 2^63 with n | mod - 1.
  BN254's r (~2^254) does not fit; use src.ntt.ntt for it.
"""

from __future__ import annotations

import functools

import numpy as np

from src.field_gpu import gpu_batch_add, gpu_batch_mul
from src.gpu_backend import GPU_AVAILABLE as _gpu_available, cp, warn_cpu_fallback
from src.ntt import get_ntt_root, ntt as ntt_cpu

_INT64_MAX = 2**63 - 1

# Each cached twiddle table holds n/2 int64 values (4 MiB at n = 2^20); a
# handful of (size, modulus, direction) combinations covers realistic use.
# Cached arrays are shared, so they must only ever be read.
_CACHE_SIZE = 16


# ------------------------------------------------------------------
# Cached per-size device data
# ------------------------------------------------------------------


@functools.lru_cache(maxsize=_CACHE_SIZE)
def _bit_reverse_index(n: int) -> "cp.ndarray":
    """Device array rev where rev[i] is i with its log2(n) bits reversed."""
    bits = n.bit_length() - 1
    idx = np.arange(n, dtype=np.int64)
    rev = np.zeros(n, dtype=np.int64)
    for _ in range(bits):
        rev = (rev << 1) | (idx & 1)
        idx >>= 1
    return cp.asarray(rev)


@functools.lru_cache(maxsize=_CACHE_SIZE)
def _twiddle_table(n: int, mod: int, invert: bool) -> "cp.ndarray":
    """Device array [ω^0, ω^1, ..., ω^(n/2 - 1)] for ω = ω_n, or ω_n^-1 when inverting."""
    omega = get_ntt_root(n, mod)
    if invert:
        omega = pow(omega, mod - 2, mod)
    # Doubling: if table = [ω^0 .. ω^(s-1)] then table · ω^s = [ω^s .. ω^(2s-1)],
    # so log2(n) kernel launches replace an n/2-step Python loop.
    table = cp.ones(1, dtype=cp.int64)
    while table.size < n // 2:
        step = cp.asarray(pow(omega, table.size, mod), dtype=cp.int64)
        table = cp.concatenate([table, gpu_batch_mul(table, step, mod)])
    return table


# ------------------------------------------------------------------
# Public GPU NTT
# ------------------------------------------------------------------


def ntt_gpu(
    a: "cp.ndarray",
    mod: int,
    invert: bool = False,
) -> "cp.ndarray":
    """
    Cooley-Tukey NTT on GPU using CuPy vectorized butterflies.

    Parameters
    ----------
    a      : cp.ndarray of int64, length a power of 2 (any values; they are
             reduced mod *mod* first)
    mod    : odd prime below 2^63 with len(a) dividing mod - 1
    invert : if True, compute the inverse NTT

    Returns
    -------
    cp.ndarray int64 of the same length with values in [0, mod)
    (np.ndarray on the CPU fallback)

    Raises
    ------
    ValueError : mod >= 2^63, or the same length/modulus errors as src.ntt.ntt
    """
    if mod > _INT64_MAX:
        raise ValueError(
            f"ntt_gpu works on int64 elements, so mod must be below 2^63, got {mod}; "
            "use src.ntt.ntt for larger moduli such as BN254's r"
        )
    if not _gpu_available:
        warn_cpu_fallback("ntt_gpu", "the CPU NTT")
        a_list = a.tolist() if hasattr(a, "tolist") else list(a)
        return np.array(ntt_cpu(a_list, mod, invert), dtype=np.int64)  # type: ignore[return-value]

    n = len(a)
    if n & (n - 1):  # lets n == 0 through
        raise ValueError(f"NTT length must be a power of 2, got {n}")
    if n == 0:
        return cp.empty(0, dtype=cp.int64)
    get_ntt_root(n, mod)  # validate mod and n up front, exactly as the CPU NTT does

    # Reduce once so every butterfly sees residues; CuPy's % floors like NumPy's
    arr = cp.asarray(a, dtype=cp.int64) % mod
    if n == 1:
        return arr

    arr = arr[_bit_reverse_index(n)]
    table = _twiddle_table(n, mod, invert)

    length = 2
    while length <= n:
        half = length >> 1
        wk = table[:: n // length]  # ω_length^j for j < half, as a strided view
        groups = arr.reshape(-1, length)
        upper, lower = groups[:, :half], groups[:, half:]

        v = gpu_batch_mul(lower, wk, mod)  # wk broadcasts across groups
        # Both halves are computed in full before either is overwritten
        new_upper = gpu_batch_add(upper, v, mod)
        new_lower = (upper - v) % mod  # upper - v lies in (-mod, mod): no overflow
        groups[:, :half] = new_upper
        groups[:, half:] = new_lower

        length <<= 1

    if invert:
        n_inv = cp.asarray(pow(n, mod - 2, mod), dtype=cp.int64)
        arr = gpu_batch_mul(arr, n_inv, mod)

    return arr


# ------------------------------------------------------------------
# Smoke test / benchmark
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    import time
    import statistics
    from src.ntt import MOD

    print("=== ntt_gpu.py benchmark ===")
    SIZES = [2**k for k in [16, 18, 20]]

    for N in SIZES:
        rng = np.random.default_rng(42)
        a_np = rng.integers(0, MOD, size=N, dtype=np.int64)
        a_list = a_np.tolist()

        cpu_times = []
        for _ in range(3):
            t0 = time.perf_counter()
            ntt_cpu(a_list, MOD)
            cpu_times.append(time.perf_counter() - t0)
        cpu_ms = statistics.median(cpu_times) * 1000

        if _gpu_available:
            # Warmup
            _ = ntt_gpu(cp.array(a_np, dtype=cp.int64), MOD)

            gpu_times = []
            for _ in range(3):
                arr = cp.array(a_np, dtype=cp.int64)
                t0 = time.perf_counter()
                ntt_gpu(arr, MOD)
                cp.cuda.Stream.null.synchronize()
                gpu_times.append(time.perf_counter() - t0)
            gpu_ms = statistics.median(gpu_times) * 1000
            print(f"N={N:>8}: CPU {cpu_ms:8.1f} ms | GPU {gpu_ms:6.1f} ms | Speedup {cpu_ms/gpu_ms:.1f}x")
        else:
            print(f"N={N:>8}: CPU {cpu_ms:8.1f} ms | GPU not available")

    print("\nntt_gpu.py benchmark complete.")
