"""
src/ntt_gpu.py — Number Theoretic Transform on GPU (CuPy).

Implements the Cooley-Tukey NTT using CuPy's vectorized array operations.
Each butterfly stage runs on the GPU using CuPy indexing and broadcasting;
only CuPy and nvrtc are needed (src.gpu_backend explains how nvrtc is found).

Algorithm (per stage)
---------------------
For stage with half = length // 2:
  1. Precompute twiddle factors wk[j] = w^j mod p  (on CPU, copied to GPU)
  2. Reshape the flat array to (-1, length)
  3. upper[:, j] = arr[:, j],  lower[:, j] = arr[:, j + half]
  4. v[:, j]     = (lower[:, j] * wk[j]) % mod
  5. arr[:, j]        = (upper + v) % mod
  6. arr[:, j + half] = (upper - v + mod) % mod
  7. Flatten back

CuPy broadcasts wk (shape: half,) across groups (shape: n//length, half)
so step 4 is a single vectorized modular multiply on the GPU.

Limitations
-----------
- Same root selection as src.ntt (PRIMITIVE_ROOT = 3), so it is only correct
  for moduli where 3 is a quadratic non-residue, such as 998244353.
- Elements are int64 and products are formed in int64, so mod must be
  below ~3.03 × 10⁹ (≈ 2^31.5).
- Twiddle factors are rebuilt in a Python loop on every call; at N = 2^20
  this host-side loop is over half of the total runtime.
"""

from __future__ import annotations

import math

import numpy as np

from src.gpu_backend import GPU_AVAILABLE as _gpu_available, cp, warn_cpu_fallback
from src.ntt import PRIMITIVE_ROOT, ntt as ntt_cpu


# ------------------------------------------------------------------
# GPU NTT kernels
# ------------------------------------------------------------------

if _gpu_available:
    # Modular multiply for mod < 2^31: values < mod so a*b < 2^62, fits in int64
    # Safe for NTT with mod=998244353 (< 2^30)
    _modmul_ek = cp.ElementwiseKernel(
        "int64 a, int64 b, int64 mod",
        "int64 c",
        "c = a * b % mod",
        "ntt_modmul",
    )


def _modmul_gpu(a: "cp.ndarray", b: "cp.ndarray", mod: int) -> "cp.ndarray":
    """Element-wise modular multiply on GPU. Safe when mod < 2^31."""
    return _modmul_ek(a, b, mod)


# ------------------------------------------------------------------
# Bit-reversal permutation (GPU via CuPy indexing)
# ------------------------------------------------------------------


def _bit_reverse_permute_gpu(arr: "cp.ndarray") -> "cp.ndarray":
    """Return a bit-reversal permuted copy of *arr* on the GPU."""
    n = len(arr)
    bits = n.bit_length() - 1
    # Build reversal index on CPU then transfer once
    idx = np.arange(n, dtype=np.int32)
    rev = np.zeros(n, dtype=np.int32)
    tmp = idx.copy()
    for _ in range(bits):
        rev = (rev << 1) | (tmp & 1)
        tmp >>= 1
    rev_gpu = cp.array(rev)
    return arr[rev_gpu]


# ------------------------------------------------------------------
# Public GPU NTT
# ------------------------------------------------------------------


def ntt_gpu(
    a: "cp.ndarray",
    mod: int,
    invert: bool = False,
) -> "cp.ndarray":
    """
    Cooley-Tukey NTT on GPU using CuPy vectorized butterfly.

    Parameters
    ----------
    a      : cp.ndarray of dtype int64, length a power of 2
    mod    : NTT-friendly prime below ~2^31.5 for which 3 is a quadratic
             non-residue (see module Limitations); 998244353 qualifies
    invert : if True, compute the inverse NTT

    Returns
    -------
    cp.ndarray int64 of the same length (np.ndarray on the CPU fallback)
    """
    if not _gpu_available:
        warn_cpu_fallback("ntt_gpu", "the CPU NTT")
        a_list = a.tolist() if hasattr(a, "tolist") else list(a)
        result = ntt_cpu(a_list, mod, invert)
        return np.array(result, dtype=np.int64)  # type: ignore[return-value]

    n = len(a)
    if n == 0:
        return a.copy()
    if n & (n - 1):
        raise ValueError(f"NTT length must be a power of 2, got {n}")
    if n == 1:
        return a.copy()

    arr = _bit_reverse_permute_gpu(cp.array(a, dtype=cp.int64))

    log_n = int(math.log2(n))
    length = 2
    for _ in range(log_n):
        half = length >> 1

        # Primitive root of order *length*
        w = pow(PRIMITIVE_ROOT, (mod - 1) // length, mod)
        if invert:
            w = pow(w, mod - 2, mod)

        # Twiddle factors [w^0, w^1, ..., w^(half-1)] — computed on CPU, tiny array
        wk_cpu = np.empty(half, dtype=np.int64)
        cur = 1
        for j in range(half):
            wk_cpu[j] = cur
            cur = cur * w % mod
        wk = cp.array(wk_cpu)  # shape (half,)

        # Reshape: (num_groups, length)
        arr2 = arr.reshape(-1, length)
        upper = arr2[:, :half].copy()   # (groups, half)
        lower = arr2[:, half:].copy()   # (groups, half)

        # v[g, j] = lower[g, j] * wk[j]  — broadcast wk across groups
        wk_broad = cp.broadcast_to(wk[cp.newaxis, :], lower.shape).copy()
        v = _modmul_gpu(lower, wk_broad, mod)

        arr2[:, :half] = (upper + v) % mod
        arr2[:, half:] = (upper - v + mod) % mod
        arr = arr2.reshape(-1)

        length <<= 1

    if invert:
        n_inv = pow(n, mod - 2, mod)
        arr = (arr * n_inv) % mod

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
