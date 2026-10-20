"""
src/field_gpu.py — Finite field arithmetic on GPU (CuPy).

Provides GPU-accelerated batch operations over a prime field F_p.
Implemented entirely with CuPy's ElementwiseKernel — no Numba required.

GPU availability is decided once in src.gpu_backend; when it is unavailable
every function here warns with the reason and falls back to NumPy.

Contract
--------
- The modulus must satisfy 2 <= mod < 2^63 so it fits in int64 (checked).
- Inputs may be any int64 values; they are reduced into [0, mod) first, so
  negative and unreduced inputs give field-correct results.
- Products use a plain int64 multiply when mod <= 3_037_000_500, where the
  largest product of two residues, (mod-1)^2, still fits in int64. Above that
  an exact 128-bit-wide routine is used (about 2x slower for gpu_batch_mul
  on an RTX 5050 at N = 10^7; addition costs the same at any modulus).
- Negative exponents in gpu_batch_pow use Fermat's little theorem, so they
  require a prime modulus.
- BN254's moduli (~2^254) cannot be stored in int64 at all; see src.field for
  arbitrary-precision arithmetic.
"""

from __future__ import annotations

import math
import operator

import numpy as np

from src.gpu_backend import GPU_AVAILABLE as _gpu_available, cp, warn_cpu_fallback

_INT64_MAX = 2**63 - 1

# Largest modulus whose residues multiply without overflowing int64: the
# biggest product is (mod - 1)^2, which must not exceed 2^63 - 1.
_INT64_PRODUCT_SAFE_MOD = math.isqrt(_INT64_MAX) + 1  # 3_037_000_500


# ------------------------------------------------------------------
# CuPy ElementwiseKernels (compiled once on first call)
# ------------------------------------------------------------------

_KERNEL_PREAMBLE = f"""
// Reduce any int64 into [0, m). CUDA's % truncates toward zero, so a negative
// x leaves a negative remainder. Values already in range skip the 64-bit
// division, which the GPU emulates in software and is comparatively slow.
__device__ long long zk_reduce(long long x, long long m) {{
    if (x >= 0 && x < m) return x;
    long long r = x % m;
    return r < 0 ? r + m : r;
}}

// (a * b) mod m for 0 <= a, b < m < 2^63, without a 128-bit integer type
// (NVRTC on Windows rejects __int128). __umul64hi gives the high 64 bits of
// the 128-bit product; since a, b < m that half is already below m, so it
// seeds the remainder. The low 64 bits are then folded in one at a time,
// as in binary long division: r = (2r + bit) mod m. Because r < m < 2^63,
// 2r + 1 never overflows 64 bits.
__device__ long long zk_mulmod_wide(unsigned long long a, unsigned long long b,
                                    unsigned long long m) {{
    unsigned long long r = __umul64hi(a, b);
    unsigned long long lo = a * b;
    for (int i = 63; i >= 0; --i) {{
        r = (r << 1) | ((lo >> i) & 1ULL);
        if (r >= m) r -= m;
    }}
    return (long long)r;
}}

#define ZK_INT64_PRODUCT_SAFE_MOD {_INT64_PRODUCT_SAFE_MOD}LL
"""

if _gpu_available:
    _modmul_kernel = cp.ElementwiseKernel(
        "int64 a, int64 b, int64 m",
        "int64 c",
        """
        long long ar = zk_reduce(a, m), br = zk_reduce(b, m);
        // m is the same for every thread, so this branch never diverges
        c = m <= ZK_INT64_PRODUCT_SAFE_MOD ? ar * br % m : zk_mulmod_wide(ar, br, m);
        """,
        "zk_modmul",
        preamble=_KERNEL_PREAMBLE,
    )

    _modadd_kernel = cp.ElementwiseKernel(
        "int64 a, int64 b, int64 m",
        "int64 c",
        """
        long long ar = zk_reduce(a, m), br = zk_reduce(b, m);
        // ar - (m - br) lies in (-m, m), so unlike ar + br it cannot
        // overflow when m is close to 2^63
        long long d = ar - (m - br);
        c = d < 0 ? d + m : d;
        """,
        "zk_modadd",
        preamble=_KERNEL_PREAMBLE,
    )


# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------


def _check_modulus(mod: int) -> int:
    """Return *mod* as an int, or raise if it cannot be an int64 field modulus."""
    mod = operator.index(mod)  # rejects floats such as 7.0 instead of truncating
    if not 2 <= mod <= _INT64_MAX:
        raise ValueError(f"mod must satisfy 2 <= mod < 2^63 to fit in int64, got {mod}")
    return mod


def _check_integer_array(x, name: str) -> None:
    """Reject non-integer arrays, which the int64 kernels would truncate silently."""
    dtype = x.dtype if hasattr(x, "dtype") else np.asarray(x).dtype
    if dtype.kind not in "iu":
        raise TypeError(f"{name} must be an integer array, got dtype {dtype}")


def _to_host_reduced(x, mod: int) -> np.ndarray:
    """int64 NumPy copy of *x* reduced into [0, mod); NumPy's % floors, so signs come out right."""
    host = cp.asnumpy(x) if hasattr(x, "get") else np.asarray(x)
    # Widen first: in a narrower dtype such as int32 the products would overflow
    return np.mod(host.astype(np.int64, copy=False), mod)


def _gpu_pow_nonnegative(base: "cp.ndarray", exp: int, mod: int) -> "cp.ndarray":
    """Square-and-multiply on reduced *base*; *exp* must be >= 0."""
    result = cp.ones_like(base)
    while exp:
        if exp & 1:
            result = _modmul_kernel(result, base, mod)
        base = _modmul_kernel(base, base, mod)
        exp >>= 1
    return result


# ------------------------------------------------------------------
# Public GPU API
# ------------------------------------------------------------------


def gpu_batch_mul(a: "cp.ndarray", b: "cp.ndarray", mod: int) -> "cp.ndarray":
    """
    Element-wise modular multiplication on GPU: result[i] = (a[i] * b[i]) % mod.

    Parameters
    ----------
    a, b : cp.ndarray of an integer dtype (any values; reduced mod *mod* first)
    mod  : modulus with 2 <= mod < 2^63

    Returns
    -------
    int64 array with every element in [0, mod) (np.ndarray on the CPU fallback)

    Raises
    ------
    ValueError : if *mod* is outside [2, 2^63)
    TypeError  : if *mod* is not an integer or an input is not an integer array
    """
    mod = _check_modulus(mod)
    _check_integer_array(a, "a")
    _check_integer_array(b, "b")
    if not _gpu_available:
        warn_cpu_fallback("gpu_batch_mul", "NumPy")
        a_r, b_r = _to_host_reduced(a, mod), _to_host_reduced(b, mod)
        if mod <= _INT64_PRODUCT_SAFE_MOD:
            return a_r * b_r % mod  # type: ignore[return-value]
        # Products would overflow int64: multiply as exact Python ints instead
        return (a_r.astype(object) * b_r.astype(object) % mod).astype(np.int64)  # type: ignore[return-value]

    # CuPy's typed kernels reject other integer dtypes rather than casting;
    # astype is a no-op for arrays that are already int64
    return _modmul_kernel(a.astype(cp.int64, copy=False), b.astype(cp.int64, copy=False), mod)


def gpu_batch_add(a: "cp.ndarray", b: "cp.ndarray", mod: int) -> "cp.ndarray":
    """
    Element-wise modular addition on GPU: result[i] = (a[i] + b[i]) % mod.

    Same contract as gpu_batch_mul.
    """
    mod = _check_modulus(mod)
    _check_integer_array(a, "a")
    _check_integer_array(b, "b")
    if not _gpu_available:
        warn_cpu_fallback("gpu_batch_add", "NumPy")
        a_r, b_r = _to_host_reduced(a, mod), _to_host_reduced(b, mod)
        d = a_r - (mod - b_r)  # in (-mod, mod): cannot overflow, unlike a_r + b_r
        return np.where(d < 0, d + mod, d)  # type: ignore[return-value]

    return _modadd_kernel(a.astype(cp.int64, copy=False), b.astype(cp.int64, copy=False), mod)


def gpu_batch_pow(bases: "cp.ndarray", exp: int, mod: int) -> "cp.ndarray":
    """
    Element-wise modular exponentiation: result[i] = bases[i]^exp % mod.

    Uses binary exponentiation via successive squarings in CuPy (all GPU).
    A negative *exp* raises the inverse to |exp|, with the inverse computed
    by Fermat's little theorem (a^-1 = a^(mod-2)), so *mod* must be prime.

    Raises
    ------
    ValueError : if exp < 0 and some base is 0 mod *mod* (zero has no inverse),
                 matching Python's pow(); also for a modulus outside [2, 2^63)
    TypeError  : if *exp* or *mod* is not an integer or *bases* is not an integer array
    """
    mod = _check_modulus(mod)
    exp = operator.index(exp)
    _check_integer_array(bases, "bases")
    if not _gpu_available:
        warn_cpu_fallback("gpu_batch_pow", "Python pow()")
        bases_r = _to_host_reduced(bases, mod)
        # Python's pow raises the same ValueError for a zero base with exp < 0
        return np.array([pow(int(x), exp, mod) for x in bases_r], dtype=np.int64)  # type: ignore[return-value]

    # Widen first (a large mod does not fit narrower dtypes); CuPy's % floors
    # like NumPy's, so the result lies in [0, mod)
    base = bases.astype(cp.int64, copy=False) % mod
    if exp < 0:
        if bool((base == 0).any()):
            raise ValueError("base is not invertible for the given modulus")
        base = _gpu_pow_nonnegative(base, mod - 2, mod)
        exp = -exp
    return _gpu_pow_nonnegative(base, exp, mod)


# ------------------------------------------------------------------
# Smoke test / quick benchmark
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    import time
    import statistics

    # Use NTT-friendly modulus that fits in int64 for benchmarking
    MOD_BENCH = 998_244_353
    N = 10_000_000
    print(f"=== field_gpu.py smoke test ({N:,} elements, mod={MOD_BENCH}) ===")

    rng = np.random.default_rng(42)
    a_np = rng.integers(1, MOD_BENCH, size=N, dtype=np.int64)
    b_np = rng.integers(1, MOD_BENCH, size=N, dtype=np.int64)

    # CPU baseline
    times_cpu = []
    for _ in range(3):
        t0 = time.perf_counter()
        cpu_result = (a_np * b_np) % MOD_BENCH
        times_cpu.append(time.perf_counter() - t0)
    cpu_ms = statistics.median(times_cpu) * 1000
    print(f"CPU (numpy):    {cpu_ms:.1f} ms")

    if _gpu_available:
        a_cp = cp.array(a_np)
        b_cp = cp.array(b_np)

        # Warmup
        _ = gpu_batch_mul(a_cp, b_cp, MOD_BENCH)

        times_gpu = []
        for _ in range(3):
            t0 = time.perf_counter()
            gpu_result = gpu_batch_mul(a_cp, b_cp, MOD_BENCH)
            cp.cuda.Stream.null.synchronize()
            times_gpu.append(time.perf_counter() - t0)
        gpu_ms = statistics.median(times_gpu) * 1000
        print(f"GPU (cupy):     {gpu_ms:.1f} ms  (median of 3)")
        print(f"Speedup:        {cpu_ms / gpu_ms:.1f}x")

        # Correctness check
        assert np.array_equal(cpu_result[:1000], cp.asnumpy(gpu_result[:1000]))
        print("GPU result matches CPU on first 1000 elements  OK")
    else:
        print("GPU not available - skipping GPU benchmark.")

    print("\nfield_gpu.py smoke test complete.")
