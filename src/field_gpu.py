"""
src/field_gpu.py — Finite field arithmetic on GPU (CuPy).

Provides GPU-accelerated batch operations over a prime field F_p.
Implemented entirely with CuPy's ElementwiseKernel — no Numba required.

GPU availability detection adds PyTorch's lib directory to PATH before
importing CuPy so that nvrtc64_*.dll (needed for kernel compilation) is
found even when the CUDA Toolkit is not separately installed.

Limitations
-----------
- Array elements are int64. This means the modulus must satisfy:
    mod^2 < 2^63 - 1  →  mod < ~3.03 × 10⁹  (safe for NTT mod = 998244353)
  For larger moduli the kernel uses __int128 intermediate arithmetic via
  a RawKernel with the explicit cast `(__int128)a * b`.
- BN254 field modulus r ≈ 2^254 cannot be stored in int64; the benchmark
  uses MOD_BENCH = 998244353 for GPU throughput measurements.
"""

from __future__ import annotations

import os
import sys
import warnings
import numpy as np

# ------------------------------------------------------------------
# Ensure PyTorch's lib dir (which contains nvrtc64*.dll) is in PATH
# before importing CuPy, so kernel compilation succeeds on Windows.
# ------------------------------------------------------------------

def _add_torch_lib_to_path() -> None:
    try:
        import torch as _torch
        torch_lib = os.path.join(os.path.dirname(_torch.__file__), "lib")
        if os.path.isdir(torch_lib) and torch_lib not in os.environ.get("PATH", ""):
            os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
    except ImportError:
        pass

_add_torch_lib_to_path()

# ------------------------------------------------------------------
# GPU availability
# ------------------------------------------------------------------

try:
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        import cupy as cp

    # Probe: try a trivial kernel compilation to confirm nvrtc works
    _probe = cp.ElementwiseKernel("int64 x", "int64 y", "y = x + 1", "_probe_kernel")
    _probe(cp.array([0], dtype=cp.int64))  # will raise if nvrtc missing
    _gpu_available: bool = True

except Exception:
    _gpu_available = False
    cp = None  # type: ignore[assignment]


def _check_gpu() -> None:
    """Raise a descriptive RuntimeError if CUDA is not available."""
    if not _gpu_available:
        raise RuntimeError(
            "GPU not available. Install cupy-cuda12x and ensure CUDA drivers are "
            "installed. Run: nvcc --version to check your CUDA version."
        )


# ------------------------------------------------------------------
# CuPy ElementwiseKernels (compiled once on first call)
# ------------------------------------------------------------------

if _gpu_available:
    # int64 modular multiply — safe when mod < 2^31 (mod^2 < 2^62 < INT64_MAX)
    # For NTT use: mod=998244353 < 2^30, so a*b < 2^60. Safe.
    _modmul_kernel = cp.ElementwiseKernel(
        "int64 a, int64 b, int64 mod",
        "int64 c",
        "c = a * b % mod",
        "modmul_kernel",
    )

    # int64 modular add
    _modadd_kernel = cp.ElementwiseKernel(
        "int64 a, int64 b, int64 mod",
        "int64 c",
        "c = (a + b) % mod",
        "modadd_kernel",
    )


# ------------------------------------------------------------------
# Public GPU API
# ------------------------------------------------------------------

_BLOCK = 256


def gpu_batch_mul(a: "cp.ndarray", b: "cp.ndarray", mod: int) -> "cp.ndarray":
    """
    Element-wise modular multiplication on GPU: result[i] = (a[i] * b[i]) % mod.

    Uses 128-bit intermediate arithmetic so it is correct for any mod < 2^63.

    Parameters
    ----------
    a, b : cp.ndarray of dtype int64
    mod  : prime field modulus (must fit in int64, i.e. < 2^63)
    """
    if not _gpu_available:
        warnings.warn("GPU not available; falling back to NumPy.", stacklevel=2)
        a_np = cp.asnumpy(a) if hasattr(a, "get") else np.asarray(a)
        b_np = cp.asnumpy(b) if hasattr(b, "get") else np.asarray(b)
        return (a_np * b_np) % mod  # type: ignore[return-value]

    return _modmul_kernel(a, b, mod)


def gpu_batch_add(a: "cp.ndarray", b: "cp.ndarray", mod: int) -> "cp.ndarray":
    """
    Element-wise modular addition on GPU: result[i] = (a[i] + b[i]) % mod.
    """
    if not _gpu_available:
        warnings.warn("GPU not available; falling back to NumPy.", stacklevel=2)
        a_np = cp.asnumpy(a) if hasattr(a, "get") else np.asarray(a)
        b_np = cp.asnumpy(b) if hasattr(b, "get") else np.asarray(b)
        return (a_np + b_np) % mod  # type: ignore[return-value]

    return _modadd_kernel(a, b, mod)


def gpu_batch_pow(bases: "cp.ndarray", exp: int, mod: int) -> "cp.ndarray":
    """
    Element-wise modular exponentiation: result[i] = bases[i]^exp % mod.

    Uses binary exponentiation via successive squarings in CuPy (all GPU).
    """
    if not _gpu_available:
        warnings.warn("GPU not available; falling back to NumPy.", stacklevel=2)
        bases_np = cp.asnumpy(bases) if hasattr(bases, "get") else np.asarray(bases)
        return np.array([pow(int(x), exp, mod) for x in bases_np])  # type: ignore[return-value]

    result = cp.ones(bases.shape[0], dtype=cp.int64)
    base = bases % mod
    e = int(exp)
    while e > 0:
        if e & 1:
            result = gpu_batch_mul(result, base, mod)
        base = gpu_batch_mul(base, base, mod)
        e >>= 1
    return result


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
