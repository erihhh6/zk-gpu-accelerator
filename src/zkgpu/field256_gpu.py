"""
zkgpu/field256_gpu.py — 256-bit prime-field arithmetic on GPU (CUDA via CuPy RawModule).

BN254's fields (r and p, both ~2^254) do not fit the int64 kernels of
zkgpu.field_gpu. Here every element is four 64-bit limbs, and multiplication
is Montgomery multiplication, the standard way GPU provers multiply large
field elements: it replaces the division in (a * b) mod p by multiplications,
additions and shifts, which GPUs execute quickly.

Representation
--------------
A batch of n elements is an (n, 4) uint64 array, least significant limb
first. Each row holds the element's **Montgomery form** a·R mod p, with
R = 2^256, never the plain value; convert with Field256.from_ints and
Field256.to_ints. In Montgomery form, addition and subtraction are the
usual modular operations, and mont_mul(aR, bR) = aR · bR · R^-1 = (ab)R, so
products stay in the same form. One row is 32 bytes, a single memory sector,
so a warp's 32 threads reading 32 consecutive rows use every byte fetched.

Contract
--------
- The modulus must be odd (Montgomery reduction needs p invertible mod 2^64)
  and below 2^255, so the sum of two residues still fits in 256 bits.
- Rows must be canonical, i.e. produced by this class: below p, for this
  modulus. They are not re-checked on the GPU, because that would cost as
  much as the arithmetic itself.
- Either operand of a binary operation may be a single row, which is
  broadcast against the other (e.g. multiplying a batch by a constant).
- GPU availability is decided once in zkgpu.gpu_backend. When it is
  unavailable, every method warns with the reason and computes the same
  rows on the CPU with Python integers.
"""

from __future__ import annotations

import operator
from collections.abc import Iterable
from typing import Any

import numpy as np

from zkgpu.curve import P as BN254_BASE_MODULUS
from zkgpu.field import FIELD_MODULUS as BN254_SCALAR_MODULUS
from zkgpu.gpu_backend import GPU_AVAILABLE as _gpu_available, cp, warn_cpu_fallback

LIMBS = 4
_LIMB_BITS = 64
_R_BITS = LIMBS * _LIMB_BITS  # R = 2^256
_THREADS_PER_BLOCK = 256

# CUDA C++ shared by every modulus; the modulus-specific constants are
# prepended as #defines (see Field256._compile), so they compile to
# immediates instead of memory loads.
_KERNEL_BODY = r"""
typedef unsigned long long u64;

// Low word of a * b + c + d, with the high word stored in *hi. This cannot
// overflow 128 bits: (2^64 - 1)^2 + 2 (2^64 - 1) = 2^128 - 1. __umul64hi
// supplies the high half of the product because NVRTC on Windows has no
// __int128.
__device__ __forceinline__ u64 zk_mac(u64 a, u64 b, u64 c, u64 d, u64* hi) {
    u64 lo = a * b;
    u64 h = __umul64hi(a, b);
    lo += c;
    h += lo < c;
    lo += d;
    h += lo < d;
    *hi = h;
    return lo;
}

// t := t - p if t >= p, for t < 2p. Every caller stays below 2p < 2^256
// (p < 2^255 is part of the contract), so t never needs a fifth word.
// Branch-free (selects, not jumps), so the threads of a warp never diverge
// on data-dependent comparisons.
__device__ __forceinline__ void zk_reduce_once(u64 t[4]) {
    const u64 p[4] = {ZK_P0, ZK_P1, ZK_P2, ZK_P3};
    u64 d[4];
    u64 borrow = 0;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        u64 pj = p[j] + borrow;
        u64 next = (pj < borrow) | (t[j] < pj);
        d[j] = t[j] - pj;
        borrow = next;
    }
    // No borrow out of the top limb means t >= p
    #pragma unroll
    for (int j = 0; j < 4; ++j) t[j] = borrow ? t[j] : d[j];
}

// r = a * b * 2^-256 mod p, for a, b < p. This is CIOS (Coarsely Integrated
// Operand Scanning; Koc, Acar and Kaliski, 1996): each outer step adds
// a * b[i] and then a multiple m * p chosen so that the lowest word becomes
// zero, and drops that word. After four steps the total is a * b * 2^-256
// mod p, so one conditional subtraction finishes it.
//
// Why five words suffice (the general algorithm needs six): between steps
// t < 2p < 2^256, i.e. four words. Adding a * b[i] < p * 2^64 keeps
// t < p (2^64 + 2) < 2^320, five words. Adding m * p makes the total exactly
// 2^64 * t_next, and t_next <= (2p - 1 + (2^64 - 1)(2p - 1)) / 2^64 = 2p - 1,
// so it is below 2^320 as well, and t_next is again four words. This needs
// only 2p < 2^256, i.e. the contract's p < 2^255.
__device__ __forceinline__ void zk_mont_mul(const u64 a[4], const u64 b[4], u64 r[4]) {
    const u64 p[4] = {ZK_P0, ZK_P1, ZK_P2, ZK_P3};
    u64 t[5] = {0, 0, 0, 0, 0};
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        u64 c = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) t[j] = zk_mac(a[j], b[i], t[j], c, &c);
        t[4] = c;  // t was four words, so the fifth is just the carry

        // m = -t[0] / p mod 2^64, so t[0] + m * p[0] = 0 mod 2^64
        u64 m = t[0] * ZK_NINV;
        zk_mac(m, p[0], t[0], 0, &c);  // low word is zero by the choice of m
        #pragma unroll
        for (int j = 1; j < 4; ++j) t[j - 1] = zk_mac(m, p[j], t[j], c, &c);
        t[3] = t[4] + c;  // no carry out: the total is below 2^320 (see above)
    }
    zk_reduce_once(t);
    #pragma unroll
    for (int j = 0; j < 4; ++j) r[j] = t[j];
}

// Both inputs are below p, so the sum is below 2p < 2^256: the carry out of
// the top limb is always zero
__device__ __forceinline__ void zk_add(const u64 a[4], const u64 b[4], u64 r[4]) {
    u64 carry = 0;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        u64 x = a[j] + carry;
        u64 c1 = x < carry;
        r[j] = x + b[j];
        carry = c1 | (r[j] < b[j]);
    }
    zk_reduce_once(r);
}

__device__ __forceinline__ void zk_sub(const u64 a[4], const u64 b[4], u64 r[4]) {
    const u64 p[4] = {ZK_P0, ZK_P1, ZK_P2, ZK_P3};
    u64 borrow = 0;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        u64 bj = b[j] + borrow;
        u64 next = (bj < borrow) | (a[j] < bj);
        r[j] = a[j] - bj;
        borrow = next;
    }
    // a < b: the difference wrapped around 2^256, so add p back (with
    // the mask instead of a branch); the final carry out cancels the wrap
    u64 mask = 0 - borrow, carry = 0;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        u64 pj = p[j] & mask;
        u64 x = r[j] + carry;
        u64 c1 = x < carry;
        r[j] = x + pj;
        carry = c1 | (r[j] < pj);
    }
}

// One thread per row. A step of 0 broadcasts a single-row operand.
#define ZK_BINARY_KERNEL(NAME, OP)                                              \
extern "C" __global__ void NAME(const u64* a, long long a_step,                 \
                                const u64* b, long long b_step,                 \
                                u64* out, long long n) {                        \
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;             \
    if (i >= n) return;                                                         \
    u64 x[4], y[4], r[4];                                                       \
    _Pragma("unroll")                                                           \
    for (int j = 0; j < 4; ++j) {                                               \
        x[j] = a[i * a_step + j];                                               \
        y[j] = b[i * b_step + j];                                               \
    }                                                                           \
    OP(x, y, r);                                                                \
    _Pragma("unroll")                                                           \
    for (int j = 0; j < 4; ++j) out[4 * i + j] = r[j];                          \
}

ZK_BINARY_KERNEL(zk256_mul, zk_mont_mul)
ZK_BINARY_KERNEL(zk256_add, zk_add)
ZK_BINARY_KERNEL(zk256_sub, zk_sub)
"""
_KERNEL_NAMES = ("zk256_mul", "zk256_add", "zk256_sub")


def _ints_to_limbs(values: Iterable[int]) -> np.ndarray:
    """(n, 4) uint64 array of 256-bit little-endian limbs; every value must be in [0, 2^256)."""
    buf = b"".join(v.to_bytes(32, "little") for v in values)
    return np.frombuffer(buf, dtype="<u8").astype(np.uint64).reshape(-1, LIMBS)


def _limbs_to_ints(limbs: np.ndarray) -> list[int]:
    buf = np.ascontiguousarray(limbs, dtype="<u8").tobytes()
    return [int.from_bytes(buf[k : k + 32], "little") for k in range(0, len(buf), 32)]


def _row_count(x: Any, name: str) -> int:
    """Rows of *x*, after checking it is an (n, 4) uint64 array of limbs."""
    if getattr(x, "dtype", None) != np.uint64 or x.ndim != 2 or x.shape[1] != LIMBS:
        raise TypeError(
            f"{name} must be an (n, {LIMBS}) uint64 array from Field256.from_ints, "
            f"got {getattr(x, 'dtype', type(x).__name__)} with shape {getattr(x, 'shape', None)}"
        )
    return int(x.shape[0])


def _broadcast_rows(a: Any, b: Any) -> int:
    rows_a, rows_b = _row_count(a, "a"), _row_count(b, "b")
    if rows_a == rows_b or rows_b == 1:
        return rows_a
    if rows_a == 1:
        return rows_b
    raise ValueError(f"cannot broadcast {rows_a} rows against {rows_b} rows")


class Field256:
    """
    Arithmetic modulo an odd *modulus* below 2^255 on (n, 4) uint64 limb arrays.

    The CUDA module is compiled on first use, once per modulus, with the
    modulus built into the code as constants.
    """

    def __init__(self, modulus: int) -> None:
        modulus = operator.index(modulus)
        if not (3 <= modulus < 2 ** (_R_BITS - 1) and modulus % 2 == 1):
            raise ValueError(f"modulus must be odd and in [3, 2^255), got {modulus}")
        self.modulus = modulus
        # Montgomery constants: R^2 converts into Montgomery form with one
        # multiplication (a * R^2 * R^-1 = aR); NINV = -p^-1 mod 2^64 makes
        # t + m * p divisible by 2^64 in each CIOS step.
        self._r2 = pow(2, 2 * _R_BITS, modulus)
        self._r_inv = pow(2**_R_BITS, -1, modulus)
        self._n_inv = -pow(modulus, -1, 2**_LIMB_BITS) % 2**_LIMB_BITS
        self._kernels: dict[str, Any] | None = None

    def __repr__(self) -> str:
        return f"Field256({self.modulus:#x})"

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def from_ints(self, values: Iterable[int]) -> Any:
        """
        Montgomery-form rows for *values* (any ints; reduced mod the modulus first).

        Returns a CuPy array on the GPU, or a NumPy array on the CPU fallback.
        """
        reduced = [operator.index(v) % self.modulus for v in values]
        if not _gpu_available:
            warn_cpu_fallback("Field256.from_ints", "Python integers")
            return _ints_to_limbs(v * 2**_R_BITS % self.modulus for v in reduced)
        # Upload the plain values, then a·R^2·R^-1 = aR on the device: one
        # multiplication per element instead of a 512-bit product on the host
        return self._launch("zk256_mul", cp.asarray(_ints_to_limbs(reduced)), self._constant(self._r2))

    def to_ints(self, x: Any) -> list[int]:
        """The values of Montgomery-form rows *x*, as Python ints in [0, modulus)."""
        _row_count(x, "x")
        if not _gpu_available:
            warn_cpu_fallback("Field256.to_ints", "Python integers")
            return [v * self._r_inv % self.modulus for v in _limbs_to_ints(_to_host(x))]
        # aR · 1 · R^-1 = a: a Montgomery multiplication by a plain 1 leaves Montgomery form
        plain = self._launch("zk256_mul", x, self._constant(1))
        return _limbs_to_ints(cp.asnumpy(plain))

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def mul(self, a: Any, b: Any) -> Any:
        """Element-wise product of Montgomery-form rows (either may be one broadcast row)."""
        if not _gpu_available:
            return self._fallback("mul", a, b, lambda x, y: x * y * self._r_inv)
        return self._launch("zk256_mul", a, b)

    def add(self, a: Any, b: Any) -> Any:
        """Element-wise sum of Montgomery-form rows (either may be one broadcast row)."""
        if not _gpu_available:
            return self._fallback("add", a, b, lambda x, y: x + y)
        return self._launch("zk256_add", a, b)

    def sub(self, a: Any, b: Any) -> Any:
        """Element-wise difference a - b of Montgomery-form rows (either may be one broadcast row)."""
        if not _gpu_available:
            return self._fallback("sub", a, b, lambda x, y: x - y)
        return self._launch("zk256_sub", a, b)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _constant(self, value: int) -> Any:
        """One device row holding *value* as plain limbs (not Montgomery form)."""
        return cp.asarray(_ints_to_limbs([value]))

    def _compile(self) -> dict[str, Any]:
        if self._kernels is None:
            limbs = [(self.modulus >> (_LIMB_BITS * j)) & (2**_LIMB_BITS - 1) for j in range(LIMBS)]
            defines = "".join(f"#define ZK_P{j} {limb:#x}ULL\n" for j, limb in enumerate(limbs))
            defines += f"#define ZK_NINV {self._n_inv:#x}ULL\n"
            module = cp.RawModule(code=defines + _KERNEL_BODY)
            self._kernels = {name: module.get_function(name) for name in _KERNEL_NAMES}
        return self._kernels

    def _launch(self, name: str, a: Any, b: Any) -> Any:
        n = _broadcast_rows(a, b)
        a, b = cp.ascontiguousarray(a), cp.ascontiguousarray(b)
        out = cp.empty((n, LIMBS), dtype=cp.uint64)
        if n:
            blocks = (n + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
            step_a, step_b = (np.int64(LIMBS if x.shape[0] > 1 else 0) for x in (a, b))
            self._compile()[name]((blocks,), (_THREADS_PER_BLOCK,), (a, step_a, b, step_b, out, np.int64(n)))
        return out

    def _fallback(self, op: str, a: Any, b: Any, combine: Any) -> np.ndarray:
        warn_cpu_fallback(f"Field256.{op}", "Python integers")
        n = _broadcast_rows(a, b)
        xs, ys = (_limbs_to_ints(_to_host(v)) for v in (a, b))
        xs, ys = (v * n if len(v) == 1 and n != 1 else v for v in (xs, ys))
        return _ints_to_limbs(combine(x, y) % self.modulus for x, y in zip(xs, ys, strict=True))


def _to_host(x: Any) -> np.ndarray:
    return cp.asnumpy(x) if hasattr(x, "get") else np.asarray(x)


BN254_FR = Field256(BN254_SCALAR_MODULUS)  # scalar field r: NTT and polynomial arithmetic
BN254_FQ = Field256(BN254_BASE_MODULUS)  # base field p: curve point coordinates
