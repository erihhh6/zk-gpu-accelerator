"""
zkgpu/field256_gpu.py — 256-bit prime-field arithmetic and NTT on GPU (CUDA via CuPy RawModule).

BN254's fields (r and p, both ~2^254) do not fit the int64 kernels of
zkgpu.field_gpu. Here every element is four 64-bit limbs, and multiplication
is Montgomery multiplication, the standard way GPU provers multiply large
field elements: it replaces the division in (a * b) mod p by multiplications,
additions and shifts, which GPUs execute quickly. Field256.ntt builds the
number-theoretic transform on the same kernels, so polynomial arithmetic
over BN254's r can stay on the GPU from start to finish.

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

import functools
import operator
from collections.abc import Iterable
from typing import Any

import numpy as np

from zkgpu.curve import P as BN254_BASE_MODULUS
from zkgpu.field import FIELD_MODULUS as BN254_SCALAR_MODULUS
from zkgpu.gpu_backend import GPU_AVAILABLE as _gpu_available, compile_module, cp, warn_cpu_fallback
from zkgpu.ntt import get_ntt_root, ntt as ntt_cpu

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
// the top limb is always zero. Like every function here, r may alias a or b
// (e.g. zk_add(x, x, x) doubles x in place): each limb of the inputs is read
// before the same limb of r is written.
__device__ __forceinline__ void zk_add(const u64 a[4], const u64 b[4], u64 r[4]) {
    u64 carry = 0;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        u64 bj = b[j];  // read before r[j] is written, in case r is b
        u64 x = a[j] + carry;
        u64 c1 = x < carry;
        u64 s = x + bj;
        carry = c1 | (s < bj);
        r[j] = s;
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

// ---- NTT (radix-2 Cooley-Tukey, decimation in time) ----

// out[i] = in[bitreverse(i)] for n = 2^log_n rows, log_n >= 1. Out of place,
// so the caller's input survives; __brevll reverses all 64 bits, and the
// shift keeps the low log_n of them.
extern "C" __global__ void zk256_bit_reverse(const u64* in, u64* out, int log_n, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    long long j = (long long)(__brevll((u64)i) >> (64 - log_n));
    #pragma unroll
    for (int k = 0; k < 4; ++k) out[4 * i + k] = in[4 * j + k];
}

// One butterfly stage in place: thread k pairs rows i0 and i1 = i0 + half
// (half = 2^log_half) and computes (u, v) -> (u + w v, u - w v). With j the
// position within its group, w = omega_n^(j * n / (2 half)), read from the
// table [omega^0, ..., omega^(n/2 - 1)] at j << (log_n - 1 - log_half). The
// pairs of one stage are disjoint, so no thread reads another's output.
extern "C" __global__ void zk256_ntt_stage(u64* a, const u64* twiddles, int log_half, int log_n,
                                           long long n_half) {
    long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_half) return;
    long long j = k & ((1LL << log_half) - 1);
    long long i0 = ((k >> log_half) << (log_half + 1)) | j;
    long long i1 = i0 + (1LL << log_half);
    const u64* tw = twiddles + 4 * (j << (log_n - 1 - log_half));
    u64 u[4], v[4], w[4], wv[4], x[4], y[4];
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        u[t] = a[4 * i0 + t];
        v[t] = a[4 * i1 + t];
        w[t] = tw[t];
    }
    zk_mont_mul(v, w, wv);
    zk_add(u, wv, x);
    zk_sub(u, wv, y);
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        a[4 * i0 + t] = x[t];
        a[4 * i1 + t] = y[t];
    }
}
"""
_KERNEL_NAMES = ("zk256_mul", "zk256_add", "zk256_sub", "zk256_bit_reverse", "zk256_ntt_stage")


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
    modulus built into the code as constants, and self-tested before use.
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
        plain = cp.asarray(_ints_to_limbs(reduced))  # gpu-only
        return self._launch("zk256_mul", plain, self._constant(self._r2))  # gpu-only

    def to_ints(self, x: Any) -> list[int]:
        """The values of Montgomery-form rows *x*, as Python ints in [0, modulus)."""
        _row_count(x, "x")
        if not _gpu_available:
            warn_cpu_fallback("Field256.to_ints", "Python integers")
            return [v * self._r_inv % self.modulus for v in _limbs_to_ints(_to_host(x))]
        # aR · 1 · R^-1 = a: a Montgomery multiplication by a plain 1 leaves Montgomery form
        return _limbs_to_ints(cp.asnumpy(self._launch("zk256_mul", x, self._constant(1))))  # gpu-only

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def mul(self, a: Any, b: Any) -> Any:
        """Element-wise product of Montgomery-form rows (either may be one broadcast row)."""
        if not _gpu_available:
            return self._fallback("mul", a, b, lambda x, y: x * y * self._r_inv)
        return self._launch("zk256_mul", a, b)  # gpu-only

    def add(self, a: Any, b: Any) -> Any:
        """Element-wise sum of Montgomery-form rows (either may be one broadcast row)."""
        if not _gpu_available:
            return self._fallback("add", a, b, lambda x, y: x + y)
        return self._launch("zk256_add", a, b)  # gpu-only

    def sub(self, a: Any, b: Any) -> Any:
        """Element-wise difference a - b of Montgomery-form rows (either may be one broadcast row)."""
        if not _gpu_available:
            return self._fallback("sub", a, b, lambda x, y: x - y)
        return self._launch("zk256_sub", a, b)  # gpu-only

    # ------------------------------------------------------------------
    # NTT
    # ------------------------------------------------------------------

    def ntt(self, x: Any, *, invert: bool = False) -> Any:
        """
        Radix-2 NTT of Montgomery-form rows *x*; returns new rows, *x* is unchanged.

        Exactly the transform of zkgpu.ntt.ntt (same roots of unity, same output
        order), and therefore of Radix2Domain.fft / .ifft for BN254_FR. On the
        GPU it is one bit-reversal launch plus one launch per butterfly stage,
        log2(n) + 1 in total; the twiddle table is built on the GPU on the
        first call for each size and cached.

        Raises
        ------
        ValueError : if the row count is not a power of two, the modulus is
                     not an odd prime, or the row count does not divide modulus - 1
        """
        n = _row_count(x, "x")
        if n == 0:
            return (cp if _gpu_available else np).empty((0, LIMBS), dtype=np.uint64)
        if n & (n - 1):
            raise ValueError(f"NTT length must be a power of 2, got {n}")
        p = self.modulus
        get_ntt_root(n, p)  # validates modulus and size before any work, as ntt() does
        if not _gpu_available:
            warn_cpu_fallback("Field256.ntt", "zkgpu.ntt.ntt on Python integers")
            values = ntt_cpu([v * self._r_inv % p for v in _limbs_to_ints(_to_host(x))], p, invert)
            return _ints_to_limbs(v * 2**_R_BITS % p for v in values)
        return self._ntt_on_gpu(x, n, invert)  # gpu-only

    def _ntt_on_gpu(self, x: Any, n: int, invert: bool) -> Any:  # gpu-only
        src = cp.ascontiguousarray(x)
        if n == 1:  # the transform of a single value is itself (and n^-1 = 1)
            return src.copy()
        log_n = n.bit_length() - 1
        twiddles, n_inv = _ntt_tables(self, n, invert)
        kernels = self._compile()
        out = cp.empty_like(src)
        _run(kernels["zk256_bit_reverse"], n, src, out, np.int32(log_n), np.int64(n))
        stage, half = kernels["zk256_ntt_stage"], n // 2  # one thread per butterfly
        for log_half in range(log_n):
            _run(stage, half, out, twiddles, np.int32(log_half), np.int32(log_n), np.int64(half))
        return self.mul(out, n_inv) if invert else out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _constant(self, value: int) -> Any:  # gpu-only
        """One device row holding *value* as plain limbs (not Montgomery form)."""
        return cp.asarray(_ints_to_limbs([value]))

    def cuda_source(self) -> str:
        """
        CUDA C++ for this field: its constants as #defines, then the shared device code.

        For modules that build on the field arithmetic (zkgpu.curve_gpu appends
        point formulas to BN254_FQ's source). Defines ZK_P0..3 (the modulus),
        ZK_NINV (-p^-1 mod 2^64) and ZK_ONE0..3 (1 in Montgomery form, R mod p).
        """

        def limbs(value: int) -> list[int]:
            return [(value >> (_LIMB_BITS * j)) & (2**_LIMB_BITS - 1) for j in range(LIMBS)]

        defines = "".join(f"#define ZK_P{j} {limb:#x}ULL\n" for j, limb in enumerate(limbs(self.modulus)))
        one = 2**_R_BITS % self.modulus
        defines += "".join(f"#define ZK_ONE{j} {limb:#x}ULL\n" for j, limb in enumerate(limbs(one)))
        defines += f"#define ZK_NINV {self._n_inv:#x}ULL\n"
        return defines + _KERNEL_BODY

    def compile(self) -> None:  # gpu-only
        """
        Compile, load and self-test the kernels now instead of on first use.

        Raises RuntimeError if they fail their self-test (see _self_test).
        """
        self._compile()

    def _compile(self) -> dict[str, Any]:  # gpu-only
        if self._kernels is None:
            module = compile_module(self.cuda_source(), "zkgpu_field256")
            self._kernels = {name: module.get_function(name) for name in _KERNEL_NAMES}
            try:
                self._self_test()
            except Exception:
                self._kernels = None  # never keep kernels that failed their self-test
                raise
        return self._kernels

    def _self_test(self) -> None:  # gpu-only
        """
        Check the freshly compiled kernels against Python ints, once per process.

        A compiler can produce wrong machine code for correct source (it has, see
        zkgpu.gpu_backend.compile_module). This makes such a miscompile fail
        loudly at load time instead of returning wrong field elements later.
        Carry-heavy operands: zero, one, p - 1, and values with all-ones limbs.
        """
        p = self.modulus
        a = [0, 1, 2, p - 1, p - 2, p // 2, (2**128 - 1) % p, (2**192 - 1) % p]
        b = [p - 1, p - 1, 3, p - 1, 5, p // 3, (2**64 - 1) % p, 2**254 % p]
        x, y = self.from_ints(a), self.from_ints(b)
        checks = {
            "mul": (self.mul(x, y), [u * v % p for u, v in zip(a, b, strict=True)]),
            "add": (self.add(x, y), [(u + v) % p for u, v in zip(a, b, strict=True)]),
            "sub": (self.sub(x, y), [(u - v) % p for u, v in zip(a, b, strict=True)]),
        }
        try:
            get_ntt_root(len(a), p)
        except ValueError:
            pass  # no NTT of this size modulo p, so nothing to check
        else:
            checks["ntt"] = (self.ntt(x), ntt_cpu(a, p))
        for op, (rows, expected) in checks.items():
            if self.to_ints(rows) != expected:
                raise RuntimeError(
                    f"{self!r}: the compiled {op} kernel failed its self-test, so the CUDA compiler "
                    "produced wrong machine code for this GPU. Please report it with the output of "
                    "`python scripts/check_gpu.py`."
                )

    def _launch(self, name: str, a: Any, b: Any) -> Any:  # gpu-only
        n = _broadcast_rows(a, b)
        a, b = cp.ascontiguousarray(a), cp.ascontiguousarray(b)
        out = cp.empty((n, LIMBS), dtype=cp.uint64)
        if n:
            step_a, step_b = (np.int64(LIMBS if x.shape[0] > 1 else 0) for x in (a, b))
            _run(self._compile()[name], n, a, step_a, b, step_b, out, np.int64(n))
        return out

    def _fallback(self, op: str, a: Any, b: Any, combine: Any) -> np.ndarray:
        warn_cpu_fallback(f"Field256.{op}", "Python integers")
        n = _broadcast_rows(a, b)
        xs, ys = (_limbs_to_ints(_to_host(v)) for v in (a, b))
        xs, ys = (v * n if len(v) == 1 and n != 1 else v for v in (xs, ys))
        return _ints_to_limbs(combine(x, y) % self.modulus for x, y in zip(xs, ys, strict=True))


def _to_host(x: Any) -> np.ndarray:
    return cp.asnumpy(x) if hasattr(x, "get") else np.asarray(x)


def _run(kernel: Any, threads: int, *args: Any) -> None:  # gpu-only
    """Launch *kernel* with *threads* threads (rounded up to whole blocks; kernels skip the extras)."""
    kernel(((threads + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK,), (_THREADS_PER_BLOCK,), args)


@functools.lru_cache(maxsize=16)
def _ntt_tables(field: Field256, n: int, invert: bool) -> tuple[Any, Any]:  # gpu-only
    """
    Device rows [w^0, ..., w^(n/2 - 1)] for w = omega_n (or omega_n^-1), and n^-1.

    Built by doubling: if the table holds w^0 .. w^(s-1), multiplying it by
    w^s gives w^s .. w^(2s-1), so log2(n) - 1 small launches replace n/2
    sequential Python multiplications.
    """
    p = field.modulus
    omega = get_ntt_root(n, p)
    if invert:
        omega = pow(omega, -1, p)
    table = field.from_ints([1])
    while table.shape[0] < n // 2:
        table = cp.concatenate([table, field.mul(table, field.from_ints([pow(omega, table.shape[0], p)]))])
    return table, field.from_ints([pow(n, -1, p)])


BN254_FR = Field256(BN254_SCALAR_MODULUS)  # scalar field r: NTT and polynomial arithmetic
BN254_FQ = Field256(BN254_BASE_MODULUS)  # base field p: curve point coordinates
