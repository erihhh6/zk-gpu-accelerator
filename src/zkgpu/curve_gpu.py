"""
zkgpu/curve_gpu.py — BN254 G1 point arithmetic on GPU, in Jacobian coordinates.

The MSM spends nearly all its time adding curve points, so moving it to the
GPU starts here: point addition, mixed addition and doubling as CUDA device
functions over BN254's base field, built on zkgpu.field256_gpu's Montgomery
arithmetic (BN254_FQ), plus element-wise kernels to run and test them.

The formulas are exactly zkgpu.curve's (EFD add-2007-bl, madd-2007-bl and
dbl-2009-l for a = 0, with the same special cases), so the GPU produces the
same Jacobian coordinates as the CPU, not merely the same affine points. The
kernels are built with gpu_backend.compile_module and checked against
zkgpu.curve when first loaded, like zkgpu.field256_gpu's.

Representation
--------------
- A batch of n Jacobian points is an (n, 3, 4) uint64 array: X, Y and Z, each
  a Montgomery-form BN254_FQ element. Z = 0 is the point at infinity, stored
  as (1, 1, 0) like zkgpu.curve.JACOBIAN_IDENTITY.
- A batch of n affine points, the second operand of add_affine, is an
  (n, 2, 4) array of x and y. Affine form cannot hold the point at infinity,
  so affine_to_device rejects it; Jacobian Z = 1 points need 1/3 more memory
  and two more field multiplications per addition.
- As in field256_gpu, either operand of a binary operation may be a single
  point, broadcast against the other, and the CPU fallback (zkgpu.curve on
  Python ints, with a GpuFallbackWarning) returns identical rows.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

from zkgpu.curve import (
    JACOBIAN_IDENTITY,
    G,
    Jacobian,
    Point,
    from_jacobian,
    jacobian_add,
    jacobian_add_affine,
    jacobian_double,
    jacobian_mul,
    to_jacobian,
)
from zkgpu.field256_gpu import BN254_FQ, LIMBS
from zkgpu.gpu_backend import GPU_AVAILABLE as _gpu_available, compile_module, cp, warn_cpu_fallback

_THREADS_PER_BLOCK = 256

# Appended to BN254_FQ.cuda_source(), which provides u64, zk_mont_mul,
# zk_add, zk_sub and the constants ZK_P*, ZK_ONE*.
_POINT_SOURCE = r"""
// ---- BN254 G1, Jacobian coordinates, a = 0 ----
// A Jacobian point is 12 limbs: X = p[0..3], Y = p[4..7], Z = p[8..11]; an
// affine point is 8: x, y. Every function reads all of its inputs before
// writing r, so r may alias an input (accumulating in place is safe).

__device__ __forceinline__ bool zk_is_zero(const u64 a[4]) {
    return (a[0] | a[1] | a[2] | a[3]) == 0;  // rows are canonical, so zero is all-zero limbs
}

__device__ __forceinline__ void zk_set_identity(u64 r[12]) {
    const u64 one[4] = {ZK_ONE0, ZK_ONE1, ZK_ONE2, ZK_ONE3};
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        r[j] = one[j];
        r[4 + j] = one[j];
        r[8 + j] = 0;
    }
}

__device__ __forceinline__ void zk_store_point(const u64 x[4], const u64 y[4], const u64 z[4], u64 r[12]) {
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        r[j] = x[j];
        r[4 + j] = y[j];
        r[8 + j] = z[j];
    }
}

// 2 (X1, Y1, Z1), EFD dbl-2009-l: A = X1^2, B = Y1^2, C = B^2,
// D = 2((X1 + B)^2 - A - C), E = 3A, F = E^2, X3 = F - 2D,
// Y3 = E (D - X3) - 8C, Z3 = 2 Y1 Z1.
__device__ void zk_jac_double(const u64 p[12], u64 r[12]) {
    const u64 *x1 = p, *y1 = p + 4, *z1 = p + 8;
    if (zk_is_zero(z1) || zk_is_zero(y1)) {
        // The identity, or a vertical tangent: only a point of order 2 has
        // one, and BN254 G1 has none (its order r is odd)
        zk_set_identity(r);
        return;
    }
    u64 a[4], b[4], c[4], d[4], e[4], f[4], t[4], x3[4], y3[4], z3[4];
    zk_mont_mul(x1, x1, a);
    zk_mont_mul(y1, y1, b);
    zk_mont_mul(b, b, c);
    zk_add(x1, b, t);
    zk_mont_mul(t, t, t);
    zk_sub(t, a, t);
    zk_sub(t, c, t);
    zk_add(t, t, d);
    zk_add(a, a, e);
    zk_add(e, a, e);
    zk_mont_mul(e, e, f);
    zk_add(d, d, t);
    zk_sub(f, t, x3);
    zk_sub(d, x3, t);
    zk_mont_mul(e, t, y3);
    zk_add(c, c, t);  // 2C
    zk_add(t, t, t);  // 4C
    zk_add(t, t, t);  // 8C
    zk_sub(y3, t, y3);
    zk_mont_mul(y1, z1, t);
    zk_add(t, t, z3);
    zk_store_point(x3, y3, z3, r);
}

// (X1, Y1, Z1) + (X2, Y2, Z2), EFD add-2007-bl: Z1Z1 = Z1^2, Z2Z2 = Z2^2,
// U1 = X1 Z2Z2, U2 = X2 Z1Z1, S1 = Y1 Z2 Z2Z2, S2 = Y2 Z1 Z1Z1, H = U2 - U1,
// I = (2H)^2, J = H I, r = 2(S2 - S1), V = U1 I, X3 = r^2 - J - 2V,
// Y3 = r (V - X3) - 2 S1 J, Z3 = ((Z1 + Z2)^2 - Z1Z1 - Z2Z2) H.
__device__ void zk_jac_add(const u64 p[12], const u64 q[12], u64 r[12]) {
    const u64 *x1 = p, *y1 = p + 4, *z1 = p + 8;
    const u64 *x2 = q, *y2 = q + 4, *z2 = q + 8;
    if (zk_is_zero(z1)) {
        zk_store_point(x2, y2, z2, r);
        return;
    }
    if (zk_is_zero(z2)) {
        zk_store_point(x1, y1, z1, r);
        return;
    }
    u64 z1z1[4], z2z2[4], u1[4], u2[4], s1[4], s2[4], h[4], rr[4], i[4], j[4], v[4], t[4];
    u64 x3[4], y3[4], z3[4];
    zk_mont_mul(z1, z1, z1z1);
    zk_mont_mul(z2, z2, z2z2);
    zk_mont_mul(x1, z2z2, u1);
    zk_mont_mul(x2, z1z1, u2);
    zk_mont_mul(y1, z2, s1);
    zk_mont_mul(s1, z2z2, s1);
    zk_mont_mul(y2, z1, s2);
    zk_mont_mul(s2, z1z1, s2);
    zk_sub(u2, u1, h);
    zk_sub(s2, s1, rr);
    zk_add(rr, rr, rr);
    if (zk_is_zero(h)) {
        // Same affine x: equal points (double) or inverses (identity). The
        // general formula would give Z3 = 0 for both, wrongly for the first
        if (zk_is_zero(rr)) {
            zk_jac_double(p, r);
        } else {
            zk_set_identity(r);
        }
        return;
    }
    zk_add(h, h, t);
    zk_mont_mul(t, t, i);
    zk_mont_mul(h, i, j);
    zk_mont_mul(u1, i, v);
    zk_mont_mul(rr, rr, x3);
    zk_sub(x3, j, x3);
    zk_add(v, v, t);
    zk_sub(x3, t, x3);
    zk_sub(v, x3, t);
    zk_mont_mul(rr, t, y3);
    zk_mont_mul(s1, j, t);
    zk_add(t, t, t);
    zk_sub(y3, t, y3);
    zk_add(z1, z2, t);
    zk_mont_mul(t, t, z3);
    zk_sub(z3, z1z1, z3);
    zk_sub(z3, z2z2, z3);
    zk_mont_mul(z3, h, z3);
    zk_store_point(x3, y3, z3, r);
}

// (X1, Y1, Z1) + (x2, y2) for an affine point (never the identity), EFD
// madd-2007-bl: Z1Z1 = Z1^2, U2 = x2 Z1Z1, S2 = y2 Z1 Z1Z1, H = U2 - X1,
// HH = H^2, I = 4 HH, J = H I, r = 2(S2 - Y1), V = X1 I, X3 = r^2 - J - 2V,
// Y3 = r (V - X3) - 2 Y1 J, Z3 = (Z1 + H)^2 - Z1Z1 - HH.
__device__ void zk_jac_add_affine(const u64 p[12], const u64 q[8], u64 r[12]) {
    const u64 *x1 = p, *y1 = p + 4, *z1 = p + 8;
    const u64 *x2 = q, *y2 = q + 4;
    if (zk_is_zero(z1)) {
        const u64 one[4] = {ZK_ONE0, ZK_ONE1, ZK_ONE2, ZK_ONE3};
        zk_store_point(x2, y2, one, r);
        return;
    }
    u64 z1z1[4], u2[4], s2[4], h[4], hh[4], i[4], j[4], rr[4], v[4], t[4], x3[4], y3[4], z3[4];
    zk_mont_mul(z1, z1, z1z1);
    zk_mont_mul(x2, z1z1, u2);
    zk_mont_mul(y2, z1, s2);
    zk_mont_mul(s2, z1z1, s2);
    zk_sub(u2, x1, h);
    zk_sub(s2, y1, rr);
    zk_add(rr, rr, rr);
    if (zk_is_zero(h)) {
        if (zk_is_zero(rr)) {
            zk_jac_double(p, r);
        } else {
            zk_set_identity(r);
        }
        return;
    }
    zk_mont_mul(h, h, hh);
    zk_add(hh, hh, i);
    zk_add(i, i, i);
    zk_mont_mul(h, i, j);
    zk_mont_mul(x1, i, v);
    zk_mont_mul(rr, rr, x3);
    zk_sub(x3, j, x3);
    zk_add(v, v, t);
    zk_sub(x3, t, x3);
    zk_sub(v, x3, t);
    zk_mont_mul(rr, t, y3);
    zk_mont_mul(y1, j, t);
    zk_add(t, t, t);
    zk_sub(y3, t, y3);
    zk_add(z1, h, t);
    zk_mont_mul(t, t, z3);
    zk_sub(z3, z1z1, z3);
    zk_sub(z3, hh, z3);
    zk_store_point(x3, y3, z3, r);
}

// Element-wise kernels, one thread per point. A step of 0 broadcasts a
// single-point operand.
extern "C" __global__ void zk_g1_add(const u64* a, long long a_step, const u64* b, long long b_step,
                                     u64* out, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    u64 p[12], q[12], r[12];
    #pragma unroll
    for (int k = 0; k < 12; ++k) {
        p[k] = a[i * a_step + k];
        q[k] = b[i * b_step + k];
    }
    zk_jac_add(p, q, r);
    #pragma unroll
    for (int k = 0; k < 12; ++k) out[12 * i + k] = r[k];
}

extern "C" __global__ void zk_g1_add_affine(const u64* a, long long a_step, const u64* b, long long b_step,
                                            u64* out, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    u64 p[12], q[8], r[12];
    #pragma unroll
    for (int k = 0; k < 12; ++k) p[k] = a[i * a_step + k];
    #pragma unroll
    for (int k = 0; k < 8; ++k) q[k] = b[i * b_step + k];
    zk_jac_add_affine(p, q, r);
    #pragma unroll
    for (int k = 0; k < 12; ++k) out[12 * i + k] = r[k];
}

extern "C" __global__ void zk_g1_double(const u64* a, u64* out, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    u64 p[12], r[12];
    #pragma unroll
    for (int k = 0; k < 12; ++k) p[k] = a[12 * i + k];
    zk_jac_double(p, r);
    #pragma unroll
    for (int k = 0; k < 12; ++k) out[12 * i + k] = r[k];
}
"""
_KERNEL_NAMES = ("zk_g1_add", "zk_g1_add_affine", "zk_g1_double")
_compiled: dict[str, Any] = {}  # filled on first use, emptied again if the self-test fails


def _kernels() -> dict[str, Any]:  # gpu-only
    if not _compiled:
        module = compile_module(BN254_FQ.cuda_source() + _POINT_SOURCE, "zkgpu_curve")
        _compiled.update({name: module.get_function(name) for name in _KERNEL_NAMES})
        try:
            _self_test()
        except Exception:
            _compiled.clear()  # never keep kernels that failed their self-test
            raise
    return _compiled


def _self_test() -> None:  # gpu-only
    """
    Check the freshly compiled kernels against zkgpu.curve, once per process.

    Same reason as Field256's self-test: correct source can still compile to
    wrong machine code (gpu_backend.compile_module), and this point code is
    where that first showed. Covers every branch of the formulas: general Z,
    equal points (doubling), inverse points (identity) and the identity on
    either side. Small multiples of G keep this to a few milliseconds.
    """
    j5, j7, j11 = (jacobian_mul(G, k) for k in (5, 7, 11))  # Z != 1, as in real accumulations
    neg5 = (j5[0], -j5[1] % BN254_FQ.modulus, j5[2])
    p7, p11 = from_jacobian(j7), from_jacobian(j11)
    pairs = [(j5, j7), (j5, j5), (j5, neg5), (JACOBIAN_IDENTITY, j7), (j5, JACOBIAN_IDENTITY)]
    mixed = [(j5, p7), (j11, p11), (jacobian_add(j11, jacobian_double(j11)), p11), (JACOBIAN_IDENTITY, p7)]
    mixed.append(((j11[0], -j11[1] % BN254_FQ.modulus, j11[2]), p11))  # -11G + 11G
    singles = [j5, j11, JACOBIAN_IDENTITY]
    checks = {
        "add": (
            add(jacobian_to_device(a for a, _ in pairs), jacobian_to_device(b for _, b in pairs)),
            [jacobian_add(a, b) for a, b in pairs],
        ),
        "add_affine": (
            add_affine(jacobian_to_device(a for a, _ in mixed), affine_to_device(q for _, q in mixed)),
            [jacobian_add_affine(a, *q.affine()) for a, q in mixed],
        ),
        "double": (double(jacobian_to_device(singles)), [jacobian_double(j) for j in singles]),
    }
    for op, (rows, expected) in checks.items():
        if jacobian_ints(rows) != expected:
            raise RuntimeError(
                f"curve_gpu: the compiled {op} kernel failed its self-test, so the CUDA compiler "
                "produced wrong machine code for this GPU. Please report it with the output of "
                "`python scripts/check_gpu.py`."
            )


def compile_kernels() -> None:  # gpu-only
    """
    Compile, load and self-test the point kernels now instead of on first use.

    Raises RuntimeError if they fail their self-test.
    """
    _kernels()


# ------------------------------------------------------------------
# Conversion
# ------------------------------------------------------------------


def jacobian_to_device(points: Iterable[Jacobian]) -> Any:
    """(n, 3, 4) rows for Jacobian (X, Y, Z) int triples, e.g. from zkgpu.curve."""
    flat = [c for point in points for c in point]
    if len(flat) % 3:
        raise ValueError("Jacobian points must be (X, Y, Z) triples")
    return BN254_FQ.from_ints(flat).reshape(-1, 3, LIMBS)


def to_device(points: Iterable[Point]) -> Any:
    """(n, 3, 4) Jacobian rows for affine Points (Z = 1; the identity becomes (1, 1, 0))."""
    return jacobian_to_device(to_jacobian(p) for p in points)


def affine_to_device(points: Iterable[Point]) -> Any:
    """(n, 2, 4) affine rows, the second operand of add_affine; the identity is rejected."""
    flat: list[int] = []
    for p in points:
        if p.is_identity():
            raise ValueError("affine rows cannot hold the point at infinity; leave it out (it adds nothing)")
        flat.extend(p.affine())
    return BN254_FQ.from_ints(flat).reshape(-1, 2, LIMBS)


def jacobian_ints(x: Any) -> list[Jacobian]:
    """The exact (X, Y, Z) coordinates of Jacobian rows, as Python ints."""
    _point_count(x, "x", 3)
    flat = BN254_FQ.to_ints(x.reshape(-1, LIMBS))
    return [(flat[k], flat[k + 1], flat[k + 2]) for k in range(0, len(flat), 3)]


def from_device(x: Any) -> list[Point]:
    """Affine Points for Jacobian rows; each is re-validated on the curve by Point."""
    return [from_jacobian(j) for j in jacobian_ints(x)]


# ------------------------------------------------------------------
# Arithmetic
# ------------------------------------------------------------------


def add(a: Any, b: Any) -> Any:
    """Element-wise a + b of Jacobian rows (either may be one broadcast point)."""
    if not _gpu_available:
        return _fallback("add", a, b, 3, jacobian_add)
    return _launch_binary("zk_g1_add", a, b, 3)  # gpu-only


def add_affine(a: Any, b: Any) -> Any:
    """Element-wise a + b of Jacobian rows *a* and affine rows *b* (the cheaper mixed addition)."""
    if not _gpu_available:
        return _fallback("add_affine", a, b, 2, lambda j, q: jacobian_add_affine(j, q[0], q[1]))
    return _launch_binary("zk_g1_add_affine", a, b, 2)  # gpu-only


def double(a: Any) -> Any:
    """Element-wise 2a of Jacobian rows."""
    n = _point_count(a, "a", 3)
    if not _gpu_available:
        warn_cpu_fallback("curve_gpu.double", "zkgpu.curve on Python integers")
        return jacobian_to_device(jacobian_double(j) for j in jacobian_ints(a))
    return _launch_unary("zk_g1_double", a, n)  # gpu-only


# ------------------------------------------------------------------
# Internals
# ------------------------------------------------------------------


def _point_count(x: Any, name: str, coords: int) -> int:
    """Points in *x*, after checking it is an (n, coords, 4) uint64 array."""
    if getattr(x, "dtype", None) != np.uint64 or x.ndim != 3 or x.shape[1:] != (coords, LIMBS):
        kind = "Jacobian" if coords == 3 else "affine"
        raise TypeError(
            f"{name} must be an (n, {coords}, {LIMBS}) uint64 array of {kind} points, "
            f"got {getattr(x, 'dtype', type(x).__name__)} with shape {getattr(x, 'shape', None)}"
        )
    return int(x.shape[0])


def _broadcast(a: Any, b: Any, b_coords: int) -> int:
    rows_a, rows_b = _point_count(a, "a", 3), _point_count(b, "b", b_coords)
    if rows_a == rows_b or rows_b == 1:
        return rows_a
    if rows_a == 1:
        return rows_b
    raise ValueError(f"cannot broadcast {rows_a} points against {rows_b} points")


def _run(kernel: Any, threads: int, *args: Any) -> None:  # gpu-only
    kernel(((threads + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK,), (_THREADS_PER_BLOCK,), args)


def _launch_unary(name: str, a: Any, n: int) -> Any:  # gpu-only
    src = cp.ascontiguousarray(a)
    out = cp.empty_like(src)
    if n:
        _run(_kernels()[name], n, src, out, np.int64(n))
    return out


def _launch_binary(name: str, a: Any, b: Any, b_coords: int) -> Any:  # gpu-only
    n = _broadcast(a, b, b_coords)
    a, b = cp.ascontiguousarray(a), cp.ascontiguousarray(b)
    out = cp.empty((n, 3, LIMBS), dtype=cp.uint64)
    if n:
        step_a = np.int64(3 * LIMBS if a.shape[0] > 1 else 0)
        step_b = np.int64(b_coords * LIMBS if b.shape[0] > 1 else 0)
        _run(_kernels()[name], n, a, step_a, b, step_b, out, np.int64(n))
    return out


def _fallback(op: str, a: Any, b: Any, b_coords: int, combine: Callable[[Jacobian, Any], Jacobian]) -> Any:
    warn_cpu_fallback(f"curve_gpu.{op}", "zkgpu.curve on Python integers")
    n = _broadcast(a, b, b_coords)
    xs = jacobian_ints(a)
    flat = BN254_FQ.to_ints(b.reshape(-1, LIMBS))
    ys = [tuple(flat[k : k + b_coords]) for k in range(0, len(flat), b_coords)]
    if len(xs) == 1 and n != 1:  # broadcast a single point
        xs = xs * n
    if len(ys) == 1 and n != 1:
        ys = ys * n
    return jacobian_to_device(combine(x, y) for x, y in zip(xs, ys, strict=True))
