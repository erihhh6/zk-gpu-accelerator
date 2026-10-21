"""
tests/test_curve_gpu.py — zkgpu/curve_gpu.py (BN254 G1 point arithmetic) on CUDA and through its CPU fallback.

The GPU uses zkgpu.curve's formulas, so every result is compared with the
CPU's exact Jacobian coordinates, not just the affine point. Inputs have
Z != 1 (from jacobian_mul), so the Z-dependent terms are exercised too.
"""

import random

import numpy as np
import pytest

from zkgpu import curve_gpu
from zkgpu.curve import (
    JACOBIAN_IDENTITY,
    G,
    P,
    Point,
    from_jacobian,
    jacobian_add,
    jacobian_add_affine,
    jacobian_double,
    jacobian_mul,
)
from zkgpu.field import FIELD_MODULUS as R
from zkgpu.field256_gpu import BN254_FQ
from zkgpu.gpu_backend import GpuFallbackWarning

pytestmark = pytest.mark.filterwarnings("ignore::zkgpu.gpu_backend.GpuFallbackWarning")

GPU = pytest.param("gpu", marks=pytest.mark.gpu)
FALLBACK = "fallback"
BACKENDS = [GPU, FALLBACK]


@pytest.fixture
def backend(request, monkeypatch):
    if request.param == FALLBACK:
        monkeypatch.setattr(curve_gpu, "_gpu_available", False)
        monkeypatch.setattr("zkgpu.field256_gpu._gpu_available", False)
    return request.param


def jacobians(n: int, seed: int) -> list[tuple[int, int, int]]:
    """Random multiples of G in Jacobian form with Z != 1."""
    rng = random.Random(seed)
    return [jacobian_mul(G * rng.randrange(1, R), rng.randrange(2, 2**64)) for _ in range(n)]


def negate(j):
    return (j[0], -j[1] % P, j[2])


# ------------------------------------------------------------------
# Exact agreement with zkgpu.curve
# ------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, indirect=True)
def test_add_matches_cpu_formulas(backend):
    js = jacobians(12, 1)
    # Random pairs, then every special case: equal (doubling), inverse
    # (identity), and the identity on either side
    a = js + js[:4] + js[4:8] + [JACOBIAN_IDENTITY] * 2 + js[8:10]
    b = js[::-1] + js[:4] + [negate(j) for j in js[4:8]] + js[10:12] + [JACOBIAN_IDENTITY] * 2
    got = curve_gpu.add(curve_gpu.jacobian_to_device(a), curve_gpu.jacobian_to_device(b))
    assert curve_gpu.jacobian_ints(got) == [jacobian_add(x, y) for x, y in zip(a, b, strict=True)]


@pytest.mark.parametrize("backend", BACKENDS, indirect=True)
def test_add_affine_matches_cpu_formulas(backend):
    js = jacobians(10, 2)
    pts = [from_jacobian(j) for j in jacobians(10, 3)]
    same = [from_jacobian(j) for j in js[:4]]  # affine forms of the Z != 1 points: doubling case
    acc = js + js[:4] + [negate(j) for j in js[4:8]] + [JACOBIAN_IDENTITY] * 2
    q = pts + same + [from_jacobian(j) for j in js[4:8]] + pts[:2]
    got = curve_gpu.add_affine(curve_gpu.jacobian_to_device(acc), curve_gpu.affine_to_device(q))
    expected = [jacobian_add_affine(j, p.x, p.y) for j, p in zip(acc, q, strict=True)]
    assert curve_gpu.jacobian_ints(got) == expected


@pytest.mark.parametrize("backend", BACKENDS, indirect=True)
def test_double_matches_cpu_formulas(backend):
    # (1, 2) is G itself, where X3 and Y3 wrap below 0. (5, 0, 1) is not a curve
    # point (BN254 G1 has no y = 0), but it pins down the vertical-tangent guard,
    # which both sides map to the identity.
    js = [*jacobians(10, 4), JACOBIAN_IDENTITY, (1, 2, 1), (5, 0, 1)]
    got = curve_gpu.double(curve_gpu.jacobian_to_device(js))
    assert curve_gpu.jacobian_ints(got) == [jacobian_double(j) for j in js]


@pytest.mark.parametrize("backend", BACKENDS, indirect=True)
def test_chained_doublings_and_additions(backend):
    """Results feed straight back in: 2^10 G + 3 G, all on the device, equals 1027 G."""
    acc = curve_gpu.to_device([G])
    for _ in range(10):
        acc = curve_gpu.double(acc)
    acc = curve_gpu.add_affine(acc, curve_gpu.affine_to_device([G * 3]))
    assert curve_gpu.from_device(acc) == [G * 1027]


# ------------------------------------------------------------------
# Conversion
# ------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, indirect=True)
def test_points_round_trip(backend):
    pts = [G, G * 5, -G, Point.identity()]
    rows = curve_gpu.to_device(pts)
    assert rows.shape == (4, 3, 4) and rows.dtype == np.uint64
    assert curve_gpu.from_device(rows) == pts
    assert curve_gpu.jacobian_ints(rows)[3] == JACOBIAN_IDENTITY


@pytest.mark.parametrize("backend", BACKENDS, indirect=True)
def test_from_device_revalidates_points(backend):
    """Rows that are not on the curve (wrong arithmetic, corrupted memory) are rejected on the way out."""
    rows = curve_gpu.jacobian_to_device([(1, 3, 1)])  # x = 1, y = 3: 9 != 1 + 3
    with pytest.raises(ValueError, match="not on the BN254 curve"):
        curve_gpu.from_device(rows)


def test_affine_rows_reject_the_identity():
    with pytest.raises(ValueError, match="point at infinity"):
        curve_gpu.affine_to_device([G, Point.identity()])


def test_jacobian_input_must_be_triples():
    with pytest.raises(ValueError, match="triples"):
        curve_gpu.jacobian_to_device([(1, 2)])


# ------------------------------------------------------------------
# Broadcasting and shapes
# ------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, indirect=True)
def test_single_point_broadcasts_on_either_side(backend):
    js = jacobians(3, 5)
    rows, one = curve_gpu.jacobian_to_device(js), curve_gpu.to_device([G])
    g = (1, 2, 1)
    assert curve_gpu.jacobian_ints(curve_gpu.add(rows, one)) == [jacobian_add(j, g) for j in js]
    assert curve_gpu.jacobian_ints(curve_gpu.add(one, rows)) == [jacobian_add(g, j) for j in js]
    got = curve_gpu.add_affine(rows, curve_gpu.affine_to_device([G]))
    assert curve_gpu.jacobian_ints(got) == [jacobian_add_affine(j, 1, 2) for j in js]


@pytest.mark.parametrize("backend", BACKENDS, indirect=True)
def test_empty_batches(backend):
    empty = curve_gpu.to_device([])
    assert empty.shape == (0, 3, 4)
    assert curve_gpu.from_device(curve_gpu.add(empty, empty)) == []
    assert curve_gpu.from_device(curve_gpu.double(empty)) == []


@pytest.mark.parametrize("backend", BACKENDS, indirect=True)
def test_mismatched_batches_raise(backend):
    with pytest.raises(ValueError, match="broadcast"):
        curve_gpu.add(curve_gpu.to_device([G, G]), curve_gpu.to_device([G, G, G]))


@pytest.mark.parametrize("backend", BACKENDS, indirect=True)
def test_malformed_rows_raise(backend):
    affine = curve_gpu.affine_to_device([G])
    with pytest.raises(TypeError, match=r"\(n, 3, 4\) uint64 array of Jacobian points"):
        curve_gpu.add(affine, affine)  # affine rows where Jacobian ones belong
    with pytest.raises(TypeError, match=r"\(n, 2, 4\) uint64 array of affine points"):
        curve_gpu.add_affine(curve_gpu.to_device([G]), curve_gpu.to_device([G]))
    with pytest.raises(TypeError, match="Jacobian points"):
        curve_gpu.double(BN254_FQ.from_ints([1, 2, 3]))  # field rows, not points


def test_fallback_warns(monkeypatch):
    monkeypatch.setattr(curve_gpu, "_gpu_available", False)
    monkeypatch.setattr("zkgpu.field256_gpu._gpu_available", False)
    with pytest.warns(GpuFallbackWarning, match="curve_gpu.double"):
        curve_gpu.double(curve_gpu.to_device([G]))


# ------------------------------------------------------------------
# Load-time self-test
# ------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("zk_add(t, t, z3);", "zk_sub(t, t, z3);"),  # double: Z3 = 0 instead of 2 Y1 Z1
        ("zk_mont_mul(z3, h, z3);", "zk_mont_mul(z3, z3, z3);"),  # add: Z3 without the H factor
        ("zk_sub(z3, hh, z3);", "zk_add(z3, hh, z3);"),  # mixed add: wrong Z3
    ],
    ids=["double", "add", "add_affine"],
)
def test_miscompiled_kernels_fail_their_self_test(monkeypatch, old, new):
    assert curve_gpu._POINT_SOURCE.count(old) == 1
    monkeypatch.setattr(curve_gpu, "_POINT_SOURCE", curve_gpu._POINT_SOURCE.replace(old, new))
    monkeypatch.setattr(curve_gpu, "_compiled", {})  # compile the patched source
    with pytest.raises(RuntimeError, match="failed its self-test"):
        curve_gpu.compile_kernels()
    assert curve_gpu._compiled == {}  # nothing that failed is kept for later calls
