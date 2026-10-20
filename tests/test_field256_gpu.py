"""
tests/test_field256_gpu.py — zkgpu/field256_gpu.py on CUDA and through its CPU fallback.

Like test_field_gpu.py, every arithmetic case runs on the GPU (skipped
without one) and through the Python-integer fallback, so CI checks the limb
packing and the Montgomery constants, and a GPU machine checks the kernels.
"""

import random

import numpy as np
import pytest

from zkgpu import field256_gpu
from zkgpu.curve import P
from zkgpu.field import FIELD_MODULUS as R_SCALAR
from zkgpu.field256_gpu import BN254_FQ, BN254_FR, Field256
from zkgpu.gpu_backend import GpuFallbackWarning

pytestmark = pytest.mark.filterwarnings("ignore::zkgpu.gpu_backend.GpuFallbackWarning")

GPU = pytest.param("gpu", marks=pytest.mark.gpu)
FALLBACK = "fallback"
BACKENDS = [GPU, FALLBACK]

# BN254's two fields, the largest allowed modulus shape (2^255 - 19, the
# Curve25519 prime), the smallest, and one whose two top limbs are zero
FIELDS = [
    pytest.param(BN254_FR, id="bn254-r"),
    pytest.param(BN254_FQ, id="bn254-p"),
    pytest.param(Field256(2**255 - 19), id="2^255-19"),
    pytest.param(Field256(3), id="3"),
    pytest.param(Field256(2**64 + 13), id="2^64+13"),
]


@pytest.fixture
def backend(request, monkeypatch):
    if request.param == FALLBACK:
        monkeypatch.setattr(field256_gpu, "_gpu_available", False)
    return request.param


def operands(p: int, seed: int) -> tuple[list[int], list[int]]:
    """Every pair of carry-sensitive edge values, plus random residues."""
    edge = sorted({e % p for e in (0, 1, 2, p - 1, p - 2, p // 2, 2**64 - 1, 2**128 - 1, 2**192 - 1)})
    rng = random.Random(seed)
    a = [x for x in edge for _ in edge] + [rng.randrange(p) for _ in range(500)]
    b = edge * len(edge) + [rng.randrange(p) for _ in range(500)]
    return a, b


# ------------------------------------------------------------------
# Montgomery constants: known answers from gnark/arkworks
# ------------------------------------------------------------------


def test_montgomery_constants_match_published_values():
    # -p^-1 mod 2^64, which gnark calls qInvNeg
    assert BN254_FR._n_inv == 0xC2E1F593EFFFFFFF
    assert BN254_FQ._n_inv == 0x87D20782E4866389
    for field in (BN254_FR, BN254_FQ):
        assert field._n_inv * field.modulus % 2**64 == 2**64 - 1
        assert field._r2 == pow(2, 512, field.modulus)


@pytest.mark.parametrize("backend", BACKENDS, indirect=True)
def test_rows_hold_montgomery_form(backend):
    """One is stored as R mod r, the value arkworks and gnark use for Fr's one."""
    rows = BN254_FR.from_ints([1, 0])
    assert rows.dtype == np.uint64 and rows.shape == (2, 4)
    r_mod = 0x0E0A77C19A07DF2F666EA36F7879462E36FC76959F60CD29AC96341C4FFFFFFB
    assert rows.tolist()[0] == [(r_mod >> (64 * j)) & (2**64 - 1) for j in range(4)]
    assert rows.tolist()[1] == [0, 0, 0, 0]


@pytest.mark.gpu
def test_backends_store_identical_rows(monkeypatch):
    """The same bits on GPU and CPU, so rows can move between the two paths."""
    values = [0, 1, R_SCALAR - 1, 12345, 2**200 + 7]
    gpu_rows = BN254_FR.from_ints(values).get()
    monkeypatch.setattr(field256_gpu, "_gpu_available", False)
    assert np.array_equal(BN254_FR.from_ints(values), gpu_rows)


@pytest.mark.parametrize("p", [R_SCALAR, P, 2**255 - 19, 2**255 - 1], ids=["r", "p", "2^255-19", "2^255-1"])
def test_cios_fits_in_five_words(p):
    """
    The kernel keeps five words of state, not CIOS's general six, citing a
    bound that needs p < 2^255. Replay its steps on Python ints and check
    that bound where it is tightest: the largest moduli and operands.
    """
    word = 2**64
    n_inv = -pow(p, -1, word) % word
    extremes = [p - 1, p - 2, 2**256 % p, sum((word - 1) << (64 * j) for j in range(3)) % p]
    rng = random.Random(p)
    randoms = [(rng.randrange(p), rng.randrange(p)) for _ in range(2000)]
    for a, b in [(x, y) for x in extremes for y in extremes] + randoms:
        t = 0
        for i in range(4):
            t += a * ((b >> (64 * i)) % word)
            assert t < 2**320
            t += (t % word) * n_inv % word * p
            assert t < 2**320 and t % word == 0
            t //= word
            assert t < 2 * p
        assert (t - p if t >= p else t) == a * b * pow(2**256, -1, p) % p


# ------------------------------------------------------------------
# Agreement with Python integers
# ------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, indirect=True)
@pytest.mark.parametrize("field", FIELDS)
def test_round_trip(backend, field):
    p = field.modulus
    values = [0, 1, p - 1, p, p + 1, -1, -p, 3 * p + 5, 2**256 + 9]  # unreduced inputs too
    assert field.to_ints(field.from_ints(values)) == [v % p for v in values]


@pytest.mark.parametrize("backend", BACKENDS, indirect=True)
@pytest.mark.parametrize("field", FIELDS)
def test_mul_matches_python(backend, field):
    p = field.modulus
    a, b = operands(p, 1)
    got = field.to_ints(field.mul(field.from_ints(a), field.from_ints(b)))
    assert got == [x * y % p for x, y in zip(a, b, strict=True)]


@pytest.mark.parametrize("backend", BACKENDS, indirect=True)
@pytest.mark.parametrize("field", FIELDS)
def test_add_matches_python(backend, field):
    p = field.modulus
    a, b = operands(p, 2)
    got = field.to_ints(field.add(field.from_ints(a), field.from_ints(b)))
    assert got == [(x + y) % p for x, y in zip(a, b, strict=True)]


@pytest.mark.parametrize("backend", BACKENDS, indirect=True)
@pytest.mark.parametrize("field", FIELDS)
def test_sub_matches_python(backend, field):
    p = field.modulus
    a, b = operands(p, 3)
    got = field.to_ints(field.sub(field.from_ints(a), field.from_ints(b)))
    assert got == [(x - y) % p for x, y in zip(a, b, strict=True)]


@pytest.mark.parametrize("backend", BACKENDS, indirect=True)
def test_operations_compose(backend):
    """Results feed straight back in: (a + b)(a - b) = a^2 - b^2, all in Montgomery form."""
    F, p = BN254_FQ, P
    a, b = operands(p, 4)
    A, B = F.from_ints(a), F.from_ints(b)
    lhs = F.to_ints(F.mul(F.add(A, B), F.sub(A, B)))
    rhs = F.to_ints(F.sub(F.mul(A, A), F.mul(B, B)))
    assert lhs == rhs == [(x * x - y * y) % p for x, y in zip(a, b, strict=True)]


# ------------------------------------------------------------------
# Broadcasting and shapes
# ------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, indirect=True)
def test_single_row_broadcasts_on_either_side(backend):
    F, p = BN254_FR, R_SCALAR
    values = [3, 5, p - 1, 0]
    rows, c = F.from_ints(values), F.from_ints([p - 2])
    assert F.to_ints(F.mul(rows, c)) == [v * (p - 2) % p for v in values]
    assert F.to_ints(F.sub(c, rows)) == [(p - 2 - v) % p for v in values]
    assert F.to_ints(F.add(c, c)) == [(2 * (p - 2)) % p]


@pytest.mark.parametrize("backend", BACKENDS, indirect=True)
def test_empty_batches(backend):
    F = BN254_FR
    empty = F.from_ints([])
    assert empty.shape == (0, 4)
    assert F.to_ints(F.mul(empty, empty)) == []
    assert F.to_ints(F.add(empty, F.from_ints([7]))) == []


@pytest.mark.parametrize("backend", BACKENDS, indirect=True)
def test_mismatched_row_counts_raise(backend):
    F = BN254_FR
    with pytest.raises(ValueError, match="broadcast"):
        F.mul(F.from_ints([1, 2]), F.from_ints([1, 2, 3]))


@pytest.mark.parametrize("backend", BACKENDS, indirect=True)
@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((3, 4), dtype=np.int64),  # signed limbs
        np.zeros((3, 3), dtype=np.uint64),  # wrong limb count
        np.zeros(12, dtype=np.uint64),  # flat
        [[0, 0, 0, 0]],  # not an array
    ],
    ids=["int64", "3-limbs", "flat", "list"],
)
def test_malformed_rows_raise(backend, bad):
    with pytest.raises(TypeError, match=r"\(n, 4\) uint64"):
        BN254_FR.mul(bad, BN254_FR.from_ints([1]))


# ------------------------------------------------------------------
# Construction and fallback
# ------------------------------------------------------------------


@pytest.mark.parametrize("modulus", [2, 1, 0, -7, 2**64, 2**255 + 1, 2**256 - 189])
def test_invalid_moduli_raise(modulus):
    with pytest.raises(ValueError, match="odd and in"):
        Field256(modulus)


def test_float_modulus_raises():
    with pytest.raises(TypeError):
        Field256(7.0)


def test_fallback_warns(monkeypatch):
    monkeypatch.setattr(field256_gpu, "_gpu_available", False)
    with pytest.warns(GpuFallbackWarning, match="Field256.mul"):
        BN254_FR.mul(BN254_FR.from_ints([2]), BN254_FR.from_ints([3]))


def test_repr_names_the_modulus():
    assert repr(Field256(3)) == "Field256(0x3)"
