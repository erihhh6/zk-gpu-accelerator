"""tests/test_ntt.py — Correctness tests for src/ntt.py (and GPU if available)."""

import random

import numpy as np
import pytest

from src import ntt_gpu as ntt_gpu_module
from src.field import FIELD_MODULUS as BN254_R
from src.gpu_backend import GpuFallbackWarning, cp
from src.ntt import ntt, polynomial_mul, MOD
from src.ntt_gpu import ntt_gpu
from tests.known_bugs import known_bug

ROOT_3_IS_INVALID = (
    "the primitive root is hardcoded to 3, a quadratic residue mod this prime, "
    "so the computed 'roots of unity' have the wrong order"
)

# NTT-friendly primes q = k·2^m + 1, each supporting power-of-two sizes up to 2^m
MODULI = [
    pytest.param(MOD, id="998244353"),  # 119·2^23 + 1, the default
    pytest.param(7_340_033, id="7340033"),  # 7·2^20 + 1
    pytest.param(12_289, id="12289", marks=known_bug(ROOT_3_IS_INVALID, raises=AssertionError)),  # 3·2^12 + 1
    pytest.param(BN254_R, id="bn254_r", marks=known_bug(ROOT_3_IS_INVALID, raises=AssertionError)),  # 2-adicity 28
]


def naive_mul(p: list[int], q: list[int], m: int) -> list[int]:
    """O(N²) schoolbook convolution — the reference for polynomial_mul."""
    out = [0] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            out[i + j] = (out[i + j] + a * b) % m
    return out


# ------------------------------------------------------------------
# CPU NTT tests
# ------------------------------------------------------------------


def test_ntt_inverse_small():
    """INTT(NTT(a)) == a for a small vector."""
    a = [1, 2, 3, 4, 5, 6, 7, 8]
    transformed = ntt(a, MOD)
    recovered = ntt(transformed, MOD, invert=True)
    assert a == recovered


def test_ntt_inverse_random():
    """INTT(NTT(a)) == a for a random vector of length 1024."""
    random.seed(7)
    n = 1024
    a = [random.randint(0, MOD - 1) for _ in range(n)]
    recovered = ntt(ntt(a, MOD), MOD, invert=True)
    assert a == recovered


def test_ntt_linearity():
    """NTT is linear: NTT(a + b) == NTT(a) + NTT(b) pointwise."""
    random.seed(13)
    n = 64
    a = [random.randint(0, MOD - 1) for _ in range(n)]
    b = [random.randint(0, MOD - 1) for _ in range(n)]
    ab = [(x + y) % MOD for x, y in zip(a, b)]
    lhs = ntt(ab, MOD)
    rhs = [(x + y) % MOD for x, y in zip(ntt(a, MOD), ntt(b, MOD))]
    assert lhs == rhs


def test_ntt_single_element():
    """NTT of a single element is the element itself."""
    assert ntt([42], MOD) == [42]
    assert ntt([42], MOD, invert=True) == [42]


def test_ntt_non_power_of_two_raises():
    with pytest.raises(ValueError):
        ntt([1, 2, 3], MOD)


def test_ntt_polynomial_mul_simple():
    """(1 + x) * (1 + x) = 1 + 2x + x²."""
    result = polynomial_mul([1, 1], [1, 1], MOD)
    assert result == [1, 2, 1]


def test_ntt_polynomial_mul_matches_naive():
    """NTT-based polynomial multiplication matches naive O(N²) convolution."""
    random.seed(31)
    p1 = [random.randint(0, MOD - 1) for _ in range(32)]
    p2 = [random.randint(0, MOD - 1) for _ in range(32)]
    assert polynomial_mul(p1, p2, MOD) == naive_mul(p1, p2, MOD)


def test_ntt_zero_vector():
    """NTT of all-zeros is all-zeros."""
    a = [0] * 8
    assert ntt(a, MOD) == a
    assert ntt(a, MOD, invert=True) == a


def test_ntt_rejects_size_the_modulus_cannot_support():
    """17 - 1 = 2^4, so F_17 has no primitive 32nd root of unity."""
    with pytest.raises(ValueError, match="not supported"):
        ntt(list(range(32)), 17)


# ------------------------------------------------------------------
# Correctness across moduli
# ------------------------------------------------------------------


@pytest.mark.parametrize("mod", MODULI)
def test_ntt_of_x_lists_the_roots_of_unity(mod):
    """
    NTT(x) evaluates the polynomial x at ω^0, ω^1, …, ω^(n-1).

    Known answer that pins ω itself: a round-trip INTT(NTT(a)) == a still
    passes with a wrong ω (the transform stays invertible), this does not.
    """
    n = 16
    out = ntt([0, 1] + [0] * (n - 2), mod)
    omega = out[1]
    assert out == [pow(omega, k, mod) for k in range(n)]
    assert pow(omega, n // 2, mod) == mod - 1  # ω is primitive: ω^(n/2) = -1


@pytest.mark.parametrize("mod", MODULI)
def test_polynomial_mul_matches_naive_for_each_modulus(mod):
    rng = random.Random(mod % 1009)
    p1 = [rng.randrange(1, mod) for _ in range(16)]
    p2 = [rng.randrange(1, mod) for _ in range(16)]
    assert polynomial_mul(p1, p2, mod) == naive_mul(p1, p2, mod)


# ------------------------------------------------------------------
# GPU NTT
# ------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize("n", [2, 8, 256, 1 << 14])
def test_ntt_gpu_matches_cpu(n):
    rng = random.Random(n)
    a = [rng.randrange(MOD) for _ in range(n)]
    assert ntt_gpu(cp.asarray(a, dtype=cp.int64), MOD).tolist() == ntt(a, MOD)


@pytest.mark.gpu
def test_ntt_gpu_inverse_roundtrip():
    rng = random.Random(55)
    a = [rng.randrange(MOD) for _ in range(1024)]
    transformed = ntt_gpu(cp.asarray(a, dtype=cp.int64), MOD)
    assert ntt_gpu(transformed, MOD, invert=True).tolist() == a


@pytest.mark.gpu
def test_ntt_gpu_reduces_unreduced_input():
    a = [MOD + 5, 2 * MOD + 1] + [0] * 6
    assert ntt_gpu(cp.asarray(a, dtype=cp.int64), MOD).tolist() == ntt(a, MOD)


@pytest.mark.gpu
@known_bug("ntt_gpu returns length-1 input unreduced; the CPU NTT reduces it", raises=AssertionError)
def test_ntt_gpu_single_element_is_reduced():
    assert ntt_gpu(cp.asarray([MOD + 5], dtype=cp.int64), MOD).tolist() == [5]


def test_ntt_gpu_fallback_matches_cpu(monkeypatch):
    monkeypatch.setattr(ntt_gpu_module, "_gpu_available", False)
    a = list(range(16))
    with pytest.warns(GpuFallbackWarning):
        out = ntt_gpu(np.array(a, dtype=np.int64), MOD)
    assert out.tolist() == ntt(a, MOD)
