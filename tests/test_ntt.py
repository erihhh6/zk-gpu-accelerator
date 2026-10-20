"""tests/test_ntt.py — Correctness tests for zkgpu/ntt.py (and GPU if available)."""

import random

import numpy as np
import pytest

from zkgpu import ntt_gpu as ntt_gpu_module
from zkgpu.field import FIELD_MODULUS as BN254_R
from zkgpu.gpu_backend import GpuFallbackWarning, cp
from zkgpu.ntt import get_ntt_root, ntt, polynomial_mul, quadratic_non_residue, MOD
from zkgpu.ntt_gpu import ntt_gpu

# Largest prime below 2^63 of the form k·2^32 + 1: exercises field_gpu's
# 128-bit-wide multiply inside the GPU NTT
LARGE_INT64_NTT_PRIME = 2_147_483_641 * 2**32 + 1
INT64_MIN, INT64_MAX = -(2**63), 2**63 - 1

# NTT-friendly primes q = k·2^m + 1, each supporting power-of-two sizes up to 2^m.
# 3 is a quadratic residue mod 12289 and mod BN254's r, which is what broke the
# old hardcoded primitive root 3 for them.
MODULI = [
    pytest.param(MOD, id="998244353"),  # 119·2^23 + 1, the default
    pytest.param(7_340_033, id="7340033"),  # 7·2^20 + 1
    pytest.param(12_289, id="12289"),  # 3·2^12 + 1
    pytest.param(BN254_R, id="bn254_r"),  # 2-adicity 28
]

# Moduli that fit the GPU NTT's int64 elements
GPU_MODULI = [
    pytest.param(MOD, id="998244353"),
    pytest.param(12_289, id="12289"),
    pytest.param(LARGE_INT64_NTT_PRIME, id="large_int64_prime"),
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


@pytest.mark.parametrize("mod", [-7, 1, 2, 15, 21])  # 21 - 1 is divisible by 4, but 21 is composite
def test_ntt_rejects_moduli_that_are_not_odd_primes(mod):
    with pytest.raises(ValueError, match="odd prime"):
        ntt([1, 2, 3, 4], mod)


# ------------------------------------------------------------------
# Root derivation
# ------------------------------------------------------------------


@pytest.mark.parametrize("mod", MODULI)
def test_generator_is_a_quadratic_non_residue(mod):
    g = quadratic_non_residue(mod)
    assert pow(g, (mod - 1) // 2, mod) == mod - 1  # Euler's criterion


def test_known_non_residues():
    assert quadratic_non_residue(MOD) == 3  # same roots as the old hardcoded 3, so outputs are unchanged
    assert quadratic_non_residue(12_289) == 11
    assert quadratic_non_residue(BN254_R) == 5


def test_bn254_supports_sizes_up_to_its_two_adicity():
    omega = get_ntt_root(2**28, BN254_R)
    assert pow(omega, 2**27, BN254_R) == BN254_R - 1
    with pytest.raises(ValueError, match="not supported"):
        get_ntt_root(2**29, BN254_R)


def test_roots_are_consistent_across_sizes():
    """Stage roots are derived from the full-size root, so ω_{n/2} must equal ω_n²."""
    for mod in (MOD, 12_289, BN254_R):
        assert get_ntt_root(8, mod) == pow(get_ntt_root(16, mod), 2, mod)


def test_explicit_residue_is_rejected():
    """3 is a quadratic residue mod BN254's r: the exact cause of the old bug."""
    with pytest.raises(ValueError, match="non-residue"):
        get_ntt_root(16, BN254_R, non_residue=3)
    assert get_ntt_root(16, BN254_R, non_residue=5) == get_ntt_root(16, BN254_R)


def test_get_ntt_root_rejects_non_power_of_two():
    with pytest.raises(ValueError, match="power of 2"):
        get_ntt_root(12, MOD)


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
@pytest.mark.parametrize("mod", GPU_MODULI)
@pytest.mark.parametrize("n", [1, 2, 8, 256, 1 << 12])
def test_ntt_gpu_matches_cpu(n, mod):
    rng = random.Random(n)
    a = [rng.randrange(mod) for _ in range(n)]
    assert ntt_gpu(cp.asarray(a, dtype=cp.int64), mod).tolist() == ntt(a, mod)


@pytest.mark.gpu
def test_ntt_gpu_matches_cpu_large_size():
    rng = random.Random(14)
    a = [rng.randrange(MOD) for _ in range(1 << 14)]
    assert ntt_gpu(cp.asarray(a, dtype=cp.int64), MOD).tolist() == ntt(a, MOD)


@pytest.mark.gpu
@pytest.mark.parametrize("mod", GPU_MODULI)
def test_ntt_gpu_inverse_roundtrip(mod):
    rng = random.Random(55)
    a = [rng.randrange(mod) for _ in range(1024)]
    transformed = ntt_gpu(cp.asarray(a, dtype=cp.int64), mod)
    assert ntt_gpu(transformed, mod, invert=True).tolist() == a


@pytest.mark.gpu
def test_ntt_gpu_repeated_calls_agree():
    """The second call runs on cached twiddles and must give the same answer."""
    a = cp.arange(4096, dtype=cp.int64)
    first = ntt_gpu(a, MOD).tolist()
    assert ntt_gpu(a, MOD).tolist() == first
    assert a.tolist() == list(range(4096))  # input left untouched


@pytest.mark.gpu
@pytest.mark.parametrize("mod", GPU_MODULI)
def test_ntt_gpu_reduces_unreduced_input(mod):
    a = [mod + 5, 2 * mod + 1 if 2 * mod + 1 <= INT64_MAX else -1, INT64_MAX, INT64_MIN] + [0] * 4
    assert ntt_gpu(cp.asarray(a, dtype=cp.int64), mod).tolist() == ntt(a, mod)


@pytest.mark.gpu
def test_ntt_gpu_single_element_is_reduced():
    assert ntt_gpu(cp.asarray([MOD + 5], dtype=cp.int64), MOD).tolist() == [5]


@pytest.mark.gpu
def test_ntt_gpu_validates_like_the_cpu_ntt():
    with pytest.raises(ValueError, match="odd prime"):
        ntt_gpu(cp.arange(4, dtype=cp.int64), 21)
    with pytest.raises(ValueError, match="power of 2"):
        ntt_gpu(cp.arange(3, dtype=cp.int64), MOD)
    with pytest.raises(ValueError, match="not supported"):
        ntt_gpu(cp.arange(1 << 13, dtype=cp.int64), 12_289)  # 12289 - 1 = 3·2^12


def test_ntt_gpu_rejects_modulus_beyond_int64():
    """Checked before any GPU work, so this runs (and must hold) on CPU-only machines too."""
    with pytest.raises(ValueError, match="below 2\\^63"):
        ntt_gpu(np.arange(4, dtype=np.int64), BN254_R)


def test_ntt_gpu_fallback_matches_cpu(monkeypatch):
    monkeypatch.setattr(ntt_gpu_module, "_gpu_available", False)
    a = list(range(16))
    with pytest.warns(GpuFallbackWarning):
        out = ntt_gpu(np.array(a, dtype=np.int64), MOD)
    assert out.tolist() == ntt(a, MOD)
