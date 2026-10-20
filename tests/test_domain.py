"""tests/test_domain.py — Radix-2 evaluation domains over BN254 Fr."""

import dataclasses
import random

import pytest

from zkgpu.domain import Radix2Domain
from zkgpu.field import FIELD_MODULUS as R
from zkgpu.ntt import MOD as NTT_MOD, get_ntt_root, polynomial_mul

# Primitive 2^28-th roots of unity of BN254 Fr published by two independent libraries.
# snarkjs/ffjavascript build theirs from the non-residue 5, as zkgpu does, so the
# values must match exactly; arkworks uses the generator 7, a different valid root.
SNARKJS_W28 = 19103219067921713944291392827692070036145651957329286315305642004821462161904
ARKWORKS_W28 = 1748695177688661943023146337482803886740723238769601073607632802312037301404


def horner(coeffs: list[int], x: int, mod: int = R) -> int:
    """Direct polynomial evaluation: the definition every transform must agree with."""
    acc = 0
    for c in reversed(coeffs):
        acc = (acc * x + c) % mod
    return acc


def random_poly(n: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(R) for _ in range(n)]


def trim(poly: list[int]) -> list[int]:
    while len(poly) > 1 and poly[-1] == 0:
        poly = poly[:-1]
    return poly


# ------------------------------------------------------------------
# Known answers and structure
# ------------------------------------------------------------------


def test_bn254_root_matches_snarkjs():
    assert Radix2Domain(2**28).omega == SNARKJS_W28


def test_explicit_generator_7_matches_arkworks():
    assert get_ntt_root(2**28, R, non_residue=7) == ARKWORKS_W28


@pytest.mark.parametrize("k", [1, 2, 10, 20, 27])
def test_smaller_domains_use_powers_of_the_same_root(k):
    """Every size derives from one root, so snarkjs's w[k] = w[28]^(2^(28-k)) holds here too."""
    assert Radix2Domain(2**k).omega == pow(SNARKJS_W28, 2 ** (28 - k), R)


def test_largest_domain_is_the_two_adicity():
    Radix2Domain(2**28)
    with pytest.raises(ValueError, match="not supported"):
        Radix2Domain(2**29)


def test_elements_form_the_subgroup():
    d = Radix2Domain(8)
    els = d.elements()
    assert len(set(els)) == 8
    assert all(pow(x, 8, R) == 1 for x in els)
    assert els[0] == 1 and els[4] == R - 1  # ω^(n/2) = -1
    assert [d.element(i) for i in range(8)] == els


def test_at_least_rounds_up_to_a_power_of_two():
    assert [Radix2Domain.at_least(n).size for n in (1, 2, 3, 5, 8, 9)] == [1, 2, 4, 8, 8, 16]
    with pytest.raises(ValueError):
        Radix2Domain.at_least(0)


def test_domain_is_immutable():
    d = Radix2Domain(8)
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.size = 16


def test_other_ntt_friendly_moduli():
    d = Radix2Domain(16, modulus=NTT_MOD)
    assert d.fft([1, 2, 3]) == [horner([1, 2, 3], x, NTT_MOD) for x in d.elements()]


# ------------------------------------------------------------------
# Transforms agree with direct evaluation
# ------------------------------------------------------------------


@pytest.mark.parametrize("n_coeffs", [1, 5, 8])  # shorter inputs are zero-padded
def test_fft_matches_direct_evaluation(n_coeffs):
    d = Radix2Domain(8)
    coeffs = random_poly(n_coeffs, seed=n_coeffs)
    assert d.fft(coeffs) == [horner(coeffs, x) for x in d.elements()]


def test_ifft_inverts_fft():
    d = Radix2Domain(16)
    coeffs = random_poly(16, seed=1)
    assert d.ifft(d.fft(coeffs)) == coeffs


@pytest.mark.parametrize("offset", [None, 7, R - 3])
def test_coset_fft_matches_direct_evaluation(offset):
    d = Radix2Domain(8)
    g = d.coset_offset if offset is None else offset
    coeffs = random_poly(6, seed=2)
    assert d.coset_fft(coeffs, offset) == [horner(coeffs, g * x % R) for x in d.elements()]
    assert d.coset_ifft(d.coset_fft(coeffs, offset), offset) == coeffs + [0, 0]


def test_vanishing_polynomial():
    d = Radix2Domain(8)
    assert all(d.vanishing_eval(x) == 0 for x in d.elements())
    g = d.coset_offset
    assert all(d.vanishing_eval(g * x % R) == (pow(g, 8, R) - 1) % R for x in d.elements())
    assert d.vanishing_eval(g) != 0


def test_groth16_quotient_via_coset_fft():
    """
    The step a Groth16 prover runs: h(X) = (A(X)·B(X) - C(X)) / Z_H(X).

    C agrees with A·B on H, so the numerator vanishes on H and is divisible
    by Z_H. On the coset g·H, Z_H is the constant g^n - 1, so h's coset
    evaluations are the pointwise numerator times one inverse. h has degree
    <= n - 2, so n coset evaluations determine it completely.
    """
    n = 8
    d = Radix2Domain(n)
    a, b = random_poly(n, seed=3), random_poly(n, seed=4)
    c = d.ifft([x * y % R for x, y in zip(d.fft(a), d.fft(b), strict=True)])

    z_inv = pow(d.vanishing_eval(d.coset_offset), -1, R)
    numerator_on_coset = [
        (x * y - z) * z_inv % R
        for x, y, z in zip(d.coset_fft(a), d.coset_fft(b), d.coset_fft(c), strict=True)
    ]
    h = d.coset_ifft(numerator_on_coset)

    assert h[-1] == 0  # deg h <= n - 2
    z_h = [R - 1] + [0] * (n - 1) + [1]  # X^n - 1
    ab = polynomial_mul(a, b, R)
    numerator = [(x - y) % R for x, y in zip(ab, c + [0] * (len(ab) - n), strict=True)]
    assert trim(numerator) == trim(polynomial_mul(trim(h), z_h, R))


# ------------------------------------------------------------------
# Coset edge cases and validation
# ------------------------------------------------------------------


def test_coset_offset_falls_back_when_the_non_residue_lies_in_the_domain():
    """For p = 41 at size 8 the smallest non-residue, 3, satisfies 3^8 = 1, so it lies in H."""
    d = Radix2Domain(8, modulus=41)
    assert pow(3, 8, 41) == 1
    assert d.coset_offset == 4
    coeffs = [1, 2, 3]
    assert d.coset_fft(coeffs) == [horner(coeffs, 4 * x % 41, 41) for x in d.elements()]


def test_full_group_domain_has_no_coset():
    """17 - 1 = 16, so a size-16 domain is all of F_17*: FFTs work, cosets cannot exist."""
    d = Radix2Domain(16, modulus=17)
    assert d.coset_offset is None
    assert d.ifft(d.fft([1, 2, 3])) == [1, 2, 3] + [0] * 13
    with pytest.raises(ValueError, match="no coset"):
        d.coset_fft([1, 2, 3])


def test_rejects_offsets_inside_the_domain():
    d = Radix2Domain(8)
    for bad in (0, 1, d.omega, R):
        with pytest.raises(ValueError, match="outside the domain"):
            d.coset_fft([1], offset=bad)


def test_rejects_bad_sizes_and_lengths():
    with pytest.raises(ValueError, match="power of 2"):
        Radix2Domain(6)
    with pytest.raises(ValueError, match="odd prime"):
        Radix2Domain(4, modulus=21)
    d = Radix2Domain(8)
    with pytest.raises(ValueError, match="do not fit"):
        d.fft(list(range(9)))  # degree >= n would silently alias, since X^n = 1 on H
    with pytest.raises(ValueError, match="expected 8 evaluations"):
        d.ifft([1, 2, 3])
