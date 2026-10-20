"""tests/test_curve.py — BN254 G1 group law, scalar multiplication and known answers."""

import dataclasses
import random

import pytest

from zkgpu.curve import G, P, Point, scalar_mul
from zkgpu.field import FIELD_MODULUS as R  # BN254 G1 is a cyclic group of prime order r

INF = Point.identity()

# 2·G on alt_bn128 as published by py_ecc, an independent reference implementation.
TWO_G = Point(
    1368015179489954701390400359078579693043519447331113978918064868415326638035,
    9918110051302171585080402603319702774565515993150576347155970296011118125764,
)


@pytest.fixture(scope="module")
def random_points():
    rng = random.Random(2024)
    return [G * rng.randrange(1, R) for _ in range(3)]


# ------------------------------------------------------------------
# Group law
# ------------------------------------------------------------------


def test_generator_is_on_curve():
    assert G.is_on_curve()
    assert (G.y**2 - G.x**3 - 3) % P == 0


def test_identity_is_neutral():
    assert G + INF == G
    assert INF + G == G
    assert INF + INF == INF
    assert INF.is_on_curve()


def test_negation():
    assert Point(1, P - 2) == -G
    assert G + (-G) == INF
    assert -INF == INF


def test_doubling_matches_reference():
    assert G.double() == TWO_G
    assert G + G == TWO_G
    assert TWO_G.is_on_curve()


def test_addition_is_commutative_and_associative(random_points):
    a, b, c = random_points
    assert a + b == b + a
    assert (a + b) + c == a + (b + c)
    assert all(p.is_on_curve() for p in (a, b, c, a + b, a + b + c))


# ------------------------------------------------------------------
# Scalar multiplication
# ------------------------------------------------------------------


def test_group_order_is_r():
    """r·G = O pins both the curve constants and the scalar field modulus."""
    assert G * R == INF
    assert G * (R + 1) == G
    assert G * (R - 1) == -G


def test_scalar_mul_small_values():
    assert G * 0 == INF
    assert G * 1 == G
    assert G * 2 == TWO_G
    assert 3 * G == G * 3 == scalar_mul(G, 3) == G + G + G


def test_scalar_mul_is_a_homomorphism():
    rng = random.Random(7)
    k1, k2 = rng.randrange(R), rng.randrange(R)
    assert G * (k1 + k2) == G * k1 + G * k2
    assert (G * k1) * k2 == G * (k1 * k2 % R)


def test_negative_scalar():
    assert G * -5 == -(G * 5)
    assert G * -5 + G * 5 == INF


def test_hash_is_consistent_with_equality():
    assert hash(Point(1, 2)) == hash(G)
    assert len({G, Point(1, 2), TWO_G}) == 2


# ------------------------------------------------------------------
# Validation and immutability
# ------------------------------------------------------------------


def test_rejects_points_off_the_curve():
    """(1, 3) lies on y² = x³ + 8, not BN254: computing with it is an invalid-curve attack."""
    with pytest.raises(ValueError, match="not on the BN254 curve"):
        Point(1, 3)


def test_coordinates_are_canonical():
    assert Point(1, 2 + P) == G
    assert Point(1 - P, 2 - P) == G  # negative representatives reduce too
    assert hash(Point(1, 2 + P)) == hash(G)
    assert len({G, Point(1, 2 + P)}) == 1


def test_points_are_immutable():
    point = Point(1, 2)  # never mutate the shared generator G
    with pytest.raises(dataclasses.FrozenInstanceError):
        point.x = 5


def test_rejects_malformed_coordinates():
    with pytest.raises(ValueError, match="identity"):
        Point(1, None)
    with pytest.raises(TypeError):
        Point(1.0, 2.0)  # floats would silently lose precision at 254 bits


def test_arithmetic_rejects_unsupported_operands():
    with pytest.raises(TypeError):
        G + 1
    with pytest.raises(TypeError):
        G * 2.0
