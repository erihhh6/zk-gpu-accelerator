"""tests/test_curve.py — BN254 G1 group law, scalar multiplication and known answers."""

import dataclasses
import random

import pytest

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
    scalar_mul,
    to_jacobian,
)
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


def test_affine_coordinates():
    assert G.affine() == (1, 2)
    with pytest.raises(ValueError, match="no affine coordinates"):
        INF.affine()


# ------------------------------------------------------------------
# Jacobian coordinates, checked against the affine formulas
# ------------------------------------------------------------------
# Point's affine + and double() share no code with the Jacobian functions, so
# they serve as the reference. Inputs are scaled by a random λ so that Z != 1:
# formulas that silently assume Z = 1 would pass on to_jacobian() output alone.


def scaled(point: Point, lam: int) -> tuple[int, int, int]:
    """The same point in Jacobian form with Z = λ: (x·λ², y·λ³, λ)."""
    x, y = point.affine()
    return (x * lam * lam % P, y * pow(lam, 3, P) % P, lam)


def affine_ladder(point: Point, k: int) -> Point:
    """Reference scalar multiplication using only affine + and double()."""
    if k < 0:
        point, k = -point, -k
    result, addend = INF, point
    while k:
        if k & 1:
            result = result + addend
        addend = addend.double()
        k >>= 1
    return result


@pytest.fixture(scope="module")
def lambdas():
    rng = random.Random(77)
    return [rng.randrange(2, P) for _ in range(3)]


def test_jacobian_round_trip(random_points, lambdas):
    for point in (G, TWO_G, INF, *random_points):
        assert from_jacobian(to_jacobian(point)) == point
    assert from_jacobian(scaled(G, lambdas[0])) == G
    assert from_jacobian(JACOBIAN_IDENTITY) == INF
    assert from_jacobian((5, 7, 0)) == INF  # any Z = 0 is the identity


def test_jacobian_double_matches_affine(random_points, lambdas):
    for point, lam in zip(random_points, lambdas, strict=True):
        assert from_jacobian(jacobian_double(scaled(point, lam))) == point.double()
    assert from_jacobian(jacobian_double(JACOBIAN_IDENTITY)) == INF


def test_jacobian_add_matches_affine(random_points, lambdas):
    a, b, c = random_points
    l1, l2, l3 = lambdas
    assert from_jacobian(jacobian_add(scaled(a, l1), scaled(b, l2))) == a + b
    assert from_jacobian(jacobian_add(scaled(b, l2), scaled(c, l3))) == b + c
    assert from_jacobian(jacobian_add_affine(scaled(a, l1), *b.affine())) == a + b


def test_jacobian_special_cases(random_points, lambdas):
    """Equal and opposite points need the explicit branch: the general formula gives Z = 0 for both."""
    a = random_points[0]
    l1, l2, _ = lambdas
    # Same point, different Z: the coordinates differ, but the sum must be a doubling
    assert from_jacobian(jacobian_add(scaled(a, l1), scaled(a, l2))) == a.double()
    assert from_jacobian(jacobian_add_affine(scaled(a, l1), *a.affine())) == a.double()
    # P + (-P) = O
    assert from_jacobian(jacobian_add(scaled(a, l1), scaled(-a, l2))) == INF
    assert from_jacobian(jacobian_add_affine(scaled(a, l1), *(-a).affine())) == INF
    # The identity on either side
    assert from_jacobian(jacobian_add(JACOBIAN_IDENTITY, scaled(a, l1))) == a
    assert from_jacobian(jacobian_add(scaled(a, l1), JACOBIAN_IDENTITY)) == a
    assert from_jacobian(jacobian_add_affine(JACOBIAN_IDENTITY, *a.affine())) == a


def test_jacobian_mul_matches_affine_ladder(random_points):
    rng = random.Random(5)
    point = random_points[0]
    for k in (0, 1, 2, 3, -1, -7, R - 1, R, R + 1, rng.randrange(R), rng.randrange(R)):
        expected = affine_ladder(point, k)
        assert from_jacobian(jacobian_mul(point, k)) == expected
        assert point * k == expected
    assert from_jacobian(jacobian_mul(INF, 12345)) == INF
