"""
zkgpu/curve.py — BN254 elliptic curve point arithmetic (CPU).

Curve equation (short Weierstrass form):
  y² = x³ + 3  mod P

where P is the BN254 base field prime. All coordinates are elements of F_P.

This module implements:
  - Point: immutable affine points with addition, doubling and negation
  - Jacobian coordinates for long chains of operations (scalar
    multiplication, MSM), which avoid a modular inverse per operation
  - Scalar multiplication via double-and-add (in Jacobian form)
  - The standard generator point G = (1, 2)

Points are immutable values and are validated on construction: coordinates
are reduced into [0, P) and must satisfy the curve equation. Rejecting
off-curve input matters for security, not just tidiness: the addition
formulas never use B, so an off-curve "point" silently computes on a
different curve, which is the basis of invalid-curve attacks.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass

# BN254 base field prime (field of definition of the curve)
P: int = 21888242871839275222246405745257275088696311157297823662689037894645226208583

# Curve parameters for y² = x³ + Ax + B. The Jacobian formulas below are the
# specialised a = 0 versions, so they are only valid because A is 0.
A: int = 0
B: int = 3

# Standard generator point of BN254 (G1)
GX: int = 1
GY: int = 2


def _modinv(a: int, m: int) -> int:
    """
    Modular inverse via Python's built-in pow(a, -1, m) (extended Euclid).

    Several times faster than Fermat's a^(m-2) for 254-bit moduli, and it
    raises ValueError instead of returning 0 when a is not invertible.
    """
    return pow(a, -1, m)


@dataclass(frozen=True, slots=True)
class Point:
    """
    Affine point on the BN254 G1 curve.
    The point at infinity (identity element) is represented as (None, None).

    Immutable: frozen, because points are hashed and shared (G, bucket
    entries in Pippenger), so mutating one would corrupt every holder.

    Raises
    ------
    ValueError : if exactly one coordinate is None, or (x, y) is not on the curve
    TypeError  : if a coordinate is not an integer
    """

    x: int | None
    y: int | None

    def __post_init__(self) -> None:
        if self.x is None and self.y is None:
            return
        if self.x is None or self.y is None:
            raise ValueError(f"only the identity has None coordinates, got ({self.x}, {self.y})")
        # Canonical coordinates in [0, P), so equal points compare and hash equal
        # (frozen dataclasses must assign through object.__setattr__)
        object.__setattr__(self, "x", operator.index(self.x) % P)
        object.__setattr__(self, "y", operator.index(self.y) % P)
        if not self.is_on_curve():
            raise ValueError(f"({self.x}, {self.y}) is not on the BN254 curve y^2 = x^3 + 3")

    # ------------------------------------------------------------------
    # Constructors / predicates
    # ------------------------------------------------------------------

    @staticmethod
    def identity() -> Point:
        """Return the additive identity (point at infinity)."""
        return Point(None, None)

    def is_identity(self) -> bool:
        """Return True if this point is the point at infinity."""
        return self.x is None and self.y is None

    def affine(self) -> tuple[int, int]:
        """
        (x, y) of a finite point as plain ints; raises ValueError for the identity.

        Returning an int pair also tells the type checker that neither
        coordinate is None, so formulas using the result need no per-use checks.
        """
        if self.x is None or self.y is None:
            raise ValueError("the point at infinity has no affine coordinates")
        return self.x, self.y

    def is_on_curve(self) -> bool:
        """Verify that the point satisfies y² = x³ + 3 mod P."""
        if self.is_identity():
            return True
        x, y = self.affine()
        return (y * y - pow(x, 3, P) - B) % P == 0

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def __add__(self, other: object) -> Point:
        """Elliptic curve point addition (affine coordinates, one inverse)."""
        if not isinstance(other, Point):
            return NotImplemented
        if self.is_identity():
            return other
        if other.is_identity():
            return self

        (x1, y1), (x2, y2) = self.affine(), other.affine()
        if x1 == x2:
            # Points are either equal (doubling) or inverses (identity)
            if y1 == y2:
                return self.double()
            # P + (-P) = identity
            return Point.identity()

        # Generic addition: λ = (y2 - y1) / (x2 - x1) mod P
        lam = ((y2 - y1) * _modinv(x2 - x1, P)) % P
        x3 = (lam * lam - x1 - x2) % P
        y3 = (lam * (x1 - x3) - y1) % P
        return Point(x3, y3)

    def double(self) -> Point:
        """Elliptic curve point doubling: λ = (3x² + A) / (2y) mod P."""
        if self.is_identity():
            return Point.identity()
        x, y = self.affine()
        if y == 0:
            # Tangent is vertical → result is identity
            return Point.identity()

        lam = ((3 * x * x + A) * _modinv(2 * y, P)) % P
        x3 = (lam * lam - 2 * x) % P
        y3 = (lam * (x - x3) - y) % P
        return Point(x3, y3)

    def __mul__(self, scalar: int) -> Point:
        """Scalar multiplication: double-and-add in Jacobian form, one inverse at the end."""
        try:
            scalar = operator.index(scalar)  # rejects floats instead of truncating
        except TypeError:
            return NotImplemented
        return from_jacobian(jacobian_mul(self, scalar))

    def __rmul__(self, scalar: int) -> Point:
        return self.__mul__(scalar)

    def __neg__(self) -> Point:
        """Additive inverse: negate the y-coordinate."""
        if self.is_identity():
            return Point.identity()
        x, y = self.affine()
        return Point(x, -y % P)

    # ------------------------------------------------------------------
    # Representation (__eq__ and __hash__ come from the frozen dataclass and
    # compare the canonical (x, y) pair)
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if self.is_identity():
            return "Point(∞)"
        return f"Point(x={self.x}, y={self.y})"


def scalar_mul(point: Point, scalar: int) -> Point:
    """Convenience wrapper for double-and-add scalar multiplication."""
    return point * scalar


# Generator point
G: Point = Point(GX, GY)


# ------------------------------------------------------------------
# Jacobian coordinates
# ------------------------------------------------------------------
#
# (X, Y, Z) stands for the affine point (X/Z², Y/Z³); Z = 0 is the point at
# infinity. Affine addition needs one modular inverse per operation, which
# costs as much as dozens of multiplications. These formulas need none, so
# long chains of additions and doublings (scalar multiplication, MSM) run in
# Jacobian form and convert back once, with a single inverse, at the end.
#
# Formulas are the Explicit-Formulas Database's for short Weierstrass curves
# with a = 0, with its variable names kept so they can be checked line by line:
#   doubling        dbl-2009-l      2M + 5S
#   addition        add-2007-bl    11M + 5S
#   mixed addition  madd-2007-bl    7M + 4S   (second operand affine: Z2 = 1)
#   https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html
# All coordinates are kept reduced into [0, P), so "is Z zero?" is z == 0.

Jacobian = tuple[int, int, int]
JACOBIAN_IDENTITY: Jacobian = (1, 1, 0)


def to_jacobian(point: Point) -> Jacobian:
    """Jacobian form of an affine point (Z = 1, or Z = 0 for the identity)."""
    if point.is_identity():
        return JACOBIAN_IDENTITY
    x, y = point.affine()
    return (x, y, 1)


def from_jacobian(j: Jacobian) -> Point:
    """
    Affine Point for (X, Y, Z): x = X/Z², y = Y/Z³, using a single inverse.

    The result goes through Point's on-curve validation, so every chain of
    Jacobian operations is checked once at the end.
    """
    x, y, z = j
    if z == 0:
        return Point.identity()
    z_inv = _modinv(z, P)
    z_inv2 = z_inv * z_inv % P
    return Point(x * z_inv2 % P, y * z_inv2 % P * z_inv % P)


def jacobian_double(j: Jacobian) -> Jacobian:
    """2·(X, Y, Z) (EFD dbl-2009-l)."""
    x1, y1, z1 = j
    if z1 == 0 or y1 == 0:
        # Identity, or a vertical tangent (only possible for a point of order
        # 2, which BN254 G1 does not have since its order r is odd)
        return JACOBIAN_IDENTITY
    a = x1 * x1 % P
    b = y1 * y1 % P
    c = b * b % P
    d = 2 * ((x1 + b) * (x1 + b) - a - c) % P
    e = 3 * a % P
    f = e * e % P
    x3 = (f - 2 * d) % P
    y3 = (e * (d - x3) - 8 * c) % P
    z3 = 2 * y1 * z1 % P
    return (x3, y3, z3)


def jacobian_add(j1: Jacobian, j2: Jacobian) -> Jacobian:
    """(X1, Y1, Z1) + (X2, Y2, Z2) (EFD add-2007-bl)."""
    x1, y1, z1 = j1
    x2, y2, z2 = j2
    if z1 == 0:
        return j2
    if z2 == 0:
        return j1
    z1z1 = z1 * z1 % P
    z2z2 = z2 * z2 % P
    u1 = x1 * z2z2 % P
    u2 = x2 * z1z1 % P
    s1 = y1 * z2 % P * z2z2 % P
    s2 = y2 * z1 % P * z1z1 % P
    h = (u2 - u1) % P
    r = 2 * (s2 - s1) % P
    if h == 0:
        # Same affine x: the points are equal (double) or inverses (identity).
        # The general formula would produce Z3 = 0 in both cases, wrongly for the first.
        return jacobian_double(j1) if r == 0 else JACOBIAN_IDENTITY
    i = 4 * h * h % P  # (2H)^2
    j = h * i % P
    v = u1 * i % P
    x3 = (r * r - j - 2 * v) % P
    y3 = (r * (v - x3) - 2 * s1 * j) % P
    z3 = ((z1 + z2) * (z1 + z2) - z1z1 - z2z2) * h % P
    return (x3, y3, z3)


def jacobian_add_affine(j1: Jacobian, x2: int, y2: int) -> Jacobian:
    """(X1, Y1, Z1) + (x2, y2) for an affine, non-identity second point (EFD madd-2007-bl)."""
    x1, y1, z1 = j1
    if z1 == 0:
        return (x2, y2, 1)
    z1z1 = z1 * z1 % P
    u2 = x2 * z1z1 % P
    s2 = y2 * z1 % P * z1z1 % P
    h = (u2 - x1) % P
    r = 2 * (s2 - y1) % P
    if h == 0:
        return jacobian_double(j1) if r == 0 else JACOBIAN_IDENTITY
    hh = h * h % P
    i = 4 * hh % P
    j = h * i % P
    v = x1 * i % P
    x3 = (r * r - j - 2 * v) % P
    y3 = (r * (v - x3) - 2 * y1 * j) % P
    z3 = ((z1 + h) * (z1 + h) - z1z1 - hh) % P
    return (x3, y3, z3)


def jacobian_mul(point: Point, scalar: int) -> Jacobian:
    """
    scalar·point in Jacobian form: left-to-right double-and-add.

    Each step doubles the accumulator and, for a 1 bit, adds the affine base
    point with the cheaper mixed addition. The scalar is used as given (not
    reduced mod r), so r·G = O exercises the full 254-bit ladder.
    """
    if scalar < 0:
        point, scalar = -point, -scalar  # -k·P = k·(-P)
    if scalar == 0 or point.is_identity():
        return JACOBIAN_IDENTITY
    x, y = point.affine()
    acc = JACOBIAN_IDENTITY
    for bit in bin(scalar)[2:]:  # most significant bit first
        acc = jacobian_double(acc)
        if bit == "1":
            acc = jacobian_add_affine(acc, x, y)
    return acc
