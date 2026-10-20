"""
zkgpu/curve.py — BN254 elliptic curve point arithmetic (CPU).

Curve equation (short Weierstrass form):
  y² = x³ + 3  mod P

where P is the BN254 base field prime. All coordinates are elements of F_P.

This module implements:
  - Point addition and doubling in affine coordinates
  - Scalar multiplication via double-and-add
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

# Curve parameters for y² = x³ + Ax + B
A: int = 0
B: int = 3

# Standard generator point of BN254 (G1)
GX: int = 1
GY: int = 2


def _modinv(a: int, m: int) -> int:
    """Modular inverse via Fermat's little theorem (m must be prime)."""
    return pow(a, m - 2, m)


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
    def identity() -> "Point":
        """Return the additive identity (point at infinity)."""
        return Point(None, None)

    def is_identity(self) -> bool:
        """Return True if this point is the point at infinity."""
        return self.x is None and self.y is None

    def is_on_curve(self) -> bool:
        """Verify that the point satisfies y² = x³ + 3 mod P."""
        if self.is_identity():
            return True
        lhs = (self.y * self.y) % P
        rhs = (pow(self.x, 3, P) + B) % P
        return lhs == rhs

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def __add__(self, other: "Point") -> "Point":
        """Elliptic curve point addition (affine coordinates)."""
        if not isinstance(other, Point):
            return NotImplemented
        if self.is_identity():
            return other
        if other.is_identity():
            return self

        if self.x == other.x:
            # Points are either equal (doubling) or inverses (identity)
            if self.y == other.y:
                return self.double()
            # P + (-P) = identity
            return Point.identity()

        # Generic addition: λ = (y2 - y1) / (x2 - x1) mod P
        lam = ((other.y - self.y) * _modinv(other.x - self.x, P)) % P
        x3 = (lam * lam - self.x - other.x) % P
        y3 = (lam * (self.x - x3) - self.y) % P
        return Point(x3, y3)

    def double(self) -> "Point":
        """Elliptic curve point doubling: λ = (3x² + A) / (2y) mod P."""
        if self.is_identity():
            return Point.identity()
        if self.y == 0:
            # Tangent is vertical → result is identity
            return Point.identity()

        lam = ((3 * self.x * self.x + A) * _modinv(2 * self.y, P)) % P
        x3 = (lam * lam - 2 * self.x) % P
        y3 = (lam * (self.x - x3) - self.y) % P
        return Point(x3, y3)

    def __mul__(self, scalar: int) -> "Point":
        """Scalar multiplication via double-and-add: O(log scalar)."""
        try:
            scalar = operator.index(scalar)  # rejects floats instead of truncating
        except TypeError:
            return NotImplemented
        if scalar < 0:
            # -k * P = k * (-P)
            return (-self) * (-scalar)
        result = Point.identity()
        addend = self  # safe to share: points are immutable
        while scalar:
            if scalar & 1:
                result = result + addend
            addend = addend.double()
            scalar >>= 1
        return result

    def __rmul__(self, scalar: int) -> "Point":
        return self.__mul__(scalar)

    def __neg__(self) -> "Point":
        """Additive inverse: negate the y-coordinate."""
        if self.is_identity():
            return Point.identity()
        return Point(self.x, (-self.y) % P)

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
