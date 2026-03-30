"""
src/curve.py — BN254 elliptic curve point arithmetic (CPU).

Curve equation (short Weierstrass form):
  y² = x³ + 3  mod P

where P is the BN254 base field prime. All coordinates are elements of F_P.

This module implements:
  - Point addition and doubling in affine coordinates
  - Scalar multiplication via double-and-add
  - The standard generator point G = (1, 2)
"""

from __future__ import annotations

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


@dataclass
class Point:
    """
    Affine point on the BN254 G1 curve.
    The point at infinity (identity element) is represented as (None, None).
    """

    x: int | None
    y: int | None

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
        if scalar < 0:
            # -k * P = k * (-P)
            return (-self) * (-scalar)
        result = Point.identity()
        addend = Point(self.x, self.y)
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
    # Comparison and representation
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Point):
            return self.x == other.x and self.y == other.y
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.x, self.y))

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
# Smoke test
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("=== curve.py smoke test ===")

    assert G.is_on_curve(), "Generator G is not on the curve!"
    print("G is on curve  OK")

    # Identity behaviour
    assert G + Point.identity() == G
    assert Point.identity() + G == G
    print("G + O = G  OK")

    # Inverse: G + (-G) = O
    neg_G = -G
    assert G + neg_G == Point.identity()
    print("G + (-G) = O  OK")

    # Doubling
    G2 = G.double()
    assert G2.is_on_curve()
    print(f"2G = {G2.x} (on curve OK)")

    # Scalar mul: 2*G should equal G.double()
    assert G * 2 == G2
    print("G*2 == G.double()  OK")

    # Linearity: (a + b)*G == a*G + b*G
    a, b = 7, 13
    assert G * (a + b) == G * a + G * b
    print(f"(a+b)*G == a*G + b*G  OK")

    print("\nAll curve.py checks passed.")
