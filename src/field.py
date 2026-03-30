"""
src/field.py — Finite field arithmetic over BN254 scalar field (CPU baseline).

BN254 scalar field modulus r (order of the base point G):
  r = 21888242871839275222246405745257275088548364400416034343698204186575808495617

All arithmetic is performed mod r. This module provides:
  - Fp class: wraps a field element with operator overloading
  - batch_mul / batch_add: element-wise operations on plain int lists
"""

from __future__ import annotations

FIELD_MODULUS: int = 21888242871839275222246405745257275088548364400416034343698204186575808495617


class Fp:
    """Element of the finite field F_r (BN254 scalar field)."""

    __slots__ = ("value",)

    def __init__(self, value: int) -> None:
        self.value: int = int(value) % FIELD_MODULUS

    # ------------------------------------------------------------------
    # Arithmetic operators
    # ------------------------------------------------------------------

    def __add__(self, other: "Fp") -> "Fp":
        return Fp(self.value + other.value)

    def __sub__(self, other: "Fp") -> "Fp":
        return Fp(self.value - other.value)

    def __mul__(self, other: "Fp") -> "Fp":
        return Fp(self.value * other.value)

    def __neg__(self) -> "Fp":
        return Fp(FIELD_MODULUS - self.value) if self.value != 0 else Fp(0)

    def __pow__(self, exp: int) -> "Fp":
        """Fast modular exponentiation using Python's built-in pow."""
        return Fp(pow(self.value, exp, FIELD_MODULUS))

    def inv(self) -> "Fp":
        """Multiplicative inverse via Fermat's little theorem: a^(p-2) mod p."""
        if self.value == 0:
            raise ZeroDivisionError("Cannot invert zero in a finite field.")
        return Fp(pow(self.value, FIELD_MODULUS - 2, FIELD_MODULUS))

    def __truediv__(self, other: "Fp") -> "Fp":
        return self * other.inv()

    # ------------------------------------------------------------------
    # Comparison and representation
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Fp):
            return self.value == other.value
        if isinstance(other, int):
            return self.value == other % FIELD_MODULUS
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.value)

    def __repr__(self) -> str:
        return f"Fp({self.value})"

    def __int__(self) -> int:
        return self.value


# ------------------------------------------------------------------
# Batch operations on plain int lists (convenience helpers)
# ------------------------------------------------------------------


def batch_mul(a: list[int], b: list[int], mod: int) -> list[int]:
    """Element-wise multiplication modulo *mod* on two int lists."""
    if len(a) != len(b):
        raise ValueError(f"Length mismatch: len(a)={len(a)}, len(b)={len(b)}")
    return [(x * y) % mod for x, y in zip(a, b)]


def batch_add(a: list[int], b: list[int], mod: int) -> list[int]:
    """Element-wise addition modulo *mod* on two int lists."""
    if len(a) != len(b):
        raise ValueError(f"Length mismatch: len(a)={len(a)}, len(b)={len(b)}")
    return [(x + y) % mod for x, y in zip(a, b)]


# ------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("=== field.py smoke test ===")

    a = Fp(5)
    a_inv = a.inv()
    product = a * a_inv
    assert product == Fp(1), f"Expected Fp(1), got {product}"
    print(f"Fp(5) * Fp(5).inv() = {product}  OK")

    # Test __pow__
    assert Fp(2) ** 10 == Fp(1024)
    print(f"Fp(2) ** 10 = {Fp(2) ** 10}  OK")

    # Test field wrap-around
    assert Fp(FIELD_MODULUS) == Fp(0)
    print(f"Fp(FIELD_MODULUS) == Fp(0)  OK")

    # Test batch_mul and batch_add
    res_mul = batch_mul([3, 4, 5], [2, 2, 2], FIELD_MODULUS)
    assert res_mul == [6, 8, 10]
    print(f"batch_mul([3,4,5], [2,2,2]) = {res_mul}  OK")

    res_add = batch_add([1, 2, 3], [9, 8, 7], FIELD_MODULUS)
    assert res_add == [10, 10, 10]
    print(f"batch_add([1,2,3], [9,8,7]) = {res_add}  OK")

    print("\nAll field.py checks passed.")
