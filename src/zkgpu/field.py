"""
zkgpu/field.py — Finite field arithmetic over BN254 scalar field (CPU baseline).

BN254 scalar field modulus r (order of the base point G):
  r = 21888242871839275222246405745257275088548364400416034343698204186575808495617

All arithmetic is performed mod r. This module provides:
  - Fp class: wraps a field element with operator overloading
  - batch_mul / batch_add: element-wise operations on plain int lists
"""

from __future__ import annotations

import operator

FIELD_MODULUS: int = 21888242871839275222246405745257275088548364400416034343698204186575808495617


def _value_of(x: object) -> int | None:
    """Canonical int for an Fp or an integer operand, or None for anything else."""
    if isinstance(x, Fp):
        return x.value
    try:
        return operator.index(x) % FIELD_MODULUS
    except TypeError:
        return None


class Fp:
    """
    Element of the finite field F_r (BN254 scalar field).

    Immutable, because it is hashable: mutating an element used as a dict key
    or set member would silently corrupt that container. Arithmetic accepts
    plain ints on either side (Fp(3) + 4, 10 - Fp(3)), treating them as the
    field elements they reduce to.
    """

    __slots__ = ("value",)
    value: int

    def __init__(self, value: int | Fp) -> None:
        canonical = _value_of(value)
        if canonical is None:
            raise TypeError(f"Fp() needs an integer or Fp, got {type(value).__name__}")
        object.__setattr__(self, "value", canonical)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Fp is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("Fp is immutable")

    def __reduce__(self) -> tuple[type[Fp], tuple[int]]:
        # copy and pickle would otherwise restore __slots__ via setattr, which is blocked
        return (Fp, (self.value,))

    # ------------------------------------------------------------------
    # Arithmetic operators (each returns NotImplemented for unsupported
    # operand types, so Python raises the usual TypeError)
    # ------------------------------------------------------------------

    def __add__(self, other: Fp | int) -> Fp:
        v = _value_of(other)
        return NotImplemented if v is None else Fp(self.value + v)

    __radd__ = __add__

    def __sub__(self, other: Fp | int) -> Fp:
        v = _value_of(other)
        return NotImplemented if v is None else Fp(self.value - v)

    def __rsub__(self, other: int) -> Fp:
        v = _value_of(other)
        return NotImplemented if v is None else Fp(v - self.value)

    def __mul__(self, other: Fp | int) -> Fp:
        v = _value_of(other)
        return NotImplemented if v is None else Fp(self.value * v)

    __rmul__ = __mul__

    def __truediv__(self, other: Fp | int) -> Fp:
        v = _value_of(other)
        return NotImplemented if v is None else self * Fp(v).inv()

    def __rtruediv__(self, other: int) -> Fp:
        v = _value_of(other)
        return NotImplemented if v is None else Fp(v) * self.inv()

    def __neg__(self) -> Fp:
        return Fp(-self.value)

    def __pow__(self, exp: int) -> Fp:
        """Fast modular exponentiation; a negative exponent inverts first."""
        try:
            exp = operator.index(exp)
        except TypeError:
            return NotImplemented
        if exp < 0:
            return self.inv() ** -exp  # raises ZeroDivisionError for 0, like inv()
        return Fp(pow(self.value, exp, FIELD_MODULUS))

    def inv(self) -> Fp:
        """Multiplicative inverse via Fermat's little theorem: a^(p-2) mod p."""
        if self.value == 0:
            raise ZeroDivisionError("Cannot invert zero in a finite field.")
        return Fp(pow(self.value, FIELD_MODULUS - 2, FIELD_MODULUS))

    # ------------------------------------------------------------------
    # Comparison and representation
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        v = _value_of(other)
        return NotImplemented if v is None else self.value == v

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
