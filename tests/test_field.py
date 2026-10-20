"""tests/test_field.py — Correctness tests for src/field.py"""

import pytest
from src.field import Fp, FIELD_MODULUS, batch_mul, batch_add
from tests.known_bugs import known_bug


# ------------------------------------------------------------------
# Fp class tests
# ------------------------------------------------------------------


def test_fp_wrap_around():
    """Values reduce modulo FIELD_MODULUS on construction."""
    assert Fp(FIELD_MODULUS) == Fp(0)
    assert Fp(FIELD_MODULUS + 1) == Fp(1)
    assert Fp(-1) == Fp(FIELD_MODULUS - 1)


def test_fp_addition():
    assert Fp(3) + Fp(4) == Fp(7)
    assert Fp(FIELD_MODULUS - 1) + Fp(1) == Fp(0)


def test_fp_subtraction():
    assert Fp(10) - Fp(3) == Fp(7)
    # Subtraction wraps around correctly
    assert Fp(0) - Fp(1) == Fp(FIELD_MODULUS - 1)


def test_fp_multiplication():
    assert Fp(6) * Fp(7) == Fp(42)
    # Large product wraps correctly
    large = Fp(FIELD_MODULUS - 1) * Fp(FIELD_MODULUS - 1)
    expected = pow(FIELD_MODULUS - 1, 2, FIELD_MODULUS)
    assert large == Fp(expected)


def test_fp_pow():
    assert Fp(2) ** 10 == Fp(1024)
    assert Fp(3) ** 0 == Fp(1)
    assert Fp(0) ** 0 == Fp(1)  # Python convention: 0^0 = 1


def test_fp_inverse_fermat():
    """a * a.inv() == 1 for any nonzero a."""
    for val in [1, 2, 5, 100, FIELD_MODULUS - 1]:
        a = Fp(val)
        assert a * a.inv() == Fp(1), f"Inverse failed for Fp({val})"


def test_fp_inverse_zero_raises():
    with pytest.raises(ZeroDivisionError):
        Fp(0).inv()


def test_fp_truediv():
    a = Fp(10)
    b = Fp(2)
    assert a / b == Fp(5)


def test_fp_equality():
    assert Fp(7) == Fp(7)
    assert Fp(7) != Fp(8)
    assert Fp(7) == 7  # int comparison


def test_fp_repr():
    assert "Fp(" in repr(Fp(42))


@known_bug("Fp arithmetic with plain ints raises AttributeError", raises=AttributeError)
def test_fp_interoperates_with_int():
    assert Fp(3) + 4 == Fp(7)
    assert 4 + Fp(3) == Fp(7)
    assert Fp(3) * 2 == Fp(6)
    assert 10 - Fp(3) == Fp(7)


# ------------------------------------------------------------------
# Batch operation tests
# ------------------------------------------------------------------


def test_batch_mul_basic():
    result = batch_mul([3, 4, 5], [2, 2, 2], FIELD_MODULUS)
    assert result == [6, 8, 10]


def test_batch_add_basic():
    result = batch_add([1, 2, 3], [9, 8, 7], FIELD_MODULUS)
    assert result == [10, 10, 10]


def test_batch_mul_wraparound():
    m = 7
    result = batch_mul([6, 5], [2, 3], m)
    assert result == [(6 * 2) % 7, (5 * 3) % 7]


def test_batch_add_wraparound():
    m = 7
    result = batch_add([6, 5], [2, 3], m)
    assert result == [(6 + 2) % 7, (5 + 3) % 7]


def test_batch_length_mismatch():
    with pytest.raises(ValueError):
        batch_mul([1, 2], [1], FIELD_MODULUS)
    with pytest.raises(ValueError):
        batch_add([1], [1, 2], FIELD_MODULUS)
