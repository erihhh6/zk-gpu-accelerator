"""
tests/test_field_gpu.py — src/field_gpu.py on CUDA and through its CPU fallback.

Every case runs twice: on the GPU (skipped without one) and through the NumPy
fallback (forced by switching the GPU flag off), so CI exercises the fallback
and a GPU machine checks that both paths agree with Python's exact integers.
"""

import random

import numpy as np
import pytest

from src import field_gpu
from src.gpu_backend import cp

pytestmark = pytest.mark.filterwarnings("ignore::src.gpu_backend.GpuFallbackWarning")

NTT_MOD = 998_244_353
INT64_SAFE_LIMIT = 3_037_000_500  # largest modulus that still uses the plain int64 multiply
LARGEST_INT64_PRIME = 2**63 - 25
INT64_MIN, INT64_MAX = -(2**63), 2**63 - 1

# Both sides of the int64-product boundary, up to the largest int64 prime
MODULI = [7, NTT_MOD, INT64_SAFE_LIMIT, INT64_SAFE_LIMIT + 1, 2**61 - 1, LARGEST_INT64_PRIME]

GPU = pytest.param("gpu", marks=pytest.mark.gpu)
FALLBACK = "fallback"
BACKENDS = [GPU, FALLBACK]


@pytest.fixture
def backend(request, monkeypatch):
    if request.param == FALLBACK:
        monkeypatch.setattr(field_gpu, "_gpu_available", False)
    return request.param


def as_input(backend: str, values, dtype=np.int64):
    arr = np.array(values, dtype=dtype)
    return cp.asarray(arr) if backend == "gpu" else arr


def field_elements(n: int, mod: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(mod) for _ in range(n)]


# ------------------------------------------------------------------
# Agreement with Python integers
# ------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, indirect=True)
@pytest.mark.parametrize("mod", MODULI)
def test_batch_mul_matches_python(backend, mod):
    a, b = field_elements(1000, mod, 1), field_elements(1000, mod, 2)
    got = field_gpu.gpu_batch_mul(as_input(backend, a), as_input(backend, b), mod)
    assert got.dtype == np.int64
    assert got.tolist() == [(x * y) % mod for x, y in zip(a, b)]


@pytest.mark.parametrize("backend", BACKENDS, indirect=True)
@pytest.mark.parametrize("mod", MODULI)
def test_batch_add_matches_python(backend, mod):
    a, b = field_elements(1000, mod, 3), field_elements(1000, mod, 4)
    got = field_gpu.gpu_batch_add(as_input(backend, a), as_input(backend, b), mod)
    assert got.dtype == np.int64
    assert got.tolist() == [(x + y) % mod for x, y in zip(a, b)]


@pytest.mark.parametrize("backend", BACKENDS, indirect=True)
@pytest.mark.parametrize("mod", [INT64_SAFE_LIMIT, INT64_SAFE_LIMIT + 1, LARGEST_INT64_PRIME])
def test_extreme_residues(backend, mod):
    """(mod-1)^2 is the largest product: exactly fits int64 at the limit, overflows just above it."""
    a = [mod - 1, mod - 1, mod - 2, 0, 1]
    b = [mod - 1, 1, mod - 1, mod - 1, mod - 1]
    a_in, b_in = as_input(backend, a), as_input(backend, b)
    assert field_gpu.gpu_batch_mul(a_in, b_in, mod).tolist() == [(x * y) % mod for x, y in zip(a, b)]
    assert field_gpu.gpu_batch_add(a_in, b_in, mod).tolist() == [(x + y) % mod for x, y in zip(a, b)]


@pytest.mark.parametrize("backend", BACKENDS, indirect=True)
@pytest.mark.parametrize("exp", [0, 1, 2, 65537, NTT_MOD - 2])  # NTT_MOD - 2: Fermat inverse
def test_batch_pow_matches_python(backend, exp):
    bases = field_elements(256, NTT_MOD, 5)
    got = field_gpu.gpu_batch_pow(as_input(backend, bases), exp, NTT_MOD)
    assert got.tolist() == [pow(x, exp, NTT_MOD) for x in bases]


@pytest.mark.parametrize("backend", BACKENDS, indirect=True)
@pytest.mark.parametrize("exp", [3, 2**61 - 3])  # 2^61 - 3 = p - 2: Fermat inverse
def test_batch_pow_large_modulus(backend, exp):
    mod = 2**61 - 1
    bases = field_elements(64, mod, 6)
    got = field_gpu.gpu_batch_pow(as_input(backend, bases), exp, mod)
    assert got.tolist() == [pow(x, exp, mod) for x in bases]


# ------------------------------------------------------------------
# Input normalisation
# ------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, indirect=True)
@pytest.mark.parametrize("mod", [7, LARGEST_INT64_PRIME])
def test_unreduced_and_negative_inputs_reduce_into_field(backend, mod):
    a = [-3, -10, INT64_MAX, INT64_MIN, 12, 0]
    b = [5, 1, INT64_MAX, INT64_MIN, -12, -1]
    a_in, b_in = as_input(backend, a), as_input(backend, b)
    assert field_gpu.gpu_batch_mul(a_in, b_in, mod).tolist() == [(x * y) % mod for x, y in zip(a, b)]
    assert field_gpu.gpu_batch_add(a_in, b_in, mod).tolist() == [(x + y) % mod for x, y in zip(a, b)]
    assert field_gpu.gpu_batch_pow(a_in, 3, mod).tolist() == [pow(x, 3, mod) for x in a]


@pytest.mark.parametrize("backend", BACKENDS, indirect=True)
def test_narrow_integer_dtypes_are_widened(backend):
    """(NTT_MOD - 1)^2 overflows int32 but not int64."""
    values = as_input(backend, [NTT_MOD - 1, 3], dtype=np.int32)
    assert field_gpu.gpu_batch_mul(values, values, NTT_MOD).tolist() == [pow(NTT_MOD - 1, 2, NTT_MOD), 9]
    assert field_gpu.gpu_batch_pow(values, 2, 2**61 - 1).tolist() == [(NTT_MOD - 1) ** 2, 9]


@pytest.mark.parametrize("backend", BACKENDS, indirect=True)
def test_batch_pow_negative_exponent_is_inverse(backend):
    bases = [3, 5, 6]
    for exp in (-1, -3):
        got = field_gpu.gpu_batch_pow(as_input(backend, bases), exp, 7)
        assert got.tolist() == [pow(x, exp, 7) for x in bases]


@pytest.mark.parametrize("backend", BACKENDS, indirect=True)
def test_batch_pow_negative_exponent_rejects_zero(backend):
    """7 ≡ 0 (mod 7): zero has no inverse, so this must raise like Python's pow()."""
    with pytest.raises(ValueError, match="not invertible"):
        field_gpu.gpu_batch_pow(as_input(backend, [3, 7]), -1, 7)


# ------------------------------------------------------------------
# Argument validation
# ------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, indirect=True)
@pytest.mark.parametrize("mod", [0, 1, -7, 2**63])
def test_rejects_modulus_outside_int64_range(backend, mod):
    x = as_input(backend, [1, 2])
    with pytest.raises(ValueError, match="2 <= mod < 2"):
        field_gpu.gpu_batch_mul(x, x, mod)
    with pytest.raises(ValueError, match="2 <= mod < 2"):
        field_gpu.gpu_batch_add(x, x, mod)
    with pytest.raises(ValueError, match="2 <= mod < 2"):
        field_gpu.gpu_batch_pow(x, 2, mod)


@pytest.mark.parametrize("backend", BACKENDS, indirect=True)
def test_rejects_non_integer_arguments(backend):
    ints = as_input(backend, [1, 2])
    floats = as_input(backend, [1.5, 2.0], dtype=np.float64)
    with pytest.raises(TypeError):
        field_gpu.gpu_batch_mul(ints, ints, 7.0)  # float modulus
    with pytest.raises(TypeError, match="integer array"):
        field_gpu.gpu_batch_mul(floats, ints, 7)
    with pytest.raises(TypeError, match="integer array"):
        field_gpu.gpu_batch_add(ints, floats, 7)
    with pytest.raises(TypeError, match="integer array"):
        field_gpu.gpu_batch_pow(floats, 2, 7)
    with pytest.raises(TypeError):
        field_gpu.gpu_batch_pow(ints, 2.0, 7)  # float exponent
