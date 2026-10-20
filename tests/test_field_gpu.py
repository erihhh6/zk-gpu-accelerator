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
from tests.known_bugs import known_bug

pytestmark = pytest.mark.filterwarnings("ignore::src.gpu_backend.GpuFallbackWarning")

NTT_MOD = 998_244_353
GPU = pytest.param("gpu", marks=pytest.mark.gpu)
FALLBACK = "fallback"


@pytest.fixture
def backend(request, monkeypatch):
    if request.param == FALLBACK:
        monkeypatch.setattr(field_gpu, "_gpu_available", False)
    return request.param


def as_input(backend: str, values: list[int]):
    arr = np.array(values, dtype=np.int64)
    return cp.asarray(arr) if backend == "gpu" else arr


def field_elements(n: int, mod: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(mod) for _ in range(n)]


# ------------------------------------------------------------------
# Agreement with Python integers
# ------------------------------------------------------------------


@pytest.mark.parametrize("backend", [GPU, FALLBACK], indirect=True)
@pytest.mark.parametrize("mod", [7, NTT_MOD, 2**31 - 1])
def test_batch_mul_matches_python(backend, mod):
    a, b = field_elements(1000, mod, 1), field_elements(1000, mod, 2)
    got = field_gpu.gpu_batch_mul(as_input(backend, a), as_input(backend, b), mod)
    assert got.tolist() == [(x * y) % mod for x, y in zip(a, b)]


@pytest.mark.parametrize("backend", [GPU, FALLBACK], indirect=True)
@pytest.mark.parametrize("mod", [7, NTT_MOD, 2**61 - 1])
def test_batch_add_matches_python(backend, mod):
    a, b = field_elements(1000, mod, 3), field_elements(1000, mod, 4)
    got = field_gpu.gpu_batch_add(as_input(backend, a), as_input(backend, b), mod)
    assert got.tolist() == [(x + y) % mod for x, y in zip(a, b)]


@pytest.mark.parametrize("backend", [GPU, FALLBACK], indirect=True)
@pytest.mark.parametrize("exp", [0, 1, 2, 65537, NTT_MOD - 2])  # NTT_MOD - 2: Fermat inverse
def test_batch_pow_matches_python(backend, exp):
    bases = field_elements(256, NTT_MOD, 5)
    got = field_gpu.gpu_batch_pow(as_input(backend, bases), exp, NTT_MOD)
    assert got.tolist() == [pow(x, exp, NTT_MOD) for x in bases]


# ------------------------------------------------------------------
# Known bugs
# ------------------------------------------------------------------


@pytest.mark.parametrize("backend", [GPU, FALLBACK], indirect=True)
@known_bug("int64 a * b overflows once mod > ~2^31.5 and returns wrong values silently", raises=AssertionError)
def test_batch_mul_large_modulus(backend):
    mod = 2**61 - 1  # Mersenne prime: (mod-1)*(mod-2) needs ~122 bits
    got = field_gpu.gpu_batch_mul(as_input(backend, [mod - 1]), as_input(backend, [mod - 2]), mod)
    assert got.tolist() == [((mod - 1) * (mod - 2)) % mod]


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(
            "gpu",
            marks=[
                pytest.mark.gpu,
                known_bug("CUDA % keeps the dividend's sign, so negative inputs leave [0, mod)", raises=AssertionError),
            ],
        ),
        FALLBACK,
    ],
    indirect=True,
)
def test_negative_inputs_reduce_into_field(backend):
    a, b = as_input(backend, [-3, -10]), as_input(backend, [5, 1])
    assert field_gpu.gpu_batch_mul(a, b, 7).tolist() == [(-15) % 7, (-10) % 7]
    assert field_gpu.gpu_batch_add(a, b, 7).tolist() == [2 % 7, (-9) % 7]


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(
            "gpu",
            marks=[pytest.mark.gpu, known_bug("negative exponents return 1 instead of an inverse power", raises=AssertionError)],
        ),
        FALLBACK,
    ],
    indirect=True,
)
def test_batch_pow_negative_exponent_is_inverse(backend):
    bases = [3, 5, 6]
    got = field_gpu.gpu_batch_pow(as_input(backend, bases), -1, 7)
    assert got.tolist() == [pow(x, -1, 7) for x in bases]
