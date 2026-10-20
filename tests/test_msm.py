"""tests/test_msm.py — Correctness tests for src/msm.py (and GPU if available)."""

import random

import numpy as np
import pytest

from src import msm_gpu as msm_gpu_module
from src.curve import G, P as CURVE_P, Point
from src.field import FIELD_MODULUS as R
from src.msm import msm_naive, msm_double_and_add
from src.msm_gpu import msm_gpu, msm_pippenger_gpu
from tests.known_bugs import known_bug

pytestmark = pytest.mark.filterwarnings("ignore::src.gpu_backend.GpuFallbackWarning")

GPU = pytest.param("gpu", marks=pytest.mark.gpu)
FALLBACK = "fallback"


@pytest.fixture
def backend(request, monkeypatch):
    """Run msm_gpu's code on CUDA ("gpu") or force its CPU fallback ("fallback")."""
    if request.param == FALLBACK:
        monkeypatch.setattr(msm_gpu_module, "_gpu_available", False)
    return request.param


def pippenger(points: list[Point], scalars: list[int], window_size: int = 8) -> Point:
    """Adapt msm_pippenger_gpu's array API to Points; (0, 0) is its identity sentinel."""
    px = np.array([p.x for p in points], dtype=object)
    py = np.array([p.y for p in points], dtype=object)
    # object dtype keeps arbitrary-precision scalars intact on the way in
    sc = np.array(scalars, dtype=object)
    x, y = msm_pippenger_gpu(px, py, sc, CURVE_P, window_size=window_size)
    return Point.identity() if (x, y) == (0, 0) else Point(x, y)


def count_doublings(monkeypatch) -> dict[str, int]:
    """Count Point.double calls from here on (build inputs before calling this)."""
    calls = {"n": 0}
    original = Point.double

    def counting_double(self):
        calls["n"] += 1
        return original(self)

    monkeypatch.setattr(Point, "double", counting_double)
    return calls


@pytest.fixture(scope="module")
def msm_inputs():
    rng = random.Random(29)
    points = [G * rng.randint(1, 2**32) for _ in range(30)]
    scalars = [rng.randint(0, 2**32) for _ in range(30)]
    return points, scalars, msm_naive(points, scalars)


# ------------------------------------------------------------------
# CPU MSM tests
# ------------------------------------------------------------------


def test_msm_single_point():
    """msm([G], [k]) == k*G for several k values."""
    for k in [0, 1, 2, 7, 42, 1000]:
        expected = G * k
        assert msm_naive([G], [k]) == expected
        assert msm_double_and_add([G], [k]) == expected


def test_msm_zero_scalar():
    """Any point multiplied by 0 contributes identity."""
    result_naive = msm_naive([G, G], [5, 0])
    result_sha = msm_double_and_add([G, G], [5, 0])
    expected = G * 5
    assert result_naive == expected
    assert result_sha == expected


def test_msm_linearity():
    """msm([G, G], [a, b]) == (a+b)*G."""
    a, b = 7, 13
    expected = G * (a + b)
    assert msm_naive([G, G], [a, b]) == expected
    assert msm_double_and_add([G, G], [a, b]) == expected


def test_msm_all_identity_points():
    """msm with identity points always returns identity."""
    O = Point.identity()
    result = msm_naive([O, O, O], [5, 10, 20])
    assert result == Point.identity()


def test_msm_empty():
    """msm on empty lists returns identity."""
    assert msm_naive([], []) == Point.identity()
    assert msm_double_and_add([], []) == Point.identity()


def test_msm_both_implementations_agree():
    """msm_naive == msm_double_and_add for random N=20 inputs."""
    random.seed(17)
    n = 20
    scalars = [random.randint(0, 2**32) for _ in range(n)]
    points = [G * random.randint(1, 2**32) for _ in range(n)]
    assert msm_naive(points, scalars) == msm_double_and_add(points, scalars)


def test_msm_length_mismatch_raises():
    with pytest.raises(ValueError):
        msm_naive([G, G], [1])
    with pytest.raises(ValueError):
        msm_double_and_add([G], [1, 2])


@known_bug(
    "msm_double_and_add doubles every input point at every bit (N·bits doublings), so it is not Shamir/Straus",
    raises=AssertionError,
)
def test_double_and_add_doubles_once_per_bit(monkeypatch):
    rng = random.Random(5)
    points = [G * rng.randint(1, 2**32) for _ in range(8)]
    scalars = [rng.randint(2**15, 2**16 - 1) for _ in range(8)]  # exactly 16 bits each
    calls = count_doublings(monkeypatch)
    msm_double_and_add(points, scalars)
    assert calls["n"] <= 16 + len(points)  # one shared accumulator doubling per bit


# ------------------------------------------------------------------
# Pippenger (msm_pippenger_gpu): GPU window extraction and CPU fallback
# ------------------------------------------------------------------


@pytest.mark.parametrize("backend", [GPU, FALLBACK], indirect=True)
@pytest.mark.parametrize("window_size", [3, 8, 11])  # 3 and 11 do not divide 256
def test_pippenger_matches_naive(backend, window_size, msm_inputs):
    points, scalars, expected = msm_inputs
    assert pippenger(points, scalars, window_size) == expected


@pytest.mark.parametrize("backend", [GPU, FALLBACK], indirect=True)
def test_pippenger_bucket_edge_cases(backend):
    """P and -P in one bucket cancel to O; P and P in one bucket take the doubling branch."""
    assert pippenger([G, -G], [7, 7]) == Point.identity()
    assert pippenger([G, G], [5, 5]) == G * 10
    assert pippenger([G, G], [0, 0]) == Point.identity()


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(
            "gpu",
            marks=[pytest.mark.gpu, known_bug("scalars are packed into int64 before window extraction", raises=OverflowError)],
        ),
        FALLBACK,
    ],
    indirect=True,
)
def test_pippenger_full_size_scalars(backend):
    """Real ZK scalars are uniformly random elements of F_r (~254 bits), not 32-bit ints."""
    rng = random.Random(41)
    points = [G * k for k in (3, 5, 7, 11)]
    scalars = [rng.randrange(R) for _ in points]
    assert pippenger(points, scalars) == msm_naive(points, scalars)


@pytest.mark.parametrize("backend", [GPU, FALLBACK], indirect=True)
@known_bug("scalars are not reduced mod r: negatives and values >= 2^256 give wrong sums")
def test_pippenger_reduces_scalars_mod_r(backend):
    point = G * 3
    for scalar in (-5, R + 7, 2**300 + 1):
        assert pippenger([point], [scalar]) == point * (scalar % R)


@known_bug(
    "each window is shifted by its own doubling loop (~3968 doublings for w=8) instead of Horner's ~256",
    raises=AssertionError,
)
def test_pippenger_doublings_are_linear_in_scalar_bits(monkeypatch):
    monkeypatch.setattr(msm_gpu_module, "_gpu_available", False)  # full-size scalars need the CPU path today
    points = [G * k for k in (3, 5, 7, 11)]
    scalars = [2**250 + k for k in range(len(points))]
    calls = count_doublings(monkeypatch)
    pippenger(points, scalars)
    assert calls["n"] <= 256 + len(points)


# ------------------------------------------------------------------
# msm_gpu (naive per-point MSM behind a GPU-shaped API)
# ------------------------------------------------------------------


@pytest.mark.gpu
def test_msm_gpu_equals_cpu():
    """msm_gpu result matches CPU MSM for N=100 random points."""
    random.seed(23)
    n = 100
    scalars = [random.randint(1, 2**32) for _ in range(n)]
    points = [G * random.randint(1, 2**32) for _ in range(n)]

    cpu_result = msm_naive(points, scalars)

    # BN254 coordinates are ~254-bit: keep as object dtype (Python ints)
    points_x = np.array([p.x for p in points], dtype=object)
    points_y = np.array([p.y for p in points], dtype=object)
    scalars_np = np.array(scalars, dtype=np.int64)

    # msm_gpu returns (x, y) as Python ints
    gx, gy = msm_gpu(points_x, points_y, scalars_np, CURVE_P)
    gpu_result = Point(gx, gy)

    assert cpu_result == gpu_result
