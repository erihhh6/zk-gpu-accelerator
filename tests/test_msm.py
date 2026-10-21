"""tests/test_msm.py — Correctness tests for zkgpu/msm.py (and GPU if available)."""

import random

import numpy as np
import pytest

from zkgpu import curve as curve_module, msm as msm_module, msm_gpu as msm_gpu_module
from zkgpu.curve import G, P as CURVE_P, Point
from zkgpu.field import FIELD_MODULUS as R
from zkgpu.msm import msm_double_and_add, msm_naive
from zkgpu.msm_gpu import msm_pippenger_gpu

pytestmark = pytest.mark.filterwarnings("ignore::zkgpu.gpu_backend.GpuFallbackWarning")

GPU = pytest.param("gpu", marks=pytest.mark.gpu)
FALLBACK = "fallback"
BACKENDS = [GPU, FALLBACK]


@pytest.fixture
def backend(request, monkeypatch):
    """Run msm_pippenger_gpu on CUDA ("gpu") or force its CPU fallback ("fallback")."""
    if request.param == FALLBACK:
        monkeypatch.setattr(msm_gpu_module, "_gpu_available", False)
    return request.param


def pippenger(points: list[Point], scalars: list[int], window_size: int = 8) -> Point:
    """Call msm_pippenger_gpu with the array layout callers use for BN254 data."""
    # object dtype: 254-bit coordinates and scalars do not fit in int64
    px = np.array([p.x for p in points], dtype=object)
    py = np.array([p.y for p in points], dtype=object)
    sc = np.array(scalars, dtype=object)
    return msm_pippenger_gpu(px, py, sc, window_size=window_size)


def count_doublings(monkeypatch) -> dict[str, int]:
    """
    Count the MSMs' Jacobian doublings from here on (build inputs before calling this).

    Patched where the MSM modules look the function up; the callers' tests also
    assert a lower bound, so a counter that is no longer wired in fails loudly
    instead of passing with a count of zero.
    """
    calls = {"n": 0}
    original = curve_module.jacobian_double

    def counting_double(j):
        calls["n"] += 1
        return original(j)

    monkeypatch.setattr(msm_module, "jacobian_double", counting_double)
    monkeypatch.setattr(msm_gpu_module, "jacobian_double", counting_double)
    return calls


@pytest.fixture(scope="module")
def msm_inputs():
    rng = random.Random(29)
    points = [G * rng.randint(1, 2**32) for _ in range(30)]
    scalars = [rng.randint(0, 2**32) for _ in range(30)]
    return points, scalars, msm_naive(points, scalars)


@pytest.fixture(scope="module")
def full_size_inputs():
    """Uniform elements of F_r (~254 bits), so every 64-bit limb is populated."""
    rng = random.Random(31)
    points = [G * rng.randint(1, 2**32) for _ in range(12)]
    scalars = [rng.randrange(R) for _ in range(12)]
    return points, scalars, msm_double_and_add(points, scalars)


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
    INF = Point.identity()
    result = msm_naive([INF, INF, INF], [5, 10, 20])
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


def test_double_and_add_doubles_once_per_bit(monkeypatch):
    """Straus/Shamir: one shared accumulator, so doublings do not grow with N."""
    rng = random.Random(5)
    points = [G * rng.randint(1, 2**32) for _ in range(8)]
    scalars = [rng.randint(2**15, 2**16 - 1) for _ in range(8)]  # exactly 16 bits each
    calls = count_doublings(monkeypatch)
    msm_double_and_add(points, scalars)
    assert 16 <= calls["n"] <= 16 + len(points)  # one shared accumulator doubling per bit


def test_double_and_add_negative_scalars():
    """-k·P = k·(-P); the old bit loop silently returned O when every scalar was negative."""
    points = [G, G * 2, G * 3]
    for scalars in ([-5, 7, 0], [-1, -2, -3]):
        assert msm_double_and_add(points, scalars) == msm_naive(points, scalars)
    assert msm_double_and_add([G], [-5]) == -(G * 5)


def test_double_and_add_full_size_and_oversized_scalars():
    rng = random.Random(43)
    points = [G * k for k in (3, 5, 7, 11)]
    scalars = [rng.randrange(R) for _ in points]  # ~254-bit, like real ZK scalars
    assert msm_double_and_add(points, scalars) == msm_naive(points, scalars)
    assert msm_double_and_add([G], [R + 3]) == G * 3  # G has order r


def test_double_and_add_with_identity_points():
    INF = Point.identity()
    assert msm_double_and_add([INF, G, INF], [5, 3, -2]) == G * 3


# ------------------------------------------------------------------
# Pippenger (msm_pippenger_gpu): GPU buckets and the CPU fallback
# ------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, indirect=True)
@pytest.mark.parametrize("window_size", [1, 3, 8, 11])
def test_pippenger_matches_naive(backend, window_size, msm_inputs):
    points, scalars, expected = msm_inputs
    assert pippenger(points, scalars, window_size) == expected


@pytest.mark.parametrize("backend", BACKENDS, indirect=True)
@pytest.mark.parametrize("window_size", [3, 5, 8])  # 3 and 5 make windows straddle 64-bit limbs
def test_pippenger_full_size_scalars(backend, window_size, full_size_inputs):
    """Real ZK scalars are uniform in F_r (~254 bits), filling all four limbs."""
    points, scalars, expected = full_size_inputs
    assert pippenger(points, scalars, window_size) == expected


@pytest.mark.parametrize("backend", BACKENDS, indirect=True)
def test_pippenger_bucket_edge_cases(backend):
    """P and -P in one bucket cancel to O; P and P in one bucket take the doubling branch."""
    assert pippenger([G, -G], [7, 7]) == Point.identity()
    assert pippenger([G, G], [5, 5]) == G * 10
    assert pippenger([G, G], [0, 0]) == Point.identity()


@pytest.mark.parametrize("backend", BACKENDS, indirect=True)
def test_pippenger_reduces_scalars_mod_r(backend):
    point = G * 3
    for scalar in (-5, R, R + 7, 2**300 + 1):
        assert pippenger([point], [scalar]) == point * (scalar % R)


@pytest.mark.parametrize("backend", BACKENDS, indirect=True)
def test_pippenger_returns_a_point(backend):
    """The old API returned (x, y) and used (0, 0), which is not on the curve, for infinity."""
    assert msm_pippenger_gpu([], [], []) == Point.identity()
    result = msm_pippenger_gpu([G.x], [G.y], [R])  # r·G = O
    assert isinstance(result, Point) and result.is_identity()
    assert msm_pippenger_gpu([G.x, G.x], [G.y, G.y], [2, 3]) == G * 5  # plain lists work too


@pytest.mark.parametrize("backend", BACKENDS, indirect=True)
def test_pippenger_doublings_are_linear_in_scalar_bits(backend, monkeypatch):
    """Horner's rule: w·(windows - 1) = 248 doublings for w = 8, not a loop per window (3968)."""
    points = [G * k for k in (3, 5, 7, 11)]
    scalars = [2**250 + k for k in range(len(points))]
    if backend == "gpu":
        msm_gpu_module.compile_kernels()  # its self-test runs an MSM too: keep it out of the count
    calls = count_doublings(monkeypatch)
    pippenger(points, scalars)
    assert 8 * 31 <= calls["n"] <= 256 + len(points)  # w·(windows - 1) = 248 for w = 8


@pytest.fixture(scope="module")
def large_inputs():
    """Hundreds of points with full 254-bit scalars: enough to fill many buckets per window."""
    rng = random.Random(37)
    step = G * rng.randrange(1, R)
    points = [G * rng.randrange(1, R)]
    for _ in range(399):
        points.append(points[-1] + step)  # distinct points, far cheaper than 400 scalar multiplications
    scalars = [rng.randrange(R) for _ in points]
    return points, scalars, msm_double_and_add(points, scalars)


@pytest.mark.gpu
@pytest.mark.parametrize("window_size", [2, 6, 9, 13, 16])
def test_gpu_pippenger_at_scale(window_size, large_inputs):
    """
    The GPU path's parallel pieces: sorted buckets with several points each,
    and windows split into reduction segments (2^w / 32 of them per window,
    so up to 2048 at w = 16).
    """
    points, scalars, expected = large_inputs
    assert pippenger(points, scalars, window_size) == expected


@pytest.mark.gpu
def test_gpu_pippenger_with_one_heavy_bucket_per_window():
    """Equal scalars put every point of a window into one bucket, which one thread adds up alone."""
    points = [G * k for k in range(1, 201)]
    scalar = 2**200 + 12345
    assert pippenger(points, [scalar] * len(points)) == G * (sum(range(1, 201)) * scalar % R)


@pytest.mark.gpu
def test_miscompiled_msm_kernels_fail_their_self_test(monkeypatch):
    """A wrong segment correction (simulating a miscompile) is caught when the kernels load."""
    old = "zk_jac_mul_small(running, (unsigned int)(first - 1), shifted);"
    assert msm_gpu_module._MSM_SOURCE.count(old) == 1
    broken = msm_gpu_module._MSM_SOURCE.replace(old, old.replace("first - 1", "first"))
    monkeypatch.setattr(msm_gpu_module, "_MSM_SOURCE", broken)
    monkeypatch.setattr(msm_gpu_module, "_compiled", {})
    with pytest.raises(RuntimeError, match="failed their self-test"):
        msm_gpu_module.compile_kernels()
    assert msm_gpu_module._compiled == {}


def test_pippenger_validates_arguments():
    with pytest.raises(ValueError, match="Length mismatch"):
        msm_pippenger_gpu([G.x], [G.y], [1, 2])
    for bad_window in (0, 17):
        with pytest.raises(ValueError, match="window_size"):
            msm_pippenger_gpu([G.x], [G.y], [1], window_size=bad_window)
    with pytest.raises(TypeError, match="scalars must contain integers"):
        msm_pippenger_gpu([G.x], [G.y], [1.5])
    # Caller-supplied coordinates are validated: off-curve input is rejected
    with pytest.raises(ValueError, match="not on the BN254 curve"):
        msm_pippenger_gpu([1], [3], [5])
    # mod was dropped and window_size is keyword-only, so the old call shape
    # fails loudly instead of silently using P as the window size
    with pytest.raises(TypeError):
        msm_pippenger_gpu([G.x], [G.y], [1], CURVE_P)
