"""tests/test_msm.py — Correctness tests for src/msm.py (and GPU if available)."""

import random
import pytest
from src.curve import G, Point
from src.msm import msm_naive, msm_double_and_add


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


# ------------------------------------------------------------------
# GPU MSM test (skipped if GPU not available)
# ------------------------------------------------------------------


def test_msm_gpu_equals_cpu():
    """msm_gpu result matches CPU MSM for N=100 random points."""
    pytest.importorskip("cupy", reason="CuPy not installed — skipping GPU test")

    import numpy as np
    from src.msm_gpu import msm_gpu, _gpu_available
    from src.curve import P as CURVE_P

    if not _gpu_available:
        pytest.skip("CUDA GPU not available")

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
