"""tests/test_ntt.py — Correctness tests for src/ntt.py (and GPU if available)."""

import random
import pytest
from src.ntt import ntt, polynomial_mul, MOD


# ------------------------------------------------------------------
# CPU NTT tests
# ------------------------------------------------------------------


def test_ntt_inverse_small():
    """INTT(NTT(a)) == a for a small vector."""
    a = [1, 2, 3, 4, 5, 6, 7, 8]
    transformed = ntt(a, MOD)
    recovered = ntt(transformed, MOD, invert=True)
    assert a == recovered


def test_ntt_inverse_random():
    """INTT(NTT(a)) == a for a random vector of length 1024."""
    random.seed(7)
    n = 1024
    a = [random.randint(0, MOD - 1) for _ in range(n)]
    recovered = ntt(ntt(a, MOD), MOD, invert=True)
    assert a == recovered


def test_ntt_linearity():
    """NTT is linear: NTT(a + b) == NTT(a) + NTT(b) pointwise."""
    random.seed(13)
    n = 64
    a = [random.randint(0, MOD - 1) for _ in range(n)]
    b = [random.randint(0, MOD - 1) for _ in range(n)]
    ab = [(x + y) % MOD for x, y in zip(a, b)]
    lhs = ntt(ab, MOD)
    rhs = [(x + y) % MOD for x, y in zip(ntt(a, MOD), ntt(b, MOD))]
    assert lhs == rhs


def test_ntt_single_element():
    """NTT of a single element is the element itself."""
    assert ntt([42], MOD) == [42]
    assert ntt([42], MOD, invert=True) == [42]


def test_ntt_non_power_of_two_raises():
    with pytest.raises(ValueError):
        ntt([1, 2, 3], MOD)


def test_ntt_polynomial_mul_simple():
    """(1 + x) * (1 + x) = 1 + 2x + x²."""
    result = polynomial_mul([1, 1], [1, 1], MOD)
    assert result == [1, 2, 1]


def test_ntt_polynomial_mul_matches_naive():
    """NTT-based polynomial multiplication matches naive O(N²) convolution."""

    def naive_mul(p: list[int], q: list[int], m: int) -> list[int]:
        out = [0] * (len(p) + len(q) - 1)
        for i, a in enumerate(p):
            for j, b in enumerate(q):
                out[i + j] = (out[i + j] + a * b) % m
        return out

    random.seed(31)
    p1 = [random.randint(0, MOD - 1) for _ in range(32)]
    p2 = [random.randint(0, MOD - 1) for _ in range(32)]
    assert polynomial_mul(p1, p2, MOD) == naive_mul(p1, p2, MOD)


def test_ntt_zero_vector():
    """NTT of all-zeros is all-zeros."""
    a = [0] * 8
    assert ntt(a, MOD) == a
    assert ntt(a, MOD, invert=True) == a


# ------------------------------------------------------------------
# GPU NTT test (skipped if GPU not available)
# ------------------------------------------------------------------


def test_ntt_gpu_equals_cpu():
    """ntt_gpu result matches CPU NTT for N=256."""
    from src.gpu_backend import GPU_AVAILABLE, GPU_UNAVAILABLE_REASON, cp
    from src.ntt_gpu import ntt_gpu

    if not GPU_AVAILABLE:
        pytest.skip(f"CUDA GPU not available: {GPU_UNAVAILABLE_REASON}")

    random.seed(55)
    n = 256
    a_list = [random.randint(0, MOD - 1) for _ in range(n)]
    a_cp = cp.array(a_list, dtype=cp.int64)

    cpu_result = ntt(a_list, MOD)
    gpu_result = ntt_gpu(a_cp, MOD)

    assert cpu_result == gpu_result.tolist()
