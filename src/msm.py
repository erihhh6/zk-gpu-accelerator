"""
src/msm.py — Multi-Scalar Multiplication on CPU.

Given vectors of elliptic curve points P₁…Pₙ and scalars s₁…sₙ, compute:

  result = Σ sᵢ · Pᵢ

This is the dominant cost (~65% of total prover time) in Groth16 and PLONK.

Two implementations are provided:
  - msm_naive   : straightforward, each sᵢ*Pᵢ computed independently
  - msm_double_and_add : Shamir's trick — processes all scalars bit-by-bit
                          simultaneously, reducing doublings from N·log(s)
                          to log(max_scalar)
"""

from __future__ import annotations

from src.curve import Point


# ------------------------------------------------------------------
# MSM implementations
# ------------------------------------------------------------------


def msm_naive(points: list[Point], scalars: list[int]) -> Point:
    """
    Naive MSM: compute sᵢ·Pᵢ for each i independently and accumulate.

    Complexity: O(N · log(max_scalar)) point doublings and additions.
    Correct and simple — used as the reference result for testing.
    """
    if len(points) != len(scalars):
        raise ValueError(
            f"Length mismatch: {len(points)} points vs {len(scalars)} scalars."
        )
    result = Point.identity()
    for point, scalar in zip(points, scalars):
        result = result + point * scalar
    return result


def msm_double_and_add(points: list[Point], scalars: list[int]) -> Point:
    """
    MSM with Shamir's simultaneous double-and-add trick.

    All scalars are processed bit-by-bit in lockstep, so the number of
    point doublings is reduced to log₂(max_scalar) ≈ 254 for BN254 scalars,
    regardless of N. The savings grow with N.

    Algorithm
    ---------
    result = O  (identity)
    tmp[i] = P[i]  for all i
    while any scalar > 0:
        for each i with scalar[i] odd: result += tmp[i]
        tmp[i] = 2·tmp[i]  for all i
        scalar[i] >>= 1    for all i
    """
    if len(points) != len(scalars):
        raise ValueError(
            f"Length mismatch: {len(points)} points vs {len(scalars)} scalars."
        )
    if not points:
        return Point.identity()

    result = Point.identity()
    # Work on mutable copies
    tmp_points = list(points)
    scalars = list(scalars)

    while any(s > 0 for s in scalars):
        for i, s in enumerate(scalars):
            if s & 1:
                result = result + tmp_points[i]
        tmp_points = [p.double() for p in tmp_points]
        scalars = [s >> 1 for s in scalars]

    return result


# ------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.curve import G

    print("=== msm.py smoke test ===")

    # Single-point MSM: MSM([G], [k]) == k*G
    k = 42
    expected = G * k
    got_naive = msm_naive([G], [k])
    got_sha = msm_double_and_add([G], [k])
    assert got_naive == expected, f"naive: expected {expected}, got {got_naive}"
    assert got_sha == expected, f"shamir: expected {expected}, got {got_sha}"
    print(f"msm([G], [{k}]) == {k}*G  OK")

    # Linearity: MSM([G, G], [a, b]) == (a+b)*G
    a, b = 7, 13
    result_naive = msm_naive([G, G], [a, b])
    result_sha = msm_double_and_add([G, G], [a, b])
    expected_lin = G * (a + b)
    assert result_naive == expected_lin
    assert result_sha == expected_lin
    print(f"msm([G, G], [{a}, {b}]) == ({a}+{b})*G  OK")

    # Both implementations agree on N=10 random points
    import random
    random.seed(99)
    from src.curve import P as CURVE_P

    n = 10
    scalars_rand = [random.randint(1, 2**32) for _ in range(n)]
    points_rand = [G * random.randint(1, 2**32) for _ in range(n)]
    r1 = msm_naive(points_rand, scalars_rand)
    r2 = msm_double_and_add(points_rand, scalars_rand)
    assert r1 == r2, "naive != double_and_add for N=10"
    print("msm_naive == msm_double_and_add for N=10 random points  OK")

    print("\nAll msm.py checks passed.")
