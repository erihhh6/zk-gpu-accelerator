"""
src/msm.py — Multi-Scalar Multiplication on CPU.

Given vectors of elliptic curve points P₁…Pₙ and scalars s₁…sₙ, compute:

  result = Σ sᵢ · Pᵢ

This is the dominant cost (~65% of total prover time) in Groth16 and PLONK.

Two implementations are provided:
  - msm_naive          : straightforward, each sᵢ·Pᵢ computed independently
  - msm_double_and_add : Straus/Shamir simultaneous double-and-add — one shared
                         accumulator, so ~max_bits doublings instead of N · bits
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
    Straus/Shamir simultaneous double-and-add.

    All scalars share one accumulator, which walks their bits from the most
    significant down: at each bit position it is doubled once, then every
    point whose scalar has that bit set is added. Horner's rule makes this
    correct — after the last bit the accumulator holds Σ sᵢ·Pᵢ.

    Cost: max_bits doublings in total (independent of N) plus one addition
    per set bit, versus N · bits doublings for msm_naive. Additions are
    unchanged, so the saving is roughly the doubling half of the work.

    Algorithm
    ---------
    result = O  (identity)
    for bit from max_bits - 1 down to 0:
        result = 2·result
        for each i with bit `bit` of sᵢ set: result += Pᵢ
    """
    if len(points) != len(scalars):
        raise ValueError(
            f"Length mismatch: {len(points)} points vs {len(scalars)} scalars."
        )

    # Fold signs into the points (-k·P = k·(-P)) so the bit walk only ever sees
    # non-negative scalars; Python's >> on a negative int never reaches 0.
    terms = [
        (-point, -scalar) if scalar < 0 else (point, scalar)
        for point, scalar in zip(points, scalars)
    ]
    max_bits = max((scalar.bit_length() for _, scalar in terms), default=0)

    result = Point.identity()
    for bit in range(max_bits - 1, -1, -1):
        result = result.double()
        for point, scalar in terms:
            if (scalar >> bit) & 1:
                result = result + point
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

    n = 10
    scalars_rand = [random.randint(1, 2**32) for _ in range(n)]
    points_rand = [G * random.randint(1, 2**32) for _ in range(n)]
    r1 = msm_naive(points_rand, scalars_rand)
    r2 = msm_double_and_add(points_rand, scalars_rand)
    assert r1 == r2, "naive != double_and_add for N=10"
    print("msm_naive == msm_double_and_add for N=10 random points  OK")

    print("\nAll msm.py checks passed.")
