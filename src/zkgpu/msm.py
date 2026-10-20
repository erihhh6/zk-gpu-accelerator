"""
zkgpu/msm.py — Multi-Scalar Multiplication on CPU.

Given vectors of elliptic curve points P₁…Pₙ and scalars s₁…sₙ, compute:

  result = Σ sᵢ · Pᵢ

This is the dominant cost (~65% of total prover time) in Groth16 and PLONK.

Two implementations are provided:
  - msm_naive          : straightforward, each sᵢ·Pᵢ computed independently
  - msm_double_and_add : Straus/Shamir simultaneous double-and-add — one shared
                         accumulator, so ~max_bits doublings instead of N · bits
"""

from __future__ import annotations

from zkgpu.curve import JACOBIAN_IDENTITY, Point, from_jacobian, jacobian_add_affine, jacobian_double

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
        raise ValueError(f"Length mismatch: {len(points)} points vs {len(scalars)} scalars.")
    result = Point.identity()
    for point, scalar in zip(points, scalars, strict=True):
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

    The accumulator lives in Jacobian coordinates, so no step needs a
    modular inverse; the inputs stay affine, so each addition is the cheaper
    mixed addition. One inverse converts the result back at the end.

    Algorithm
    ---------
    result = O  (identity)
    for bit from max_bits - 1 down to 0:
        result = 2·result
        for each i with bit `bit` of sᵢ set: result += Pᵢ
    """
    if len(points) != len(scalars):
        raise ValueError(f"Length mismatch: {len(points)} points vs {len(scalars)} scalars.")

    # Fold signs into the points (-k·P = k·(-P)) so the bit walk only ever sees
    # non-negative scalars; Python's >> on a negative int never reaches 0.
    # Identity points contribute nothing and have no affine coordinates.
    terms = [
        ((-point if scalar < 0 else point).affine(), abs(scalar))
        for point, scalar in zip(points, scalars, strict=True)
        if not point.is_identity()
    ]
    max_bits = max((scalar.bit_length() for _, scalar in terms), default=0)

    acc = JACOBIAN_IDENTITY
    for bit in range(max_bits - 1, -1, -1):
        acc = jacobian_double(acc)
        for (x, y), scalar in terms:
            if (scalar >> bit) & 1:
                acc = jacobian_add_affine(acc, x, y)
    return from_jacobian(acc)
