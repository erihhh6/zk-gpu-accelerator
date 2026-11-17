"""
zkgpu/msm.py — Multi-Scalar Multiplication on CPU.

Given vectors of elliptic curve points P₁…Pₙ and scalars s₁…sₙ, compute:

  result = Σ sᵢ · Pᵢ

It is usually the largest cost of a Groth16 or PLONK prover.

Three implementations are provided:
  - msm_naive          : straightforward, each sᵢ·Pᵢ computed independently
  - msm_double_and_add : Straus/Shamir simultaneous double-and-add — one shared
                         accumulator, so ~max_bits doublings instead of N · bits
  - msm_pippenger      : Pippenger's bucket method, about N + 2^(w+1) additions
                         per w-bit window; the fastest here from ~100 points.
                         zkgpu.msm_gpu runs the same algorithm on the GPU.
"""

from __future__ import annotations

import operator

from zkgpu.curve import (
    JACOBIAN_IDENTITY,
    Jacobian,
    Point,
    from_jacobian,
    jacobian_add,
    jacobian_add_affine,
    jacobian_double,
)
from zkgpu.field import FIELD_MODULUS as _GROUP_ORDER  # BN254's r: order of G1

# Reduced scalars are < r < 2^254
_SCALAR_BITS = _GROUP_ORDER.bit_length()

# 2^16 buckets per window is far past the useful range for either the CPU
# or the GPU bucket accumulation; larger windows only burn memory and time.
MAX_WINDOW_SIZE = 16

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


# ------------------------------------------------------------------
# Pippenger (bucket method)
# ------------------------------------------------------------------


def pippenger_window_size(n: int) -> int:
    """
    Window size w minimising Pippenger's estimated point additions for n points.

    Per window: n additions to fill the buckets plus about 2·2^w for the
    running sums; ceil(254 / w) windows. A fixed w is far from optimal at
    small or large n (at 100 points, w = 8 needs about 2.3x the additions of w = 5).
    """
    n = operator.index(n)
    return min(range(1, MAX_WINDOW_SIZE + 1), key=lambda w: -(-_SCALAR_BITS // w) * (n + 2 ** (w + 1)))


def _check_window_size(window_size: int) -> int:
    """window_size as an int, after checking it is in 1..MAX_WINDOW_SIZE."""
    window_size = operator.index(window_size)
    if not 1 <= window_size <= MAX_WINDOW_SIZE:
        raise ValueError(f"window_size must be in 1..{MAX_WINDOW_SIZE}, got {window_size}")
    return window_size


def _window_digits(scalars: list[int], w: int, num_windows: int) -> list[list[int]]:
    """digits[win][i] = bits [win·w, win·w + w) of scalars[i] (all scalars in [0, r))."""
    mask = (1 << w) - 1
    return [[(s >> (win * w)) & mask for s in scalars] for win in range(num_windows)]


def _window_sum(coords: list[tuple[int, int]], digits: list[int], bucket_count: int) -> Jacobian:
    """Σ digitᵢ·Pᵢ for one window, using 2^w buckets instead of N scalar multiplications."""
    # Buckets are Jacobian and inputs affine, so filling them uses mixed additions
    buckets = [JACOBIAN_IDENTITY] * bucket_count  # bucket 0 contributes nothing
    for (x, y), digit in zip(coords, digits, strict=True):
        if digit:
            buckets[digit] = jacobian_add_affine(buckets[digit], x, y)

    # Running-sum trick: Σ k·B_k = Σ_{j≥1} (B_j + B_{j+1} + ...), so two
    # additions per bucket replace a scalar multiplication per bucket.
    running = JACOBIAN_IDENTITY
    total = JACOBIAN_IDENTITY
    for k in range(bucket_count - 1, 0, -1):
        running = jacobian_add(running, buckets[k])
        total = jacobian_add(total, running)
    return total


def _combine_windows(window_sums: list[Jacobian], w: int) -> Jacobian:
    """
    Σ 2^(w·win) · window_sums[win], by Horner's rule from the most significant window.

    Shifting the running result by w bits between windows costs
    w·(windows - 1) ≈ 254 doublings in total, instead of shifting each window
    sum separately.
    """
    acc = window_sums[-1]
    for win in range(len(window_sums) - 2, -1, -1):
        for _ in range(w):
            acc = jacobian_double(acc)
        acc = jacobian_add(acc, window_sums[win])
    return acc


def _pippenger_affine(coords: list[tuple[int, int]], scalars: list[int], w: int) -> Point:
    """Pippenger on affine coordinates of valid points, with scalars already reduced mod r."""
    num_windows = -(-_SCALAR_BITS // w)  # ceil(254 / w)
    digits = _window_digits(scalars, w, num_windows)
    return from_jacobian(_combine_windows([_window_sum(coords, d, 1 << w) for d in digits], w))


def msm_pippenger(points: list[Point], scalars: list[int], *, window_size: int | None = None) -> Point:
    """
    Pippenger's bucket method on the CPU.

    Each w-bit window sorts the points into 2^w buckets by their digit, adds
    each bucket up, and weighs the buckets with two running sums; Horner's
    rule then combines the windows. That is about N + 2^(w+1) additions per
    window instead of the N · bits of msm_naive. zkgpu.msm_gpu.msm_pippenger_gpu
    runs the same algorithm on the GPU, so the two are directly comparable.

    Parameters
    ----------
    points      : the points Pᵢ (identity points are allowed and add nothing)
    scalars     : any integers; they are reduced mod r, which leaves sᵢ·Pᵢ unchanged
    window_size : bits per window, 1..16; by default pippenger_window_size(N)

    Raises
    ------
    ValueError : if the lengths differ or window_size is out of range
    """
    if len(points) != len(scalars):
        raise ValueError(f"Length mismatch: {len(points)} points vs {len(scalars)} scalars.")
    w = pippenger_window_size(len(points)) if window_size is None else _check_window_size(window_size)
    coords, reduced = [], []
    for point, scalar in zip(points, scalars, strict=True):
        if not point.is_identity():  # the identity adds nothing and has no affine coordinates
            coords.append(point.affine())
            reduced.append(scalar % _GROUP_ORDER)
    return _pippenger_affine(coords, reduced, w) if coords else Point.identity()
