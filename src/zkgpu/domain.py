"""
zkgpu/domain.py — Radix-2 evaluation domains over BN254's scalar field (CPU).

ZK provers (Groth16, PLONK) represent polynomials by their values on a
multiplicative subgroup H = {1, ω, ω², ..., ω^(n-1)} of F_r*, where n is a power
of two and ω a primitive n-th root of unity. Moving between coefficient form
and evaluation form is an NTT (zkgpu.ntt.ntt). Two more pieces complete the
toolbox a prover needs:

- The vanishing polynomial Z_H(X) = X^n - 1, which is zero exactly on H. A
  circuit is satisfied iff a certain polynomial combination is divisible by
  Z_H, and the prover has to compute that quotient.
- Evaluation on a coset g·H, for a g outside H. There Z_H(g·ω^i) = g^n - 1 is a
  non-zero constant, so dividing by Z_H becomes a multiplication by a single
  inverse. This is how Groth16 computes h(X) = (A(X)·B(X) - C(X)) / Z_H(X).

BN254's r - 1 is divisible by 2^28, so domains of up to 2^28 points exist.
Roots come from zkgpu.ntt.get_ntt_root, which derives them from the smallest
quadratic non-residue (5). That is the same ω snarkjs/circom use, so
evaluations computed here are interchangeable with theirs.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from dataclasses import dataclass, field

from zkgpu.field import FIELD_MODULUS
from zkgpu.ntt import get_ntt_root, ntt, quadratic_non_residue


def _scale_by_powers(values: Sequence[int], g: int, mod: int) -> list[int]:
    """[v_0·g^0, v_1·g^1, v_2·g^2, ...] mod *mod*."""
    out, power = [], 1
    for v in values:
        out.append(v * power % mod)
        power = power * g % mod
    return out


@dataclass(frozen=True)
class Radix2Domain:
    """
    The subgroup of n-th roots of unity of F_modulus, for n a power of two.

    Immutable; omega (the generator ω of H) and coset_offset (the default g
    for coset evaluation) are derived on construction. coset_offset is None
    only when the domain is all of F_p* (p - 1 a power of two, such as 17 at
    size 16): no coset outside it exists, so the coset_* methods raise.

    Raises
    ------
    ValueError : if size is not a power of two, the modulus is not an odd
                 prime, or the modulus has no subgroup of this size
    """

    size: int
    modulus: int = FIELD_MODULUS
    omega: int = field(init=False)
    coset_offset: int | None = field(init=False)

    def __post_init__(self) -> None:
        size = operator.index(self.size)
        if size < 1 or size & (size - 1):
            raise ValueError(f"domain size must be a power of 2, got {size}")
        object.__setattr__(self, "size", size)
        # Validates the modulus (odd prime) and that size divides modulus - 1
        object.__setattr__(self, "omega", get_ntt_root(size, self.modulus))

        # Default coset shift: the smallest quadratic non-residue g (5 for BN254).
        # 2^s divides ord(g), where 2^s is the largest power of 2 dividing p - 1,
        # so g^n = 1 is impossible for n < 2^s. At the maximum size n = 2^s it
        # happens exactly when ord(g) = 2^s; then g lies in H, and the next
        # candidate outside H is used instead (e.g. 4 for p = 41 at size 8).
        # If every element lies in H, H is all of F_p* and no coset exists.
        # (For BN254, 5 is outside even the 2^28 subgroup.)
        start = quadratic_non_residue(self.modulus)
        offset = next(
            (g for g in range(start, self.modulus) if pow(g, size, self.modulus) != 1),
            None,
        )
        object.__setattr__(self, "coset_offset", offset)

    @classmethod
    def at_least(cls, n: int, modulus: int = FIELD_MODULUS) -> Radix2Domain:
        """Smallest domain with at least n points, e.g. one point per constraint."""
        if n < 1:
            raise ValueError(f"a domain needs at least one point, got {n}")
        return cls(1 << (n - 1).bit_length(), modulus)

    # ------------------------------------------------------------------
    # Elements and the vanishing polynomial
    # ------------------------------------------------------------------

    def element(self, i: int) -> int:
        """ω^i."""
        return pow(self.omega, i, self.modulus)

    def elements(self) -> list[int]:
        """[ω^0, ω^1, ..., ω^(n-1)], the points evaluations refer to."""
        return _scale_by_powers([1] * self.size, self.omega, self.modulus)

    def vanishing_eval(self, x: int) -> int:
        """Z_H(x) = x^n - 1, which is zero exactly on the domain's elements."""
        return (pow(x, self.size, self.modulus) - 1) % self.modulus

    # ------------------------------------------------------------------
    # Coefficient form <-> evaluation form
    # ------------------------------------------------------------------

    def fft(self, coeffs: Sequence[int]) -> list[int]:
        """[f(ω^0), ..., f(ω^(n-1))] for f with these coefficients (constant term first)."""
        return ntt(self._padded(coeffs), self.modulus)

    def ifft(self, evals: Sequence[int]) -> list[int]:
        """Coefficients of the unique f of degree < n with f(ω^i) = evals[i]."""
        if len(evals) != self.size:
            raise ValueError(f"expected {self.size} evaluations, got {len(evals)}")
        return ntt(list(evals), self.modulus, invert=True)

    def coset_fft(self, coeffs: Sequence[int], offset: int | None = None) -> list[int]:
        """
        [f(g·ω^0), ..., f(g·ω^(n-1))] on the coset g·H (g = coset_offset by default).

        f(g·x) = Σ (c_j·g^j)·x^j, so scaling coefficient j by g^j turns coset
        evaluation into an ordinary FFT.
        """
        g = self._offset(offset)
        return self.fft(_scale_by_powers(self._padded(coeffs), g, self.modulus))

    def coset_ifft(self, evals: Sequence[int], offset: int | None = None) -> list[int]:
        """Inverse of coset_fft: interpolate on H, then undo the scaling by g^-j."""
        g = self._offset(offset)
        return _scale_by_powers(self.ifft(evals), pow(g, -1, self.modulus), self.modulus)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _padded(self, coeffs: Sequence[int]) -> list[int]:
        # More than n coefficients would silently alias (X^n ≡ 1 on H), so reject them
        if len(coeffs) > self.size:
            raise ValueError(
                f"{len(coeffs)} coefficients do not fit a domain of size {self.size}; "
                f"use Radix2Domain.at_least({len(coeffs)})"
            )
        return [*coeffs, *[0] * (self.size - len(coeffs))]

    def _offset(self, offset: int | None) -> int:
        """The explicit *offset* (validated), or the default coset shift."""
        if offset is not None:
            return self._checked_offset(offset)
        if self.coset_offset is None:
            raise ValueError(
                f"the domain of size {self.size} is all of F_{self.modulus}*, so no coset outside it exists"
            )
        return self.coset_offset

    def _checked_offset(self, offset: int) -> int:
        g = operator.index(offset) % self.modulus
        if g == 0 or pow(g, self.size, self.modulus) == 1:
            # g = 0 collapses the coset; g in H makes the coset H itself, where
            # Z_H vanishes and division by it is impossible
            raise ValueError(f"coset offset {offset} must be non-zero and outside the domain")
        return g
