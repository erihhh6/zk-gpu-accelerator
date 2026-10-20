"""
src/ntt.py — Number Theoretic Transform (CPU, Cooley-Tukey iterative).

Theory
------
NTT is the analog of FFT over a finite field F_p:

  NTT(a)[k] = Σᵢ aᵢ · ωⁱᵏ  mod p

where ω is a primitive N-th root of unity in F_p (i.e. ω^N ≡ 1 mod p and
ω^k ≢ 1 mod p for 0 < k < N).

Instead of complex roots e^(2πi/N) used in FFT, we use the *number-theoretic*
root ω = g^((p-1)/N) mod p, where g is a quadratic non-residue of F_p (see
quadratic_non_residue for why that suffices for power-of-2 sizes).

NTT-friendly primes
-------------------
A prime p supports power-of-2 sizes N up to the largest power of 2 dividing
p - 1 (its "2-adicity"). The default MOD = 998 244 353 = 119 · 2^23 + 1 (a.k.a.
"the competitive-programming NTT prime") supports N ≤ 2^23.

BN254's scalar field r is NTT-friendly too: r - 1 is divisible by 2^28, so it
supports N ≤ 2^28, which is why Groth16/PLONK over BN254 use it. This module
works over r directly (Python ints); src.ntt_gpu cannot, because its elements
are int64.

Applications in ZK
------------------
In PLONK, witness polynomials are evaluated over a multiplicative subgroup of
size N. Polynomial multiplication (needed for quotient polynomial computation)
is done in O(N log N) via NTT instead of O(N²) naive convolution.
"""

from __future__ import annotations

import functools

MOD: int = 998_244_353  # NTT-friendly prime: 119 * 2^23 + 1


# ------------------------------------------------------------------
# Root computation helpers
# ------------------------------------------------------------------

# Miller-Rabin with these bases is deterministic for every n < 3.3 · 10^24
# (Sorenson & Webster, 2015). Above that it is a strong probable-prime test:
# a composite passing all 13 bases is vanishingly unlikely unless deliberately
# constructed, and the moduli used here are published primes.
_MILLER_RABIN_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)


def _is_probable_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in _MILLER_RABIN_BASES:
        if n % p == 0:
            return n == p
    s = ((n - 1) & -(n - 1)).bit_length() - 1  # n - 1 = d · 2^s with d odd
    d = (n - 1) >> s
    for a in _MILLER_RABIN_BASES:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False  # a is a witness that n is composite
    return True


@functools.lru_cache(maxsize=None)
def quadratic_non_residue(mod: int) -> int:
    """
    Smallest quadratic non-residue g of F_mod, the generator of every NTT root.

    Why a non-residue suffices: for n = 2^k dividing mod - 1, the root
    ω = g^((mod-1)/n) satisfies ω^(n/2) = g^((mod-1)/2) = -1 by Euler's
    criterion, so ω has order exactly n, i.e. it is a primitive n-th root of
    unity. A full primitive root of F_mod would also work, but finding one
    needs the factorisation of mod - 1, which is infeasible for 254-bit moduli.
    Testing a candidate costs one exponentiation and half of all residues
    qualify, so the search ends almost immediately (3 for 998244353, 5 for
    BN254's r). Cached, so callers can ask for it on every transform.

    Raises ValueError unless *mod* is an odd prime.
    """
    if mod < 3 or not _is_probable_prime(mod):
        raise ValueError(f"NTT modulus must be an odd prime, got {mod}")
    for g in range(2, mod):
        if pow(g, (mod - 1) // 2, mod) == mod - 1:
            return g
    raise AssertionError("unreachable: every odd prime has a quadratic non-residue")


def get_ntt_root(n: int, mod: int, non_residue: int | None = None) -> int:
    """
    Return a primitive n-th root of unity ω in F_mod, for n a power of 2.

    ω = g^((mod-1)/n), where g is *non_residue* or, by default, the smallest
    quadratic non-residue. Deriving every size from the same g keeps roots
    consistent across sizes: get_ntt_root(n/2) == get_ntt_root(n)^2.

    Raises ValueError if n is not a power of 2, if mod is not an odd prime, if
    n does not divide mod - 1, or if *non_residue* gives no primitive root.
    """
    if n < 1 or n & (n - 1):
        raise ValueError(f"NTT size must be a power of 2, got {n}")
    default_g = quadratic_non_residue(mod)  # also validates mod
    g = default_g if non_residue is None else non_residue
    if (mod - 1) % n != 0:
        raise ValueError(
            f"NTT of size {n} not supported for mod={mod}: "
            f"(mod-1) = {mod-1} is not divisible by n."
        )
    omega = pow(g, (mod - 1) // n, mod)
    # Primitive iff ω^(n/2) = -1. Always true for a non-residue; this catches an
    # explicit non_residue that is really a residue (as 3 is mod BN254's r).
    if n > 1 and pow(omega, n // 2, mod) != mod - 1:
        raise ValueError(
            f"{g} gives no primitive {n}-th root of unity mod {mod}; "
            "it must be a quadratic non-residue"
        )
    return omega


# ------------------------------------------------------------------
# Bit-reversal permutation
# ------------------------------------------------------------------


def _bit_reverse_copy(a: list[int]) -> list[int]:
    """Return a copy of *a* with indices bit-reversed (in-place on the copy)."""
    n = len(a)
    bits = n.bit_length() - 1  # log2(n)
    result = a[:]
    for i in range(n):
        rev = int(f"{i:0{bits}b}"[::-1], 2)
        if rev > i:
            result[i], result[rev] = result[rev], result[i]
    return result


# ------------------------------------------------------------------
# Core NTT
# ------------------------------------------------------------------


def ntt(a: list[int], mod: int = MOD, invert: bool = False) -> list[int]:
    """
    Cooley-Tukey iterative NTT.

    Parameters
    ----------
    a      : input coefficient vector; length must be a power of 2
    mod    : odd prime with len(a) dividing mod - 1 (default 998244353)
    invert : if True, compute the inverse NTT (INTT)

    Returns
    -------
    list[int] : transformed vector (same length as input), all values in [0, mod)

    Raises
    ------
    ValueError : if the length is not a power of 2, mod is not an odd prime,
                 or the length does not divide mod - 1
    """
    n = len(a)
    if n == 0:
        return []
    if n & (n - 1):
        raise ValueError(f"NTT length must be a power of 2, got {n}")

    omega = get_ntt_root(n, mod)  # validates mod and n before doing any work
    if invert:
        omega = pow(omega, mod - 2, mod)
    if n == 1:
        return [a[0] % mod]

    a = _bit_reverse_copy(a)

    length = 2
    while length <= n:
        # ω_length = ω_n^(n/length): exactly the root get_ntt_root(length) gives
        w_len = pow(omega, n // length, mod)

        half = length >> 1
        for i in range(0, n, length):
            w = 1
            for j in range(half):
                u = a[i + j]
                v = (a[i + j + half] * w) % mod
                a[i + j] = (u + v) % mod
                a[i + j + half] = (u - v) % mod
                w = (w * w_len) % mod

        length <<= 1

    if invert:
        n_inv = pow(n, mod - 2, mod)
        a = [(x * n_inv) % mod for x in a]

    return a


# ------------------------------------------------------------------
# Polynomial multiplication via NTT
# ------------------------------------------------------------------


def polynomial_mul(p1: list[int], p2: list[int], mod: int = MOD) -> list[int]:
    """
    Multiply two polynomials via NTT in O(N log N) instead of O(N²).

    The result has degree deg(p1) + deg(p2), so we pad both inputs to
    the next power of 2 that is at least len(p1) + len(p2) - 1.
    """
    result_len = len(p1) + len(p2) - 1
    n = 1
    while n < result_len:
        n <<= 1

    a = p1 + [0] * (n - len(p1))
    b = p2 + [0] * (n - len(p2))

    fa = ntt(a, mod)
    fb = ntt(b, mod)
    fc = [(x * y) % mod for x, y in zip(fa, fb)]
    result = ntt(fc, mod, invert=True)

    # Trim trailing zeros
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result


# ------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------

if __name__ == "__main__":
    import random

    print("=== ntt.py smoke test ===")

    # Verify INTT(NTT(a)) == a for a random vector
    random.seed(42)
    N = 1024
    a = [random.randint(0, MOD - 1) for _ in range(N)]
    transformed = ntt(a, MOD)
    recovered = ntt(transformed, MOD, invert=True)
    assert a == recovered, "INTT(NTT(a)) != a"
    print(f"INTT(NTT(a)) == a for N={N}  OK")

    # Polynomial multiplication: (1 + x)(1 + x) = 1 + 2x + x^2
    p1 = [1, 1]
    p2 = [1, 1]
    prod = polynomial_mul(p1, p2, MOD)
    assert prod == [1, 2, 1], f"Expected [1, 2, 1], got {prod}"
    print(f"(1+x)*(1+x) = {prod}  OK")

    # Verify poly mul matches naive convolution
    def naive_poly_mul(p: list[int], q: list[int], m: int) -> list[int]:
        result = [0] * (len(p) + len(q) - 1)
        for i, a in enumerate(p):
            for j, b in enumerate(q):
                result[i + j] = (result[i + j] + a * b) % m
        return result

    p3 = [random.randint(0, MOD - 1) for _ in range(16)]
    p4 = [random.randint(0, MOD - 1) for _ in range(16)]
    assert polynomial_mul(p3, p4, MOD) == naive_poly_mul(p3, p4, MOD)
    print("NTT poly_mul == naive poly_mul (degree-15 polynomials)  OK")

    print("\nAll ntt.py checks passed.")
