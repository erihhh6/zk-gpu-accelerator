"""
src/ntt.py — Number Theoretic Transform (CPU, Cooley-Tukey iterative).

Theory
------
NTT is the analog of FFT over a finite field F_p:

  NTT(a)[k] = Σᵢ aᵢ · ωⁱᵏ  mod p

where ω is a primitive N-th root of unity in F_p (i.e. ω^N ≡ 1 mod p and
ω^k ≢ 1 mod p for 0 < k < N).

Instead of complex roots e^(2πi/N) used in FFT, we use the *number-theoretic*
root ω = g^((p-1)/N) mod p, where g is a primitive root of F_p.

NTT-friendly prime
------------------
We use MOD = 998 244 353 = 119 · 2^23 + 1 (a.k.a. "the competitive-programming
NTT prime"). Its factorisation p - 1 = 2^23 · 7 · 17 means the field supports
NTT for any power-of-2 length N ≤ 2^23.

Note: BN254's scalar field modulus r is NOT NTT-friendly for power-of-2 sizes
used in practice, so we use MOD = 998244353 throughout this module.

Applications in ZK
------------------
In PLONK, witness polynomials are evaluated over a multiplicative subgroup of
size N. Polynomial multiplication (needed for quotient polynomial computation)
is done in O(N log N) via NTT instead of O(N²) naive convolution.
"""

from __future__ import annotations

MOD: int = 998_244_353  # NTT-friendly prime: 119 * 2^23 + 1
PRIMITIVE_ROOT: int = 3  # Primitive root of F_{998244353}


# ------------------------------------------------------------------
# Root computation helpers
# ------------------------------------------------------------------


def find_primitive_root(mod: int) -> int:
    """
    Find a primitive root (generator) of the multiplicative group F_mod*.

    A primitive root g satisfies ord(g) = mod - 1, i.e. g generates the
    entire group. We test small candidates against the prime factors of
    mod - 1.
    """
    if mod == 2:
        return 1

    phi = mod - 1  # mod is assumed prime, so φ(mod) = mod - 1

    # Factorise phi (trial division; fine for moderate-sized primes)
    factors: list[int] = []
    n = phi
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.append(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.append(n)

    # Check candidates
    for g in range(2, mod):
        if all(pow(g, phi // f, mod) != 1 for f in factors):
            return g

    raise ValueError(f"Could not find primitive root for mod={mod}")


def get_ntt_root(n: int, mod: int, primitive_root: int = PRIMITIVE_ROOT) -> int:
    """
    Return ω = primitive_root^((mod-1)/n) mod mod — the principal n-th root of unity.

    Preconditions:
      - n must be a power of 2
      - (mod - 1) must be divisible by n
    """
    if (mod - 1) % n != 0:
        raise ValueError(
            f"NTT of size {n} not supported for mod={mod}: "
            f"(mod-1) = {mod-1} is not divisible by n."
        )
    return pow(primitive_root, (mod - 1) // n, mod)


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
    mod    : field modulus (default 998244353)
    invert : if True, compute the inverse NTT (INTT)

    Returns
    -------
    list[int] : transformed vector (same length as input)
    """
    n = len(a)
    if n == 0:
        return []
    if n & (n - 1):
        raise ValueError(f"NTT length must be a power of 2, got {n}")
    if n == 1:
        return [a[0] % mod]

    a = _bit_reverse_copy(a)

    # Build up the transform butterfly-by-butterfly
    length = 2
    while length <= n:
        # ω of order *length* in F_mod
        w_len = get_ntt_root(length, mod)
        if invert:
            w_len = pow(w_len, mod - 2, mod)  # w_len^(-1) mod mod

        for i in range(0, n, length):
            w = 1
            half = length // 2
            for j in range(half):
                u = a[i + j]
                v = (a[i + j + half] * w) % mod
                a[i + j] = (u + v) % mod
                a[i + j + half] = (u - v) % mod
            w = (w * w_len) % mod  # not used after inner loop, harmless

        length <<= 1

    if invert:
        n_inv = pow(n, mod - 2, mod)
        a = [(x * n_inv) % mod for x in a]

    return a


# Fix: the butterfly's twiddle factor update was inside the wrong scope.
# Re-implement cleanly:

def ntt(a: list[int], mod: int = MOD, invert: bool = False) -> list[int]:  # noqa: F811
    """
    Cooley-Tukey iterative NTT.

    Parameters
    ----------
    a      : input coefficient vector; length must be a power of 2
    mod    : field modulus (default 998244353)
    invert : if True, compute the inverse NTT (INTT)

    Returns
    -------
    list[int] : transformed vector (same length as input), all values in [0, mod)
    """
    n = len(a)
    if n == 0:
        return []
    if n & (n - 1):
        raise ValueError(f"NTT length must be a power of 2, got {n}")
    if n == 1:
        return [a[0] % mod]

    a = _bit_reverse_copy(a)

    length = 2
    while length <= n:
        w_len = get_ntt_root(length, mod)
        if invert:
            w_len = pow(w_len, mod - 2, mod)

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
