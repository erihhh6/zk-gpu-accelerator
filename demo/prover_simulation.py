"""
demo/prover_simulation.py — Where a Groth16 prover's time goes, on this package's CPU and GPU code.

What a Groth16 prover computes, for an R1CS with m constraints and a witness
z of N values (Groth 2016, as arkworks and snarkjs implement it):

1. The quotient h(X) = (A(X)·B(X) - C(X)) / Z_H(X), on a domain H of
   n >= m + public inputs + 1 points: 3 inverse NTTs (the values of A·z,
   B·z and C·z on H become coefficients), 3 coset NTTs (values on g·H,
   where Z_H is the non-zero constant g^n - 1), a pointwise
   (A·B - C) / (g^n - 1), and 1 inverse coset NTT, giving h's n - 1
   coefficients. Seven NTTs of size n.
2. Four MSMs in G1: Σ zᵢ·[Aᵢ(τ)], Σ zᵢ·[Bᵢ(τ)] (N points each), Σ zᵢ·[Lᵢ(τ)]
   over the private witness values, and Σ hᵢ·[τⁱ·Z_H(τ)/δ] (n - 1 points).
3. One MSM in G2, Σ zᵢ·[Bᵢ(τ)]₂. Not modelled: this package has no G2
   arithmetic.

This script runs (1) and the G1 MSMs of (2) on inputs of the right shapes,
on the CPU (Radix2Domain, msm_pippenger) and on the GPU (BN254_FR.ntt,
msm_pippenger_gpu: the same algorithms), checks that they agree, and
reports the time of each step. Also not modelled: witness generation, the
sparse products A·z, B·z and C·z (cheap next to the rest), and the few
scalar multiplications that assemble and blind the proof. The totals are
therefore lower bounds for a real prover.

Stand-ins for a real circuit:
- Constraints: random values of A·z and B·z on H, and C·z = (A·z)(B·z), so
  the circuit is satisfied. Then A·B - C vanishes on H and h has degree at
  most n - 2; the script checks that h's top coefficient is zero.
- Witness: N = m uniformly random field elements. Circuits often have about
  as many variables as constraints. Real witness values are often bits or
  small numbers, which makes MSMs cheaper; random ones are the costly case.
- Proving key: one pseudo-random progression of points P₀ + i·S, shared by
  the four queries. An MSM costs the same for any valid points.

Usage (with the package installed):
    python demo/prover_simulation.py                        # 1,000 / 10,000 / 100,000 constraints
    python demo/prover_simulation.py --constraints 300000 --cpu-max 0   # GPU only
"""

from __future__ import annotations

import argparse
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from itertools import accumulate
from typing import Any, TypeVar

from zkgpu.curve import G, Point
from zkgpu.domain import Radix2Domain
from zkgpu.field import FIELD_MODULUS as R
from zkgpu.gpu_backend import GPU_AVAILABLE, GPU_UNAVAILABLE_REASON, cp
from zkgpu.msm import msm_pippenger

if GPU_AVAILABLE:
    from zkgpu import msm_gpu
    from zkgpu.field256_gpu import BN254_FR as Fr
    from zkgpu.msm_gpu import msm_pippenger_gpu

NUM_PUBLIC = 1  # public inputs, besides the constant 1 that every witness starts with
MSM_NAMES = ("A", "B", "L", "H")
T = TypeVar("T")


# ------------------------------------------------------------------
# Inputs
# ------------------------------------------------------------------


@dataclass(frozen=True)
class ProofInputs:
    """Everything one proof needs, for a circuit of a given size (see the module docstring)."""

    constraints: int
    domain: Radix2Domain
    a: list[int]  # A·z on H (zero on padding rows), likewise b and c
    b: list[int]
    c: list[int]
    witness: list[int]  # z: N = constraints values, z[0] = 1
    points: list[Point]  # proving-key stand-ins: max(N, n - 1) of them

    def msm_scalars(self, h: list[int]) -> dict[str, list[int]]:
        """The scalars of the four G1 MSMs, given the quotient's coefficients h."""
        private = self.witness[NUM_PUBLIC + 1 :]
        return {"A": self.witness, "B": self.witness, "L": private, "H": h[: self.domain.size - 1]}


def make_inputs(constraints: int, seed: int = 0) -> ProofInputs:
    if constraints < NUM_PUBLIC + 2:
        raise ValueError(f"need at least {NUM_PUBLIC + 2} constraints, got {constraints}")
    rng = random.Random(seed)
    rows = constraints + NUM_PUBLIC + 1  # arkworks adds a row per public input and one for z[0]
    domain = Radix2Domain.at_least(rows)
    padding = [0] * (domain.size - rows)
    a = [rng.randrange(R) for _ in range(rows)] + padding
    b = [rng.randrange(R) for _ in range(rows)] + padding
    c = [x * y % R for x, y in zip(a, b, strict=True)]
    witness = [1] + [rng.randrange(R) for _ in range(constraints - 1)]

    step = G * rng.randrange(1, R)
    points = [G * rng.randrange(1, R)]
    for _ in range(max(constraints, domain.size - 1) - 1):
        points.append(points[-1] + step)
    return ProofInputs(constraints, domain, a, b, c, witness, points)


# ------------------------------------------------------------------
# The two phases, on the CPU and on the GPU
# ------------------------------------------------------------------


def quotient_cpu(inputs: ProofInputs) -> list[int]:
    """h's n coefficients via Radix2Domain: 3 inverse NTTs, 3 coset NTTs, 1 inverse coset NTT."""
    d = inputs.domain
    assert d.coset_offset is not None  # BN254's domains always have a coset
    z_inv = pow(d.vanishing_eval(d.coset_offset), -1, R)
    a, b, c = (d.coset_fft(d.ifft(v)) for v in (inputs.a, inputs.b, inputs.c))
    return d.coset_ifft([(x * y - z) * z_inv % R for x, y, z in zip(a, b, c, strict=True)])


class GpuDomain:
    """
    Per-domain tables a GPU prover builds once per circuit, not per proof (setup, untimed).

    The powers g^j and g^-j turn a coset NTT into a plain one (f(g·x) has
    coefficients c_j·g^j), and the twiddle tables are built by one NTT of each
    direction. Radix2Domain computes the same powers on every call instead.
    """

    def __init__(self, domain: Radix2Domain) -> None:
        g, n = domain.coset_offset, domain.size
        assert g is not None
        g_inv = pow(g, -1, R)
        self.coset = Fr.from_ints(accumulate([1] * n, lambda acc, _: acc * g % R))
        self.coset_inv = Fr.from_ints(accumulate([1] * n, lambda acc, _: acc * g_inv % R))
        self.z_inv = Fr.from_ints([pow(domain.vanishing_eval(g), -1, R)])
        Fr.ntt(Fr.ntt(self.coset), invert=True)  # build and cache both twiddle tables


def quotient_gpu(inputs: ProofInputs, tables: GpuDomain) -> list[int]:
    """The same seven NTTs with BN254_FR on the GPU, from and back to Python ints."""

    def to_coset(evals: Any) -> Any:
        return Fr.ntt(Fr.mul(Fr.ntt(evals, invert=True), tables.coset))

    a, b, c = (to_coset(Fr.from_ints(v)) for v in (inputs.a, inputs.b, inputs.c))
    h_on_coset = Fr.mul(Fr.sub(Fr.mul(a, b), c), tables.z_inv)
    return Fr.to_ints(Fr.mul(Fr.ntt(h_on_coset, invert=True), tables.coset_inv))


# ------------------------------------------------------------------
# One proof's worth of work, timed
# ------------------------------------------------------------------


@dataclass(frozen=True)
class Run:
    """One backend's timings (ms) and results for one circuit."""

    quotient_ms: float
    msm_ms: dict[str, float]
    h: list[int]
    msm_results: dict[str, Point]

    @property
    def msm_total_ms(self) -> float:
        return sum(self.msm_ms.values())

    @property
    def total_ms(self) -> float:
        return self.quotient_ms + self.msm_total_ms


def _timed(fn: Callable[[], T]) -> tuple[float, T]:
    t0 = time.perf_counter()
    result = fn()  # every step returns host data (ints or a Point), so the GPU is done by now
    return (time.perf_counter() - t0) * 1e3, result


def _check_quotient(h: list[int], n: int, backend: str) -> None:
    # A satisfied circuit makes A·B - C divisible by Z_H, so h has degree <= n - 2.
    # A wrong NTT, coset or division almost surely leaves a non-zero top coefficient.
    if len(h) != n or h[-1] != 0:
        raise RuntimeError(f"{backend}: the quotient is not a polynomial of degree <= n - 2")


def prove_cpu(inputs: ProofInputs) -> Run:
    quotient_ms, h = _timed(lambda: quotient_cpu(inputs))
    _check_quotient(h, inputs.domain.size, "CPU")
    msm_ms, results = {}, {}
    for name, scalars in inputs.msm_scalars(h).items():
        points = inputs.points[: len(scalars)]
        msm_ms[name], results[name] = _timed(lambda p=points, s=scalars: msm_pippenger(p, s))
    return Run(quotient_ms, msm_ms, h, results)


def prove_gpu(inputs: ProofInputs) -> Run:
    """
    The GPU pipeline after one untimed warm-up proof, as a prover that runs repeatedly sees it.

    The warm-up compiles nothing (main compiles the kernels first), but it
    lets CuPy's memory pool reach this size, which the first call pays for.
    Before the timed proof, the GPU is kept busy for a moment: after idling
    through the CPU proof, a laptop GPU sits in a low power state that a few
    short kernels do not lift it out of.
    """
    tables = GpuDomain(inputs.domain)
    xs, ys = [p.x for p in inputs.points], [p.y for p in inputs.points]

    def proof() -> Run:
        quotient_ms, h = _timed(lambda: quotient_gpu(inputs, tables))
        _check_quotient(h, inputs.domain.size, "GPU")
        msm_ms, results = {}, {}
        for name, scalars in inputs.msm_scalars(h).items():
            k = len(scalars)
            msm_ms[name], results[name] = _timed(lambda k=k, s=scalars: msm_pippenger_gpu(xs[:k], ys[:k], s))
        return Run(quotient_ms, msm_ms, h, results)

    proof()
    _spin_up_gpu()
    return proof()


def _spin_up_gpu(seconds: float = 0.3) -> None:
    """Continuous memory-bound GPU work, as benchmarks/_harness.py does before every GPU timing."""
    x = cp.zeros(2**23, dtype=cp.uint64)  # 64 MiB
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        x += 1
        cp.cuda.Device().synchronize()


def check_agreement(cpu: Run, gpu: Run) -> None:
    """The GPU must reproduce the CPU's quotient and all four MSM results exactly."""
    if cpu.h != gpu.h:
        raise RuntimeError("the GPU quotient differs from the CPU quotient")
    for name in MSM_NAMES:
        if cpu.msm_results[name] != gpu.msm_results[name]:
            raise RuntimeError(f"MSM {name}: the GPU result differs from the CPU result")


# ------------------------------------------------------------------
# Report
# ------------------------------------------------------------------


def _row(label: str, run: Run) -> str:
    msms = "".join(f"{run.msm_ms[name]:>10.1f}" for name in MSM_NAMES)
    share = run.msm_total_ms / run.total_ms
    return f"{label:>21}  {run.quotient_ms:>11.1f}{msms}  {run.total_ms:>10.1f}  {share:>10.0%}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("--constraints", type=int, nargs="+", default=[1_000, 10_000, 100_000])
    parser.add_argument("--cpu-max", type=int, default=100_000, help="no CPU run above this size")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    print("Groth16 prover: the quotient polynomial (7 NTTs) and the four G1 MSMs, on BN254")
    if GPU_AVAILABLE:
        print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
        Fr.compile()  # compile and self-test the kernels before anything is timed
        msm_gpu.compile_kernels()
    else:
        print(f"GPU: not available ({GPU_UNAVAILABLE_REASON}); CPU only")
    print(
        "Times in ms. Not included: witness generation, the products Az/Bz/Cz, the G2 MSM, proof assembly.\n"
    )
    print(
        f"{'constraints, backend':>21}  {'quotient':>11}"
        + "".join(f"{'MSM ' + n:>10}" for n in MSM_NAMES)
        + f"  {'total':>10}  {'MSM share':>10}"
    )

    for m in args.constraints:
        inputs = make_inputs(m, args.seed)
        label = f"{m:,} (n = 2^{inputs.domain.size.bit_length() - 1})"
        cpu = prove_cpu(inputs) if m <= args.cpu_max else None
        gpu = prove_gpu(inputs) if GPU_AVAILABLE else None
        if cpu:
            print(_row(f"{label}, CPU", cpu))
        if gpu:
            print(_row("GPU" if cpu else f"{label}, GPU", gpu))
        if cpu and gpu:
            check_agreement(cpu, gpu)
            print(
                f"{'GPU speedup':>21}  {cpu.quotient_ms / gpu.quotient_ms:>10.1f}x"
                + "".join(f"{cpu.msm_ms[n] / gpu.msm_ms[n]:>9.1f}x" for n in MSM_NAMES)
                + f"  {cpu.total_ms / gpu.total_ms:>9.1f}x   (results identical)"
            )
        print(flush=True)


if __name__ == "__main__":
    main()
