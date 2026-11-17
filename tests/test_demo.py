"""tests/test_demo.py — The Groth16 prover demo computes what it claims to, on both backends."""

import dataclasses
import importlib.util
import random
import sys
from pathlib import Path

import pytest

from zkgpu.field import FIELD_MODULUS as R
from zkgpu.msm import msm_double_and_add

_spec = importlib.util.spec_from_file_location(
    "prover_simulation", Path(__file__).resolve().parents[1] / "demo" / "prover_simulation.py"
)
assert _spec is not None and _spec.loader is not None
demo = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = demo  # dataclasses look their module up there
_spec.loader.exec_module(demo)


@pytest.fixture(scope="module")
def inputs():
    return demo.make_inputs(13, seed=5)  # 13 + 2 rows: a domain of 16 with one padding row


def horner(coeffs: list[int], x: int) -> int:
    acc = 0
    for c in reversed(coeffs):
        acc = (acc * x + c) % R
    return acc


def test_inputs_have_groth16_shapes(inputs):
    assert inputs.domain.size == 16
    assert len(inputs.witness) == 13 and inputs.witness[0] == 1
    assert len(inputs.points) == 15  # max(N, n - 1) points
    assert all(c == a * b % R for a, b, c in zip(inputs.a, inputs.b, inputs.c, strict=True))
    sizes = {name: len(s) for name, s in inputs.msm_scalars([0] * 16).items()}
    assert sizes == {"A": 13, "B": 13, "L": 11, "H": 15}


def test_cpu_quotient_satisfies_the_groth16_identity(inputs):
    """h(τ)·Z_H(τ) = A(τ)·B(τ) - C(τ) at a random τ: what the verifier's pairing check relies on."""
    d = inputs.domain
    h = demo.quotient_cpu(inputs)
    a, b, c = (d.ifft(v) for v in (inputs.a, inputs.b, inputs.c))
    tau = random.Random(9).randrange(R)
    lhs = horner(h, tau) * d.vanishing_eval(tau) % R
    assert lhs == (horner(a, tau) * horner(b, tau) - horner(c, tau)) % R
    assert h[-1] == 0


def test_unsatisfied_circuit_is_rejected(inputs):
    """If C ≠ A·B somewhere on H, Z_H does not divide A·B - C, and the degree check notices."""
    broken = dataclasses.replace(inputs, c=[(inputs.c[0] + 1) % R, *inputs.c[1:]])
    with pytest.raises(RuntimeError, match="not a polynomial"):
        demo.prove_cpu(broken)


def test_cpu_msms_match_straus(inputs):
    run = demo.prove_cpu(inputs)
    for name, scalars in inputs.msm_scalars(run.h).items():
        assert run.msm_results[name] == msm_double_and_add(inputs.points[: len(scalars)], scalars)


@pytest.mark.gpu
def test_gpu_reproduces_the_cpu(inputs):
    demo.check_agreement(demo.prove_cpu(inputs), demo.prove_gpu(inputs))


@pytest.mark.gpu
def test_gpu_agreement_check_catches_a_difference(inputs):
    cpu = demo.prove_cpu(inputs)
    wrong = dataclasses.replace(cpu, msm_results={**cpu.msm_results, "L": cpu.msm_results["A"]})
    with pytest.raises(RuntimeError, match="MSM L"):
        demo.check_agreement(wrong, demo.prove_gpu(inputs))


def test_main_prints_a_row_per_backend(capsys):
    demo.main(["--constraints", "6"])
    out = capsys.readouterr().out
    assert "6 (n = 2^3), CPU" in out
    assert ("GPU speedup" in out) == demo.GPU_AVAILABLE


def test_too_small_circuit_is_rejected():
    with pytest.raises(ValueError, match="at least"):
        demo.make_inputs(2)
