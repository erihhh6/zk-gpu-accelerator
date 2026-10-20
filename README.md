# ZK GPU Accelerator

[![CI](https://github.com/erihhh6/zk-gpu-accelerator/actions/workflows/ci.yml/badge.svg)](https://github.com/erihhh6/zk-gpu-accelerator/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)


> The building blocks of a zero-knowledge prover (BN254 field and curve arithmetic, NTT, MSM) in Python, with CUDA kernels (via CuPy) for 64-bit and 254-bit modular arithmetic and the NTT, and benchmarks that measure what the GPU actually buys.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="results/speedup_curve_dark.png">
  <img alt="GPU speedup over CPU code running the same algorithm. 64-bit field multiplication: 44.3x with data on the GPU, 1.4x including copies, at 10M elements. 254-bit field multiplication over BN254 Fr: 909.8x with data on the GPU, 0.6x from and to Python ints, at 1M elements. NTT: 146.9x and 67.3x at 2^20. MSM Pippenger: 1.0x at every size." src="results/speedup_curve.png">
</picture>

---

## What this does

The two most expensive steps of a Groth16 or PLONK prover are the MSM and the NTT. This project implements both, plus the field arithmetic underneath them. It also measures each GPU path against CPU code that runs the same algorithm, and checks every result.

| Operation | ZK usage | What runs on the GPU | GPU speedup at the largest size measured |
|---|---|---|---|
| **Field multiplication**, 64-bit | Building block of everything else | The whole operation (CuPy kernel), for moduli below 2⁶³ | **44.3x** with data on the GPU, **1.4x** including copies (10M elements) |
| **Field multiplication**, 254-bit | BN254's real fields: Fr for polynomials, Fq for curve points | The whole operation (CUDA Montgomery kernel) | **910x** with data on the GPU, but **0.59x** from and to Python ints (1M elements), against Python ints |
| **NTT** | Polynomial arithmetic (Groth16, PLONK) | Every butterfly stage, for moduli below 2⁶³ | **146.9x** with data on the GPU, **67.3x** including copies (2²⁰ elements) |
| **MSM**, BN254 G1 | Groth16's linear combinations of curve points | Only the window-digit extraction | **1.0x** (10,000 points); all curve arithmetic is Python on the CPU |

Measured on an NVIDIA GeForce RTX 5050 Laptop GPU against NumPy and Python ints on an AMD Ryzen 7 260; see [Benchmark results](#benchmark-results).

### Scope and limits

This is a tested prototype, not yet a GPU prover for BN254:

- **The GPU NTT uses 64-bit integers**, so it works over primes below 2⁶³; the benchmarks use 998244353. BN254's two 254-bit fields now have GPU multiplication, addition and subtraction (`zkgpu.field256_gpu`), but the NTT and MSM do not use them yet, so BN254's NTT (`Radix2Domain`) still runs on the CPU.
- **The MSM gains are algorithmic.** Pippenger is 7.0x faster than the naive method at 1,000 points, and Jacobian coordinates made every MSM 18–28x faster. The GPU contributes nothing measurable yet.
- **The baselines are fair, but they are not the fastest CPU code possible.** Each GPU path is compared with the best CPU code this package has for the same job, which is the right comparison for it. Optimised native libraries (C++/Rust, multithreaded) would be considerably faster and would shrink these ratios. That matters most for the 254-bit field, where the baseline is interpreted Python integers, so its 910x overstates the gain over a good CPU library.
- **The 254-bit kernels only pay off on data that stays on the GPU.** Converting between Python ints and the GPU's format costs more than the multiplication itself (see [below](#field-multiplication-254-bit-bn254-fr)).

The next steps build BN254's NTT and MSM point arithmetic on those 254-bit kernels.

---

## Background

Most of a Groth16 or PLONK prover's time goes into two operations:

- **MSM** (multi-scalar multiplication): given curve points P₁…Pₙ and scalars s₁…sₙ, compute Σ sᵢPᵢ. It usually dominates proving time.
- **NTT** (number-theoretic transform): the FFT over a finite field, used to multiply and divide the constraint polynomials.

Both apply the same operation to millions of independent values at each step: every butterfly within an NTT stage, and every point within an MSM window. That is why GPU provers target them.

**Curve**: BN254 (aka alt_bn128) — the curve used by Ethereum's zkSNARK precompile.
- Base field prime: `p = 21888242871839275222246405745257275088696311157297823662689037894645226208583`
- Scalar field prime: `r = 21888242871839275222246405745257275088548364400416034343698204186575808495617`

---

## Benchmark results

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="results/latency_comparison_dark.png">
  <img alt="Time per call by implementation, log scales. 64-bit field multiplication: the GPU with data on the device is fastest, while the GPU including copies roughly matches NumPy from 1M elements up. 254-bit field multiplication: 0.31 ms on the GPU at 1M elements against 281 ms for Python ints, while starting and ending with Python ints is slower than Python ints alone. NTT: GPU time stays near 1-4 ms up to 2^20, against 278 ms for NumPy and 3.1 s for pure Python. MSM: Pippenger is fastest, and its CPU and GPU-digit lines coincide." src="results/latency_comparison.png">
</picture>

> Measured at commit `df75d1a` with a clean working tree, on AC power. **GPU:** NVIDIA GeForce RTX 5050 Laptop GPU (compute capability 12.0, 8 GiB), CUDA runtime 12.9, CuPy 14.0.1. **CPU:** AMD Ryzen 7 260 (16 logical cores). **Software:** Windows 11, Python 3.13.9, NumPy 2.3.4. The raw numbers and full metadata are in [`results/all_results.json`](results/all_results.json).

How to read the tables:

- **GPU** means the input is already on the GPU and the output stays there. That is the cost for a prover that keeps its data on the device.
- **GPU + copies** means the input starts in host memory and the output is copied back.
- Every time is a median in milliseconds, and every implementation's output was checked against a reference before its time was accepted (see [Method](#method)).
- **Expect some variation between runs.** Across three runs of this commit, the GPU speedups varied by up to 20% at the largest sizes (the NTT at 2²⁰ measured 123–147x) and by about 50% at the smallest, where a call takes microseconds. The 254-bit field and MSM ratios were the steadiest, within 5%. The tables show one of those runs.

### Field multiplication, 64-bit (mod 998244353)

| N | NumPy (ms) | GPU (ms) | GPU + copies (ms) | GPU speedup | With copies |
|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.019 | 0.017 | 0.135 | 1.1x | 0.14x |
| 100,000 | 0.207 | 0.018 | 0.395 | 11.5x | 0.52x |
| 1,000,000 | 3.26 | 0.053 | 2.81 | 61.8x | 1.2x |
| 10,000,000 | 33.3 | 0.751 | 24.6 | 44.3x | 1.4x |

One multiplication per element is too little work to pay for moving the data, so even at 10 million elements the GPU including copies is only 1.4x faster. Field kernels pay off inside a longer computation whose data stays on the GPU.

### Field multiplication, 254-bit (BN254 Fr)

| N | Python ints (ms) | GPU (ms) | GPU from/to Python ints (ms) | GPU speedup | From/to Python ints |
|---:|---:|---:|---:|---:|---:|
| 10,000 | 2.63 | 0.021 | 4.06 | 125.9x | 0.65x |
| 100,000 | 28.3 | 0.038 | 43.0 | 753.9x | 0.66x |
| 1,000,000 | 281 | 0.309 | 476 | 909.8x | 0.59x |

- **The baseline** is Python's own integers on NumPy object arrays, about 280 ns per multiplication. That is the fastest pure-Python option, and the only CPU implementation of 254-bit arithmetic in this package. Interpreted big integers allocate a new object for every result, which native fixed-width Montgomery code never does, so the 910x overstates what the GPU gains over an optimised CPU library.
- **With data on the GPU,** 1M multiplications take 0.31 ms, about 3.2 billion per second.
- **Starting and ending with Python ints is slower than the CPU** (0.59x). Converting each element between a Python int and four 64-bit limbs costs about 150 ns in each direction, more than the multiplication itself. That is close to the limit of pure Python: the alternatives tried were at most 20% faster. These kernels are meant for data that stays on the GPU across many operations, which is how the NTT and MSM will use them.

### NTT (radix-2 Cooley-Tukey, mod 998244353)

| N | Pure Python (ms) | NumPy (ms) | GPU (ms) | GPU + copies (ms) | GPU first call (ms) | GPU speedup | With copies |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2¹⁴ | 32.6 | 1.23 | 1.32 | 1.52 | 2.84 | 0.93x | 0.81x |
| 2¹⁶ | 144 | 5.42 | 1.56 | 1.81 | 4.53 | 3.5x | 3.0x |
| 2¹⁸ | 686 | 39.2 | 1.58 | 3.48 | 22.4 | 24.9x | 11.3x |
| 2²⁰ | 3,122 | 278 | 1.89 | 4.13 | 114 | 146.9x | 67.3x |

- **The GPU time hardly grows** between 2¹⁴ and 2²⁰. It costs a near-constant 0.09–0.10 ms per butterfly stage, and there are log₂ N stages. Each stage dispatches about six separate kernels from Python, so the time goes into that per-stage overhead rather than arithmetic. A single fused kernel would remove most of it.
- **The first call for each size** builds the twiddle table and bit-reversal index on the GPU and caches them; later calls reuse them. The NumPy baseline caches its tables the same way.
- **Pure Python** (`zkgpu.ntt.ntt`) is shown for scale and as the reference result, not as a baseline.

### MSM (BN254 G1, uniformly random 254-bit scalars)

| N | Window | Naive (ms) | Straus/Shamir (ms) | Pippenger, CPU (ms) | Pippenger, GPU digits (ms) | GPU contribution |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 5 | 99.2 | 43.5 | 24.1 | 24.4 | 0.99x |
| 1,000 | 7 | 1,018 | 431 | 146 | 146 | 1.00x |
| 10,000 | 10 | — | 4,265 | 1,041 | 1,012 | 1.0x |

- **All the point arithmetic is Python integers on the CPU, in every column.** The differences between columns are algorithmic. Pippenger is 4.1x faster than naive at 100 points and 7.0x at 1,000, and it is 4.1x faster than Straus/Shamir at 10,000.
- **The window size** follows Pippenger's cost model for each N. A fixed w = 8 would make it look slower than it is at small N.
- **Moving digit extraction to the GPU changes nothing** (0.98–1.03x across three runs, within noise), because the bucket accumulation (point additions) dominates the cost.
- **Naive is skipped at 10,000 points.** It costs a steady 1.0 ms per point, so about 10 s.

### Method

- Each implementation is timed as the median of repeated calls after one untimed warm-up call, within a time budget.
- If the median exceeds 1.25x the fastest sample, a sign of interference from other processes, the measurement is repeated, up to three attempts.
- GPU timings synchronise the device before the clock stops.
- The NumPy NTT is the same algorithm as the GPU version, vectorised one stage at a time, with the same per-size caches.

The details are in `benchmarks/_harness.py`. To reproduce on your own machine, see [Run benchmarks](#run-benchmarks).

---

## Installation

### Prerequisites

- Python 3.11+
- For the GPU paths: an NVIDIA GPU, and nvrtc for CuPy's kernel compilation, from either the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) or a CUDA build of PyTorch (which bundles it)

Check your CUDA version:
```bash
nvidia-smi
```
Look for `CUDA Version: XX.Y` in the top-right of the output.

### Install

The project is a regular Python package (`zkgpu`, configured in `pyproject.toml`). Install it in editable mode, so your edits take effect without reinstalling, together with the optional extras you need:

```bash
git clone https://github.com/erihhh6/zk-gpu-accelerator.git
cd zk-gpu-accelerator
pip install -e ".[gpu,bench,dev]"
```

| Extra | Installs | Needed for |
|---|---|---|
| `gpu` | `cupy-cuda12x` | GPU code paths (CUDA 12.x) |
| `bench` | `matplotlib` | `benchmarks/plot_results.py` |
| `dev` | `pytest` | running the tests |

> **CUDA 11.x:** install `cupy-cuda11x` yourself and leave out the `gpu` extra.
> **No GPU:** leave out the `gpu` extra. Every GPU function falls back to the CPU automatically.

### Windows: how CuPy finds nvrtc

CuPy needs `nvrtc64_*.dll` to compile GPU kernels. If a CUDA build of PyTorch is installed, `zkgpu` adds its bundled `torch/lib` directory to the DLL search path automatically, without importing torch. Otherwise, install the CUDA Toolkit, which includes nvrtc.

### Verify the GPU setup

```bash
python scripts/check_gpu.py
```

This checks the NVIDIA driver, the CuPy import, CUDA device visibility and kernel compilation (nvrtc), and prints the first layer that fails. It exits with code 1 if the GPU paths are disabled. When any layer fails, the GPU functions still run on the CPU but emit a `GpuFallbackWarning` with the reason. If GPU tests are being skipped, `pytest -rs` lists why.

> **Windows 11:** Smart App Control can refuse to load CuPy's compiled `.pyd` files ("An Application Control policy has blocked this file"), sometimes only intermittently. The script detects this and shows how to confirm it in the Code Integrity event log.

---

## Usage

### Field Arithmetic

```python
from zkgpu.field import Fp, batch_mul, FIELD_MODULUS

a = Fp(12345)
b = Fp(67890)
print(a * b)       # Fp multiplication mod r
print(a.inv())     # Fermat's little theorem inverse
print(a ** 254)    # fast exponentiation
print(1 - a / 2)   # plain ints mix in on either side; a ** -1 == a.inv()

# Fp values are immutable (and therefore safe as dict keys / set members)

# Batch operations on int lists
result = batch_mul([1, 2, 3], [4, 5, 6], FIELD_MODULUS)
```

### GPU Field Arithmetic

```python
import cupy as cp
from zkgpu.field_gpu import gpu_batch_mul, gpu_batch_add, gpu_batch_pow

MOD = 998_244_353
a = cp.array([3, 5, 7], dtype=cp.int64)
b = cp.array([4, 6, 8], dtype=cp.int64)

print(gpu_batch_mul(a, b, MOD))   # [12, 30, 56]
print(gpu_batch_add(a, b, MOD))   # [7, 11, 15]
print(gpu_batch_pow(a, 3, MOD))   # [27, 125, 343]
```

### GPU Field Arithmetic, 254-bit (BN254 Fr and Fq)

```python
from zkgpu.field256_gpu import BN254_FR as Fr   # BN254_FQ for the base field

a = Fr.from_ints([3, 5, -1])    # any ints: reduced mod r, stored on the GPU in Montgomery form
b = Fr.from_ints([4, 6, -1])
print(Fr.to_ints(Fr.mul(a, b)))                    # [12, 30, 1]
print(Fr.to_ints(Fr.add(a, Fr.from_ints([1]))))    # [4, 6, 0]: a single row broadcasts
```

A batch of n elements is an `(n, 4)` uint64 array of 64-bit limbs. Results stay in that form on the GPU, so they can feed further operations without copying, and `to_ints` reads them back. `Field256(modulus)` works the same way for any odd modulus below 2²⁵⁵.

### BN254 Curve Operations

```python
from zkgpu.curve import G, P, Point   # P is the BN254 base-field prime

A = G * 42
B = G * 7
print(A + B == G * 49)   # True

print(Point(1, 2 + P) == G)   # True: coordinates are reduced mod P
Point(1, 3)              # ValueError: (1, 3) is not on the BN254 curve
```

Points are immutable and validated on construction. Rejecting off-curve coordinates is a security property, not just tidiness: the addition formulas never use the curve constant `b = 3`, so an off-curve input would silently compute on a different, possibly weak, curve (an *invalid-curve attack*). Every `Point`, including those Pippenger builds from caller-supplied coordinates, is checked.

### NTT (CPU)

```python
from zkgpu.ntt import ntt, polynomial_mul, MOD

a = [1, 2, 3, 4, 5, 6, 7, 8]
A = ntt(a, MOD)
assert ntt(A, MOD, invert=True) == a   # perfect round-trip

# Polynomial multiplication O(N log N)
print(polynomial_mul([1, 1], [1, 1], MOD))   # [1, 2, 1] = 1 + 2x + x²
```

### Evaluation domains (BN254 Fr)

`Radix2Domain` is the subgroup H of n-th roots of unity that Groth16 and PLONK provers evaluate polynomials on. It supports sizes up to 2²⁸, with the same roots snarkjs/circom use. It provides FFT, inverse FFT, coset FFT and the vanishing polynomial Z_H(X) = Xⁿ − 1. Computing Groth16's quotient h = (A·B − C) / Z_H looks like this:

```python
from zkgpu import FIELD_MODULUS as r, Radix2Domain

d = Radix2Domain.at_least(5)               # 8 points: one per constraint, rounded up
a, b = [3, 1, 4, 1, 5], [9, 2, 6, 5, 3]    # coefficients of A(X), B(X)
c = d.ifft([x * y % r for x, y in zip(d.fft(a), d.fft(b))])   # C agrees with A·B on H

# Z_H is zero on H but the constant g^n - 1 on the coset g·H, so divide there
z_inv = pow(d.vanishing_eval(d.coset_offset), -1, r)
h = d.coset_ifft([(x * y - z) * z_inv % r
                  for x, y, z in zip(d.coset_fft(a), d.coset_fft(b), d.coset_fft(c))])
```

`tests/test_domain.py` checks that h·Z_H equals A·B − C exactly.

### NTT (GPU)

```python
import cupy as cp
from zkgpu.ntt_gpu import ntt_gpu
from zkgpu.ntt import MOD

a = cp.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=cp.int64)
A = ntt_gpu(a, MOD)
recovered = ntt_gpu(A, MOD, invert=True)
print(recovered.tolist())   # [1, 2, 3, 4, 5, 6, 7, 8]
```

### MSM (CPU)

```python
from zkgpu.curve import G
from zkgpu.msm import msm_naive, msm_double_and_add

points = [G * i for i in range(1, 11)]
scalars = list(range(1, 11))

result = msm_naive(points, scalars)
result2 = msm_double_and_add(points, scalars)
assert result == result2
```

### MSM (GPU — Pippenger)

```python
import random
import numpy as np
from zkgpu.curve import G
from zkgpu.field import FIELD_MODULUS
from zkgpu.msm import msm_double_and_add
from zkgpu.msm_gpu import msm_pippenger_gpu

points = [G * i for i in range(1, 101)]
scalars = [random.randrange(FIELD_MODULUS) for _ in points]   # full 254-bit scalars

# BN254 coordinates do not fit in int64, so keep them as Python ints (object dtype)
px = np.array([p.x for p in points], dtype=object)
py = np.array([p.y for p in points], dtype=object)

result = msm_pippenger_gpu(px, py, scalars, window_size=8)   # returns a Point
assert result == msm_double_and_add(points, scalars)
```

Scalars may be any integers (negative, or larger than r); they are reduced mod r first. The result is `Point.identity()` for the point at infinity.

### Run benchmarks

```bash
python -m benchmarks.run_all           # full run (~25 s) -> results/all_results.json + summary table
python -m benchmarks.run_all --quick   # smoke test in seconds -> results/quick_results.json (gitignored)
python benchmarks/plot_results.py      # charts, light and dark, next to the JSON
python demo/prover_simulation.py       # Groth16 prover timing simulation
```

Each run checks every implementation's output against a reference before accepting its timing. It compares against fair CPU baselines (vectorised NumPy for field ops and NTT, the same MSM algorithms on the CPU) on real 254-bit MSM scalars. It also records the hardware, library versions and git commit in the JSON. GPU numbers come in two forms: data already on the device (`gpu`), and including host↔device copies (`gpu_e2e`). The timing method is described in `benchmarks/_harness.py`.

### Run tests

```bash
pytest -v -rxs
```

- Tests marked `gpu` skip automatically when no usable GPU is found, and the skip message names the reason.
- A newly found bug is first recorded as a **known bug**: a test that asserts the *correct* behaviour, marked with `known_bug` (see `tests/known_bugs.py`) so it is reported as `xfail` until fixed. The marker is strict, so the commit that fixes the bug must also remove it.
- `-rxs` prints the reason for every skip and xfail.
- `pytest --cov` adds a line-and-branch coverage report for the `zkgpu` package. CI runs it on Linux (Python 3.11–3.13) and Windows, and fails below the floor set in `pyproject.toml`. CI has no GPU, so the CUDA paths only count on a GPU machine: about 83% total coverage there versus about 97% locally on an RTX 5050.

### Lint, format and type-check

```bash
ruff check .          # lint (pyflakes, bugbear, import order, ...)
ruff format .         # format
mypy                  # strict type-check of the zkgpu package
```

CI runs the same three commands. Their versions are pinned in the `dev` extra, and the configuration lives in `pyproject.toml`. To run them automatically before every commit:

```bash
pre-commit install    # once per clone; uses .pre-commit-config.yaml
```

After that, a commit that fails a check is stopped, and ruff fixes what it safely can.

---

## Project Structure

```
zk-gpu-accelerator/
├── pyproject.toml         # package metadata, dependency extras, pytest/ruff/mypy config
├── .pre-commit-config.yaml  # runs ruff + mypy before each commit (pre-commit install)
├── src/zkgpu/             # the importable package: `import zkgpu`
│   ├── gpu_backend.py     # One-time CUDA/CuPy detection + reason when unavailable
│   ├── field.py           # Fp arithmetic CPU — BN254 scalar field
│   ├── field_gpu.py       # Batch modular arithmetic GPU — CuPy ElementwiseKernel
│   ├── field256_gpu.py    # 254-bit field arithmetic GPU — CUDA Montgomery kernels (BN254)
│   ├── curve.py           # BN254 G1 points (affine) + Jacobian formulas for long chains
│   ├── ntt.py             # Cooley-Tukey NTT CPU (any NTT-friendly prime)
│   ├── domain.py          # Radix-2 evaluation domains: FFT, coset FFT, Z_H (BN254 Fr)
│   ├── ntt_gpu.py         # NTT GPU — vectorized butterflies, cached twiddles
│   ├── msm.py             # MSM CPU — naive + Straus/Shamir double-and-add
│   └── msm_gpu.py         # MSM — Pippenger (CPU buckets) + GPU window extraction
├── benchmarks/
│   ├── _harness.py        # timing method, result checks, run metadata
│   ├── bench_field.py
│   ├── bench_ntt.py       # includes the NumPy NTT baseline
│   ├── bench_msm.py
│   ├── run_all.py         # runs all three, writes one JSON file
│   └── plot_results.py    # charts, light and dark
├── tests/
│   ├── known_bugs.py        # Strict-xfail marker for documented, unfixed bugs
│   ├── test_field.py        # Fp arithmetic (CPU)
│   ├── test_field_gpu.py    # GPU field ops vs Python ints, GPU + fallback paths
│   ├── test_field256_gpu.py # 254-bit kernels vs Python ints, gnark/arkworks constants
│   ├── test_curve.py        # BN254 group law, py_ecc known answers
│   ├── test_ntt.py          # NTT across moduli, roots-of-unity known answer, GPU
│   ├── test_domain.py       # domains: snarkjs/arkworks roots, cosets, Groth16 quotient
│   ├── test_msm.py          # naive / Shamir / Pippenger MSM, GPU + fallback paths
│   ├── test_gpu_backend.py  # GPU detection reports why it is disabled
│   └── test_benchmarks.py   # harness runs end to end; baselines match references
├── scripts/
│   └── check_gpu.py   # Layer-by-layer GPU setup diagnostic
├── demo/
│   └── prover_simulation.py
└── results/           # the benchmark run quoted above: JSON + charts (light and dark)
```

The `src/` layout means the code is only importable once installed (`pip install -e .`), so tests always exercise the package as users get it rather than whatever happens to be in the working directory.

---

## Implementation Notes

- **Field**: BN254 scalar field (r ≈ 2²⁵⁴). All Fp ops use Python's built-in `pow()` for fast modular exponentiation. The GPU field ops (`zkgpu/field_gpu.py`) accept any modulus `2 ≤ mod < 2⁶³` and any int64 inputs, which they reduce first. Products use a plain int64 multiply up to `mod ≤ 3,037,000,500`, and an exact 128-bit-wide routine above that, built on `__umul64hi` because NVRTC on Windows has no `__int128`. The GPU benchmark uses `MOD = 998244353`.
- **NTT**: Cooley-Tukey iterative with bit-reversal permutation. The default modulus is `998244353 = 119·2²³ + 1` (N ≤ 2²³), but any odd prime works up to its 2-adicity, including BN254's scalar field r (2-adicity 28, N ≤ 2²⁸). Roots of unity come from the field's smallest quadratic non-residue, which yields primitive power-of-two roots without factoring p − 1, and every root is checked to be primitive. The GPU version (`zkgpu/ntt_gpu.py`) handles primes below 2⁶³: it caches its twiddle table and bit-reversal index on the device for each size, and uses `zkgpu/field_gpu.py` for exact modular arithmetic. BN254's r does not fit in int64, so it is CPU-only for now.
- **MSM**: Naive and Straus/Shamir simultaneous double-and-add on CPU (one shared accumulator: ~254 doublings in total instead of N·254, 2.3× faster than naive for 254-bit scalars in the benchmark above); Pippenger bucket method: scalars are reduced mod r and split into four 64-bit limbs, one GPU kernel extracts every window digit at once (a single device-to-host copy), and bucket accumulation plus Horner's-rule window combination (~254 doublings) stay on the CPU. All BN254 point arithmetic uses Python's arbitrary-precision integers (254-bit coordinates).
- **Curve arithmetic**: `Point` is affine, but scalar multiplication and every MSM run in Jacobian coordinates (X, Y, Z) ↦ (X/Z², Y/Z³). Those additions and doublings need no modular inverse, so a whole chain of operations pays for one inverse at the end, when converting back. The formulas are the Explicit-Formulas Database's a = 0 versions (`dbl-2009-l`, `add-2007-bl`, `madd-2007-bl`). Remaining inverses use `pow(a, -1, p)` (extended Euclid, about 42 multiplications) instead of Fermat's `a^(p−2)` (about 251). Together these made scalar multiplication about 30× faster and the MSMs 18–28× faster than affine arithmetic, with identical results. Every result is converted back through `Point`, so it is re-validated to lie on the curve.
- **254-bit field (GPU)**: `zkgpu/field256_gpu.py` stores each element as four 64-bit limbs, in Montgomery form (a·R mod p, R = 2²⁵⁶).
  - **Multiplication** is CIOS Montgomery multiplication (Koç, Acar and Kaliski), one thread per element. Each step adds a·bᵢ plus a multiple of p that clears the lowest word, so no division is ever needed. 64×64→128-bit products come from `__umul64hi`.
  - **Five words of temporary state are enough, not six**, because the moduli are below 2²⁵⁵. The kernel source carries the bound proof, and a test replays the algorithm word by word to check it, including at the extreme 2²⁵⁵−19.
  - **Reductions are branch-free selects**, so the threads of a warp never diverge on data.
  - **The modulus is compiled into the kernel** as constants, giving one CUDA module per field, built on first use.
  - **Memory reads stay coalesced:** one 32-byte row is exactly one memory sector, so a warp reading consecutive rows uses every byte it fetches.
  - **Tests** check gnark's published Montgomery constants for both BN254 fields. Mutation checks confirm that breaking any carry, the constant's sign or the final reduction makes them fail.
- **GPU kernels**: Written with CuPy's `ElementwiseKernel` and, for the 254-bit field, `RawModule` (CUDA C++ compiled at runtime via nvrtc). Works without the full CUDA Toolkit — only nvrtc is needed (bundled with PyTorch on Windows).
- **Fallback**: GPU availability is decided once, at import, in `zkgpu/gpu_backend.py`. When it is unavailable, every GPU function falls back to NumPy/CPU and emits a `GpuFallbackWarning` naming the reason, so no code changes are needed on CPU-only machines.
- **Benchmarking**: See [Method](#method). Kernels are compiled before any timing starts, so the timings never include CuPy's JIT compilation.

---

## References

- Groth16: <https://eprint.iacr.org/2016/260.pdf>
- Pippenger MSM: <https://cr.yp.to/papers/pippenger.pdf>
- BN254 curve parameters: <https://neuromancer.sk/std/bn/bn254>
- NTT primer: <https://cp-algorithms.com/algebra/fft.html>
- CuPy documentation: <https://docs.cupy.dev>
