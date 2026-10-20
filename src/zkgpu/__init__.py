"""
zk-gpu-accelerator — GPU-accelerated ZK proof primitives (NTT and MSM).

Submodules
----------
gpu_backend : one-time CUDA/CuPy detection; records why the GPU is unavailable
field       : Finite field arithmetic (CPU baseline) over BN254 scalar field
field_gpu   : Batch modular arithmetic on GPU (CuPy ElementwiseKernel, int64 moduli)
curve       : BN254 elliptic curve point operations (affine coordinates)
ntt         : Number Theoretic Transform — Cooley-Tukey CPU implementation
ntt_gpu     : NTT on GPU via vectorized CuPy butterflies
msm         : Multi-Scalar Multiplication on CPU (naive + Straus/Shamir)
msm_gpu     : Pippenger MSM with GPU scalar-window extraction (point math on CPU)

GPU modules are not imported here: importing them probes CUDA, which should
only happen for callers that actually want the GPU paths.
"""

__version__ = "0.1.0"  # single source of truth; pyproject.toml reads it

from zkgpu.field import Fp, FIELD_MODULUS, batch_mul, batch_add
from zkgpu.curve import Point, G, P, scalar_mul
from zkgpu.ntt import ntt, polynomial_mul, MOD as NTT_MOD
from zkgpu.msm import msm_naive, msm_double_and_add

__all__ = [
    # field
    "Fp",
    "FIELD_MODULUS",
    "batch_mul",
    "batch_add",
    # curve
    "Point",
    "G",
    "P",
    "scalar_mul",
    # ntt
    "ntt",
    "polynomial_mul",
    "NTT_MOD",
    # msm
    "msm_naive",
    "msm_double_and_add",
]
