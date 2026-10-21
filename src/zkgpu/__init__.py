"""
zk-gpu-accelerator — GPU-accelerated ZK proof primitives (NTT and MSM).

Submodules
----------
gpu_backend : one-time CUDA/CuPy detection; records why the GPU is unavailable
field       : Finite field arithmetic (CPU baseline) over BN254 scalar field
field_gpu   : Batch modular arithmetic on GPU (CuPy ElementwiseKernel, int64 moduli)
field256_gpu: 254-bit field arithmetic and NTT on GPU (CUDA Montgomery kernels; BN254 Fr and Fq)
curve_gpu   : BN254 G1 point addition and doubling on GPU (Jacobian coordinates over Fq)
curve       : BN254 elliptic curve point operations (affine coordinates)
ntt         : Number Theoretic Transform — Cooley-Tukey CPU implementation
domain      : Radix-2 evaluation domains over BN254 Fr: FFT, coset FFT, Z_H
ntt_gpu     : NTT on GPU via vectorized CuPy butterflies
msm         : Multi-Scalar Multiplication on CPU (naive + Straus/Shamir)
msm_gpu     : Pippenger MSM on GPU: digits, bucket accumulation and reduction (CPU fallback)

GPU modules are not imported here: importing them probes CUDA, which should
only happen for callers that actually want the GPU paths.
"""

__version__ = "0.1.0"  # single source of truth; pyproject.toml reads it

from zkgpu.curve import G, P, Point, scalar_mul
from zkgpu.domain import Radix2Domain
from zkgpu.field import FIELD_MODULUS, Fp, batch_add, batch_mul
from zkgpu.msm import msm_double_and_add, msm_naive
from zkgpu.ntt import MOD as NTT_MOD, ntt, polynomial_mul

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
    # domain
    "Radix2Domain",
    # msm
    "msm_naive",
    "msm_double_and_add",
]
