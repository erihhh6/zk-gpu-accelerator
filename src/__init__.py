"""
zk-gpu-accelerator — GPU-accelerated ZK proof primitives (NTT and MSM).

Submodules
----------
field       : Finite field arithmetic (CPU baseline) over BN254 scalar field
field_gpu   : GPU-accelerated field arithmetic (CuPy + Numba)
curve       : BN254 elliptic curve point operations (affine coordinates)
ntt         : Number Theoretic Transform — Cooley-Tukey CPU implementation
ntt_gpu     : NTT on GPU via Numba CUDA butterfly kernels
msm         : Multi-Scalar Multiplication on CPU (naive + Shamir's trick)
msm_gpu     : MSM on GPU (parallel scalar-mul + Pippenger)
"""

from src.field import Fp, FIELD_MODULUS, batch_mul, batch_add
from src.curve import Point, G, P, scalar_mul
from src.ntt import ntt, polynomial_mul, MOD as NTT_MOD
from src.msm import msm_naive, msm_double_and_add

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
