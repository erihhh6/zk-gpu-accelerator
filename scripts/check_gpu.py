"""
scripts/check_gpu.py — Diagnose whether this machine can run the GPU code paths.

Usage (after installing the package, e.g. pip install -e ".[gpu,dev]"):
    python scripts/check_gpu.py

Every GPU function in zkgpu falls back to a CPU implementation when CUDA is
unusable (emitting a GpuFallbackWarning), so this script reports each layer
of the stack explicitly. The first [FAIL] line is the one to fix:

  1. NVIDIA driver   — nvidia-smi can see a GPU
  2. CUDA backend    — zkgpu.gpu_backend: CuPy imports, a device is visible,
                       and nvrtc can JIT-compile and run a probe kernel; then
                       the 256-bit field kernels compile and pass their self-test
  3. PyTorch         — informational only; optional, but on Windows its
                       bundled nvrtc DLL stands in for a full CUDA Toolkit

Exit code is 0 when the project's GPU backend is usable, 1 otherwise.
"""

from __future__ import annotations

import importlib.util
import platform
import subprocess
import sys
from dataclasses import dataclass

OK, FAIL, INFO = "OK", "FAIL", "INFO"

# Substrings Windows uses when Code Integrity (Smart App Control / WDAC) refuses
# to load an unsigned DLL or .pyd — the failure mode seen with CuPy wheels.
_CODE_INTEGRITY_MARKERS = ("Application Control policy", "Enterprise signing level")


@dataclass
class Check:
    status: str
    name: str
    detail: str


def check_driver() -> Check:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        return Check(FAIL, "NVIDIA driver", "nvidia-smi not found - is the NVIDIA driver installed?")
    except subprocess.TimeoutExpired:
        return Check(FAIL, "NVIDIA driver", "nvidia-smi timed out after 30 s")
    if proc.returncode != 0:
        message = (proc.stderr or proc.stdout).strip() or f"exit code {proc.returncode}"
        return Check(FAIL, "NVIDIA driver", f"nvidia-smi failed: {message}")
    return Check(OK, "NVIDIA driver", proc.stdout.strip().replace("\n", " | "))


def check_backend() -> list[Check]:
    from zkgpu import gpu_backend

    if not gpu_backend.GPU_AVAILABLE:
        return [Check(FAIL, "CUDA backend", str(gpu_backend.GPU_UNAVAILABLE_REASON))]

    cp = gpu_backend.cp
    props = cp.cuda.runtime.getDeviceProperties(0)
    runtime = cp.cuda.runtime.runtimeGetVersion()  # encoded as 1000*major + 10*minor
    device = (
        f"{props['name'].decode()} (compute capability {props['major']}.{props['minor']}), "
        f"CUDA runtime {runtime // 1000}.{runtime % 1000 // 10}"
    )
    return [
        Check(OK, "CuPy import", f"cupy {cp.__version__}"),
        Check(OK, "CUDA device", device),
        Check(OK, "Kernel compile", "nvrtc compiled and ran the probe kernel"),
        check_field256(),
    ]


def check_field256() -> Check:
    """Compile and self-test the 256-bit kernels, and say how their machine code was produced."""
    from zkgpu import gpu_backend
    from zkgpu.field256_gpu import BN254_FQ, BN254_FR

    major, minor = gpu_backend.nvrtc_version()
    driver = gpu_backend.cp.cuda.runtime.driverGetVersion()
    how = "PTX finished by the driver" if gpu_backend.driver_compiles_ptx() else "NVRTC machine code"
    versions = f"NVRTC {major}.{minor}, driver CUDA {driver // 1000}.{driver % 1000 // 10}"
    try:
        for field in (BN254_FR, BN254_FQ):
            field.compile()
    except RuntimeError as exc:
        return Check(FAIL, "256-bit kernels", f"{exc} ({how}; {versions})")
    return Check(OK, "256-bit kernels", f"compiled ({how}; {versions}) and passed their self-test")


def check_torch() -> Check:
    if importlib.util.find_spec("torch") is None:
        return Check(INFO, "PyTorch", "not installed (optional; on Windows it can supply nvrtc for CuPy)")
    try:
        import torch
    except Exception as exc:
        return Check(INFO, "PyTorch", f"installed but failed to import: {type(exc).__name__}: {exc}")
    if not torch.cuda.is_available():
        return Check(INFO, "PyTorch", f"torch {torch.__version__}, but torch.cuda.is_available() is False")
    device = torch.cuda.get_device_name(0)
    return Check(INFO, "PyTorch", f"torch {torch.__version__} (CUDA {torch.version.cuda}) sees {device}")


def _print_hints(checks: list[Check]) -> None:
    failures = [c for c in checks if c.status == FAIL]
    if any(marker in c.detail for c in failures for marker in _CODE_INTEGRITY_MARKERS):
        print(
            "\nHint: Windows Code Integrity (usually Smart App Control) blocked a CuPy file.\n"
            "  - These blocks can be intermittent; re-running may succeed.\n"
            "  - List the blocked files (PowerShell):\n"
            '      Get-WinEvent -LogName "Microsoft-Windows-CodeIntegrity/Operational" -MaxEvents 300 |\n'
            '        Where-Object Message -match "cupy" | Select-Object -First 5 TimeCreated, Message\n'
            "  - Setting: Windows Security > App & browser control > Smart App Control."
        )
    if any("nvrtc" in c.detail for c in failures):
        print(
            "\nHint: CuPy needs nvrtc64_*.dll to compile kernels. Install the CUDA Toolkit, or install\n"
            "  PyTorch with CUDA (its torch/lib folder ships nvrtc and is picked up automatically)."
        )


def main() -> int:
    print(f"Python {platform.python_version()} on {platform.platform()}\n")

    if importlib.util.find_spec("zkgpu") is None:
        print('The zkgpu package is not installed. From the repo root run:\n    pip install -e ".[gpu,dev]"')
        return 1

    # The backend check runs before PyTorch is imported so the result reflects
    # what the project itself sees: torch loads its own CUDA DLLs into the
    # process, which can hide a DLL that CuPy alone would fail to find.
    checks = [check_driver(), *check_backend(), check_torch()]

    for check in checks:
        print(f"[{check.status:^4}] {check.name:<15} {check.detail}")
    _print_hints(checks)

    from zkgpu import gpu_backend

    if gpu_backend.GPU_AVAILABLE:
        print("\nGPU code paths are enabled.")
        return 0
    print("\nGPU code paths are DISABLED - every *_gpu function will fall back to the CPU.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
