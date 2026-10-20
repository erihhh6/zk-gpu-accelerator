"""
zkgpu/gpu_backend.py — Single source of truth for whether the CUDA backend is usable.

Every GPU module imports ``cp``, ``GPU_AVAILABLE`` and ``GPU_UNAVAILABLE_REASON``
from here instead of probing CuPy on its own. Detection runs once, at import,
and records *why* it failed so callers can surface the reason rather than
silently falling back to the CPU.

Detection stages (the first failure wins):
  1. ``import cupy``               — package installed and its compiled extensions load
  2. CUDA device query             — driver present and at least one GPU visible
  3. Compile and run a tiny kernel — nvrtc available (CuPy JIT-compiles every kernel)

Run ``python scripts/check_gpu.py`` for a human-readable report of each stage.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import warnings
from typing import Any


class GpuFallbackWarning(UserWarning):
    """Emitted when a GPU function runs its CPU fallback instead of CUDA."""


def _add_torch_lib_to_dll_search_path() -> Any:
    """
    Let CuPy find PyTorch's bundled nvrtc64_*.dll on Windows.

    CuPy compiles kernels at runtime with nvrtc, which normally ships with the
    CUDA Toolkit. PyTorch wheels bundle the same DLL in ``torch/lib``, so when
    torch is installed we expose that directory instead of requiring a full
    Toolkit install. torch is located with ``find_spec`` rather than imported:
    importing it costs seconds and loads hundreds of MB this package never uses.

    Returns the ``os.add_dll_directory`` handle (the directory is removed from
    the search path if the handle is closed), or None when nothing was added.
    """
    if sys.platform != "win32":
        return None
    spec = importlib.util.find_spec("torch")
    if spec is None or spec.origin is None:
        return None
    torch_lib = os.path.join(os.path.dirname(spec.origin), "lib")
    if not os.path.isdir(torch_lib):
        return None

    # PATH covers CuPy's by-name LoadLibrary lookup of nvrtc; add_dll_directory
    # covers DLLs resolved as dependencies of extension modules (Python >= 3.8
    # no longer consults PATH for those).
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if torch_lib not in path_entries:
        os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
    return os.add_dll_directory(torch_lib)


def _summarize(exc: BaseException) -> str:
    """
    One-line description of *exc*.

    CuPy's ImportError wraps the real cause in a multi-line installation guide
    and a DLL dependency dump; the actionable part is the first line after
    "Original error:" (which already starts with the original exception type).
    """
    text = str(exc)
    if "Original error:" in text:
        tail = text.split("Original error:", 1)[1]
        first = next((line.strip() for line in tail.splitlines() if line.strip()), "")
        if first:
            return first
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return f"{type(exc).__name__}: {first}" if first else type(exc).__name__


def _detect() -> tuple[Any, str | None]:
    """Return (cupy module or None, reason the GPU is unusable or None)."""
    try:
        import cupy
    except ModuleNotFoundError:
        return None, "CuPy is not installed (pip install cupy-cuda12x)"
    except Exception as exc:  # DLL load failures surface as ImportError or OSError
        return None, f"CuPy failed to import: {_summarize(exc)}"

    try:
        if cupy.cuda.runtime.getDeviceCount() == 0:
            return cupy, "CUDA runtime found no GPU device"
    except Exception as exc:
        return cupy, f"CUDA device query failed (is the NVIDIA driver installed?): {_summarize(exc)}"

    try:
        probe = cupy.ElementwiseKernel("int64 x", "int64 y", "y = x + 1", "zkgpu_backend_probe")
        if probe(cupy.arange(4, dtype=cupy.int64)).get().tolist() != [1, 2, 3, 4]:
            return cupy, "probe kernel ran but returned wrong values"
    except Exception as exc:
        return cupy, f"CUDA kernel compilation failed (is nvrtc available?): {_summarize(exc)}"

    return cupy, None


_TORCH_DLL_DIR = _add_torch_lib_to_dll_search_path()

# cp is the CuPy module whenever it imported, even if a later stage failed, so
# CPU fallbacks can still convert CuPy arrays the caller already created.
cp, GPU_UNAVAILABLE_REASON = _detect()
GPU_AVAILABLE: bool = GPU_UNAVAILABLE_REASON is None


def warn_cpu_fallback(func_name: str, fallback: str) -> None:
    """Warn that *func_name* is running *fallback* on the CPU, and say why."""
    warnings.warn(
        f"{func_name}: GPU unavailable ({GPU_UNAVAILABLE_REASON}); falling back to {fallback}. "
        "Run `python scripts/check_gpu.py` for details.",
        GpuFallbackWarning,
        stacklevel=3,  # point at the caller of the public GPU function
    )
