"""tests/test_gpu_backend.py — GPU detection always explains a disabled backend."""

import re
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from zkgpu import field_gpu, gpu_backend

# What CuPy's ImportError looks like when Windows Smart App Control blocks a DLL
BLOCKED_IMPORT = ImportError(
    "\n====\nFailed to import CuPy.\n\nOriginal error:\n"
    "  ImportError: DLL load failed while importing _runtime_enum: "
    "An Application Control policy has blocked this file.\n===="
)


def fake_cupy(*, devices=1, device_error=None, kernel_error=None, probe_values=(1, 2, 3, 4)):
    """Stand-in for the cupy module, exposing only what gpu_backend._detect() touches."""

    def get_device_count():
        if device_error is not None:
            raise device_error
        return devices

    class ElementwiseKernel:
        def __init__(self, *_args):
            if kernel_error is not None:
                raise kernel_error

        def __call__(self, _x):
            return SimpleNamespace(get=lambda: np.array(probe_values))

    return SimpleNamespace(
        cuda=SimpleNamespace(runtime=SimpleNamespace(getDeviceCount=get_device_count)),
        ElementwiseKernel=ElementwiseKernel,
        arange=np.arange,
        int64=np.int64,
    )


class _BlockedImportFinder:
    """Meta-path finder that makes `import cupy` fail the way a blocked DLL does."""

    def find_spec(self, name, path=None, target=None):
        if name == "cupy":
            raise BLOCKED_IMPORT
        return None


def test_reason_is_set_exactly_when_gpu_unavailable():
    assert (gpu_backend.GPU_UNAVAILABLE_REASON is None) == gpu_backend.GPU_AVAILABLE


def test_summarize_extracts_cupy_original_error():
    """CuPy buries the real cause after 'Original error:' in a long install guide."""
    exc = ImportError(
        "\n====\nFailed to import CuPy.\n\nCheck the Installation Guide.\n\n"
        "Original error:\n  ImportError: DLL load failed: blocked by policy\n====\n"
    )
    assert gpu_backend._summarize(exc) == "ImportError: DLL load failed: blocked by policy"


def test_summarize_plain_exception_keeps_type_and_first_line():
    assert gpu_backend._summarize(RuntimeError("boom\nsecond line")) == "RuntimeError: boom"
    assert gpu_backend._summarize(RuntimeError()) == "RuntimeError"


def test_detect_reports_missing_cupy(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", None)  # makes `import cupy` raise ModuleNotFoundError
    assert gpu_backend._detect() == (None, "CuPy is not installed (pip install cupy-cuda12x)")


def test_detect_reports_blocked_import(monkeypatch):
    monkeypatch.delitem(sys.modules, "cupy", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_BlockedImportFinder(), *sys.meta_path])
    module, reason = gpu_backend._detect()
    assert module is None
    assert reason == (
        "CuPy failed to import: ImportError: DLL load failed while importing _runtime_enum: "
        "An Application Control policy has blocked this file."
    )


@pytest.mark.parametrize(
    ("stage", "expected_reason"),
    [
        ({}, None),
        ({"devices": 0}, "found no GPU device"),
        (
            {"device_error": RuntimeError("cudaErrorInsufficientDriver")},
            r"device query failed .*Insufficient",
        ),
        ({"kernel_error": RuntimeError("nvrtc64_120_0.dll not found")}, r"compilation failed .*nvrtc64"),
        ({"probe_values": (0, 0, 0, 0)}, "wrong values"),
    ],
    ids=["working", "no-device", "driver-error", "nvrtc-missing", "bad-probe"],
)
def test_detect_reports_each_failure_stage(monkeypatch, stage, expected_reason):
    module = fake_cupy(**stage)
    monkeypatch.setitem(sys.modules, "cupy", module)
    detected, reason = gpu_backend._detect()
    assert detected is module  # CuPy imported, so fallbacks can still convert its arrays
    if expected_reason is None:
        assert reason is None
    else:
        assert re.search(expected_reason, reason)


def test_cpu_fallback_warns_with_reason(monkeypatch):
    """Forcing the GPU off must warn with the reason and still return correct values."""
    monkeypatch.setattr(field_gpu, "_gpu_available", False)
    monkeypatch.setattr(gpu_backend, "GPU_UNAVAILABLE_REASON", "simulated outage")

    with pytest.warns(gpu_backend.GpuFallbackWarning, match="simulated outage"):
        result = field_gpu.gpu_batch_mul(np.array([3, 5]), np.array([4, 6]), 7)

    assert result.tolist() == [5, 2]
