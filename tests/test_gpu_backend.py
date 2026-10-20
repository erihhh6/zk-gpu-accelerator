"""tests/test_gpu_backend.py — GPU detection always explains a disabled backend."""

import numpy as np
import pytest

from zkgpu import field_gpu, gpu_backend


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


def test_cpu_fallback_warns_with_reason(monkeypatch):
    """Forcing the GPU off must warn with the reason and still return correct values."""
    monkeypatch.setattr(field_gpu, "_gpu_available", False)
    monkeypatch.setattr(gpu_backend, "GPU_UNAVAILABLE_REASON", "simulated outage")

    with pytest.warns(gpu_backend.GpuFallbackWarning, match="simulated outage"):
        result = field_gpu.gpu_batch_mul(np.array([3, 5]), np.array([4, 6]), 7)

    assert result.tolist() == [5, 2]
