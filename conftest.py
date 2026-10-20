"""conftest.py — pytest configuration for zk-gpu-accelerator."""

import sys
import os

# Ensure the project root is on sys.path so `from src.xxx import ...` works
# regardless of where pytest is invoked from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest  # noqa: E402

from src.gpu_backend import GPU_AVAILABLE, GPU_UNAVAILABLE_REASON  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "gpu: needs a working CUDA GPU; skipped automatically (with the reason) otherwise"
    )


def pytest_collection_modifyitems(config, items):
    # One central skip instead of per-test checks, so every GPU test reports
    # the same, specific reason (e.g. "CuPy failed to import: ...").
    if GPU_AVAILABLE:
        return
    skip_gpu = pytest.mark.skip(reason=f"CUDA GPU not available: {GPU_UNAVAILABLE_REASON}")
    for item in items:
        if item.get_closest_marker("gpu") is not None:
            item.add_marker(skip_gpu)
