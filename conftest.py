"""conftest.py — pytest configuration for zk-gpu-accelerator.

Tests import the installed `zkgpu` package (pip install -e ".[dev]"), so no
sys.path manipulation is needed; the `gpu` marker is registered in pyproject.toml.
"""

import pytest

from zkgpu.gpu_backend import GPU_AVAILABLE, GPU_UNAVAILABLE_REASON


def pytest_collection_modifyitems(config, items):
    # One central skip instead of per-test checks, so every GPU test reports
    # the same, specific reason (e.g. "CuPy failed to import: ...").
    if GPU_AVAILABLE:
        return
    skip_gpu = pytest.mark.skip(reason=f"CUDA GPU not available: {GPU_UNAVAILABLE_REASON}")
    for item in items:
        if item.get_closest_marker("gpu") is not None:
            item.add_marker(skip_gpu)
