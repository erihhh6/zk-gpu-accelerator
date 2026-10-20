"""tests/test_benchmarks.py — the benchmark harness runs end to end and checks its own results."""

import json

import numpy as np
import pytest

from benchmarks import run_all
from benchmarks._harness import check_equal
from benchmarks.bench_msm import pippenger_window
from benchmarks.bench_ntt import ntt_numpy
from zkgpu.gpu_backend import GPU_AVAILABLE
from zkgpu.ntt import MOD, ntt

pytestmark = pytest.mark.filterwarnings("ignore::zkgpu.gpu_backend.GpuFallbackWarning")


@pytest.fixture(scope="module")
def quick_results(tmp_path_factory):
    """One quick run (a few seconds) shared by the tests below; it never touches results/."""
    out = tmp_path_factory.mktemp("bench") / "quick.json"
    run_all.run(quick=True, output=out)
    return out, json.loads(out.read_text(encoding="utf-8"))


def test_quick_run_writes_verified_results(quick_results):
    _, data = quick_results
    assert set(data) == {"meta", "field", "ntt", "msm"}
    meta = data["meta"]
    assert meta["quick"] is True
    assert meta["python"] and meta["cpu"] and meta["method"]
    assert (meta["gpu"] is not None) == GPU_AVAILABLE
    for section in ("field", "ntt", "msm"):
        assert data[section], f"{section} produced no rows"
        for row in data[section]:
            assert row["verified"] is True
            assert row["ms"] and all(ms > 0 for ms in row["ms"].values())
            # GPU columns exist exactly when a GPU does; CPU-only machines still benchmark the CPU
            has_gpu_column = any(k.startswith(("gpu", "pippenger_gpu")) for k in row["ms"])
            assert has_gpu_column == GPU_AVAILABLE


def test_mismatched_results_fail_the_run():
    with pytest.raises(RuntimeError, match="mismatch"):
        check_equal("broken", [1, 2], [1, 3])


@pytest.mark.parametrize("n", [1, 2, 64, 1024])
def test_numpy_baseline_matches_reference_ntt(n):
    """The fair CPU baseline must compute the same transform, or the comparison is meaningless."""
    a = np.random.default_rng(n).integers(0, MOD, size=n, dtype=np.int64)
    assert ntt_numpy(a, MOD).tolist() == ntt(a.tolist(), MOD)


def test_numpy_baseline_caches_survive_repeated_calls():
    """Warm calls reuse the cached index and twiddles; they must stay correct for new inputs."""
    rng = np.random.default_rng(7)
    for _ in range(3):
        a = rng.integers(0, MOD, size=256, dtype=np.int64)
        assert ntt_numpy(a, MOD).tolist() == ntt(a.tolist(), MOD)


def test_pippenger_window_follows_cost_model():
    assert [pippenger_window(n) for n in (100, 1_000, 10_000)] == [5, 7, 10]


def test_plots_render_in_both_themes(quick_results):
    pytest.importorskip("matplotlib")  # the bench extra; CI installs only dev
    from benchmarks import plot_results

    path, _ = quick_results
    written = plot_results.main(path)
    assert {p.name for p in written} == {
        "latency_comparison.png",
        "speedup_curve.png",
        "latency_comparison_dark.png",
        "speedup_curve_dark.png",
    }
    assert all(p.stat().st_size > 10_000 for p in written)
