"""
benchmarks/_harness.py — Timing and metadata helpers shared by every benchmark.

Timing method
-------------
- Every measurement is preceded by one untimed warm-up call (JIT compilation,
  caches, page faults), then repeated while it fits a small time budget; the
  median is reported. If that median sits well above the fastest sample,
  something else disturbed the run and the measurement is repeated (up to
  three attempts, keeping the least disturbed one).
- GPU timings are wall-clock with a device synchronisation before the clock
  stops, i.e. what a caller actually waits for. CUDA events would only time
  the device timeline, and for these Python-orchestrated multi-kernel
  operations that timeline includes the host's launch gaps anyway.
- "gpu" timings start from data already on the device; "gpu_e2e" timings
  start from host data and include both copies (host to device and back).
"""

from __future__ import annotations

import contextlib
import datetime
import importlib.metadata
import os
import platform
import statistics
import subprocess
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from zkgpu.gpu_backend import GPU_AVAILABLE, GPU_UNAVAILABLE_REASON, cp

REPO_ROOT = Path(__file__).resolve().parent.parent

# A measurement whose median exceeds its fastest sample by more than this is
# treated as disturbed and repeated, up to MAX_ATTEMPTS times in total.
NOISE_RATIO = 1.25
MAX_ATTEMPTS = 3


# ------------------------------------------------------------------
# Timing
# ------------------------------------------------------------------


def time_ms(
    fn: Callable[[], Any],
    *,
    budget_s: float,
    max_repeat: int = 7,
    gpu: bool = False,
    warmup: bool = True,
) -> tuple[float, Any]:
    """
    (median wall-clock milliseconds of fn(), fn's last return value).

    Runs one untimed warm-up call unless warmup=False (pure-Python code has
    nothing to warm up, and the slow baselines would double the run time).
    Repeats until *budget_s* of measured time is used (at least once, at most
    *max_repeat* times). With gpu=True the device is synchronised before each
    clock read, so queued kernels are included. The return value lets callers
    verify the result they timed instead of computing it a second time.
    """
    result: Any = None

    def run() -> float:
        nonlocal result
        t0 = time.perf_counter()
        result = fn()
        if gpu:
            cp.cuda.Device().synchronize()
        return time.perf_counter() - t0

    if warmup:
        run()
    best: list[float] = []
    for _ in range(MAX_ATTEMPTS):
        samples = [run()]
        while len(samples) < max_repeat and sum(samples) < budget_s:
            samples.append(run())
        if not best or statistics.median(samples) < statistics.median(best):
            best = samples
        # Interference (other processes, the display, power management) only ever
        # adds time. A median well above the fastest sample means it hit this
        # attempt (seen in practice: 28.7 ms medians for a 1.9 ms NTT), so measure
        # again rather than publish it. Single-shot timings cannot be judged this way.
        if len(samples) < 3 or statistics.median(samples) <= NOISE_RATIO * min(samples):
            break
    return statistics.median(best) * 1e3, result


def check_equal(name: str, got: Any, expected: Any) -> None:
    """Fail the benchmark run (not just a skipped assert under -O) if *name* disagrees."""
    if got != expected:
        raise RuntimeError(f"benchmark result mismatch: {name} disagrees with the reference implementation")


@contextlib.contextmanager
def forced_cpu_fallback(module: Any) -> Iterator[None]:
    """Temporarily run *module*'s CPU fallback even though a GPU is present."""
    saved = module._gpu_available
    module._gpu_available = False
    try:
        yield
    finally:
        module._gpu_available = saved


# ------------------------------------------------------------------
# Metadata: enough to know what produced a result file
# ------------------------------------------------------------------


def _git_commit() -> tuple[str | None, bool | None]:
    """(HEAD commit, working tree dirty?); reads .git directly if git is not on PATH."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return sha, bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        pass
    git_dir = REPO_ROOT / ".git"
    try:
        head = (git_dir / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head, None  # detached HEAD stores the commit directly
        ref = head.removeprefix("ref: ")
        loose = git_dir / ref
        if loose.exists():
            return loose.read_text().strip(), None
        for line in (git_dir / "packed-refs").read_text().splitlines():
            if line.endswith(" " + ref):
                return line.split()[0], None
    except OSError:
        pass
    return None, None


def _cpu_name() -> str:
    """Marketing name of the CPU where the OS exposes it, else platform.processor()."""
    if platform.system() == "Windows":
        try:
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    elif platform.system() == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or "unknown"


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _cuda_version(encoded: int) -> str:
    """CUDA encodes versions as 1000·major + 10·minor."""
    return f"{encoded // 1000}.{encoded % 1000 // 10}"


def _gpu_info() -> dict[str, Any] | None:
    if not GPU_AVAILABLE:
        return None
    props = cp.cuda.runtime.getDeviceProperties(0)
    return {
        "name": props["name"].decode(),
        "compute_capability": f"{props['major']}.{props['minor']}",
        "memory_gib": round(props["totalGlobalMem"] / 2**30, 1),
        "cuda_runtime": _cuda_version(cp.cuda.runtime.runtimeGetVersion()),
        "cuda_driver_supports": _cuda_version(cp.cuda.runtime.driverGetVersion()),
        "cupy": cp.__version__,
    }


def collect_metadata(*, quick: bool, runs: int = 1) -> dict[str, Any]:
    commit, dirty = _git_commit()
    method = (
        "median wall-clock time after one warm-up call, repeated within a time budget; "
        f"re-measured (up to {MAX_ATTEMPTS} attempts) when the median exceeds {NOISE_RATIO}x the "
        "fastest sample; GPU timings synchronise the device before the clock stops; 'gpu' "
        "starts from device-resident data, 'gpu_e2e' includes host<->device copies"
    )
    if runs > 1:
        method += f"; the suite ran {runs} complete passes and each time is their median (all in 'ms_runs')"
    return {
        "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "git_commit": commit,
        "git_dirty": dirty,
        "zkgpu": _version("zk-gpu-accelerator"),
        "python": platform.python_version(),
        "numpy": _version("numpy"),
        "os": platform.platform(),
        "cpu": _cpu_name(),
        "cpu_logical_cores": os.cpu_count(),
        "gpu": _gpu_info(),
        "gpu_unavailable_reason": GPU_UNAVAILABLE_REASON,
        "quick": quick,
        "runs": runs,
        "method": method,
    }
