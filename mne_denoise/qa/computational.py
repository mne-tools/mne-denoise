"""Computational instrumentation — two reporting modes.

* **algorithmic-fairness** — 1 core, fixed BLAS threads, cached input, warm-up
  excluded (set the thread env *before* importing numpy; see ``record_thread_env``).
* **practical-deployment** — recommended allocation, multithread, wall time, peak
  RAM, throughput.

:class:`Timer` captures wall + CPU time and peak Python allocation (via
``tracemalloc``); resident-set memory is reported when ``psutil`` is available
(optional dependency).
"""

from __future__ import annotations

import os
import time
import tracemalloc
from dataclasses import dataclass, field


@dataclass
class Timing:
    wall_seconds: float = 0.0
    cpu_seconds: float = 0.0
    peak_python_mb: float = 0.0
    peak_rss_mb: float | None = None
    extra: dict = field(default_factory=dict)


class Timer:
    """Context manager capturing wall/CPU time and peak memory.

    >>> with Timer() as t:
    ...     _ = sum(range(1000))
    >>> t.timing.wall_seconds >= 0
    True
    """

    def __init__(self) -> None:
        self.timing = Timing()
        self._psutil = None
        try:  # optional
            import psutil  # type: ignore
            self._psutil = psutil.Process(os.getpid())
        except Exception:
            self._psutil = None

    def __enter__(self) -> "Timer":
        self._w0 = time.perf_counter()
        self._c0 = time.process_time()
        tracemalloc.start()
        if self._psutil is not None:
            self._rss0 = self._psutil.memory_info().rss
        return self

    def __exit__(self, *exc) -> None:
        cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.timing.wall_seconds = round(time.perf_counter() - self._w0, 6)
        self.timing.cpu_seconds = round(time.process_time() - self._c0, 6)
        self.timing.peak_python_mb = round(peak / 1e6, 3)
        if self._psutil is not None:
            self.timing.peak_rss_mb = round(self._psutil.memory_info().rss / 1e6, 3)


def record_thread_env() -> dict:
    """Snapshot BLAS/thread environment + cpu count for reproducible timings."""
    keys = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "SLURM_CPUS_PER_TASK")
    env = {k: os.environ.get(k) for k in keys}
    env["cpu_count"] = os.cpu_count()
    env["node"] = os.environ.get("SLURMD_NODENAME") or os.environ.get("HOSTNAME")
    return env


def throughput(n_channels: int, n_samples: int, wall_seconds: float) -> float:
    """Samples·channels processed per second (deployment-mode metric)."""
    if wall_seconds <= 0:
        return float("inf")
    return float(n_channels * n_samples / wall_seconds)
