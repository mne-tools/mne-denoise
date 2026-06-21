"""Computational instrumentation tests."""

from mne_denoise.qa import computational as comp


def test_timer_captures_time_and_memory():
    with comp.Timer() as t:
        _ = [i * i for i in range(10000)]
    assert t.timing.wall_seconds >= 0.0
    assert t.timing.cpu_seconds >= 0.0
    assert t.timing.peak_python_mb >= 0.0


def test_record_thread_env_keys():
    env = comp.record_thread_env()
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "cpu_count"):
        assert k in env


def test_throughput():
    assert comp.throughput(32, 100000, 1.0) == 32 * 100000
    assert comp.throughput(1, 1, 0.0) == float("inf")
