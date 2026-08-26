"""Regression tests for the frozen ASR benchmark runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_runner():
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "run_asr_transient_ground_truth.py"
    )
    spec = importlib.util.spec_from_file_location("asr_ground_truth_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_method_failure_is_an_accepted_benchmark_outcome():
    runner = _load_runner()

    rows = [
        {"method": "asr", "status": "success"},
        {
            "method": "juggler_asr",
            "status": "failed",
            "error": "ValueError: insufficient selected calibration samples",
        },
    ]

    assert runner._runner_exit_status(rows) == 0


def test_missing_dependency_still_fails_closed():
    runner = _load_runner()

    assert runner._runner_exit_status([]) == 1
    assert runner._runner_exit_status(
        [{"method": "wavelet_threshold", "status": "unavailable_dependency"}]
    ) == 1
    assert runner._runner_exit_status([{"method": "asr", "status": "unknown"}]) == 1
