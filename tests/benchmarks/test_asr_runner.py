"""Regression tests for the frozen ASR benchmark runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


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


@pytest.mark.parametrize(
    "method,attribute,value",
    [
        ("asr_standard", "method", "standard"),
        ("asr_standard_auto", "calibration", "auto"),
        ("rasr_windowed", "method", "riemannian_windowed"),
        ("rasr_windowed_auto", "calibration", "auto"),
        ("rasr_legacy", "method", "riemannian"),
        ("adaptive_psp", "variant", "psp"),
        ("adaptive_psw", "variant", "psw"),
        ("adaptive_mw_final_state", "mw_mode", "final_state"),
        ("adaptive_mw_sliding", "mw_mode", "sliding"),
        ("juggler_dbscan", "strategy", "dbscan"),
        ("juggler_gev", "strategy", "gev"),
    ],
)
def test_explicit_asr_variant_factory(method, attribute, value):
    runner = _load_runner()

    model = runner._method(method, "blink", 250.0, 20.0)

    assert getattr(model, attribute) == value


class _RecordingModel:
    def __init__(self):
        self.blocksize = 1
        self.fit_inputs = []
        self.update_inputs = []
        self.transform_inputs = []
        self.fit_transform_inputs = []
        self.reset_count = 0

    def fit(self, data):
        self.fit_inputs.append(np.asarray(data).copy())
        return self

    def partial_fit(self, data):
        self.update_inputs.append(np.asarray(data).copy())
        return self

    def transform(self, data):
        self.transform_inputs.append(np.asarray(data).copy())
        return np.asarray(data).copy()

    def fit_transform(self, data):
        self.fit_transform_inputs.append(np.asarray(data).copy())
        return np.asarray(data).copy()

    def reset_process_state(self):
        self.reset_count += 1

    def get_diagnostics(self):
        return {"sample_mask": np.zeros(25, dtype=bool)}


def _mixture():
    clean = np.tile(np.linspace(-1.0, 1.0, 25), (2, 1))
    contaminated = clean.copy()
    contaminated[:, :5] += 2.0
    mask = np.zeros(25, dtype=bool)
    mask[:5] = True
    return SimpleNamespace(
        clean=clean,
        contaminated=contaminated,
        calibration=clean[:, :10],
        artifact_mask=mask,
    )


def test_adaptive_variant_score_updates_then_reconstructs_final_state():
    runner = _load_runner()
    model = _RecordingModel()
    mixture = _mixture()

    _, metrics = runner._score(
        model,
        mixture,
        "adaptive_psw",
        1.0,
        adaptive_update_chunk_s=10.0,
    )

    np.testing.assert_array_equal(model.fit_inputs[0], mixture.calibration)
    assert [chunk.shape[1] for chunk in model.update_inputs] == [10, 10, 5]
    assert model.reset_count == 1
    np.testing.assert_array_equal(model.transform_inputs[0], mixture.contaminated)
    assert metrics["status"] == "success"


@pytest.mark.parametrize(
    "method,entrypoint",
    [
        ("asr_standard_auto", "fit"),
        ("juggler_gev", "fit"),
        ("adaptive_mw_sliding", "fit_transform"),
    ],
)
def test_recording_local_variants_calibrate_on_target(method, entrypoint):
    runner = _load_runner()
    model = _RecordingModel()
    mixture = _mixture()

    runner._score(model, mixture, method, 1.0)

    inputs = getattr(model, f"{entrypoint}_inputs")
    np.testing.assert_array_equal(inputs[0], mixture.contaminated)
