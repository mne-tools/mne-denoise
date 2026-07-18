"""Tests for the Kaya adaptive-ASR replication runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "run_aasr_kaya_mi.py"
    spec = importlib.util.spec_from_file_location("run_aasr_kaya_mi", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extract_trials_uses_marker_onsets_and_chronology() -> None:
    """Hand epochs must be ordered by onset rather than marker code."""
    module = _load_module()
    data = np.arange(2 * 30, dtype=float).reshape(2, 30)
    marker = np.zeros(30, dtype=int)
    marker[3:8] = 2
    marker[15:20] = 1
    trials, labels, onsets = module._extract_trials(
        data, marker, samples_per_trial=5, codes=(1, 2)
    )
    assert labels == [2, 1]
    assert onsets == [3, 15]
    np.testing.assert_array_equal(trials[0], data[:, 3:8])
    np.testing.assert_array_equal(trials[1], data[:, 15:20])


def test_connectivity_features_are_upper_triangle_per_trial() -> None:
    """Nineteen channels must produce the paper's 171 correlations."""
    module = _load_module()
    rng = np.random.default_rng(42)
    trials = rng.normal(size=(4, 19, 20))
    stream = trials.transpose(1, 0, 2).reshape(19, -1)
    features = module._connectivity_features(stream, 4, 20)
    assert features.shape == (4, 171)
    assert np.isfinite(features).all()
    expected = np.corrcoef(trials[0])[np.triu_indices(19, k=1)]
    np.testing.assert_allclose(features[0], expected, atol=1e-12)


def test_config_is_submission_ready() -> None:
    """The paper arm must pass the generic frozen-config gate."""
    module = _load_module()
    config = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "benchmarks"
        / "asr_tsai_kaya_mi.yaml"
    )
    loaded = module._load_config(config)
    assert loaded["dataset"]["public_trial_count"] == 9224
    assert loaded["dataset"]["public_subject_count"] == 12
