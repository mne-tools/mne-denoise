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
    """Nineteen channels have 171 unique off-diagonal correlations."""
    module = _load_module()
    rng = np.random.default_rng(42)
    trials = rng.normal(size=(4, 19, 20))
    stream = trials.transpose(1, 0, 2).reshape(19, -1)
    features = module._connectivity_features(
        stream, 4, 20, layout="unique_upper_triangle"
    )
    assert features.shape == (4, 171)
    assert np.isfinite(features).all()
    expected = np.corrcoef(trials[0])[np.triu_indices(19, k=1)]
    np.testing.assert_allclose(features[0], expected, atol=1e-12)


def test_connectivity_features_can_follow_published_c_by_c_layout() -> None:
    """The paper declares C x C Pearson-correlation features."""
    module = _load_module()
    rng = np.random.default_rng(7)
    trials = rng.normal(size=(3, 19, 20))
    stream = trials.transpose(1, 0, 2).reshape(19, -1)
    features = module._connectivity_features(
        stream, 3, 20, layout="full_matrix_row_major"
    )
    assert features.shape == (3, 361)
    np.testing.assert_allclose(features[0].reshape(19, 19), np.corrcoef(trials[0]))


def test_paired_contrasts_preserve_subject_pairing() -> None:
    """Contrasts must be calculated within subjects, not from marginal means."""
    module = _load_module()
    rows = [
        {"subject": "A", "variant": "init", "cutoff": 10, "accuracy": 0.5},
        {"subject": "B", "variant": "init", "cutoff": 10, "accuracy": 0.8},
        {"subject": "A", "variant": "psw", "cutoff": 10, "accuracy": 0.7},
        {"subject": "B", "variant": "psw", "cutoff": 10, "accuracy": 0.7},
    ]
    config = {"processing": {"cutoffs": [10], "random_seed": 11}}
    contrasts = module._paired_contrasts(rows, config)
    psw = next(row for row in contrasts if row["variant"] == "psw")
    assert psw["n_pairs"] == 2
    assert np.isclose(psw["mean_accuracy_difference"], 0.05)
    assert psw["better_count"] == 1
    assert psw["worse_count"] == 1


def test_record_provenance_rejects_mixed_feature_protocols() -> None:
    """A merger must reject records generated under another feature layout."""
    module = _load_module()
    record = {
        "config_sha256": "config-v2",
        "prepared_manifest_hash": "manifest-v2",
        "repository_commit": "execution-commit",
        "dirty_worktree": False,
        "feature_layout": "unique_upper_triangle",
        "feature_count": 171,
    }
    errors = module._record_provenance_errors(
        record,
        config_sha256="config-v2",
        prepared_manifest_hash="manifest-v2",
        execution_commit="execution-commit",
        feature_layout="full_matrix_row_major",
        feature_count=361,
    )
    assert len(errors) == 2
    assert errors[0].startswith("feature_layout:")
    assert errors[1].startswith("feature_count:")


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


def test_v2_config_matches_published_feature_layout() -> None:
    """The corrected protocol must freeze the paper's declared C x C layout."""
    module = _load_module()
    config = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "benchmarks"
        / "asr_tsai_kaya_mi_v2.yaml"
    )
    loaded = module._load_config(config)
    assert loaded["classification"]["feature_layout"] == "full_matrix_row_major"
    assert loaded["classification"]["feature_count"] == 361
