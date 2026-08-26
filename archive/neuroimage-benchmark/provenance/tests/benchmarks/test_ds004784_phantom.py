"""Unit tests for the ds004784 locked-replication metrics."""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np

SCRIPT = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "run_ds004784_phantom.py"
SPEC = importlib.util.spec_from_file_location("run_ds004784_phantom", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_corrected_dqs_identity_has_unit_correction():
    rng = np.random.default_rng(13)
    brain = rng.standard_normal((10, 2000))
    mixing = rng.standard_normal((16, 10))
    raw = mixing @ brain + 0.01 * rng.standard_normal((16, brain.shape[1]))
    score, uncorrected, correction = MODULE._corrected_data_quality_score(raw, raw, brain)
    assert np.isclose(correction, 1.0)
    assert np.isclose(score, uncorrected)


def test_corrected_dqs_penalizes_destroyed_brain_subspace():
    rng = np.random.default_rng(27)
    brain = rng.standard_normal((4, 2500))
    # The production metric uses ten brain sources; append independent ones.
    brain = np.vstack([brain, rng.standard_normal((6, brain.shape[1]))])
    raw = rng.standard_normal((12, 10)) @ brain
    destroyed = np.repeat(raw.mean(axis=1, keepdims=True), raw.shape[1], axis=1)
    score, _, correction = MODULE._corrected_data_quality_score(destroyed, raw, brain)
    assert correction < 1e-8
    assert abs(score) < 1e-8


def test_lagged_reference_does_not_wrap_boundaries():
    reference = np.arange(5.0)[None]
    lagged = MODULE._lagged(reference, 1)
    np.testing.assert_array_equal(lagged[0], [1, 2, 3, 4, 0])
    np.testing.assert_array_equal(lagged[1], [0, 1, 2, 3, 4])
    np.testing.assert_array_equal(lagged[2], [0, 0, 1, 2, 3])
