"""Tests for locked known-target benchmark substrates."""

import numpy as np
import pytest

from mne_denoise.benchmarks.intended import (
    locked_seed,
    reference_mixture,
    transient_mixture,
)


@pytest.mark.parametrize(
    "artifact_type",
    ["blink", "emg_burst", "electrode_pop", "low_rank_covariance_burst"],
)
def test_transient_mixture_has_exact_pairing_and_mask(artifact_type):
    mixture = transient_mixture(
        artifact_type=artifact_type,
        artifact_to_signal_db=0,
        n_channels=16,
        sfreq=250,
        duration_s=8,
        calibration_s=3,
        seed=4,
    )
    np.testing.assert_allclose(
        mixture.contaminated - mixture.clean, mixture.artifact
    )
    assert mixture.artifact_mask.any() and (~mixture.artifact_mask).any()
    assert mixture.clean.shape == (16, 2000)


def test_locked_seed_depends_on_complete_cell_and_is_stable():
    assert locked_seed(10, "blink", 0, 3) == locked_seed(10, "blink", 0, 3)
    assert locked_seed(10, "blink", 0, 3) != locked_seed(10, "blink", 0, 4)


def test_reference_score_is_disjoint_and_leakage_changes_only_fit_reference():
    common = {
        "reference_snr_db": 0,
        "reference_count": 4,
        "coupling": "stationary",
        "n_channels": 16,
        "n_times": 2000,
        "sfreq": 250,
        "seed": 9,
    }
    clean_reference = reference_mixture(neural_leakage_fraction=0.0, **common)
    leaked_reference = reference_mixture(neural_leakage_fraction=0.2, **common)
    np.testing.assert_allclose(clean_reference.clean, leaked_reference.clean)
    np.testing.assert_allclose(
        clean_reference.reference_score, leaked_reference.reference_score
    )
    assert not np.allclose(
        clean_reference.reference_fit, leaked_reference.reference_fit
    )
