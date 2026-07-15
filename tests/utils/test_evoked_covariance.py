"""Evoked covariance stays on the native MNE path."""

import mne
import numpy as np
import pytest

from mne_denoise.dss import BandpassBias, DSS
from mne_denoise.dss.utils import compute_evoked_covariance


def test_compute_evoked_covariance_returns_named_mne_covariance():
    rng = np.random.default_rng(8)
    info = mne.create_info(["Fz", "Cz", "Pz"], 200.0, "eeg")
    evoked = mne.EvokedArray(rng.standard_normal((3, 400)), info, tmin=-0.2)
    covariance = compute_evoked_covariance(evoked, method="empirical", verbose=False)
    assert isinstance(covariance, mne.Covariance)
    assert covariance.ch_names == evoked.ch_names
    np.testing.assert_allclose(covariance.data, covariance.data.T)


def test_dss_evoked_fit_uses_native_covariance_and_preserves_shape():
    rng = np.random.default_rng(18)
    info = mne.create_info(["Fz", "Cz", "Pz"], 250.0, "eeg")
    evoked = mne.EvokedArray(rng.standard_normal((3, 500)), info, tmin=-0.1)
    model = DSS(
        BandpassBias((8.0, 12.0), info["sfreq"]),
        return_type="raw",
        normalize_input=False,
        verbose=False,
    ).fit(evoked)
    cleaned = model.transform(evoked)
    assert isinstance(cleaned, mne.Evoked)
    assert cleaned.data.shape == evoked.data.shape


def test_compute_evoked_covariance_rejects_one_sample():
    info = mne.create_info(["Cz"], 100.0, "eeg")
    evoked = mne.EvokedArray(np.ones((1, 1)), info)
    with pytest.raises(ValueError, match="at least 2"):
        compute_evoked_covariance(evoked)
