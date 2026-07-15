"""Tests for the explicit joint multi-sensor DSS path."""

import mne
import numpy as np
import pytest

from mne_denoise.dss import DSS, BandpassBias


def _mixed_raw(seed=0, n_times=1200, sfreq=200.0):
    rng = np.random.default_rng(seed)
    time = np.arange(n_times) / sfreq
    target = np.sin(2 * np.pi * 10 * time)
    types = ["mag"] * 3 + ["grad"] * 3 + ["eeg"] * 3
    scales = np.array([1e-12] * 3 + [1e-11] * 3 + [1e-5] * 3)
    data = (
        np.outer(rng.normal(size=len(types)), target)
        + 0.25 * rng.normal(size=(len(types), n_times))
    ) * scales[:, None]
    info = mne.create_info([f"C{index}" for index in range(len(types))], sfreq, types)
    return mne.io.RawArray(data, info, verbose=False), target, data


def test_multimodal_whitening_uses_every_data_channel_and_reconstructs():
    raw, target, data = _mixed_raw()
    model = DSS(
        BandpassBias((8, 12), raw.info["sfreq"]),
        whiten=True,
        return_type="raw",
        verbose=False,
    ).fit(raw)
    reconstructed = model.transform(raw).get_data()
    assert model.filters_.shape == (9, 9)
    assert np.linalg.norm(reconstructed - data) / np.linalg.norm(data) < 1e-6
    model.return_type = "sources"
    sources = model.transform(raw)
    assert abs(np.corrcoef(sources[0], target)[0, 1]) > 0.9


def test_multimodal_whitening_noise_cov_requires_named_input():
    raw, _, data = _mixed_raw()
    covariance = mne.Covariance(
        np.eye(9), raw.ch_names, [], [], nfree=raw.n_times
    )
    with pytest.raises(ValueError, match="named channels"):
        DSS(
            BandpassBias((8, 12), raw.info["sfreq"]),
            whiten=True,
            noise_cov=covariance,
        ).fit(data)
