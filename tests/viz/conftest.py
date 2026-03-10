"""Shared fixtures for visualization tests."""

import mne
import numpy as np
import pytest

from mne_denoise.dss import DSS
from mne_denoise.zapline import ZapLine


@pytest.fixture(scope="module")
def synthetic_data():
    """Create synthetic epochs with signal and noise."""
    info = mne.create_info(["Fz", "Cz", "Pz", "Oz", "F3"], 100.0, "eeg")
    info.set_montage("standard_1020")

    n_epochs = 10
    n_times = 200
    data = np.random.randn(n_epochs, 5, n_times)

    t = np.linspace(0, 2, n_times)
    signal = np.sin(2 * np.pi * 10 * t)
    data[:, 0:3, :] += signal * 0.5

    return mne.EpochsArray(data, info)


@pytest.fixture(scope="module")
def fitted_dss(synthetic_data):
    """Return a fitted DSS estimator."""

    def bias_func(d):
        return mne.filter.filter_data(d, 100.0, 8, 12, verbose=False)

    dss = DSS(n_components=3, bias=bias_func, return_type="epochs")
    dss.fit(synthetic_data)
    return dss


@pytest.fixture(scope="module")
def zapline_data():
    """Create synthetic data with line noise for ZapLine testing."""
    rng = np.random.default_rng(42)
    sfreq = 500
    n_channels = 8
    n_times = 2500
    t = np.arange(n_times) / sfreq

    data = rng.standard_normal((n_channels, n_times)) * 0.5
    line_noise = 2.0 * np.sin(2 * np.pi * 50 * t)
    for i in range(n_channels):
        data[i] += line_noise * (i + 1) / n_channels

    return data, sfreq


@pytest.fixture(scope="module")
def fitted_zapline(zapline_data):
    """Return a fitted ZapLine estimator."""
    data, sfreq = zapline_data
    zapline = ZapLine(sfreq=sfreq, line_freq=50.0, n_remove=2)
    zapline.fit(data)
    return zapline
