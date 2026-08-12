"""Fixtures shared by SSA tests."""

import numpy as np
import pytest


@pytest.fixture()
def rng():
    """Shared random generator."""
    return np.random.default_rng(0)


@pytest.fixture()
def drift_data(rng):
    """Synthetic EEG with a strong slow drift and alpha rhythm."""
    sfreq = 250.0
    n_times = 2000
    n_channels = 6
    times = np.arange(n_times) / sfreq
    data = np.empty((n_channels, n_times))
    for channel in range(n_channels):
        drift = 4.0 * np.sin(2 * np.pi * 0.4 * times + rng.uniform(0, 2 * np.pi))
        alpha = np.sin(2 * np.pi * 10.0 * times + rng.uniform(0, 2 * np.pi))
        data[channel] = drift + alpha + 0.05 * rng.standard_normal(n_times)
    return data, sfreq
