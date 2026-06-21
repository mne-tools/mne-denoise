import numpy as np
import pytest


@pytest.fixture()
def rng():
    """Shared deterministic RNG."""
    return np.random.default_rng(42)


@pytest.fixture()
def synthetic_burst_data(rng):
    """Create multichannel EEG-like data with spatial burst artifacts."""
    sfreq = 250.0
    duration = 12.0
    n_times = int(sfreq * duration)
    n_channels = 8
    t = np.arange(n_times) / sfreq

    brain = np.zeros((n_channels, n_times), dtype=np.float64)
    for ch in range(n_channels):
        phase = rng.uniform(0, 2 * np.pi)
        brain[ch] = (
            0.5 * np.sin(2 * np.pi * 10 * t + phase)
            + 0.2 * np.sin(2 * np.pi * 6 * t + 0.5 * phase)
            + 0.05 * rng.standard_normal(n_times)
        )

    data = brain.copy()
    spatial = rng.standard_normal((n_channels, 2))
    spatial /= np.linalg.norm(spatial, axis=0, keepdims=True)
    burst_mask = np.zeros(n_times, dtype=bool)
    for onset, stop in ((4.0, 4.8), (8.0, 8.6)):
        start_samp = int(onset * sfreq)
        stop_samp = int(stop * sfreq)
        burst_mask[start_samp:stop_samp] = True
        source = rng.standard_normal((2, stop_samp - start_samp)) * 8.0
        data[:, start_samp:stop_samp] += spatial @ source

    return data, brain, burst_mask, sfreq
