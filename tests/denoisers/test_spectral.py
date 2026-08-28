"""Unit tests for spectral denoisers (BandpassBias, NotchBias)."""

import numpy as np
import pytest
from scipy.signal import welch

from mne_denoise.dss.denoisers.spectral import BandpassBias, LineNoiseBias


def test_bandpass_bias():
    """Test BandpassBias for rhythm extraction."""
    sfreq = 250
    times = np.arange(1000) / sfreq

    # Signal: 10 Hz Alpha
    alpha = np.sin(2 * np.pi * 10 * times)
    # Noise: 50 Hz Line
    line = np.sin(2 * np.pi * 50 * times)

    data = (alpha + line)[np.newaxis, :]

    # Extract Alpha (8-12 Hz)
    bias = BandpassBias(freq_band=(8, 12), sfreq=sfreq, order=4)
    biased_data = bias.apply(data)

    # 1. 50 Hz should be gone
    _, psd = welch(biased_data, fs=sfreq)

    corr_alpha = np.corrcoef(biased_data[0], alpha)[0, 1]
    corr_line = np.corrcoef(biased_data[0], line)[0, 1]

    assert abs(corr_alpha) > 0.95, "Bandpass should preserve target rhythm"
    assert abs(corr_line) < 0.1, "Bandpass should attenuate out-of-band noise"


def test_notch_bias_isolation():
    """Test LineNoiseBias isolates the target frequency (for later removal)."""
    sfreq = 250
    times = np.arange(1000) / sfreq

    # Main signal (slow)
    slow = np.sin(2 * np.pi * 5 * times)
    # Line noise (50 Hz)
    line = np.sin(2 * np.pi * 50 * times)

    data = (slow + line)[np.newaxis, :]

    # Bias to FIND the line noise (so we isolate 50 Hz)
    # LineNoiseBias replaces NotchBias logic for DSS
    bias = LineNoiseBias(freq=50, sfreq=sfreq, bandwidth=2.0)
    biased_data = bias.apply(data)

    # Output should correspond to the line noise, NOT the slow signal
    corr_line = np.corrcoef(biased_data[0], line)[0, 1]
    corr_slow = np.corrcoef(biased_data[0], slow)[0, 1]

    assert abs(corr_line) > 0.95, "LineNoiseBias should isolate the target frequency"
    assert abs(corr_slow) < 0.1, "LineNoiseBias should reject other frequencies"


def test_bandpass_bias_low_freq_error():
    """Test BandpassBias raises error when low freq <= 0."""
    with pytest.raises(ValueError, match="Low frequency must be > 0"):
        BandpassBias(freq_band=(0, 10), sfreq=250)

    with pytest.raises(ValueError, match="Low frequency must be > 0"):
        BandpassBias(freq_band=(-5, 10), sfreq=250)


def test_bandpass_bias_high_freq_error():
    """Test BandpassBias raises error when high freq >= Nyquist."""
    # Nyquist = 125 for sfreq=250
    with pytest.raises(ValueError, match="must be < Nyquist"):
        BandpassBias(freq_band=(10, 130), sfreq=250)


def test_bandpass_bias_unknown_method():
    """Test BandpassBias raises error for unknown method."""
    with pytest.raises(ValueError, match="Unknown filter method"):
        BandpassBias(freq_band=(8, 12), sfreq=250, method="unknown")
