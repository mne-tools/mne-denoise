"""Unit tests for spectrogram denoisers."""

import numpy as np
from scipy import signal

from mne_denoise.dss.denoisers.spectrogram import SpectrogramBias, SpectrogramDenoiser


def test_spectrogram_denoiser():
    """Test SpectrogramDenoiser for adaptive time-frequency masking."""
    sfreq = 250
    times = np.arange(1000) / sfreq

    # 1. Burst of 50 Hz oscillation in the middle
    # Center at 2s (index 500)
    burst = np.sin(2 * np.pi * 50 * times) * np.exp(-0.5 * (times - 2) ** 2 / 0.1)
    # 2. Background noise
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.1, 1000)

    data = burst + noise
    data_2d = data[:, np.newaxis]  # (n_times, 1)

    # Adaptive thresholding: should enhance the high-amplitude burst
    denoiser = SpectrogramDenoiser(threshold_percentile=95, nperseg=128)
    denoised = denoiser.denoise(data_2d)[:, 0]

    # Extract the burst part of denoised signal
    denoised_burst = denoised[400:600]
    expected_burst = burst[400:600]

    corr = np.corrcoef(denoised_burst, expected_burst)[0, 1]

    # Should have high correlation with the original burst shape
    assert corr > 0.8, f"Denoised burst should match ground truth (corr={corr:.3f})"

    # Check noise suppression in a quiet region (e.g. t=0.0 to 0.5s)
    quiet_in = data[0:100]
    quiet_out = denoised[0:100]

    rms_in = np.sqrt(np.mean(quiet_in**2))
    rms_out = np.sqrt(np.mean(quiet_out**2))
    assert rms_out < rms_in, "Denoiser should reduce RMS in quiet regions"


def test_spectrogram_bias():
    """Test SpectrogramBias (Linear) with fixed mask."""
    sfreq = 100
    times = np.arange(200) / sfreq
    n_times = len(times)

    # Create a dummy signal (2 channels)
    # Ch 0: 10 Hz sine
    # Ch 1: Random noise
    data = np.zeros((2, n_times))
    data[0] = np.sin(2 * np.pi * 10 * times)
    data[1] = np.random.randn(n_times)

    # Calculate STFT to get mask shape
    nperseg = 64
    f, t, Zxx = signal.stft(data[0], fs=sfreq, nperseg=nperseg)

    # Create a mask that keeps only ~10 Hz
    # Freq bins: f[k] = k * fs / nperseg
    # 10 Hz bin index approx: 10 * 64 / 100 = 6.4 -> bin 6 or 7
    mask = np.zeros_like(Zxx, dtype=float)
    freq_indices = np.where((f >= 8) & (f <= 12))[0]
    mask[freq_indices, :] = 1.0

    bias = SpectrogramBias(mask=mask, nperseg=nperseg)
    biased_data = bias.apply(data)

    # Check shapes
    assert biased_data.shape == data.shape

    # Verify effect:
    # Ch 0 (10 Hz signal) should be largely preserved
    # Ch 1 (White noise) should be low-pass/band-pass filtered

    energy_in_0 = np.sum(data[0] ** 2)
    energy_out_0 = np.sum(biased_data[0] ** 2)

    energy_in_1 = np.sum(data[1] ** 2)
    energy_out_1 = np.sum(biased_data[1] ** 2)

    # Ratio of preserved energy
    ratio_0 = energy_out_0 / energy_in_0
    ratio_1 = energy_out_1 / energy_in_1

    assert ratio_0 > 0.8, "Target signal should be preserved"
    assert ratio_1 < 0.3, "Broadband noise should be filtered"
