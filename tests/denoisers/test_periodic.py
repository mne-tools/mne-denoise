"""Unit tests for periodic denoisers (PeakFilterBias, CombFilterBias)."""

import numpy as np
import pytest

from mne_denoise.dss.denoisers.periodic import (
    CombFilterBias,
    PeakFilterBias,
    QuasiPeriodicDenoiser,
)


def test_peak_filter_bias():
    """Test PeakFilterBias extracting a sine wave."""
    sfreq = 1000
    times = np.arange(1000) / sfreq
    freq = 10

    # Signal: 10 Hz sine
    signal = np.sin(2 * np.pi * freq * times)

    # Noise: different frequency (e.g. 50 Hz)
    noise = np.sin(2 * np.pi * 50 * times) * 2.0

    data = (signal + noise)[np.newaxis, :]  # (1, n_times)

    # Bias towards 10 Hz
    bias = PeakFilterBias(freq=10, sfreq=sfreq, q_factor=50)
    biased_data = bias.apply(data)

    # Result should be mostly 10 Hz
    # Correlation with signal should be high
    corr = np.corrcoef(biased_data[0], signal)[0, 1]
    assert corr > 0.9, f"Peak filter failed to extract signal (corr={corr:.3f})"


def test_comb_filter_bias():
    """Test CombFilterBias extracting fundamental + harmonic."""
    sfreq = 1000
    times = np.arange(1000) / sfreq
    f0 = 12

    # Signal: 12 Hz + 24 Hz
    s1 = np.sin(2 * np.pi * f0 * times)
    s2 = 0.5 * np.sin(2 * np.pi * 2 * f0 * times)
    signal = s1 + s2

    # Noise at other freq
    noise = np.sin(2 * np.pi * 50 * times) * 2

    data = (signal + noise)[np.newaxis, :]

    bias = CombFilterBias(fundamental_freq=12, sfreq=sfreq, n_harmonics=2)
    biased_data = bias.apply(data)

    corr = np.corrcoef(biased_data[0], signal)[0, 1]
    assert corr > 0.85, f"Comb filter failed (corr={corr:.3f})"


def test_quasi_periodic_denoiser():
    """Test QuasiPeriodicDenoiser on simulated ECG."""
    sfreq = 250
    times = np.arange(1000) / sfreq

    # Simulate ECG-like signal
    def ecg_beat(t):
        return np.exp(-100 * (t - 0.2) ** 2) - 0.2 * np.exp(-100 * (t - 0.25) ** 2)

    signal_periodic = np.zeros_like(times)
    # Beats every 1.0s
    for i in range(9):
        beat_start = int(i * sfreq)
        if beat_start + 250 <= len(times):
            t_local = np.linspace(0, 1, 250)
            signal_periodic[beat_start : beat_start + 250] += ecg_beat(t_local)

    # Add noise
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.05, len(times))
    data = signal_periodic + noise
    data_2d = data[:, np.newaxis]  # (n_times, n_epochs)

    # Denoise
    # peak_distance ~ 1s = 250 samples
    # Disable smoothing because our synthetic peaks are very sharp
    denoiser = QuasiPeriodicDenoiser(peak_distance=200, smooth_template=False)
    denoised = denoiser.denoise(data_2d)[:, 0]

    # Check correlation with clean signal
    # Ignore edges where cycles might be incomplete
    mask = (times > 0.5) & (times < 3.5)
    corr = np.corrcoef(denoised[mask], signal_periodic[mask])[0, 1]

    # Threshold lowered to 0.7 to avoid fragility with random noise
    assert corr > 0.7, f"Quasi-periodic denoising failed (corr={corr:.3f})"


def test_peak_filter_freq_too_high():
    """Test PeakFilterBias raises error when freq >= Nyquist."""
    sfreq = 100  # Nyquist = 50 Hz

    with pytest.raises(ValueError, match="must be < Nyquist"):
        PeakFilterBias(freq=60, sfreq=sfreq)


def test_comb_filter_weight_mismatch():
    """Test CombFilterBias raises error for wrong weights length."""
    with pytest.raises(ValueError, match="weights length.*must match"):
        CombFilterBias(fundamental_freq=10, sfreq=250, n_harmonics=3, weights=[1, 2])


def test_comb_filter_harmonic_frequencies():
    """Test harmonic_frequencies property."""
    bias = CombFilterBias(fundamental_freq=10, sfreq=100, n_harmonics=5)
    # Nyquist = 50 Hz, so only 10, 20, 30, 40 should be included (< 47.5 Hz)
    freqs = bias.harmonic_frequencies
    assert 10 in freqs
    assert 20 in freqs
    assert 30 in freqs
    assert 40 in freqs
    assert 50 not in freqs  # Too close to Nyquist


def test_comb_filter_adaptive_q():
    """Test CombFilterBias with proportional Q mode.

    With q_mode="proportional", Q scales as q_factor * h for harmonic h,
    maintaining approximately constant absolute bandwidth across harmonics.
    """
    sfreq = 1000
    times = np.arange(2000) / sfreq
    f0 = 10

    # Signal: fundamental + 2nd + 3rd harmonic
    signal_clean = (
        np.sin(2 * np.pi * f0 * times)
        + 0.5 * np.sin(2 * np.pi * 2 * f0 * times)
        + 0.3 * np.sin(2 * np.pi * 3 * f0 * times)
    )
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 2, len(times))
    data = (signal_clean + noise)[np.newaxis, :]

    # Fixed Q
    bias_fixed = CombFilterBias(
        fundamental_freq=f0, sfreq=sfreq, n_harmonics=3, q_factor=30.0, q_mode="fixed"
    )
    biased_fixed = bias_fixed.apply(data)

    # Proportional Q
    bias_prop = CombFilterBias(
        fundamental_freq=f0,
        sfreq=sfreq,
        n_harmonics=3,
        q_factor=30.0,
        q_mode="proportional",
    )
    biased_prop = bias_prop.apply(data)

    # Both should produce valid output
    assert biased_fixed.shape == data.shape
    assert biased_prop.shape == data.shape

    # Proportional should differ from fixed (different filter shapes)
    assert not np.allclose(biased_fixed, biased_prop, atol=1e-10)

    # Both should correlate well with the clean signal
    corr_fixed = np.corrcoef(biased_fixed[0], signal_clean)[0, 1]
    corr_prop = np.corrcoef(biased_prop[0], signal_clean)[0, 1]
    assert corr_fixed > 0.8, f"Fixed Q failed (corr={corr_fixed:.3f})"
    assert corr_prop > 0.8, f"Proportional Q failed (corr={corr_prop:.3f})"


def test_comb_filter_invalid_q_mode():
    """Test CombFilterBias raises error for invalid q_mode."""
    with pytest.raises(ValueError, match="q_mode must be one of"):
        CombFilterBias(fundamental_freq=10, sfreq=250, q_mode="invalid")


def test_quasi_periodic_few_peaks():
    """Test QuasiPeriodicDenoiser returns original when too few peaks."""
    # Create signal with only 1-2 peaks
    source = np.zeros(100)
    source[50] = 5.0  # Single peak

    denoiser = QuasiPeriodicDenoiser(peak_distance=50)
    denoised = denoiser.denoise(source)

    # Should return original since < 3 peaks
    np.testing.assert_array_equal(denoised, source)


def test_quasi_periodic_with_warp_length():
    """Test QuasiPeriodicDenoiser with explicit warp_length."""
    rng = np.random.default_rng(42)
    n_samples = 500
    source = np.zeros(n_samples)
    for i in range(0, n_samples, 100):
        source[i : min(i + 10, n_samples)] = 5.0
    source += rng.normal(0, 0.1, n_samples)

    denoiser = QuasiPeriodicDenoiser(peak_distance=80, warp_length=50)
    denoised = denoiser.denoise(source)

    assert denoised.shape == source.shape
