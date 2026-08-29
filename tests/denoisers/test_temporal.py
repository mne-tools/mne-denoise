"""Unit tests for temporal denoisers."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mne_denoise.dss.denoisers.temporal import (
    DCTDenoiser,
    LagAverageBias,
    SmoothingBias,
)


def test_lag_average_3d_never_crosses_epoch_boundaries():
    """Every epoch is lagged independently along its time axis."""
    data = np.array([[[0, 100], [1, 101], [2, 102], [3, 103], [4, 104], [5, 105]]])

    biased = LagAverageBias(lags=[1]).apply(data)

    np.testing.assert_array_equal(biased[0, :, 0], [0, 2, 3, 4, 5, 0])
    np.testing.assert_array_equal(biased[0, :, 1], [0, 102, 103, 104, 105, 0])
    assert np.issubdtype(biased.dtype, np.floating)


def test_lag_average_lags_and_weighting():
    """Explicit lags and inverse-lag weighting have the documented algebra."""
    data = np.arange(20.0)[np.newaxis, :]
    uniform = LagAverageBias(lags=np.array([1, 2])).apply(data)
    inverse = LagAverageBias(lags=np.array([1, 2]), weighting="inverse_lag").apply(data)

    expected_uniform = (data[:, 3:19] + data[:, 4:20]) / 2
    expected_inverse = (data[:, 3:19] + 0.5 * data[:, 4:20]) / 1.5
    assert_allclose(uniform[:, 2:18], expected_uniform)
    assert_allclose(inverse[:, 2:18], expected_inverse)


def test_unknown_weighting_error():
    """Reject unknown weighting rules."""
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, (2, 100))

    bias = LagAverageBias(lags=5, weighting="unknown")
    with pytest.raises(ValueError, match="weighting must be"):
        bias.apply(data)


def test_invalid_lags():
    """Reject empty or non-operational lag declarations."""
    for _label, lags in (
        ("zero", 0),
        ("negative", -1),
        ("empty", []),
        ("zero list", [0]),
    ):
        with pytest.raises(ValueError, match="lags must"):
            LagAverageBias(lags=lags).apply(np.ones((2, 20)))


def test_non_integer_lags():
    """Reject booleans and fractional lags."""
    for _label, lags in (
        ("boolean", True),
        ("fractional", [1.5]),
        ("boolean list", [False, 1]),
    ):
        with pytest.raises(TypeError, match="lags must"):
            LagAverageBias(lags=lags).apply(np.ones((2, 20)))


def test_shift_too_large_error():
    """Test LagAverageBias raises error when lag is too large."""
    data = np.ones((2, 20))

    bias = LagAverageBias(lags=15)  # Too large for data length 20
    with pytest.raises(ValueError, match="too large"):
        bias.apply(data)


def test_data_too_short_error():
    """Test LagAverageBias raises error when data is too short."""
    data = np.ones((2, 10))

    bias = LagAverageBias(lags=5)
    with pytest.raises(ValueError, match="too short|too large"):
        bias.apply(data)


def test_autocorrelated_signal_preserved():
    """Test LagAverageBias preserves autocorrelated signals."""
    n_times = 500
    times = np.arange(n_times) / 100

    # Slow signal (high autocorrelation)
    slow_signal = np.sin(2 * np.pi * 0.5 * times)  # 0.5 Hz

    # Fast noise (low autocorrelation)
    rng = np.random.default_rng(42)
    fast_noise = rng.normal(0, 0.1, n_times)

    data = (slow_signal + fast_noise)[np.newaxis, :]

    bias = LagAverageBias(lags=10)
    biased = bias.apply(data)

    # Slow signal should be better correlated with biased output
    # (ignoring edges where padding is zero)
    center = slice(50, -50)
    corr = np.corrcoef(biased[0, center], slow_signal[center])[0, 1]
    assert corr > 0.9, f"Autocorrelated signal should be preserved (corr={corr:.3f})"


def test_smoothing_reduces_variance():
    """Test that smoothing reduces signal variance."""
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, (2, 500))

    bias = SmoothingBias(window=20)
    biased = bias.apply(data)

    # Smoothed data should have lower variance
    orig_var = np.var(data)
    smooth_var = np.var(biased)
    assert smooth_var < orig_var, "Smoothing should reduce variance"


def test_slow_signal_preserved():
    """Test that slow signals are preserved by smoothing."""
    n_times = 500
    times = np.arange(n_times) / 100

    # Slow signal
    slow_signal = np.sin(2 * np.pi * 0.5 * times)

    # Fast noise
    rng = np.random.default_rng(42)
    fast_noise = rng.normal(0, 0.3, n_times)

    data = (slow_signal + fast_noise)[np.newaxis, :]

    bias = SmoothingBias(window=20)
    biased = bias.apply(data)

    # Slow signal should correlate highly with smoothed output
    corr = np.corrcoef(biased[0], slow_signal)[0, 1]
    assert corr > 0.95, f"Slow signal should be preserved (corr={corr:.3f})"


def test_dct_denoiser():
    """Test DCTDenoiser (frequency domain filtering)."""
    # Create signal: low frequency (first few coeffs)
    n = 100
    times = np.linspace(0, 1, n)
    signal_low = np.cos(np.pi * times)  # Half-cycle cosine

    # Noise: high frequency (checkerboard)
    noise_high = np.ones(n)
    noise_high[::2] = -1

    data = signal_low + noise_high

    # DCTDenoiser cutoff=0.5 (lowpass)
    denoiser = DCTDenoiser(cutoff_fraction=0.5)
    denoised = denoiser.denoise(data)

    # Low freq should be preserved
    # High freq noise (nyquist) should be removed
    corr = np.corrcoef(denoised, signal_low)[0, 1]
    assert corr > 0.99

    # RMS check
    noise_rms = np.std(noise_high)
    residual_rms = np.std(denoised - signal_low)
    assert residual_rms < noise_rms * 0.1


def test_dct_denoiser_with_mask():
    """A custom DCT mask is applied to the coefficients without approximation."""
    from scipy.fftpack import dct, idct

    rng = np.random.default_rng(42)
    source = rng.normal(0, 1, 100)

    mask = np.zeros(100)
    mask[:20] = 1.0

    denoiser = DCTDenoiser(mask=mask)
    denoised = denoiser.denoise(source)
    expected = idct(dct(source, type=2, norm="ortho") * mask, type=2, norm="ortho")

    assert_allclose(denoised, expected)
