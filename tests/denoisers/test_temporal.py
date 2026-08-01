"""Unit tests for lag-averaging and smoothing denoisers."""

import numpy as np
import pytest

from mne_denoise.dss.denoisers.temporal import (
    DCTDenoiser,
    LagAveragingBias,
    SmoothingBias,
    TimeShiftBias,
)


def test_lag_averaging_basic_2d():
    """Test LagAveragingBias with 2D data."""
    rng = np.random.default_rng(42)
    n_ch, n_times = 3, 200
    data = rng.normal(0, 1, (n_ch, n_times))

    bias = LagAveragingBias(shifts=5)
    biased = bias.apply(data)

    assert biased.shape == data.shape


def test_lag_averaging_basic_3d():
    """Test LagAveragingBias with 3D epoched data."""
    rng = np.random.default_rng(42)
    n_ch, n_times, n_epochs = 3, 200, 4
    data = rng.normal(0, 1, (n_ch, n_times, n_epochs))

    bias = LagAveragingBias(shifts=5)
    biased = bias.apply(data)

    assert biased.shape == data.shape
    assert biased.ndim == 3


def test_lag_averaging_shifts_as_array():
    """Test LagAveragingBias with shifts specified as an array."""
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, (2, 100))

    shifts = np.array([1, 2, 5, 10])
    bias = LagAveragingBias(shifts=shifts)
    biased = bias.apply(data)

    assert biased.shape == data.shape


def test_lag_averaging_autocorrelation_method():
    """Test LagAveragingBias with the autocorrelation method."""
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, (2, 100))

    bias = LagAveragingBias(shifts=5, method="autocorrelation")
    biased = bias.apply(data)

    assert biased.shape == data.shape


def test_lag_averaging_prediction_method():
    """Test LagAveragingBias with the prediction method."""
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, (2, 100))

    bias = LagAveragingBias(shifts=5, method="prediction")
    biased = bias.apply(data)

    assert biased.shape == data.shape


def test_lag_averaging_unknown_method_error():
    """Reject an unknown lag-averaging method at construction."""
    with pytest.raises(ValueError, match="method must be"):
        LagAveragingBias(shifts=5, method="unknown")


def test_lag_averaging_shift_too_large_error():
    """Test LagAveragingBias raises when a shift is too large."""
    data = np.ones((2, 20))

    bias = LagAveragingBias(shifts=15)  # Too large for data length 20
    with pytest.raises(ValueError, match="too large"):
        bias.apply(data)


def test_lag_averaging_data_too_short_error():
    """Test LagAveragingBias raises when data are too short."""
    data = np.ones((2, 10))

    bias = LagAveragingBias(shifts=5)
    with pytest.raises(ValueError, match="too short|too large"):
        bias.apply(data)


def test_lag_averaging_preserves_autocorrelated_signal():
    """Test LagAveragingBias preserves autocorrelated signals."""
    n_times = 500
    times = np.arange(n_times) / 100

    # Slow signal (high autocorrelation)
    slow_signal = np.sin(2 * np.pi * 0.5 * times)  # 0.5 Hz

    # Fast noise (low autocorrelation)
    rng = np.random.default_rng(42)
    fast_noise = rng.normal(0, 0.1, n_times)

    data = (slow_signal + fast_noise)[np.newaxis, :]

    bias = LagAveragingBias(shifts=10)
    biased = bias.apply(data)

    # Slow signal should be better correlated with biased output
    # (ignoring edges where padding is zero)
    center = slice(50, -50)
    corr = np.corrcoef(biased[0, center], slow_signal[center])[0, 1]
    assert corr > 0.9, f"Autocorrelated signal should be preserved (corr={corr:.3f})"


@pytest.mark.parametrize(
    "shifts, match",
    [
        (0, "positive"),
        ([], "non-empty"),
        ([1, 1], "duplicate"),
        ([0, 1], "zero"),
        ([1.5], "integer"),
    ],
)
def test_lag_averaging_rejects_ambiguous_shifts(shifts, match):
    """Lag specifications are finite, unique, non-zero integer samples."""
    with pytest.raises(ValueError, match=match):
        LagAveragingBias(shifts=shifts)


def test_lag_averaging_epochs_are_independent():
    """Lag averaging does not carry samples across epoch boundaries."""
    data = np.zeros((1, 20, 2))
    data[:, :, 0] = 1.0
    data[:, :, 1] = 100.0
    bias = LagAveragingBias(shifts=2)

    observed = bias.apply(data)

    for epoch in range(data.shape[2]):
        expected = bias.apply(data[:, :, epoch])
        np.testing.assert_allclose(observed[:, :, epoch], expected)


def test_time_shift_bias_compatibility_warning_and_parity():
    """The released name warns while retaining lag-average numerics."""
    data = np.arange(40.0).reshape(2, 20)
    canonical = LagAveragingBias(shifts=[-2, 1], method="prediction")
    with pytest.warns(FutureWarning, match="LagAveragingBias"):
        legacy = TimeShiftBias(shifts=[-2, 1], method="prediction")

    np.testing.assert_allclose(legacy.apply(data), canonical.apply(data))


def test_time_shift_dss_name_is_reserved_for_true_estimator():
    """The canonical class is distinct from the deprecated bias name."""
    import mne_denoise.dss as dss_module

    assert dss_module.TimeShiftDSS is not TimeShiftBias


def test_smoothing_basic_2d():
    """Test SmoothingBias with 2D data."""
    rng = np.random.default_rng(42)
    n_ch, n_times = 3, 200
    data = rng.normal(0, 1, (n_ch, n_times))

    bias = SmoothingBias(window=10)
    biased = bias.apply(data)

    assert biased.shape == data.shape


def test_smoothing_basic_3d():
    """Test SmoothingBias with 3D epoched data."""
    rng = np.random.default_rng(42)
    n_ch, n_times, n_epochs = 3, 200, 4
    data = rng.normal(0, 1, (n_ch, n_times, n_epochs))

    bias = SmoothingBias(window=10)
    biased = bias.apply(data)

    assert biased.shape == data.shape
    assert biased.ndim == 3


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


def test_different_window_sizes():
    """Test SmoothingBias with different window sizes."""
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, (2, 200))

    for window in [5, 10, 20, 50]:
        bias = SmoothingBias(window=window)
        biased = bias.apply(data)
        assert biased.shape == data.shape


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


def test_dct_denoiser_2d_data():
    """Test DCTDenoiser with 2D epoched data."""
    rng = np.random.default_rng(42)
    n_times, n_epochs = 100, 4
    data = rng.normal(0, 1, (n_times, n_epochs))

    denoiser = DCTDenoiser(cutoff_fraction=0.5)
    denoised = denoiser.denoise(data)

    assert denoised.shape == data.shape


def test_dct_denoiser_with_mask():
    """Test DCTDenoiser with custom mask."""
    rng = np.random.default_rng(42)
    source = rng.normal(0, 1, 100)

    # Custom mask (lowpass)
    mask = np.zeros(100)
    mask[:20] = 1.0

    denoiser = DCTDenoiser(mask=mask)
    denoised = denoiser.denoise(source)

    assert denoised.shape == source.shape


def test_dct_denoiser_mask_resampling():
    """Test DCTDenoiser resamples mask when lengths don't match."""
    rng = np.random.default_rng(42)
    source = rng.normal(0, 1, 100)

    # Mask with different length
    mask = np.ones(50)  # Will be resampled to 100
    mask[25:] = 0.5

    denoiser = DCTDenoiser(mask=mask)
    denoised = denoiser.denoise(source)

    assert denoised.shape == source.shape


def test_dct_denoiser_cached_mask():
    """Test DCTDenoiser uses cached mask correctly."""
    rng = np.random.default_rng(42)

    denoiser = DCTDenoiser(cutoff_fraction=0.3)

    # First call - creates cache
    source1 = rng.normal(0, 1, 100)
    denoised1 = denoiser.denoise(source1)

    # Second call - should use cache
    source2 = rng.normal(0, 1, 100)
    denoised2 = denoiser.denoise(source2)

    assert denoised1.shape == source1.shape
    assert denoised2.shape == source2.shape


def test_dct_denoiser_invalid_ndim():
    """Test DCTDenoiser raises error for 3D data."""
    denoiser = DCTDenoiser()
    data = np.zeros((10, 10, 10))

    with pytest.raises(ValueError, match="must be 1D or 2D"):
        denoiser.denoise(data)
