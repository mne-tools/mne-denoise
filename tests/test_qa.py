"""Tests for mne_denoise.qa module."""

from unittest.mock import MagicMock

import numpy as np
import pytest

import mne_denoise.qa as qa
from mne_denoise.qa import (
    below_noise_distortion_db,
    compute_all_qa_metrics,
    geometric_mean_psd_ratio,
    noise_surround_ratio,
    overclean_proportion,
    peak_attenuation_db,
    spectral_distortion,
    suppression_ratio,
    underclean_proportion,
    variance_removed,
)

# --- peak_attenuation_db ---


def test_peak_attenuation_1d_basic():
    """Peak is halved → ~3 dB attenuation."""
    freqs = np.arange(0, 100, 0.5)
    psd_before = np.ones_like(freqs) * 0.01
    psd_after = np.ones_like(freqs) * 0.01
    # Add peak at 50 Hz
    mask = (freqs >= 48) & (freqs <= 52)
    psd_before[mask] = 1.0
    psd_after[mask] = 0.5
    result = peak_attenuation_db(freqs, psd_before, psd_after, 50.0)
    assert isinstance(result, (float, np.floating))
    expected = 10 * np.log10(1.0 / 0.5)
    np.testing.assert_allclose(result, expected, atol=0.01)


def test_peak_attenuation_2d_input():
    """Test with (n_channels, n_freqs) shaped PSD."""
    freqs = np.arange(0, 100, 0.5)
    n_ch = 4
    psd_before = np.ones((n_ch, len(freqs))) * 0.01
    psd_after = np.ones((n_ch, len(freqs))) * 0.01
    mask = (freqs >= 48) & (freqs <= 52)
    psd_before[:, mask] = 1.0
    psd_after[:, mask] = 0.1
    result = peak_attenuation_db(freqs, psd_before, psd_after, 50.0)
    assert result.shape == (n_ch,)
    expected = 10 * np.log10(1.0 / 0.1)
    np.testing.assert_allclose(result, expected, atol=0.01)


def test_peak_attenuation_out_of_range():
    """Target outside frequency range → NaN (1D) or array of NaN (2D)."""
    freqs = np.arange(0, 50, 0.5)
    psd_before = np.ones_like(freqs)
    psd_after = np.ones_like(freqs)

    # 1D
    assert np.isnan(peak_attenuation_db(freqs, psd_before, psd_after, 100.0))

    # 2D
    psd_before_2d = np.ones((3, len(freqs)))
    psd_after_2d = np.ones((3, len(freqs)))
    result = peak_attenuation_db(freqs, psd_before_2d, psd_after_2d, 100.0)
    assert result.shape == (3,)
    assert np.all(np.isnan(result))


def test_noise_surround_ratio_basic():
    """Flat spectrum → ratio ≈ 1."""
    freqs = np.arange(0, 100, 0.5)
    psd = np.ones((4, len(freqs)))
    result = noise_surround_ratio(freqs, psd, 50.0)
    np.testing.assert_allclose(result, 1.0, atol=0.01)


def test_below_noise_distortion_db_basic():
    """Same PSD → 0 dB distortion."""
    freqs = np.arange(0, 100, 0.5)
    psd = np.ones((3, len(freqs)))
    result = below_noise_distortion_db(freqs, psd, psd)
    np.testing.assert_allclose(result, 0.0, atol=1e-10)


def test_overclean_proportion_basic():
    """Identical spectral floor → proportion = 0."""
    freqs = np.arange(0, 100, 0.5)
    psd = np.ones((5, len(freqs)))
    result = overclean_proportion(freqs, psd, psd, 50.0)
    assert result == pytest.approx(0.0)


def test_underclean_proportion_basic():
    """Flat spectrum → no channels under-cleaned."""
    freqs = np.arange(0, 100, 0.5)
    psd = np.ones((5, len(freqs)))
    result = underclean_proportion(freqs, psd, 50.0)
    assert result == pytest.approx(0.0)


def test_geometric_mean_psd_ratio_basic():
    """Same PSD → ratio = 1."""
    freqs = np.arange(0, 100, 0.5)
    psd = np.ones((3, len(freqs)))
    result = geometric_mean_psd_ratio(freqs, psd, psd)
    np.testing.assert_allclose(result, 1.0, atol=1e-6)


def test_compute_all_qa_metrics_mock():
    """Test the main entry point with mocks."""
    raw_b = MagicMock()
    raw_a = MagicMock()
    psd_obj = MagicMock()
    psd_obj.freqs = np.linspace(0, 125, 250)
    psd_obj.get_data.return_value = np.ones((4, 250))
    raw_b.compute_psd.return_value = psd_obj
    raw_a.compute_psd.return_value = psd_obj

    metrics = compute_all_qa_metrics(raw_b, raw_a, line_freq=50.0)
    assert "peak_attenuation_db" in metrics
    assert "R_f0" in metrics
    assert "below_noise_distortion_db" in metrics
    assert "below_noise_pct" not in metrics
    assert "overclean_proportion" in metrics
    assert "underclean_proportion" in metrics


def test_suppression_ratio_basic():
    """Test suppression_ratio with 1D and 2D input."""
    freqs = np.arange(0, 100, 0.5)
    psd_before = np.ones_like(freqs)
    psd_after = np.ones_like(freqs) * 0.1
    # Average before=1.0, after=0.1 -> Ratio=10 -> 10 dB
    res = suppression_ratio(freqs, psd_before, psd_after, 50.0)
    np.testing.assert_allclose(res, 10.0)


def test_suppression_ratio_out_of_range():
    """Target frequency outside sampled range returns NaN."""
    freqs = np.arange(0, 50, 0.5)
    psd = np.ones_like(freqs)
    assert np.isnan(suppression_ratio(freqs, psd, psd, 100.0))


def test_suppression_ratio_full_suppression():
    """Zero after-band power returns +inf."""
    freqs = np.arange(0, 100, 0.5)
    psd_before = np.ones_like(freqs)
    psd_after = np.zeros_like(freqs)
    assert suppression_ratio(freqs, psd_before, psd_after, 50.0) == np.inf


def test_spectral_distortion_basic():
    """Identical PSDs outside harmonics → 0 distortion."""
    freqs = np.arange(0, 200, 0.5)
    psd = np.ones((2, len(freqs)))
    # Exclude harmonics of 50 Hz
    res = spectral_distortion(freqs, psd, psd, line_freq=50.0, n_harmonics=3)
    np.testing.assert_allclose(res, 0.0, atol=1e-10)


def test_spectral_distortion_no_safe_freqs():
    """If all bins are excluded, distortion falls back to 0."""
    freqs = np.linspace(48, 52, 10)
    psd = np.ones_like(freqs)
    res = spectral_distortion(freqs, psd, psd, line_freq=50.0, bandwidth=5.0)
    assert res == 0.0


def test_variance_removed_basic():
    """Test basic variance removal calculation."""
    x = np.array([1.0, -1.0, 1.0, -1.0])
    # Halving amplitude -> 1/4 variance -> 75% removed
    assert variance_removed(x, 0.5 * x) == 75.0


def test_variance_removed_zero_variance_before():
    """Zero baseline variance returns 0.0 by design."""
    x = np.zeros(10)
    assert variance_removed(x, x) == 0.0


def test_reexport_matches():
    """Ensure direct import matches the module attribute."""
    assert peak_attenuation_db is qa.peak_attenuation_db
