"""Scientific contracts for :mod:`mne_denoise.qa`."""

import mne
import numpy as np
import pytest

from mne_denoise.qa import (
    below_noise_distortion_db,
    channel_variance_ratio,
    compute_all_qa_metrics,
    geometric_mean_psd_ratio,
    max_abs_change,
    noise_surround_ratio,
    overclean_proportion,
    peak_attenuation_db,
    rms_change,
    spectral_distortion,
    suppression_ratio,
    underclean_proportion,
    variance_removed,
)


def test_peak_attenuation_matches_maximum_band_power():
    """Peak attenuation uses the maximum power in the requested band."""
    freqs = np.arange(0.0, 100.0, 0.5)
    before = np.full((2, freqs.size), 0.01)
    after = before.copy()
    band = (freqs >= 48.0) & (freqs <= 52.0)
    before[:, band] = [[1.0] * band.sum(), [2.0] * band.sum()]
    after[:, band] = [[0.5] * band.sum(), [0.25] * band.sum()]

    result = peak_attenuation_db(freqs, before, after, target_freq=50.0)
    np.testing.assert_allclose(result, 10.0 * np.log10([2.0, 8.0]))

    scalar = peak_attenuation_db(freqs, before[0], after[0], target_freq=50.0)
    assert scalar == pytest.approx(10.0 * np.log10(2.0))


def test_peak_attenuation_out_of_range_returns_nan():
    """A target with no sampled bins has an undefined attenuation."""
    freqs = np.arange(0.0, 50.0, 0.5)
    before = np.ones_like(freqs)
    after = before.copy()

    assert np.isnan(peak_attenuation_db(freqs, before, after, target_freq=100.0))
    result = peak_attenuation_db(
        freqs, before[np.newaxis, :], after[np.newaxis, :], target_freq=100.0
    )
    assert result.shape == (1,)
    assert np.isnan(result[0])


def test_suppression_ratio_matches_mean_band_power():
    """Suppression is the dB ratio of channel-averaged band powers."""
    freqs = np.arange(0.0, 100.0, 0.5)
    before = np.ones((2, freqs.size))
    after = np.full_like(before, 0.1)

    assert suppression_ratio(freqs, before, after, 50.0) == pytest.approx(10.0)
    assert suppression_ratio(freqs, before[0], after[0], 50.0) == pytest.approx(10.0)

    assert np.isnan(suppression_ratio(freqs, before, after, 103.0))
    assert suppression_ratio(freqs, before, np.zeros_like(after), 50.0) == np.inf


def test_noise_surround_ratio_and_underclean_proportion():
    """Residual peak ratios and their thresholded aggregate are consistent."""
    freqs = np.arange(40.0, 61.0)
    psd = np.ones((3, freqs.size))
    peak = (freqs >= 48.0) & (freqs <= 52.0)
    psd[:, peak] = [[5.0] * peak.sum(), [2.0] * peak.sum(), [1.0] * peak.sum()]

    np.testing.assert_allclose(noise_surround_ratio(freqs, psd, 50.0), [5.0, 2.0, 1.0])
    assert underclean_proportion(
        freqs, psd, 50.0, threshold_ratio=2.0
    ) == pytest.approx(1 / 3)


def test_below_noise_distortion_is_mean_absolute_log_ratio():
    """Broadband distortion is measured in dB and aggregated per channel."""
    freqs = np.arange(1.0, 5.0)
    before = np.ones((2, freqs.size))
    after = np.array([[0.1, 10.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])

    expected = np.array([5.0, 0.0])
    np.testing.assert_allclose(
        below_noise_distortion_db(freqs, before, after, fmin=1.0, fmax=4.0),
        expected,
    )
    assert (
        below_noise_distortion_db(freqs, before[0], before[0], fmin=100.0, fmax=120.0)
        == 0.0
    )


def test_spectral_distortion_is_rms_db_outside_harmonics():
    """Spectral distortion ignores harmonic bins and uses RMS dB change."""
    freqs = np.array([2.0, 10.0, 20.0, 50.0, 60.0])
    before = np.ones_like(freqs)
    after = np.array([1.0, 2.0, 0.5, 100.0, 1.0])
    db_changes = 10.0 * np.log10(after[[0, 1, 2, 4]])

    expected = np.sqrt(np.mean(db_changes**2))
    assert spectral_distortion(
        freqs, before, after, line_freq=50.0, n_harmonics=1
    ) == pytest.approx(expected)
    assert (
        spectral_distortion(
            np.array([48.0, 50.0, 52.0]),
            np.ones(3),
            np.ones(3),
            line_freq=50.0,
            bandwidth=2.0,
        )
        == 0.0
    )


def test_overclean_proportion_flags_suppressed_spectral_floor():
    """Only channels whose surrounding floor crosses the dB threshold are flagged."""
    freqs = np.arange(40.0, 61.0)
    before = np.ones((4, freqs.size))
    after = before.copy()
    surround = ((freqs >= 46.0) & (freqs < 48.0)) | ((freqs > 52.0) & (freqs <= 54.0))
    after[:2, surround] = 0.1

    assert overclean_proportion(freqs, before, after, 50.0) == pytest.approx(0.5)
    assert overclean_proportion(freqs, before[0], before[0], 50.0) == 0.0


def test_geometric_mean_psd_ratio_is_channelwise():
    """Broadband ratios use the geometric mean rather than the arithmetic mean."""
    freqs = np.arange(1.0, 5.0)
    before = np.ones((3, freqs.size))
    after = np.array(
        [
            [0.5, 0.5, 0.5, 0.5],
            [2.0, 2.0, 2.0, 2.0],
            [0.5, 2.0, 0.5, 2.0],
        ]
    )

    np.testing.assert_allclose(
        geometric_mean_psd_ratio(freqs, before, after, fmin=1.0, fmax=4.0),
        [0.5, 2.0, 1.0],
    )
    assert (
        geometric_mean_psd_ratio(freqs, before[0], before[0], fmin=100.0, fmax=120.0)
        == 1.0
    )


def test_variance_and_signal_change_metrics_match_definitions():
    """Variance, RMS, and maximum-change metrics have hand-computable values."""
    before = np.array([1.0, -1.0, 1.0, -1.0])
    after = 0.5 * before

    assert variance_removed(before, after) == pytest.approx(75.0)
    assert rms_change(before, after) == pytest.approx(0.5)
    assert max_abs_change(before, after) == pytest.approx(0.5)
    assert variance_removed(10.0 * before, 10.0 * after) == pytest.approx(75.0)
    assert rms_change(-before, -after) == pytest.approx(0.5)
    assert max_abs_change(-before, -after) == pytest.approx(0.5)
    assert variance_removed(np.zeros(4), np.zeros(4)) == 0.0


def test_channel_variance_ratio_aggregates_epochs_and_channels():
    """Variance ratios preserve the documented channel dimension for 2D/3D data."""
    before_2d = np.array([[1.0, -1.0, 1.0, -1.0], [2.0, -2.0, 2.0, -2.0]])
    after_2d = np.array([[0.5, -0.5, 0.5, -0.5], [0.0, 0.0, 0.0, 0.0]])
    np.testing.assert_allclose(channel_variance_ratio(before_2d, after_2d), [0.25, 0.0])

    before_3d = np.array([[[1.0, -1.0], [1.0, -1.0]], [[2.0, -2.0], [2.0, -2.0]]])
    after_3d = np.array([[[0.5, -0.5], [0.5, -0.5]], [[0.0, 0.0], [0.0, 0.0]]])
    np.testing.assert_allclose(
        channel_variance_ratio(before_3d, after_3d), [0.05, 0.05]
    )


def test_compute_all_qa_metrics_returns_aggregated_public_metrics():
    """The Raw-level helper aggregates per-channel and per-harmonic metrics."""
    rng = np.random.default_rng(7)
    sfreq = 100.0
    times = np.arange(1000) / sfreq
    line = np.sin(2.0 * np.pi * 20.0 * times)
    noise = 0.05 * rng.standard_normal((2, times.size))
    before = noise + line
    after = noise + 0.1 * line
    info = mne.create_info(["Fz", "Cz"], sfreq, ch_types="eeg")
    raw_before = mne.io.RawArray(before, info)
    raw_after = mne.io.RawArray(after, info.copy())

    metrics = compute_all_qa_metrics(
        raw_before,
        raw_after,
        line_freq=20.0,
        n_harmonics=1,
        fmax=40.0,
    )

    expected_keys = {
        "peak_attenuation_db",
        "R_f0",
        "below_noise_distortion_db",
        "overclean_proportion",
        "underclean_proportion",
        "geometric_mean_psd_ratio",
        "harmonics_hz",
        "per_harmonic_attenuation_db",
        "per_harmonic_R",
    }
    assert expected_keys <= metrics.keys()
    assert metrics["harmonics_hz"] == [20.0, 40.0]
    assert len(metrics["per_harmonic_attenuation_db"]) == 2
    assert len(metrics["per_harmonic_R"]) == 2
    assert metrics["peak_attenuation_db"] > 0.0
    assert 0.0 <= metrics["overclean_proportion"] <= 1.0
    assert 0.0 <= metrics["underclean_proportion"] <= 1.0
