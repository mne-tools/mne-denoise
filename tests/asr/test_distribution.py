import numpy as np
import pytest

from mne_denoise.asr._distribution import (
    _histc_scaled_bins,
    _robust_location_scale,
    fit_rms_distribution,
)


def test_histc_scaled_bins():
    """Test histogram scaling and clipping."""
    # Two columns:
    # Col 0: values from -1 to 5. nbins=4.
    # Col 1: all NaNs
    values = np.array(
        [
            [-1.5, np.nan],
            [0.0, np.nan],
            [1.2, np.nan],
            [2.9, np.nan],
            [3.1, np.nan],
            [5.5, np.nan],
        ]
    )
    nbins = 4
    counts = _histc_scaled_bins(values, nbins)

    assert counts.shape == (4, 2)
    # Col 1 should be all zeros
    np.testing.assert_array_equal(counts[:, 1], 0)

    # Col 0 values:
    # -1.5 -> floor(-1.5) = -2 -> clip to 0
    # 0.0 -> floor(0) = 0 -> clip to 0
    # 1.2 -> floor(1.2) = 1 -> clip to 1
    # 2.9 -> floor(2.9) = 2 -> clip to 2
    # 3.1 -> floor(3.1) = 3 -> clip to 3
    # 5.5 -> floor(5.5) = 5 -> clip to 3 (nbins-1)

    expected_col0 = np.array([2, 1, 1, 2])
    np.testing.assert_array_equal(counts[:, 0], expected_col0)


def test_robust_location_scale_normal():
    """Test robust location scale on normal data."""
    rng = np.random.default_rng(42)
    values = rng.normal(loc=5.0, scale=2.0, size=10000)
    mu, sigma = _robust_location_scale(values)
    assert np.isclose(mu, 5.0, atol=0.1)
    assert np.isclose(sigma, 2.0, atol=0.1)


def test_robust_location_scale_fallback_std():
    """Test fallback to standard deviation when MAD is 0."""
    # e.g., 5, 5, 5, 10
    # median = 5, MAD = 0
    values = np.array([5.0, 5.0, 5.0, 10.0])
    mu, sigma = _robust_location_scale(values)
    assert mu == 5.0
    # std fallback is std(ddof=1)
    expected_sigma = np.std(values, ddof=1)
    assert np.isclose(sigma, expected_sigma)


def test_robust_location_scale_fallback_tiny():
    """Test fallback to tiny epsilon when std is also 0."""
    values = np.array([5.0, 5.0, 5.0, 5.0])
    mu, sigma = _robust_location_scale(values)
    assert mu == 5.0
    # sigma should fall back to max(abs(mu) * 1e-6, eps)
    assert np.isclose(sigma, 5.0 * 1e-6)

    # test with exactly zero
    values_zero = np.array([0.0, 0.0])
    mu_z, sigma_z = _robust_location_scale(values_zero)
    assert mu_z == 0.0
    assert sigma_z == np.finfo(float).eps


def test_fit_rms_distribution_empty():
    """Test fitting on empty or all-NaN arrays raises an error."""
    with pytest.raises(
        ValueError, match="Cannot fit ASR thresholds from empty RMS distribution"
    ):
        fit_rms_distribution(np.array([]))
    with pytest.raises(
        ValueError, match="Cannot fit ASR thresholds from empty RMS distribution"
    ):
        fit_rms_distribution(np.array([np.nan, np.inf]))


def test_fit_rms_distribution_invalid_params():
    """Test parameter validation for fit_rms_distribution."""
    valid_data = np.ones(100)

    with pytest.raises(ValueError, match="max_dropout_fraction must be in"):
        fit_rms_distribution(valid_data, max_dropout_fraction=-0.1)

    with pytest.raises(ValueError, match="min_clean_fraction must be in"):
        fit_rms_distribution(valid_data, min_clean_fraction=0.0)

    with pytest.raises(ValueError, match="max_dropout_fraction \\+ min_clean_fraction"):
        fit_rms_distribution(
            valid_data, max_dropout_fraction=0.6, min_clean_fraction=0.5
        )

    with pytest.raises(ValueError, match="fit_quantiles must satisfy"):
        fit_rms_distribution(valid_data, fit_quantiles=(0.8, 0.2))

    with pytest.raises(ValueError, match="beta_grid must contain positive values"):
        fit_rms_distribution(valid_data, beta_grid=np.array([]))

    with pytest.raises(
        ValueError, match="beta_grid values must be in the open interval"
    ):
        fit_rms_distribution(valid_data, beta_grid=np.array([0.5, 2.0]))


def test_fit_rms_distribution_valid_data():
    """Test the grid search distribution fitter on realistic valid data."""
    rng = np.random.default_rng(42)
    # Simulate strictly positive RMS data
    values = np.abs(rng.normal(loc=5.0, scale=1.0, size=5000))

    mu, sigma, info = fit_rms_distribution(values, return_info=True)

    # We don't assert exact values since generalized Gaussian fitting is complex,
    # but we assert the shape and bounds of the output are mathematically sound.
    assert mu > 0
    assert sigma > 0
    assert "beta" in info
    assert "score" in info
    assert "fit_interval" in info
    assert "fit_error" in info
    assert info["beta"] > 0
    assert info["score"] >= 0


def test_fit_rms_distribution_robust_to_tail_and_dropouts(rng):
    """Clean RMS fitting resists high-tail artifacts and low dropouts."""
    clean = rng.normal(loc=1.0, scale=0.08, size=800)
    high_tail = rng.uniform(4.0, 9.0, size=120)
    dropouts = rng.uniform(0.01, 0.08, size=40)
    values = np.concatenate([clean, high_tail, dropouts])

    mu, sigma, info = fit_rms_distribution(
        values,
        min_clean_fraction=0.25,
        max_dropout_fraction=0.1,
        return_info=True,
    )

    assert 0.9 < mu < 1.1
    assert 0.02 < sigma < 0.2
    assert sigma < np.std(values) * 0.2
    assert np.isfinite(info["beta"])
    assert info["n_fit_samples"] > 0


def test_fit_rms_distribution_validation():
    """Clean RMS fitter validates empty and invalid parameter cases."""
    with pytest.raises(ValueError, match="empty RMS"):
        fit_rms_distribution(np.array([np.nan, np.inf]))
    with pytest.raises(ValueError, match="fit_quantiles"):
        fit_rms_distribution(np.ones(10), fit_quantiles=(0.7, 0.6))
    with pytest.raises(ValueError, match="beta_grid"):
        fit_rms_distribution(np.ones(10), beta_grid=np.array([]))


def test_fit_rms_distribution_returns_params():
    rng = np.random.default_rng(1)
    rms = np.abs(rng.standard_normal(2000)) + 0.5
    mu, sigma, *_ = fit_rms_distribution(rms)
    assert np.isfinite(mu) and sigma > 0


def test_fit_adaptive_thresholds() -> None:
    from mne_denoise.asr._distribution import _fit_adaptive_thresholds

    rng = np.random.default_rng(42)
    # 3 channels, 1000 samples
    X = rng.standard_normal((3, 1000))
    # Add a burst so the distribution isn't totally perfectly normal,
    # ensuring fit_rms_distribution doesn't fall back completely to zeros.
    X[0, 400:450] += 5.0
    V = np.eye(3)

    thresholds, info = _fit_adaptive_thresholds(
        X=X,
        V=V,
        sfreq=100.0,
        window_length=0.5,
        window_overlap=0.5,
        cutoff=5.0,
        min_clean_fraction=0.25,
        max_dropout_fraction=0.1,
    )

    assert thresholds.shape == (3,)
    assert isinstance(info, dict)
    for key in [
        "mu",
        "sigma",
        "beta",
        "fit_error",
        "fit_interval",
        "window_starts",
        "window_length_samples",
    ]:
        assert key in info

    assert np.all(thresholds > 0)
    assert info["mu"].shape == (3,)
    assert info["sigma"].shape == (3,)
    assert info["beta"].shape == (3,)


def test_fit_rms_distribution_fallback_to_robust():
    """When the grid search fails to find a valid fit, fallback to robust stats."""
    # Only 2 values — too few for any grid search window to succeed
    values = np.array([1.0, 1.0])
    mu, sigma, info = fit_rms_distribution(values, return_info=True)
    assert np.isfinite(mu)
    assert sigma > 0
    # The grid search should have failed, so beta should be NaN
    assert np.isnan(info["beta"])


def test_fit_rms_distribution_constant_values():
    """Constant values: all denominators are zero, grid search skips everything."""
    values = np.full(100, 5.0)
    mu, sigma, info = fit_rms_distribution(values, return_info=True)
    assert mu == 5.0
    assert sigma > 0
    # Grid search cannot distinguish bins, falls back
    assert np.isnan(info["beta"])
