import numpy as np
import pytest

from mne_denoise.asr._distribution import (
    _histc_scaled_bins,
    _robust_location_scale,
    fit_rms_distribution,
)


def test_histc_scaled_bins():
    """Histogram scaling clips out-of-range values and ignores non-finite data."""
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
    counts = _histc_scaled_bins(values, 4)
    assert counts.shape == (4, 2)
    np.testing.assert_array_equal(counts[:, 1], 0)
    np.testing.assert_array_equal(counts[:, 0], [2, 1, 1, 2])


def test_robust_location_scale_contract():
    """Robust scale uses MAD, then standard-deviation and epsilon fallbacks."""
    rng = np.random.default_rng(42)
    values = rng.normal(loc=5.0, scale=2.0, size=10000)
    mu, sigma = _robust_location_scale(values)
    assert np.isclose(mu, 5.0, atol=0.1)
    assert np.isclose(sigma, 2.0, atol=0.1)

    repeated = np.array([5.0, 5.0, 5.0, 10.0])
    mu, sigma = _robust_location_scale(repeated)
    assert mu == 5.0
    assert np.isclose(sigma, np.std(repeated, ddof=1))
    mu, sigma = _robust_location_scale(np.full(4, 5.0))
    assert mu == 5.0
    assert np.isclose(sigma, 5e-6)
    mu, sigma = _robust_location_scale(np.zeros(2))
    assert mu == 0.0
    assert sigma == np.finfo(float).eps


def test_fit_rms_distribution_validation_contract():
    """RMS fitting rejects empty data and invalid configuration combinations."""
    for values in (np.array([]), np.array([np.nan, np.inf])):
        with pytest.raises(ValueError, match="empty RMS"):
            fit_rms_distribution(values)
    valid = np.ones(100)
    cases = [
        ({"max_dropout_fraction": -0.1}, "max_dropout_fraction must be in"),
        ({"min_clean_fraction": 0.0}, "min_clean_fraction must be in"),
        (
            {"max_dropout_fraction": 0.6, "min_clean_fraction": 0.5},
            "max_dropout_fraction \\+ min_clean_fraction",
        ),
        ({"fit_quantiles": (0.8, 0.2)}, "fit_quantiles must satisfy"),
        ({"beta_grid": np.array([])}, "beta_grid must contain positive values"),
        (
            {"beta_grid": np.array([0.5, 2.0])},
            "beta_grid values must be in the open interval",
        ),
    ]
    for kwargs, message in cases:
        with pytest.raises(ValueError, match=message):
            fit_rms_distribution(valid, **kwargs)


def test_fit_rms_distribution_numerical_contract():
    """Valid RMS fitting returns finite parameters and diagnostics."""
    rng = np.random.default_rng(42)
    values = np.abs(rng.normal(loc=5.0, scale=1.0, size=5000))
    mu, sigma, info = fit_rms_distribution(values, return_info=True)
    assert mu > 0 and sigma > 0
    assert {"beta", "score", "fit_interval", "fit_error"} <= info.keys()
    assert info["beta"] > 0 and info["score"] >= 0
    other = np.abs(rng.standard_normal(2000)) + 0.5
    mu, sigma = fit_rms_distribution(other)
    assert np.isfinite(mu) and sigma > 0


def test_fit_rms_distribution_resists_tails_and_falls_back(rng):
    """Heavy tails are rejected while underspecified/constant data stay finite."""
    clean = rng.normal(loc=1.0, scale=0.08, size=800)
    values = np.concatenate(
        [clean, rng.uniform(4.0, 9.0, 120), rng.uniform(0.01, 0.08, 40)]
    )
    mu, sigma, info = fit_rms_distribution(
        values, min_clean_fraction=0.25, max_dropout_fraction=0.1, return_info=True
    )
    assert 0.9 < mu < 1.1
    assert 0.02 < sigma < 0.2
    assert sigma < np.std(values) * 0.2
    assert np.isfinite(info["beta"]) and info["n_fit_samples"] > 0

    for values in (np.array([1.0, 1.0]), np.full(100, 5.0)):
        mu, sigma, info = fit_rms_distribution(values, return_info=True)
        assert np.isfinite(mu) and sigma > 0
        assert np.isnan(info["beta"])


def test_fit_adaptive_thresholds() -> None:
    """Adaptive thresholds contain one finite per-channel fit and diagnostics."""
    from mne_denoise.asr._distribution import _fit_adaptive_thresholds

    rng = np.random.default_rng(42)
    data = rng.standard_normal((3, 1000))
    data[0, 400:450] += 5.0
    thresholds, info = _fit_adaptive_thresholds(
        X=data,
        V=np.eye(3),
        sfreq=100.0,
        window_length=0.5,
        window_overlap=0.5,
        cutoff=5.0,
        min_clean_fraction=0.25,
        max_dropout_fraction=0.1,
    )
    assert thresholds.shape == (3,)
    assert np.all(thresholds > 0)
    assert info["mu"].shape == info["sigma"].shape == info["beta"].shape == (3,)
    assert {
        "fit_error",
        "fit_interval",
        "window_starts",
        "window_length_samples",
    } <= info.keys()
