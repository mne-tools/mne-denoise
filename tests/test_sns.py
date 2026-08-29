"""Tests for the mne_denoise.sns module (Sensor Noise Suppression)."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from mne_denoise.sns import SNS, compute_sns, compute_sns_weights

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def rng():
    """Shared random generator."""
    return np.random.default_rng(0)


@pytest.fixture()
def sensor_noise_data(rng):
    """Low-rank spatially-correlated signal + independent per-sensor noise.

    Returns ``(X, shared)`` where ``shared`` (rank 5, mixed across all channels)
    is predictable from other channels and should be recovered by SNS, while the
    independent per-sensor noise should be suppressed.
    """
    n_ch = 20
    n_times = 5000
    n_src = 5
    sources = rng.standard_normal((n_src, n_times))
    mixing = rng.standard_normal((n_ch, n_src))
    shared = mixing @ sources
    noise = 0.5 * rng.standard_normal((n_ch, n_times))  # independent per sensor
    X = shared + noise
    return X, shared


def _rrmse(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


# ---------------------------------------------------------------------------
# covariance-level and array-level SNS
# ---------------------------------------------------------------------------


def test_compute_sns_shape_and_zero_diagonal(rng):
    """The operator is square with a zero diagonal (no self-regeneration)."""
    X = rng.standard_normal((12, 3000))
    Xd = X - X.mean(axis=1, keepdims=True)
    cov = (Xd @ Xd.T) / Xd.shape[1]
    W, k, ranks = compute_sns_weights(cov, n_neighbors=0)
    assert W.shape == (12, 12)
    assert k == 11  # all others
    assert ranks.shape == (12,)
    np.testing.assert_allclose(np.diag(W), 0.0, atol=1e-12)


def test_compute_sns_zero_diagonal_with_duplicate_channel(rng):
    """Self is excluded even when a duplicate channel ties the correlation (W[k,k]==0)."""
    X = rng.standard_normal((6, 3000))
    X[4] = X[0].copy()  # exact-duplicate channel -> corr**2 tie at 1.0
    Xd = X - X.mean(axis=1, keepdims=True)
    cov = (Xd @ Xd.T) / Xd.shape[1]
    W, _k, _ranks = compute_sns_weights(cov, n_neighbors=0)
    np.testing.assert_allclose(np.diag(W), 0.0, atol=1e-12)


def test_compute_sns_caps_neighbors():
    """n_neighbors and skip are capped at n_channels - skip - 1."""
    cov = np.eye(6)
    _W, k, _ranks = compute_sns_weights(cov, n_neighbors=100, skip=2)
    assert k == 6 - 2 - 1  # == 3


def test_compute_sns_recovers_analytical_projection():
    """Local least squares recovers known coefficients from covariance."""
    # Channel 0 is exactly channel 1 + 2 * channel 2.
    cov = np.array([[5.0, 1.0, 2.0], [1.0, 1.0, 0.0], [2.0, 0.0, 1.0]])
    weights, _k, _ranks = compute_sns_weights(cov, n_neighbors=2)
    np.testing.assert_allclose(weights[0], [0.0, 1.0, 2.0], atol=1e-12)


def test_compute_sns_weights_progress_callback():
    """Standalone SNS weights report each completed channel solve."""
    cov = np.array([[5.0, 1.0, 2.0], [1.0, 1.0, 0.0], [2.0, 0.0, 1.0]])
    events = []
    _weights, _n_neighbors, ranks = compute_sns_weights(
        cov, n_neighbors=2, callback=events.append
    )

    assert len(events) == cov.shape[0]
    assert all(event.method == "sns" for event in events)
    assert all(event.stage == "channel" for event in events)
    assert [event.current for event in events] == [1, 2, 3]
    assert all(event.total == cov.shape[0] for event in events)
    assert all(event.component is None for event in events)
    np.testing.assert_array_equal(
        [event.metric for event in events], ranks.astype(float)
    )


def test_compute_sns_neighbor_selection_and_skip():
    """Squared correlation orders neighbors and skip advances that order."""
    cov = np.array([[5.0, 1.0, 2.0], [1.0, 1.0, 0.0], [2.0, 0.0, 1.0]])
    closest, _k, _ranks = compute_sns_weights(cov, n_neighbors=1)
    skipped, _k, _ranks = compute_sns_weights(cov, n_neighbors=1, skip=1)
    assert np.flatnonzero(closest[0]).tolist() == [2]
    assert np.flatnonzero(skipped[0]).tolist() == [1]


def test_compute_sns_is_unit_scale_invariant(rng):
    """The relative pseudoinverse cutoff must not depend on physical units."""
    X = rng.standard_normal((8, 1000))
    cov = np.cov(X)
    weights, _k, _ranks = compute_sns_weights(cov, n_neighbors=4)
    scaled_weights, _k, _ranks = compute_sns_weights(cov * 1e-12, n_neighbors=4)
    np.testing.assert_allclose(weights, scaled_weights, rtol=1e-10, atol=1e-12)


def test_compute_sns_rejects_invalid_covariance():
    """Covariance preconditions fail explicitly before local regressions."""
    invalid = (
        (np.ones(3), "square"),
        (np.ones((2, 3)), "square"),
        (np.ones((1, 1)), "at least two"),
        (np.array([[1.0, np.nan], [np.nan, 1.0]]), "finite"),
        (np.array([[1.0, 0.0], [0.5, 1.0]]), "symmetric"),
        (np.array([[1.0, 2.0], [2.0, 1.0]]), "positive semidefinite"),
    )
    for cov, match in invalid:
        with pytest.raises(ValueError, match=match):
            compute_sns_weights(cov)


def test_compute_sns_mean_policy(rng):
    """Reference centering and explicit mean preservation are distinct policies."""
    X = rng.standard_normal((6, 500)) + np.arange(6)[:, np.newaxis]
    centered, _ = compute_sns(X, n_neighbors=3)
    preserved, _ = compute_sns(X, n_neighbors=3, preserve_mean=True)
    np.testing.assert_allclose(centered.mean(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(preserved.mean(axis=1), X.mean(axis=1), atol=1e-12)


def test_compute_sns_clean_low_rank_is_preserved(rng):
    """A spatially redundant clean signal is regenerated to numerical precision."""
    sources = rng.standard_normal((3, 1000))
    clean = rng.standard_normal((12, 3)) @ sources
    regenerated, info = compute_sns(clean, n_neighbors=8)
    np.testing.assert_allclose(
        regenerated, clean - clean.mean(axis=1, keepdims=True), atol=2e-11
    )
    assert info["input_rank"] == 3
    np.testing.assert_array_equal(info["neighbor_ranks"], 3)


def test_compute_sns_manual_weights_match_retained_fit(rng):
    """Zero-weight samples are exactly equivalent to omitting them from fitting."""
    X = rng.standard_normal((8, 600))
    keep = np.ones(600, dtype=bool)
    keep[100:170] = False
    weighted, info = compute_sns(X, n_neighbors=5, sample_weight=keep)
    _, retained_info = compute_sns(X[:, keep], n_neighbors=5)
    np.testing.assert_allclose(info["weights"], retained_info["weights"], atol=1e-12)
    expected = retained_info["weights"] @ (X - X[:, keep].mean(axis=1, keepdims=True))
    np.testing.assert_allclose(weighted, expected, atol=1e-12)


def test_compute_sns_robust_mask_limits_glitch_bias(rng):
    """Automatic masking keeps glitches from biasing the learned projection."""
    sources = rng.standard_normal((3, 1200))
    clean = rng.standard_normal((10, 3)) @ sources
    noisy = clean + 0.1 * rng.standard_normal(clean.shape)
    glitched = noisy.copy()
    glitched[0, 100:110] += 100.0
    _, baseline = compute_sns(noisy, n_neighbors=6)
    _, ordinary = compute_sns(glitched, n_neighbors=6)
    _, robust = compute_sns(glitched, n_neighbors=6, outlier_threshold=8.0)
    ordinary_error = np.linalg.norm(ordinary["weights"] - baseline["weights"])
    robust_error = np.linalg.norm(robust["weights"] - baseline["weights"])
    assert robust["rejected_sample_count"] >= 10
    assert robust_error < ordinary_error


def test_compute_sns_iterations_compose_exactly(rng):
    """The reported composite is the ordered product of per-pass operators."""
    X = rng.standard_normal((9, 800))
    cleaned, info = compute_sns(X, n_neighbors=5, n_iter=3)
    composite = np.eye(9)
    for matrix in info["denoising_matrices"]:
        composite = matrix @ composite
    np.testing.assert_allclose(info["weights"], composite)
    np.testing.assert_allclose(cleaned, composite @ (X - X.mean(axis=1)[:, None]))
    assert len(info["neighbor_ranks_per_iteration"]) == 3


def test_compute_sns_progress_callback_has_flat_iteration_counter(rng):
    """SNS channel events use one counter across all learning iterations."""
    X = rng.standard_normal((3, 180))
    for n_iter in (1, 2):
        events = []
        _cleaned, info = compute_sns(
            X, n_neighbors=1, n_iter=n_iter, callback=events.append
        )

        n_channels = X.shape[0]
        total = n_iter * n_channels
        assert len(events) == total
        assert [event.current for event in events] == list(range(1, total + 1))
        assert [event.total for event in events] == [total] * total
        assert all(event.method == "sns" for event in events)
        assert all(event.stage == "channel" for event in events)
        expected_metrics = np.concatenate(info["neighbor_ranks_per_iteration"])
        np.testing.assert_array_equal(
            [event.metric for event in events], expected_metrics.astype(float)
        )


def test_compute_sns_automatic_mask_is_fixed_across_iterations(rng):
    """Robust rejection is derived once and reused for every pass."""
    X = rng.standard_normal((7, 500))
    X[2, 100:106] += 80.0
    threshold = 8.0
    center = np.median(X, axis=1, keepdims=True)
    scale = 1.4826 * np.median(np.abs(X - center), axis=1, keepdims=True)
    fallback = np.std(X, axis=1, keepdims=True)
    scale = np.where(scale > 0, scale, fallback)
    scale = np.where(scale > 0, scale, 1.0)
    keep = np.max(np.abs((X - center) / scale), axis=0) <= threshold
    _, automatic = compute_sns(X, n_neighbors=4, n_iter=3, outlier_threshold=threshold)
    _, explicit = compute_sns(X, n_neighbors=4, n_iter=3, sample_weight=keep)
    np.testing.assert_allclose(automatic["weights"], explicit["weights"])
    assert automatic["rejected_sample_count"] == np.count_nonzero(~keep)


def test_compute_sns_chunked_matches_unchunked(rng):
    """Chunking changes memory use, not numerical behavior."""
    X = rng.standard_normal((10, 777))
    weight = rng.uniform(0.1, 1.0, X.shape[1])
    full, full_info = compute_sns(X, n_neighbors=6, n_iter=2, sample_weight=weight)
    chunked, chunked_info = compute_sns(
        X,
        n_neighbors=6,
        n_iter=2,
        sample_weight=weight,
        chunk_size=73,
    )
    np.testing.assert_allclose(chunked, full, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(
        chunked_info["weights"], full_info["weights"], rtol=1e-11, atol=1e-12
    )


def test_compute_sns_rejects_invalid_or_insufficient_weights(rng):
    """Weights must be finite, nonnegative, correctly shaped, and sufficient."""
    X = rng.standard_normal((5, 100))
    invalid = (
        np.ones((2, 50)),
        np.full(100, np.nan),
        -np.ones(100),
        np.r_[1.0, np.zeros(99)],
    )
    for weight in invalid:
        with pytest.raises(ValueError, match="sample|positively"):
            compute_sns(X, sample_weight=weight)


def test_compute_sns_all_samples_rejected(rng):
    """An overly strict automatic mask fails clearly."""
    X = rng.standard_normal((5, 100))
    X[:, 0] = 0.0
    with pytest.raises(ValueError, match="after rejection"):
        compute_sns(X, outlier_threshold=1e-9)


def test_compute_sns_constant_channel_is_finite(rng):
    """Constant channels and robust-scale fallbacks remain finite."""
    X = rng.standard_normal((6, 300))
    X[0] = 2.0
    cleaned, info = compute_sns(X, outlier_threshold=10.0)
    assert np.isfinite(cleaned).all()
    assert np.isfinite(info["weights"]).all()


# ---------------------------------------------------------------------------
# SNS estimator
# ---------------------------------------------------------------------------


def test_compute_sns_suppresses_independent_sensor_noise(sensor_noise_data):
    """SNS recovers the shared low-rank signal, suppressing per-sensor noise."""
    X, shared = sensor_noise_data
    cleaned, _info = compute_sns(X, n_neighbors=0)  # all others
    err_before = _rrmse(X, shared)
    err_after = _rrmse(cleaned, shared)
    assert err_after < 0.9 * err_before  # closer to the clean shared signal


def test_sns_fitted_attributes(rng):
    """Fitted attributes are populated with correct shapes."""
    X = rng.standard_normal((14, 3000))
    est = SNS(n_neighbors=5).fit(X)
    assert est.denoising_matrix_.shape == (14, 14)
    assert est.n_neighbors_ == 5
    assert est.neighbor_ranks_.shape == (14,)
    assert est.training_mean_.shape == (14, 1)
    assert len(est.denoising_matrices_) == 1
    assert len(est.neighbor_ranks_per_iteration_) == 1
    assert est.input_rank_ > 0


def test_sns_training_mean_is_weighted(rng):
    """The fitted centering statistic honors global sample weights."""
    X = rng.standard_normal((6, 300)) + np.arange(6)[:, None]
    weight = np.linspace(0.1, 1.0, X.shape[1])
    est = SNS(n_neighbors=4).fit(X, sample_weight=weight)
    expected = np.average(X, axis=1, weights=weight)[:, None]
    np.testing.assert_allclose(est.training_mean_, expected)


def test_sns_leakage_split_applies_operator(rng):
    """transform applies the operator learned in fit (train != eval)."""
    train = rng.standard_normal((10, 4000))
    evalu = rng.standard_normal((10, 800))
    est = SNS(n_neighbors=6).fit(train)
    cleaned = est.transform(evalu)
    expected = est.denoising_matrix_ @ (evalu - est.training_mean_)
    np.testing.assert_allclose(cleaned, expected)


def test_sns_continuous_transform_is_batch_invariant(rng):
    """Transforming a temporal chunk alone gives the same samples as a full call."""
    train = rng.standard_normal((8, 1000)) + np.arange(8)[:, None]
    evalu = rng.standard_normal((8, 400)) + 3.0
    for preserve_mean in (False, True):
        est = SNS(n_neighbors=5, preserve_mean=preserve_mean).fit(train)
        full = est.transform(evalu)
        np.testing.assert_allclose(est.transform(evalu[:, 100:250]), full[:, 100:250])


def test_sns_epoch_transform_is_batch_invariant(rng):
    """An epoch result is independent of the other epochs in the transform call."""
    train = rng.standard_normal((5, 7, 120))
    evalu = rng.standard_normal((4, 7, 80)) + 2.0
    for preserve_mean in (False, True):
        est = SNS(n_neighbors=4, preserve_mean=preserve_mean).fit(train)
        full = est.transform(evalu)
        one = est.transform(evalu[[2]])
        np.testing.assert_allclose(one[0], full[2])


def test_sns_preserve_mean_restores_training_mean(rng):
    """Mean preservation uses fitted statistics, never evaluation statistics."""
    train = rng.standard_normal((6, 700)) + np.arange(6)[:, None]
    evalu = rng.standard_normal((6, 200)) + 20.0
    est = SNS(n_neighbors=4, preserve_mean=True).fit(train)
    expected = est.denoising_matrix_ @ (evalu - est.training_mean_)
    expected += est.training_mean_
    np.testing.assert_allclose(est.transform(evalu), expected)


def test_sns_epoch_sample_weights_and_chunking(rng):
    """Epoch-shaped sample weights flatten in the same order as epoch data."""
    epochs = rng.standard_normal((3, 6, 101))
    weights = rng.uniform(0.2, 1.0, (3, 101))
    est = SNS(n_neighbors=4, n_iter=2, chunk_size=37).fit(epochs, sample_weight=weights)
    continuous = epochs.transpose(1, 0, 2).reshape(6, -1)
    flat = SNS(n_neighbors=4, n_iter=2, chunk_size=37).fit(
        continuous, sample_weight=weights.reshape(-1)
    )
    np.testing.assert_allclose(est.training_mean_, flat.training_mean_)
    np.testing.assert_allclose(est.denoising_matrix_, flat.denoising_matrix_)


def test_sns_emits_one_aggregate_summary(rng, caplog):
    """The SNS core owns one summary that includes learned neighbours."""
    with caplog.at_level(logging.INFO, logger="mne_denoise"):
        compute_sns(
            rng.standard_normal((5, 300)),
            n_neighbors=2,
            n_iter=2,
            verbose=True,
        )
    summaries = [r for r in caplog.records if r.message.startswith("SNS:")]
    assert len(summaries) == 1
    assert "neighbours each" in summaries[0].message
    assert "iteration(s)" in summaries[0].message
