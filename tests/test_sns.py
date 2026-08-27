"""Tests for the mne_denoise.sns module (Sensor Noise Suppression)."""

from __future__ import annotations

import logging

import numpy as np
import pytest
from sklearn.base import clone

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
# compute_sns + compute_sns
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


@pytest.mark.parametrize(
    ("cov", "match"),
    [
        (np.ones(3), "square"),
        (np.ones((2, 3)), "square"),
        (np.ones((1, 1)), "at least two"),
        (np.array([[1.0, np.nan], [np.nan, 1.0]]), "finite"),
        (np.array([[1.0, 0.0], [0.5, 1.0]]), "symmetric"),
        (np.array([[1.0, 2.0], [2.0, 1.0]]), "positive semidefinite"),
    ],
)
def test_compute_sns_rejects_invalid_covariance(cov, match):
    """Covariance preconditions fail explicitly before local regressions."""
    with pytest.raises(ValueError, match=match):
        compute_sns_weights(cov)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"n_neighbors": -1}, ValueError, "n_neighbors"),
        ({"n_neighbors": 1.5}, TypeError, "n_neighbors"),
        ({"skip": -1}, ValueError, "skip"),
        ({"skip": 5}, ValueError, "leave at least one"),
        ({"rcond": 0}, ValueError, "strictly between"),
        ({"rcond": np.inf}, ValueError, "strictly between"),
    ],
)
def test_compute_sns_rejects_invalid_operating_points(kwargs, error, match):
    """Invalid integer and numerical operating points cannot be silently coerced."""
    with pytest.raises(error, match=match):
        compute_sns_weights(np.eye(6), **kwargs)


def test_compute_sns_shapes_and_info(rng):
    """compute_sns returns cleaned data + diagnostics."""
    X = rng.standard_normal((10, 2000))
    X_clean, info = compute_sns(X, n_neighbors=6)
    assert X_clean.shape == X.shape
    assert info["weights"].shape == (10, 10)
    assert info["n_neighbors"] == 6
    assert info["neighbor_ranks"].shape == (10,)
    assert info["input_rank"] > 0
    centered = X - X.mean(axis=1, keepdims=True)
    expected, _k, _ranks = compute_sns_weights(centered @ centered.T / X.shape[1], 6)
    np.testing.assert_allclose(info["weights"], expected)
    assert len(info["denoising_matrices"]) == 1
    np.testing.assert_allclose(X_clean.mean(axis=1), 0.0, atol=1e-12)


def test_compute_sns_rejects_1d():
    """A 1-D input raises a clear error."""
    with pytest.raises(ValueError, match="2-D"):
        compute_sns(np.zeros(100))


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
    regenerated, _ = compute_sns(clean, n_neighbors=8)
    np.testing.assert_allclose(
        regenerated, clean - clean.mean(axis=1, keepdims=True), atol=2e-11
    )


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


def test_compute_sns_iterations_have_diminishing_changes(sensor_noise_data):
    """Successive projections converge on representative redundant data."""
    X, _ = sensor_noise_data
    outputs = [X - X.mean(axis=1, keepdims=True)]
    for n_iter in range(1, 4):
        outputs.append(compute_sns(X, n_neighbors=12, n_iter=n_iter)[0])
    changes = [
        np.linalg.norm(outputs[index] - outputs[index - 1])
        for index in range(1, len(outputs))
    ]
    assert changes[2] < changes[1] < changes[0]


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


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"n_iter": 0}, ValueError, "n_iter"),
        ({"n_iter": 1.5}, TypeError, "n_iter"),
        ({"chunk_size": 0}, ValueError, "chunk_size"),
        ({"chunk_size": 2.5}, TypeError, "chunk_size"),
        ({"outlier_threshold": 0}, ValueError, "outlier_threshold"),
        ({"outlier_threshold": np.inf}, ValueError, "outlier_threshold"),
        ({"outlier_threshold": "bad"}, TypeError, "outlier_threshold"),
    ],
)
def test_compute_sns_rejects_invalid_refinements(rng, kwargs, error, match):
    """New operating points have explicit validation."""
    with pytest.raises(error, match=match):
        compute_sns(rng.standard_normal((5, 100)), **kwargs)


@pytest.mark.parametrize(
    "weight",
    [
        np.ones((2, 50)),
        np.full(100, np.nan),
        -np.ones(100),
        np.r_[1.0, np.zeros(99)],
    ],
)
def test_compute_sns_rejects_invalid_weights(rng, weight):
    """Weights must be finite, nonnegative, correctly shaped, and sufficient."""
    with pytest.raises(ValueError, match="sample|positively"):
        compute_sns(rng.standard_normal((5, 100)), sample_weight=weight)


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


def test_sns_estimator_uses_public_compute_sns(monkeypatch, rng):
    """The estimator delegates fitting to the public array algorithm."""
    import mne_denoise.sns as sns_core

    original = sns_core.compute_sns
    calls = []

    def recording_compute_sns(*args, **kwargs):
        calls.append(args[0].shape)
        return original(*args, **kwargs)

    monkeypatch.setattr(sns_core, "compute_sns", recording_compute_sns)
    SNS(n_neighbors=4, n_iter=2).fit(rng.standard_normal((8, 500)))
    assert calls == [(8, 500)]


def test_sns_suppresses_independent_sensor_noise(sensor_noise_data):
    """SNS recovers the shared low-rank signal, suppressing per-sensor noise."""
    X, shared = sensor_noise_data
    cleaned = SNS(n_neighbors=0).fit_transform(X)  # all others
    err_before = _rrmse(X, shared)
    err_after = _rrmse(cleaned, shared)
    assert err_after < 0.9 * err_before  # closer to the clean shared signal


def test_sns_fit_transform_numpy_shape(rng):
    """fit_transform on a NumPy array returns an array of the same shape."""
    X = rng.standard_normal((16, 4000))
    cleaned = SNS(n_neighbors=8).fit_transform(X)
    assert isinstance(cleaned, np.ndarray)
    assert cleaned.shape == X.shape


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


@pytest.mark.parametrize("preserve_mean", [False, True])
def test_sns_continuous_transform_is_batch_invariant(rng, preserve_mean):
    """Transforming a temporal chunk alone gives the same samples as a full call."""
    train = rng.standard_normal((8, 1000)) + np.arange(8)[:, None]
    evalu = rng.standard_normal((8, 400)) + 3.0
    est = SNS(n_neighbors=5, preserve_mean=preserve_mean).fit(train)
    full = est.transform(evalu)
    np.testing.assert_allclose(est.transform(evalu[:, 100:250]), full[:, 100:250])


@pytest.mark.parametrize("preserve_mean", [False, True])
def test_sns_epoch_transform_is_batch_invariant(rng, preserve_mean):
    """An epoch result is independent of the other epochs in the transform call."""
    train = rng.standard_normal((5, 7, 120))
    evalu = rng.standard_normal((4, 7, 80)) + 2.0
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


def test_sns_fit_transform_composes_and_clones(rng):
    """The sklearn composition is exact and all operating parameters clone."""
    X = rng.standard_normal((8, 500))
    estimator = SNS(n_neighbors=4, rcond=1e-10, preserve_mean=True)
    direct = estimator.fit_transform(X)
    separate = clone(estimator).fit(X).transform(X)
    np.testing.assert_allclose(direct, separate)
    assert clone(estimator).get_params() == estimator.get_params()


def test_sns_numpy_epochs_and_channel_count(rng):
    """Three-dimensional arrays concatenate only during fit and retain shape."""
    epochs = rng.standard_normal((3, 6, 100))
    estimator = SNS(n_neighbors=3).fit(epochs)
    cleaned = estimator.transform(epochs)
    assert cleaned.shape == epochs.shape
    np.testing.assert_allclose(cleaned.mean(axis=(0, 2)), 0.0, atol=1e-12)
    with pytest.raises(ValueError, match="fitted data had"):
        estimator.transform(rng.standard_normal((5, 100)))


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


@pytest.mark.parametrize(
    "X",
    [
        np.ones((1, 100)),
        np.ones((3, 1)),
        np.full((3, 100), np.nan),
    ],
)
def test_sns_rejects_invalid_data(X):
    """Data preconditions fail before covariance construction."""
    with pytest.raises(ValueError):
        SNS().fit(X)


def test_sns_transform_before_fit_raises(rng):
    """transform before fit raises NotFittedError."""
    from sklearn.exceptions import NotFittedError

    with pytest.raises(NotFittedError):
        SNS().transform(rng.standard_normal((8, 100)))


# ---------------------------------------------------------------------------
# MNE round-trip
# ---------------------------------------------------------------------------


def test_sns_mne_raw_roundtrip(sensor_noise_data):
    """fit_transform on an MNE Raw returns a Raw of identical shape."""
    mne = pytest.importorskip("mne")
    X, _shared = sensor_noise_data
    info = mne.create_info([f"EEG{i:02d}" for i in range(X.shape[0])], 250.0, "eeg")
    raw = mne.io.RawArray(X, info, verbose=False)

    cleaned = SNS(n_neighbors=0).fit_transform(raw)
    assert isinstance(cleaned, mne.io.BaseRaw)
    assert cleaned.get_data().shape == X.shape
    assert not np.allclose(cleaned.get_data(), X)


def test_sns_mne_preserves_metadata_and_unpicked_channel(sensor_noise_data):
    """SNS updates a copy without rebuilding Raw or touching excluded channels."""
    mne = pytest.importorskip("mne")
    X, _shared = sensor_noise_data
    sfreq = 250.0
    stim = np.arange(X.shape[1], dtype=float) % 2
    data = np.vstack((X, stim))
    info = mne.create_info(
        [*[f"EEG{i:02d}" for i in range(X.shape[0])], "STI 014"],
        sfreq,
        [*(["eeg"] * X.shape[0]), "stim"],
    )
    raw = mne.io.RawArray(data, info, first_samp=29, verbose=False)
    raw.set_annotations(mne.Annotations([0.5], [0.1], ["marker"]))

    cleaned = SNS(n_neighbors=8).fit_transform(raw)

    assert cleaned.first_samp == raw.first_samp
    assert cleaned.annotations == raw.annotations
    np.testing.assert_array_equal(cleaned.get_data(picks=["STI 014"])[0], stim)
    np.testing.assert_array_equal(raw.get_data(), data)


def test_sns_mne_mixed_types_uses_shared_channel_policy(rng):
    """Shared MNE selection cleans one type and preserves every other channel."""
    mne = pytest.importorskip("mne")
    data = rng.standard_normal((4, 500))
    info = mne.create_info(
        ["MAG1", "MAG2", "GRAD", "EEG"],
        250.0,
        ["mag", "mag", "grad", "eeg"],
    )
    raw = mne.io.RawArray(data, info, verbose=False)
    with pytest.warns(UserWarning, match="multiple data channel types"):
        cleaned = SNS(n_neighbors=1).fit_transform(raw)
    np.testing.assert_array_equal(cleaned.get_data()[2:], data[2:])
    assert not np.allclose(cleaned.get_data()[:2], data[:2])


def test_sns_mne_channel_order_must_match_fit(sensor_noise_data):
    """A learned spatial operator cannot be applied to reordered named channels."""
    mne = pytest.importorskip("mne")
    X, _shared = sensor_noise_data
    names = [f"EEG{i:02d}" for i in range(X.shape[0])]
    raw = mne.io.RawArray(X, mne.create_info(names, 250.0, "eeg"), verbose=False)
    estimator = SNS(n_neighbors=8).fit(raw)
    reordered = raw.copy().reorder_channels(names[::-1])
    with pytest.raises(ValueError, match="names/order"):
        estimator.transform(reordered)


def test_sns_mne_epochs_preserves_events_and_metadata(sensor_noise_data):
    """Epoch identities survive fit/apply and are not reconstructed from scratch."""
    pd = pytest.importorskip("pandas")
    mne = pytest.importorskip("mne")
    X, _shared = sensor_noise_data
    data = np.stack((X[:, :400], X[:, 400:800]))
    names = [f"EEG{i:02d}" for i in range(X.shape[0])]
    info = mne.create_info(names, 250.0, "eeg")
    events = np.array([[100, 0, 1], [700, 0, 2]])
    metadata = pd.DataFrame({"trial": ["a", "b"]})
    epochs = mne.EpochsArray(
        data,
        info,
        events=events,
        event_id={"a": 1, "b": 2},
        tmin=-0.1,
        metadata=metadata,
        verbose=False,
    )

    cleaned = SNS(n_neighbors=8).fit_transform(epochs)

    np.testing.assert_array_equal(cleaned.events, epochs.events)
    assert cleaned.event_id == epochs.event_id
    assert cleaned.metadata.equals(metadata)
    assert cleaned.tmin == epochs.tmin


def test_sns_mne_evoked_preserves_identity_fields(sensor_noise_data):
    """Evoked timing, averaging count, comment, and excluded channels survive."""
    mne = pytest.importorskip("mne")
    X, _shared = sensor_noise_data
    stim = np.arange(X.shape[1], dtype=float) % 2
    data = np.vstack((X, stim))
    names = [*[f"EEG{i:02d}" for i in range(X.shape[0])], "STI 014"]
    types = [*(["eeg"] * X.shape[0]), "stim"]
    evoked = mne.EvokedArray(
        data,
        mne.create_info(names, 250.0, types),
        tmin=-0.2,
        nave=17,
        comment="condition-a",
        verbose=False,
    )

    cleaned = SNS(n_neighbors=8).fit_transform(evoked)

    assert isinstance(cleaned, mne.Evoked)
    assert cleaned.nave == 17
    assert cleaned.comment == "condition-a"
    assert cleaned.tmin == evoked.tmin
    np.testing.assert_array_equal(cleaned.get_data(picks=["STI 014"])[0], stim)
    np.testing.assert_array_equal(evoked.data, data)


def test_sns_verbose_uses_package_logging(rng):
    """MNE-style string verbosity is scoped to the SNS operation."""

    package_logger = logging.getLogger("mne_denoise")
    previous = package_logger.level
    try:
        SNS(n_neighbors=3, verbose="ERROR").fit(rng.standard_normal((5, 100)))
        assert package_logger.level == previous
    finally:
        package_logger.setLevel(previous)


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
