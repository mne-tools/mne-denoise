"""Tests for the mne_denoise.sns module (Sensor Noise Suppression)."""

from __future__ import annotations

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
# compute_sns_weights + compute_sns
# ---------------------------------------------------------------------------


def test_compute_sns_weights_shape_and_zero_diagonal(rng):
    """The operator is square with a zero diagonal (no self-regeneration)."""
    X = rng.standard_normal((12, 3000))
    Xd = X - X.mean(axis=1, keepdims=True)
    cov = (Xd @ Xd.T) / Xd.shape[1]
    W, k = compute_sns_weights(cov, n_neighbors=0)
    assert W.shape == (12, 12)
    assert k == 11  # all others
    np.testing.assert_allclose(np.diag(W), 0.0, atol=1e-12)


def test_compute_sns_weights_zero_diagonal_with_duplicate_channel(rng):
    """Self is excluded even when a duplicate channel ties the correlation (W[k,k]==0)."""
    X = rng.standard_normal((6, 3000))
    X[4] = X[0].copy()  # exact-duplicate channel -> corr**2 tie at 1.0
    Xd = X - X.mean(axis=1, keepdims=True)
    cov = (Xd @ Xd.T) / Xd.shape[1]
    W, _k = compute_sns_weights(cov, n_neighbors=0)
    np.testing.assert_allclose(np.diag(W), 0.0, atol=1e-12)


def test_compute_sns_weights_caps_neighbors():
    """n_neighbors and skip are capped at n_channels - skip - 1."""
    cov = np.eye(6)
    _W, k = compute_sns_weights(cov, n_neighbors=100, skip=2)
    assert k == 6 - 2 - 1  # == 3


def test_compute_sns_weights_is_unit_scale_invariant(rng):
    """The relative pseudoinverse cutoff must not depend on physical units."""
    X = rng.standard_normal((8, 1000))
    cov = np.cov(X)
    weights, _ = compute_sns_weights(cov, n_neighbors=4)
    scaled_weights, _ = compute_sns_weights(cov * 1e-12, n_neighbors=4)
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
def test_compute_sns_weights_rejects_invalid_covariance(cov, match):
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
def test_compute_sns_weights_rejects_invalid_operating_points(kwargs, error, match):
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
    np.testing.assert_allclose(X_clean.mean(axis=1), 0.0, atol=1e-12)


def test_compute_sns_meegkit_noisetools_behavior_bridge():
    """A frozen small case matches the NoiseTools-compatible MEEGkit operator."""
    # Generated independently with meegkit/sns.py at commit 8c9d44963e881409.
    X = np.array(
        [
            [1.0, 2.0, 0.0, 1.0, 3.0, 2.0, -1.0, 0.0],
            [0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 0.0, -1.0],
            [2.0, 3.0, 1.0, 0.0, 2.0, 1.0, -2.0, 1.0],
            [-1.0, 0.0, 2.0, 1.0, 1.0, 0.0, 3.0, 2.0],
        ]
    )
    expected_weights = np.array(
        [
            [0.0, 0.0, 0.4347826086956522, -0.3043478260869564],
            [0.8, 0.0, 0.0, 0.2],
            [0.5, 0.0, 0.0, -0.5],
            [-0.3043478260869564, 0.0, -0.4347826086956522, 0.0],
        ]
    )
    expected_cleaned = np.array(
        [
            [
                1.043478260869565,
                1.1739130434782608,
                -0.3043478260869564,
                -0.4347826086956522,
                0.4347826086956522,
                0.3043478260869564,
                -1.9130434782608692,
                -0.3043478260869564,
            ],
            [-0.4, 0.6, -0.6, 0.0, 1.6, 0.6, -1.2, -0.6],
            [1.0, 1.0, -1.0, 0.0, 1.0, 1.0, -2.0, -1.0],
            [
                -0.4347826086956522,
                -1.1739130434782608,
                0.3043478260869564,
                0.4347826086956522,
                -1.043478260869565,
                -0.3043478260869564,
                1.9130434782608694,
                0.3043478260869564,
            ],
        ]
    )

    cleaned, info = compute_sns(X, n_neighbors=2)

    np.testing.assert_allclose(info["weights"], expected_weights, atol=1e-12)
    np.testing.assert_allclose(cleaned, expected_cleaned, atol=1e-12)


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


# ---------------------------------------------------------------------------
# SNS estimator
# ---------------------------------------------------------------------------


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
    assert est.input_rank_ > 0


def test_sns_leakage_split_applies_operator(rng):
    """transform applies the operator learned in fit (train != eval)."""
    train = rng.standard_normal((10, 4000))
    evalu = rng.standard_normal((10, 800))
    est = SNS(n_neighbors=6).fit(train)
    cleaned = est.transform(evalu)
    expected = est.denoising_matrix_ @ (evalu - evalu.mean(axis=1, keepdims=True))
    np.testing.assert_allclose(cleaned, expected)


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
    with pytest.raises(ValueError, match="channel count"):
        estimator.transform(rng.standard_normal((5, 100)))


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
