"""Tests for the mne_denoise.sns module (Sensor Noise Suppression)."""

from __future__ import annotations

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


def test_compute_sns_shapes_and_info(rng):
    """compute_sns returns cleaned data + diagnostics."""
    X = rng.standard_normal((10, 2000))
    X_clean, info = compute_sns(X, n_neighbors=6)
    assert X_clean.shape == X.shape
    assert info["weights"].shape == (10, 10)
    assert info["n_neighbors"] == 6


def test_compute_sns_rejects_1d():
    """A 1-D input raises a clear error."""
    with pytest.raises(ValueError, match="2-D"):
        compute_sns(np.zeros(100))


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


def test_sns_leakage_split_applies_operator(rng):
    """transform applies the operator learned in fit (train != eval)."""
    train = rng.standard_normal((10, 4000))
    evalu = rng.standard_normal((10, 800))
    est = SNS(n_neighbors=6).fit(train)
    cleaned = est.transform(evalu)
    np.testing.assert_allclose(cleaned, est.denoising_matrix_ @ evalu)


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
