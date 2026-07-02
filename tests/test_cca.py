"""Tests for the mne_denoise.cca module (reference-free BSS-CCA / auto-CCA)."""

from __future__ import annotations

import numpy as np
import pytest

from mne_denoise.cca import AutoCCA, compute_autocca

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def rng():
    """Shared random generator."""
    return np.random.default_rng(0)


@pytest.fixture()
def muscle_data(rng):
    """Synthetic EEG with autocorrelated neural sources + broadband EMG.

    Returns ``(X, clean, sfreq)`` where ``X`` is the contaminated recording,
    ``clean`` is the neural-only ground truth, and the artifact is broadband
    (weakly autocorrelated) EMG that auto-CCA should drop.
    """
    from scipy.signal import butter, filtfilt

    sfreq = 250.0
    n_times = 2000
    n_ch = 12
    t = np.arange(n_times) / sfreq

    # Neural sources: narrowband sinusoids -> autocorrelation ~1 (kept).
    neural = np.vstack(
        [np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi)) for f in (10.0, 11.0, 9.0)]
    )

    # EMG sources: high-passed white noise -> broadband, low autocorrelation (dropped).
    b, a = butter(4, 30.0 / (0.5 * sfreq), btype="high")
    emg = filtfilt(b, a, rng.standard_normal((3, n_times)), axis=-1)

    M_neural = rng.standard_normal((n_ch, 3))
    M_emg = rng.standard_normal((n_ch, 3)) * 1.5

    clean = M_neural @ neural
    artifact = M_emg @ emg
    X = clean + artifact + 0.05 * rng.standard_normal((n_ch, n_times))
    return X, clean, sfreq


def _band_power(X, sfreq, fmin, fmax):
    """Total power in [fmin, fmax) across channels."""
    spec = np.abs(np.fft.rfft(X, axis=-1)) ** 2
    freqs = np.fft.rfftfreq(X.shape[-1], 1.0 / sfreq)
    band = (freqs >= fmin) & (freqs < fmax)
    return float(spec[:, band].sum())


# ---------------------------------------------------------------------------
# compute_autocca (one-shot functional API)
# ---------------------------------------------------------------------------


def test_compute_autocca_shapes_and_info(rng):
    """compute_autocca returns cleaned data of the same shape + a diagnostics dict."""
    X = rng.standard_normal((8, 1000))
    X_clean, info = compute_autocca(X, rho_threshold=0.9)
    assert X_clean.shape == X.shape
    for key in ("cleaning_matrix", "filters", "patterns", "correlations",
                "kept_mask", "n_kept", "n_removed"):
        assert key in info
    assert info["cleaning_matrix"].shape == (8, 8)
    assert info["n_kept"] + info["n_removed"] == info["kept_mask"].size


def test_compute_autocca_rejects_1d():
    """A 1-D input raises a clear error."""
    with pytest.raises(ValueError, match="2-D"):
        compute_autocca(np.zeros(100))


# ---------------------------------------------------------------------------
# AutoCCA estimator
# ---------------------------------------------------------------------------


def test_autocca_fit_transform_numpy_shape(rng):
    """fit_transform on a NumPy array returns an array of the same shape."""
    X = rng.standard_normal((10, 1500))
    cleaned = AutoCCA(rho_threshold=0.9).fit_transform(X)
    assert isinstance(cleaned, np.ndarray)
    assert cleaned.shape == X.shape


def test_autocca_fitted_attributes(rng):
    """Fitted attributes are populated with correct shapes/properties."""
    X = rng.standard_normal((9, 1200))
    est = AutoCCA(rho_threshold=0.8).fit(X)
    assert est.cleaning_matrix_.shape == (9, 9)
    d = est.correlations_.size
    assert est.filters_.shape == (d, 9)
    assert est.patterns_.shape == (9, d)
    assert np.all(np.diff(est.correlations_) <= 1e-9)  # descending
    assert np.all(est.correlations_ >= -1e-9)
    assert np.all(est.correlations_ <= 1.0 + 1e-9)
    assert est.n_kept_ + est.n_removed_ == d


def test_autocca_reduces_broadband_preserves_alpha(muscle_data):
    """Auto-CCA drops broadband EMG power while preserving the alpha rhythm."""
    X, _clean, sfreq = muscle_data
    cleaned = AutoCCA(rho_threshold=0.9).fit_transform(X)

    hf_before = _band_power(X, sfreq, 30.0, 120.0)
    hf_after = _band_power(cleaned, sfreq, 30.0, 120.0)
    alpha_before = _band_power(X, sfreq, 8.0, 12.0)
    alpha_after = _band_power(cleaned, sfreq, 8.0, 12.0)

    assert hf_after < 0.8 * hf_before  # broadband muscle attenuated
    assert alpha_after > 0.7 * alpha_before  # alpha largely preserved


def test_autocca_leakage_split_matches_operator(rng):
    """transform applies the operator learned in fit (train != eval)."""
    train = rng.standard_normal((8, 1000))
    evalu = rng.standard_normal((8, 500))
    est = AutoCCA(rho_threshold=0.9).fit(train)
    cleaned = est.transform(evalu)
    m = evalu.mean(axis=1, keepdims=True)
    expected = est.cleaning_matrix_ @ (evalu - m) + m
    np.testing.assert_allclose(cleaned, expected)


def test_autocca_n_keep_keeps_exactly_k(rng):
    """n_keep keeps exactly k components regardless of threshold."""
    X = rng.standard_normal((10, 1500))
    est = AutoCCA(n_keep=3).fit(X)
    assert est.n_kept_ == 3


def test_autocca_transform_before_fit_raises(rng):
    """transform before fit raises NotFittedError."""
    from sklearn.exceptions import NotFittedError

    with pytest.raises(NotFittedError):
        AutoCCA().transform(rng.standard_normal((8, 100)))


# ---------------------------------------------------------------------------
# MNE round-trip
# ---------------------------------------------------------------------------


def test_autocca_mne_raw_roundtrip(muscle_data):
    """fit_transform on an MNE Raw returns a Raw of identical shape."""
    mne = pytest.importorskip("mne")
    X, _clean, sfreq = muscle_data
    info = mne.create_info(
        [f"EEG{i:02d}" for i in range(X.shape[0])], sfreq, "eeg"
    )
    raw = mne.io.RawArray(X, info, verbose=False)

    cleaned = AutoCCA(rho_threshold=0.9).fit_transform(raw)
    assert isinstance(cleaned, mne.io.BaseRaw)
    assert cleaned.get_data().shape == X.shape
    # Data was actually modified.
    assert not np.allclose(cleaned.get_data(), X)
