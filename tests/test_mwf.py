"""Tests for the mne_denoise.mwf module (multi-channel Wiener filter)."""

from __future__ import annotations

import numpy as np
import pytest

from mne_denoise.mwf import MWF, compute_mwf, hf_power_mask, mwf_filter

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def rng():
    """Shared random generator."""
    return np.random.default_rng(0)


@pytest.fixture()
def burst_data(rng):
    """Synthetic EEG with intermittent broadband high-frequency bursts.

    Returns ``(X, sfreq, burst)`` where ``burst`` is the boolean sample mask of
    the artifact segments. The clean background is low-frequency; the artifact is
    a broadband HF burst injected into a subset of samples.
    """
    sfreq = 250.0
    n_times = 4000
    n_ch = 12
    t = np.arange(n_times) / sfreq

    # Low-frequency neural background (well separated from the HF artifact).
    neural = np.vstack(
        [np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi)) for f in (6.0, 8.0, 10.0)]
    )
    M_neural = rng.standard_normal((n_ch, 3))
    X = M_neural @ neural

    # Broadband HF bursts on ~30% of samples.
    burst = np.zeros(n_times, dtype=bool)
    for start in (400, 1500, 2600, 3300):
        burst[start : start + 300] = True
    hf = rng.standard_normal((n_ch, n_times)) * 3.0
    from scipy.signal import butter, filtfilt

    b, a = butter(4, 40.0 / (0.5 * sfreq), btype="high")
    hf = filtfilt(b, a, hf, axis=-1)
    X = X + hf * burst
    return X, sfreq, burst


def _band_power(X, sfreq, fmin, fmax):
    spec = np.abs(np.fft.rfft(np.atleast_2d(X), axis=-1)) ** 2
    freqs = np.fft.rfftfreq(X.shape[-1], 1.0 / sfreq)
    band = (freqs >= fmin) & (freqs < fmax)
    return float(spec[:, band].sum())


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------


def test_hf_power_mask_flags_bursts(burst_data):
    """The HF-power mask overlaps the injected burst samples."""
    X, sfreq, burst = burst_data
    mask = hf_power_mask(X, sfreq, hf_hz=20.0, quantile=0.6)
    assert mask.shape == (X.shape[-1],)
    # Detected artifact samples are enriched inside the true bursts.
    overlap = (mask & burst).sum() / max(mask.sum(), 1)
    assert overlap > 0.5


def test_compute_mwf_shapes_and_info(burst_data):
    """compute_mwf returns cleaned data + a diagnostics dict."""
    X, sfreq, _burst = burst_data
    cleaned, info = compute_mwf(X, sfreq)
    assert cleaned.shape == X.shape
    assert info["mask"].shape == (X.shape[-1],)
    assert 0.0 <= info["artifact_fraction"] <= 1.0


def test_compute_mwf_rejects_1d():
    """A 1-D input raises a clear error."""
    with pytest.raises(ValueError, match="2-D"):
        compute_mwf(np.zeros(100), 250.0)


def test_mwf_filter_identity_when_covariances_match(rng):
    """With a random mask on stationary data the filter is ~identity (no over-clean)."""
    X = rng.standard_normal((8, 4000))
    mask = np.zeros(4000, dtype=bool)
    mask[::2] = True  # arbitrary split of stationary data
    cleaned = mwf_filter(X, mask, reg=1e-6)
    # Stationary data -> R_clean ~ R_artifact -> filter ~ identity.
    assert np.corrcoef(cleaned.ravel(), X.ravel())[0, 1] > 0.9


# ---------------------------------------------------------------------------
# MWF estimator
# ---------------------------------------------------------------------------


def test_mwf_fit_transform_numpy_shape(burst_data):
    """fit_transform on a NumPy array returns an array of the same shape."""
    X, sfreq, _burst = burst_data
    cleaned = MWF(sfreq=sfreq).fit_transform(X)
    assert isinstance(cleaned, np.ndarray)
    assert cleaned.shape == X.shape


def test_mwf_reduces_burst_hf_power(burst_data):
    """MWF attenuates the broadband HF burst power."""
    X, sfreq, _burst = burst_data
    cleaned = MWF(sfreq=sfreq, hf_hz=20.0, quantile=0.6).fit_transform(X)
    hf_before = _band_power(X, sfreq, 40.0, 120.0)
    hf_after = _band_power(cleaned, sfreq, 40.0, 120.0)
    assert hf_after < hf_before


def test_mwf_fitted_attributes(burst_data):
    """Fitted attributes are populated with correct shapes."""
    X, sfreq, _burst = burst_data
    est = MWF(sfreq=sfreq).fit(X)
    assert est.spatial_filter_.shape == (X.shape[0], X.shape[0])
    assert est.artifact_mask_.shape == (X.shape[-1],)
    assert 0.0 <= est.artifact_fraction_ <= 1.0


def test_mwf_leakage_split_applies_operator(rng):
    """transform applies the operator learned in fit (train != eval)."""
    train = rng.standard_normal((8, 4000))
    evalu = rng.standard_normal((8, 1000))
    mask = np.zeros(4000, dtype=bool)
    mask[:1500] = True
    est = MWF().fit(train, mask=mask)  # explicit mask -> no sfreq needed
    cleaned = est.transform(evalu)
    np.testing.assert_allclose(cleaned, est.spatial_filter_ @ evalu)


def test_mwf_requires_sfreq_for_array_without_mask(burst_data):
    """Array input without sfreq or mask raises a clear error."""
    X, _sfreq, _burst = burst_data
    with pytest.raises(ValueError, match="sfreq is required"):
        MWF().fit(X)


def test_mwf_transform_before_fit_raises(rng):
    """transform before fit raises NotFittedError."""
    from sklearn.exceptions import NotFittedError

    with pytest.raises(NotFittedError):
        MWF().transform(rng.standard_normal((8, 100)))


# ---------------------------------------------------------------------------
# MNE round-trip
# ---------------------------------------------------------------------------


def test_mwf_mne_raw_roundtrip_infers_sfreq(burst_data):
    """fit_transform on an MNE Raw returns a Raw of identical shape; sfreq inferred."""
    mne = pytest.importorskip("mne")
    X, sfreq, _burst = burst_data
    info = mne.create_info([f"EEG{i:02d}" for i in range(X.shape[0])], sfreq, "eeg")
    raw = mne.io.RawArray(X, info, verbose=False)

    cleaned = MWF(hf_hz=20.0, quantile=0.6).fit_transform(raw)  # sfreq from info
    assert isinstance(cleaned, mne.io.BaseRaw)
    assert cleaned.get_data().shape == X.shape
    assert not np.allclose(cleaned.get_data(), X)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_hf_power_mask_short_signal_length(rng):
    """The mask length equals n_times even when the smoothing window is longer."""
    # 20 samples at 250 Hz -> default smooth window (25 samples) exceeds length.
    X = rng.standard_normal((8, 20))
    mask = hf_power_mask(X, 250.0)
    assert mask.shape == (20,)


def test_mwf_short_signal_no_crash(rng):
    """compute_mwf / MWF.fit_transform do not crash on very short signals."""
    X = rng.standard_normal((8, 20))
    cleaned, info = compute_mwf(X, 250.0)
    assert cleaned.shape == X.shape
    assert info["mask"].shape == (20,)
    cleaned2 = MWF(sfreq=250.0).fit_transform(X)
    assert cleaned2.shape == X.shape


def test_mwf_single_channel_is_identity(rng):
    """A single channel is degenerate for a spatial filter -> returned unchanged."""
    X = rng.standard_normal((1, 2000))
    cleaned = MWF(sfreq=250.0).fit_transform(X)
    assert cleaned.shape == X.shape
    np.testing.assert_allclose(cleaned, X)
    # functional API too
    cleaned_f, _ = compute_mwf(X, 250.0)
    assert cleaned_f.shape == X.shape


def test_mwf_int_mask_matches_bool_mask(rng):
    """An integer 0/1 mask gives the same result as the equivalent bool mask."""
    X = rng.standard_normal((8, 2000))
    bool_mask = np.zeros(2000, dtype=bool)
    bool_mask[:800] = True
    int_mask = bool_mask.astype(int)
    cleaned_bool, _ = compute_mwf(X, 250.0, mask=bool_mask)
    cleaned_int, _ = compute_mwf(X, 250.0, mask=int_mask)
    np.testing.assert_allclose(cleaned_bool, cleaned_int)
