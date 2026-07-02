"""Tests for the mne_denoise.ssa module (Singular Spectrum Analysis)."""

from __future__ import annotations

import numpy as np
import pytest

from mne_denoise.ssa import SSA, compute_ssa, ssa_clean_channel

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def rng():
    """Shared random generator."""
    return np.random.default_rng(0)


@pytest.fixture()
def drift_data(rng):
    """Synthetic EEG with a strong slow drift + an alpha rhythm.

    Returns ``(X, sfreq)``. The slow drift (0.4 Hz) should be dropped by SSA
    (dominant frequency below ``drop_freq_max``) while the 10 Hz alpha rhythm is
    preserved.
    """
    sfreq = 250.0
    n_times = 2000
    n_ch = 6
    t = np.arange(n_times) / sfreq
    X = np.empty((n_ch, n_times))
    for c in range(n_ch):
        drift = 4.0 * np.sin(2 * np.pi * 0.4 * t + rng.uniform(0, 2 * np.pi))
        alpha = 1.0 * np.sin(2 * np.pi * 10.0 * t + rng.uniform(0, 2 * np.pi))
        X[c] = drift + alpha + 0.05 * rng.standard_normal(n_times)
    return X, sfreq


def _band_power(x, sfreq, fmin, fmax):
    """Total power in [fmin, fmax) for a 1-D or 2-D array (summed over channels)."""
    x = np.atleast_2d(x)
    spec = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    freqs = np.fft.rfftfreq(x.shape[-1], 1.0 / sfreq)
    band = (freqs >= fmin) & (freqs < fmax)
    return float(spec[:, band].sum())


# ---------------------------------------------------------------------------
# ssa_clean_channel + compute_ssa
# ---------------------------------------------------------------------------


def test_ssa_clean_channel_removes_drift(drift_data):
    """Single-channel SSA removes the slow drift, preserves alpha."""
    X, sfreq = drift_data
    x = X[0]
    cleaned = ssa_clean_channel(x, sfreq, drop_freq_max=3.0)
    assert cleaned.shape == x.shape
    low_before = _band_power(x, sfreq, 0.0, 3.0)
    low_after = _band_power(cleaned, sfreq, 0.0, 3.0)
    alpha_before = _band_power(x, sfreq, 8.0, 12.0)
    alpha_after = _band_power(cleaned, sfreq, 8.0, 12.0)
    assert low_after < 0.5 * low_before
    assert alpha_after > 0.7 * alpha_before


def test_compute_ssa_shapes_and_info(drift_data):
    """compute_ssa returns cleaned data + per-channel diagnostics."""
    X, sfreq = drift_data
    cleaned, info = compute_ssa(X, sfreq, drop_freq_max=3.0)
    assert cleaned.shape == X.shape
    assert info["dropped_counts"].shape == (X.shape[0],)
    assert len(info["dropped_freqs"]) == X.shape[0]
    assert np.all(info["dropped_counts"] >= 1)  # drift dropped in every channel


def test_compute_ssa_rejects_1d():
    """A 1-D input to compute_ssa raises a clear error."""
    with pytest.raises(ValueError, match="2-D"):
        compute_ssa(np.zeros(100), 250.0)


def test_compute_ssa_drop_band(rng):
    """drop_band targets a specific frequency band instead of the low end."""
    sfreq = 250.0
    n = 2000
    t = np.arange(n) / sfreq
    # 1.2 Hz "cardiac-like" component + 10 Hz alpha.
    x = 3.0 * np.sin(2 * np.pi * 1.2 * t) + np.sin(2 * np.pi * 10.0 * t)
    X = np.vstack([x, x])
    cleaned, _ = compute_ssa(X, sfreq, drop_band=(0.8, 1.6))
    band_before = _band_power(X, sfreq, 0.8, 1.6)
    band_after = _band_power(cleaned, sfreq, 0.8, 1.6)
    alpha_after = _band_power(cleaned, sfreq, 8.0, 12.0)
    alpha_before = _band_power(X, sfreq, 8.0, 12.0)
    assert band_after < 0.5 * band_before
    assert alpha_after > 0.7 * alpha_before


# ---------------------------------------------------------------------------
# SSA estimator
# ---------------------------------------------------------------------------


def test_ssa_fit_transform_numpy(drift_data):
    """fit_transform on a NumPy array returns an array of the same shape."""
    X, sfreq = drift_data
    cleaned = SSA(sfreq=sfreq, drop_freq_max=3.0).fit_transform(X)
    assert isinstance(cleaned, np.ndarray)
    assert cleaned.shape == X.shape


def test_ssa_attributes_after_transform(drift_data):
    """Diagnostics attributes are populated after transform."""
    X, sfreq = drift_data
    est = SSA(sfreq=sfreq).fit(X)
    est.transform(X)
    assert est.n_channels_ == X.shape[0]
    assert est.dropped_counts_.shape == (X.shape[0],)


def test_ssa_requires_sfreq_for_array(drift_data):
    """Array input without sfreq raises a clear error."""
    X, _sfreq = drift_data
    with pytest.raises(ValueError, match="sfreq is required"):
        SSA().fit_transform(X)


def test_ssa_positional_sfreq_backward_compat(drift_data):
    """SSA(sfreq) positional call still works (backward compatible)."""
    X, sfreq = drift_data
    cleaned = SSA(sfreq).fit_transform(X)
    assert cleaned.shape == X.shape


# ---------------------------------------------------------------------------
# MNE round-trip
# ---------------------------------------------------------------------------


def test_ssa_mne_raw_roundtrip_infers_sfreq(drift_data):
    """fit_transform on an MNE Raw returns a Raw of identical shape; sfreq inferred."""
    mne = pytest.importorskip("mne")
    X, sfreq = drift_data
    info = mne.create_info([f"EEG{i:02d}" for i in range(X.shape[0])], sfreq, "eeg")
    raw = mne.io.RawArray(X, info, verbose=False)

    # No sfreq passed -> read from info.
    cleaned = SSA(drop_freq_max=3.0).fit_transform(raw)
    assert isinstance(cleaned, mne.io.BaseRaw)
    assert cleaned.get_data().shape == X.shape
    low_before = _band_power(X, sfreq, 0.0, 3.0)
    low_after = _band_power(cleaned.get_data(), sfreq, 0.0, 3.0)
    assert low_after < low_before
