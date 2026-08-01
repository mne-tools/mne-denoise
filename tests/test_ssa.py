"""Tests for the mne_denoise.ssa module (Singular Spectrum Analysis)."""

from __future__ import annotations

import logging

import numpy as np
import pytest
from sklearn.base import clone

from mne_denoise.ssa import (
    SSA,
    SingularSpectrumAnalysis,
    compute_ssa,
    ssa_clean_channel,
)

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
# SingularSpectrumAnalysis estimator
# ---------------------------------------------------------------------------


def test_ssa_fit_transform_numpy(drift_data):
    """fit_transform on a NumPy array returns an array of the same shape."""
    X, sfreq = drift_data
    cleaned = SingularSpectrumAnalysis(sfreq=sfreq, drop_freq_max=3.0).fit_transform(X)
    assert isinstance(cleaned, np.ndarray)
    assert cleaned.shape == X.shape


def test_ssa_attributes_after_transform(drift_data):
    """Diagnostics attributes are populated after transform."""
    X, sfreq = drift_data
    est = SingularSpectrumAnalysis(sfreq=sfreq).fit(X)
    est.transform(X)
    assert est.n_channels_ == X.shape[0]
    assert est.dropped_counts_.shape == (X.shape[0],)


def test_ssa_requires_sfreq_for_array(drift_data):
    """Array input without sfreq raises a clear error."""
    X, _sfreq = drift_data
    with pytest.raises(ValueError, match="sfreq is required"):
        SingularSpectrumAnalysis().fit_transform(X)


def test_ssa_alias_and_positional_sfreq(drift_data):
    """The concise SSA alias exposes the canonical estimator unchanged."""
    X, sfreq = drift_data
    assert SSA is SingularSpectrumAnalysis
    cleaned = SSA(sfreq).fit_transform(X)
    assert cleaned.shape == X.shape


def test_ssa_fit_transform_composes_and_clones(drift_data):
    """fit_transform is exactly fit followed by transform and clones cleanly."""
    X, sfreq = drift_data
    estimator = SingularSpectrumAnalysis(sfreq=sfreq, n_check=5)
    direct = estimator.fit_transform(X)
    separate = clone(estimator).fit(X).transform(X)
    np.testing.assert_allclose(direct, separate)
    assert clone(estimator).get_params() == estimator.get_params()


def test_ssa_numpy_epochs_retain_all_diagnostics(drift_data):
    """Three-dimensional arrays are cleaned without overwriting epoch diagnostics."""
    X, sfreq = drift_data
    epochs = np.stack((X, 0.5 * X))
    estimator = SingularSpectrumAnalysis(sfreq=sfreq, n_check=5)
    cleaned = estimator.fit_transform(epochs)
    assert cleaned.shape == epochs.shape
    assert estimator.dropped_counts_.shape == epochs.shape[:2]
    assert len(estimator.dropped_frequencies_) == epochs.shape[0]
    assert all(
        len(freqs) == epochs.shape[1] for freqs in estimator.dropped_frequencies_
    )


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"sfreq": 0}, ValueError, "positive"),
        ({"sfreq": np.inf}, ValueError, "finite"),
        ({"sfreq": 250, "window_length": 1}, ValueError, "window_length"),
        ({"sfreq": 250, "window_length": 101}, ValueError, "half"),
        ({"sfreq": 250, "n_check": 0}, ValueError, "n_check"),
        ({"sfreq": 250, "max_window": 1}, ValueError, "max_window"),
        ({"sfreq": 250, "drop_freq_max": 126}, ValueError, "Nyquist"),
        ({"sfreq": 250, "drop_band": (2.0, 1.0)}, ValueError, "drop_band"),
    ],
)
def test_ssa_rejects_invalid_operating_points(kwargs, error, match):
    """Mathematically invalid operating points fail before decomposition."""
    with pytest.raises(error, match=match):
        SingularSpectrumAnalysis(**kwargs).fit_transform(np.ones((2, 200)))


def test_ssa_rejects_short_and_nonfinite_inputs():
    """Inputs incapable of a finite trajectory decomposition fail explicitly."""
    with pytest.raises(ValueError, match="at least 8"):
        compute_ssa(np.ones((2, 7)), 250.0)
    nonfinite = np.ones((2, 100))
    nonfinite[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        compute_ssa(nonfinite, 250.0)


def test_ssa_zero_hz_component_is_not_hidden():
    """DC is a valid dominant frequency and can be targeted explicitly."""
    x = np.full(200, 3.0)
    cleaned = ssa_clean_channel(x, 100.0, drop_freq_max=0.0, n_check=1)
    assert np.linalg.norm(cleaned) < 1e-10


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
    cleaned = SingularSpectrumAnalysis(drop_freq_max=3.0).fit_transform(raw)
    assert isinstance(cleaned, mne.io.BaseRaw)
    assert cleaned.get_data().shape == X.shape
    low_before = _band_power(X, sfreq, 0.0, 3.0)
    low_after = _band_power(cleaned.get_data(), sfreq, 0.0, 3.0)
    assert low_after < low_before


def test_ssa_mne_raw_preserves_container_and_unpicked_channel(drift_data):
    """Cleaning copies Raw metadata and leaves auto-excluded channels untouched."""
    mne = pytest.importorskip("mne")
    X, sfreq = drift_data
    stim = np.arange(X.shape[1], dtype=float) % 2
    data = np.vstack((X, stim))
    info = mne.create_info(
        [*[f"EEG{i:02d}" for i in range(X.shape[0])], "STI 014"],
        sfreq,
        [*["eeg"] * X.shape[0], "stim"],
    )
    raw = mne.io.RawArray(data, info, first_samp=37, verbose=False)
    raw.set_annotations(mne.Annotations([0.5], [0.1], ["marker"]))

    cleaned = SingularSpectrumAnalysis(drop_freq_max=3.0).fit_transform(raw)

    assert cleaned is not raw
    assert cleaned.first_samp == raw.first_samp
    assert cleaned.annotations == raw.annotations
    np.testing.assert_array_equal(cleaned.get_data(picks=["STI 014"])[0], stim)
    np.testing.assert_array_equal(raw.get_data(), data)


def test_ssa_mne_epochs_preserves_events_metadata_and_diagnostics(drift_data):
    """Epoch cleaning preserves identities and records every epoch/channel."""
    pd = pytest.importorskip("pandas")
    mne = pytest.importorskip("mne")
    X, sfreq = drift_data
    epoch_data = np.stack((X[:, :500], X[:, 500:1000]))
    info = mne.create_info([f"EEG{i:02d}" for i in range(X.shape[0])], sfreq, "eeg")
    events = np.array([[100, 0, 1], [800, 0, 2]])
    metadata = pd.DataFrame({"trial": ["a", "b"]})
    epochs = mne.EpochsArray(
        epoch_data,
        info,
        events=events,
        event_id={"a": 1, "b": 2},
        tmin=-0.2,
        metadata=metadata,
        verbose=False,
    )
    estimator = SingularSpectrumAnalysis(drop_freq_max=3.0, n_check=5)

    cleaned = estimator.fit_transform(epochs)

    np.testing.assert_array_equal(cleaned.events, epochs.events)
    assert cleaned.event_id == epochs.event_id
    assert cleaned.metadata.equals(metadata)
    assert cleaned.tmin == epochs.tmin
    assert estimator.dropped_counts_.shape == epoch_data.shape[:2]


def test_ssa_rejects_conflicting_mne_sfreq(drift_data):
    """An explicit sampling frequency cannot silently override MNE metadata."""
    mne = pytest.importorskip("mne")
    X, sfreq = drift_data
    raw = mne.io.RawArray(
        X,
        mne.create_info([f"EEG{i:02d}" for i in range(X.shape[0])], sfreq, "eeg"),
        verbose=False,
    )
    with pytest.raises(ValueError, match="disagrees"):
        SingularSpectrumAnalysis(sfreq=sfreq / 2).fit(raw)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"sfreq": True}, TypeError, "sfreq"),
        ({"sfreq": 250.0, "drop_freq_max": True}, TypeError, "drop_freq_max"),
        ({"sfreq": 250.0, "drop_freq_max": np.nan}, ValueError, "finite"),
        ({"sfreq": 250.0, "drop_band": [1.0, 2.0]}, TypeError, "drop_band"),
        ({"sfreq": 250.0, "drop_band": (True, 2.0)}, TypeError, "bounds"),
        ({"sfreq": 250.0, "drop_band": (1.0, np.nan)}, TypeError, "bounds"),
    ],
)
def test_ssa_scalar_contracts_reject_ambiguous_values(kwargs, error, match):
    """Boolean, non-finite, and structurally ambiguous parameters fail clearly."""
    with pytest.raises(error, match=match):
        SingularSpectrumAnalysis(**kwargs).fit_transform(np.ones((2, 200)))


def test_single_channel_primitive_validates_shape_and_finiteness():
    """The single-channel API rejects multidimensional and non-finite series."""
    with pytest.raises(TypeError, match="sfreq"):
        compute_ssa(np.ones((2, 20)), True)
    with pytest.raises(ValueError, match="one-dimensional"):
        ssa_clean_channel(np.ones((2, 20)), 100.0)
    nonfinite = np.ones(20)
    nonfinite[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        ssa_clean_channel(nonfinite, 100.0)


def test_zero_singular_values_are_skipped_without_inventing_energy():
    """Zero-energy eigentriples remain zero and are not reported as artifacts."""
    cleaned = ssa_clean_channel(np.zeros(100), 100.0, drop_freq_max=3.0, n_check=5)
    np.testing.assert_array_equal(cleaned, np.zeros(100))


def test_empty_channels_and_fit_input_validation():
    """Functional and estimator entry points reject empty or non-finite channels."""
    with pytest.raises(ValueError, match="at least one channel"):
        compute_ssa(np.empty((0, 100)), 100.0)
    with pytest.raises(ValueError, match="at least one channel"):
        SingularSpectrumAnalysis(sfreq=100.0).fit(np.empty((0, 100)))
    nonfinite = np.ones((2, 100))
    nonfinite[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        SingularSpectrumAnalysis(sfreq=100.0).fit(nonfinite)


def test_ssa_mne_evoked_preserves_metadata_and_stim_channel(drift_data):
    """Evoked cleaning copies metadata and leaves auto-excluded channels untouched."""
    mne = pytest.importorskip("mne")
    X, sfreq = drift_data
    stim = np.arange(X.shape[1], dtype=float) % 2
    data = np.vstack((X[:3], stim))
    info = mne.create_info(
        ["EEG0", "EEG1", "EEG2", "STI 014"],
        sfreq,
        ["eeg", "eeg", "eeg", "stim"],
    )
    evoked = mne.EvokedArray(
        data, info, tmin=-0.2, nave=14, comment="condition", verbose=False
    )
    cleaned = SingularSpectrumAnalysis(drop_freq_max=3.0).fit_transform(evoked)
    assert isinstance(cleaned, mne.Evoked)
    assert cleaned.comment == evoked.comment
    assert cleaned.nave == evoked.nave
    assert cleaned.first == evoked.first
    np.testing.assert_array_equal(cleaned.data[-1], stim)


def test_ssa_verbose_reports_dropped_component_summary(drift_data, caplog):
    """Opt-in logging emits the descriptive fitted-run summary."""
    X, sfreq = drift_data
    with caplog.at_level(logging.INFO, logger="mne_denoise.ssa.core"):
        SingularSpectrumAnalysis(sfreq=sfreq, verbose=True).fit_transform(X[:2])
    assert "SSA: dropped a mean" in caplog.text
