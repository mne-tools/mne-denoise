"""Tests for Basic SSA decomposition, grouping, and integration."""

from __future__ import annotations

import inspect
import logging

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from mne_denoise.ssa import (
    LocalSingularSpectrumAnalysis,
    SingularSpectrumAnalysis,
    compute_basic_ssa,
    ssa_clean_channel,
    ssa_decompose,
    ssa_w_correlation,
)
from mne_denoise.ssa._common import _diagonal_average

from ._utils import band_power

# ---------------------------------------------------------------------------
# Mathematical core
# ---------------------------------------------------------------------------


def test_diagonal_average_includes_edge_weights():
    """A hand-computed matrix verifies every anti-diagonal and edge weight."""
    matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    np.testing.assert_array_equal(_diagonal_average(matrix), [1.0, 3.0, 4.0, 6.0])


@pytest.mark.parametrize(("n_times", "window"), [(7, 3), (30, 15), (31, 16)])
def test_complete_elementary_reconstruction(n_times, window, rng):
    """Every valid L/K orientation reconstructs the source additively."""
    x = rng.standard_normal(n_times)
    components, info = ssa_decompose(x, window)
    np.testing.assert_allclose(components.sum(axis=0), x, rtol=2e-14, atol=2e-14)
    assert info["trajectory_shape"] == (window, n_times - window + 1)
    assert np.all(np.diff(info["singular_values"]) <= 0)


def test_constant_decomposition_and_w_correlation():
    """A rank-one constant series has one exact component and finite diagnostics."""
    x = np.full(20, 3.0)
    components, info = ssa_decompose(x, 5)
    assert info["rank"] == 1
    np.testing.assert_allclose(components[0], x, atol=1e-13)
    np.testing.assert_allclose(components[1:], 0.0, atol=1e-13)
    correlation = ssa_w_correlation(components, 5)
    assert correlation.shape == (5, 5)
    assert correlation[0, 0] == pytest.approx(1.0)
    assert np.isfinite(correlation).all()


def test_sample_and_second_windows_are_equivalent(rng):
    """Time- and sample-based embedding specifications resolve identically."""
    x = rng.standard_normal(80)
    by_sample, sample_info = ssa_decompose(x, 20, sfreq=100.0)
    by_time, time_info = ssa_decompose(x, window_seconds=0.2, sfreq=100.0)
    np.testing.assert_allclose(by_sample, by_time)
    assert sample_info["window_length"] == time_info["window_length"] == 20


def test_transform_before_fit_is_rejected(drift_data):
    """The transductive estimator still follows the sklearn fitted-state contract."""
    X, sfreq = drift_data
    with pytest.raises(NotFittedError):
        SingularSpectrumAnalysis(sfreq=sfreq).transform(X)


@pytest.mark.parametrize(
    ("estimator_type", "kwargs"),
    [
        (SingularSpectrumAnalysis, {"sfreq": 100.0}),
        (LocalSingularSpectrumAnalysis, {}),
    ],
)
def test_ssa_fit_transform_validates_callback_before_fit(estimator_type, kwargs):
    """Invalid SSA callbacks cannot leave a transformer fitted."""
    estimator = estimator_type(**kwargs)
    with pytest.raises(TypeError, match="callback must be callable or None"):
        estimator.fit_transform(np.ones((2, 100)), callback=1)
    assert not hasattr(estimator, "is_fitted_")


def test_array_input_is_immutable(drift_data):
    """Functional and estimator entry points never mutate caller-owned arrays."""
    X, sfreq = drift_data
    original = X.copy()
    compute_basic_ssa(X, sfreq, window_length=40)
    np.testing.assert_array_equal(X, original)
    SingularSpectrumAnalysis(sfreq=sfreq, window_length=40).fit_transform(X)
    np.testing.assert_array_equal(X, original)


def test_epoch_boundaries_are_explicitly_transductive(drift_data):
    """Independent epoch decompositions need not equal one continuous decomposition."""
    X, sfreq = drift_data
    continuous = SingularSpectrumAnalysis(sfreq=sfreq, window_length=40).fit_transform(
        X[:2, :800]
    )
    epochs = X[:2, :800].reshape(2, 2, 400).transpose(1, 0, 2)
    estimator = SingularSpectrumAnalysis(sfreq=sfreq, window_length=40)
    split = estimator.fit_transform(epochs).transpose(1, 0, 2).reshape(2, 800)
    assert not np.allclose(continuous, split)


def test_estimator_delegates_to_public_compute(monkeypatch, drift_data):
    """The estimator's canonical numerical path is the public compute function."""
    X, sfreq = drift_data
    calls = []

    def fake_compute(data, passed_sfreq, *args, **kwargs):
        calls.append((data.copy(), passed_sfreq))
        return data.copy(), {
            "dropped_counts": np.zeros(data.shape[0], dtype=int),
            "dropped_frequencies": [np.empty(0) for _ in data],
        }

    monkeypatch.setattr("mne_denoise.ssa.basic.compute_basic_ssa", fake_compute)
    result = SingularSpectrumAnalysis(sfreq=sfreq).fit_transform(X)
    np.testing.assert_array_equal(result, X)
    assert len(calls) == 1
    assert calls[0][1] == sfreq


# ---------------------------------------------------------------------------
# ssa_clean_channel + compute_basic_ssa
# ---------------------------------------------------------------------------


def test_ssa_clean_channel_removes_drift(drift_data):
    """Single-channel SSA removes the slow drift, preserves alpha."""
    X, sfreq = drift_data
    x = X[0]
    cleaned = ssa_clean_channel(x, sfreq, drop_freq_max=3.0)
    assert cleaned.shape == x.shape
    low_before = band_power(x, sfreq, 0.0, 3.0)
    low_after = band_power(cleaned, sfreq, 0.0, 3.0)
    alpha_before = band_power(x, sfreq, 8.0, 12.0)
    alpha_after = band_power(cleaned, sfreq, 8.0, 12.0)
    assert low_after < 0.5 * low_before
    assert alpha_after > 0.7 * alpha_before


def test_compute_basic_ssa_shapes_and_info(drift_data):
    """compute_basic_ssa returns cleaned data and per-channel diagnostics."""
    X, sfreq = drift_data
    cleaned, info = compute_basic_ssa(X, sfreq, drop_freq_max=3.0)
    assert cleaned.shape == X.shape
    assert info["dropped_counts"].shape == (X.shape[0],)
    assert len(info["dropped_freqs"]) == X.shape[0]
    assert np.all(info["dropped_counts"] >= 1)  # drift dropped in every channel


def test_compute_basic_ssa_progress_callback(drift_data):
    """Basic SSA reports one completed channel with its drop count."""
    X, sfreq = drift_data
    X = X[:3, :240]
    events = []
    cleaned, info = compute_basic_ssa(
        X,
        sfreq,
        window_length=20,
        callback=events.append,
    )

    assert cleaned.shape == X.shape
    assert len(events) == X.shape[0]
    assert [event.method for event in events] == ["basic_ssa"] * X.shape[0]
    assert [event.stage for event in events] == ["channel"] * X.shape[0]
    assert [event.current for event in events] == list(range(1, X.shape[0] + 1))
    assert [event.total for event in events] == [X.shape[0]] * X.shape[0]
    assert all(event.component is None for event in events)
    np.testing.assert_array_equal(
        [event.metric for event in events], info["dropped_counts"].astype(float)
    )


def test_compute_basic_ssa_callback_is_numerically_transparent(drift_data):
    """Basic SSA callbacks do not alter cleaned data or diagnostics."""
    X, sfreq = drift_data
    X = X[:3, :240]
    without, without_info = compute_basic_ssa(X, sfreq, window_length=20)
    events = []
    with_callback, with_info = compute_basic_ssa(
        X, sfreq, window_length=20, callback=events.append
    )

    np.testing.assert_allclose(with_callback, without)
    np.testing.assert_array_equal(
        with_info["dropped_counts"], without_info["dropped_counts"]
    )
    for expected, actual in zip(
        without_info["dropped_frequencies"],
        with_info["dropped_frequencies"],
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)


def test_compute_basic_ssa_rejects_1d():
    """A 1-D input to compute_basic_ssa raises a clear error."""
    with pytest.raises(ValueError, match="2-D"):
        compute_basic_ssa(np.zeros(100), 250.0)


def test_compute_basic_ssa_drop_band(rng):
    """drop_band targets a specific frequency band instead of the low end."""
    sfreq = 250.0
    n = 2000
    t = np.arange(n) / sfreq
    # 1.2 Hz "cardiac-like" component + 10 Hz alpha.
    x = 3.0 * np.sin(2 * np.pi * 1.2 * t) + np.sin(2 * np.pi * 10.0 * t)
    X = np.vstack([x, x])
    cleaned, _ = compute_basic_ssa(X, sfreq, drop_band=(0.8, 1.6))
    band_before = band_power(X, sfreq, 0.8, 1.6)
    band_after = band_power(cleaned, sfreq, 0.8, 1.6)
    alpha_after = band_power(cleaned, sfreq, 8.0, 12.0)
    alpha_before = band_power(X, sfreq, 8.0, 12.0)
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
    assert est.n_channels_in_ == X.shape[0]
    assert est.dropped_counts_.shape == (X.shape[0],)


def test_ssa_continuous_transform_progress_callback(drift_data):
    """Continuous SSA emits channel events owned by its functional core."""
    X, sfreq = drift_data
    X = X[:3, :240]
    estimator = SingularSpectrumAnalysis(sfreq=sfreq, window_length=20).fit(X)
    events = []
    cleaned = estimator.transform(X, callback=events.append)

    assert cleaned.shape == X.shape
    assert len(events) == X.shape[0]
    assert all(event.method == "basic_ssa" for event in events)
    assert all(event.stage == "channel" for event in events)
    assert [event.current for event in events] == list(range(1, X.shape[0] + 1))
    assert all(event.total == X.shape[0] for event in events)
    assert all(event.component is None for event in events)
    np.testing.assert_array_equal(
        [event.metric for event in events], estimator.dropped_counts_.astype(float)
    )


def test_ssa_fit_transform_callback_matches_direct_transform(drift_data):
    """SSA fit_transform forwards callbacks only to transform."""
    X, sfreq = drift_data
    X = X[:3, :240]
    direct_events = []
    direct_model = SingularSpectrumAnalysis(sfreq=sfreq, window_length=20)
    direct = direct_model.fit(X).transform(X, callback=direct_events.append)

    composed_events = []
    composed_model = SingularSpectrumAnalysis(sfreq=sfreq, window_length=20)
    composed = composed_model.fit_transform(X, callback=composed_events.append)

    np.testing.assert_allclose(composed, direct)
    assert composed_events == direct_events


def test_ssa_epoched_transform_reports_epochs_only(drift_data):
    """Epoched SSA emits one event per completed epoch, never nested channels."""
    X, sfreq = drift_data
    epochs = np.stack((X[:3, :240], 0.5 * X[:3, :240]))
    estimator = SingularSpectrumAnalysis(sfreq=sfreq, window_length=20).fit(epochs)
    events = []
    cleaned = estimator.transform(epochs, callback=events.append)

    assert cleaned.shape == epochs.shape
    assert len(events) == epochs.shape[0]
    assert all(event.method == "basic_ssa" for event in events)
    assert [event.stage for event in events] == ["epoch"] * epochs.shape[0]
    assert [event.current for event in events] == list(range(1, epochs.shape[0] + 1))
    assert all(event.total == epochs.shape[0] for event in events)
    assert all(event.component is None for event in events)
    assert all(event.metric is None for event in events)


def test_ssa_callback_exception_propagates_unchanged(drift_data):
    """An SSA integration callback exception aborts the transform unchanged."""
    X, sfreq = drift_data
    X = X[:3, :240]
    estimator = SingularSpectrumAnalysis(sfreq=sfreq, window_length=20).fit(X)

    class SentinelError(RuntimeError):
        pass

    error = SentinelError("stop SSA")

    def callback(event):
        raise error

    with pytest.raises(SentinelError) as caught:
        estimator.transform(X, callback=callback)
    assert caught.value is error


def test_ssa_callback_api_is_runtime_only():
    """SSA fit and single-channel helpers remain callback-free APIs."""
    assert (
        "callback"
        not in inspect.signature(SingularSpectrumAnalysis.__init__).parameters
    )
    assert "callback" not in inspect.signature(SingularSpectrumAnalysis.fit).parameters
    for function in (ssa_decompose, ssa_clean_channel, ssa_w_correlation):
        assert "callback" not in inspect.signature(function).parameters


def test_compute_basic_ssa_rejects_invalid_callback():
    """Basic SSA validates callbacks at its public functional boundary."""
    with pytest.raises(TypeError, match="callback must be callable or None"):
        compute_basic_ssa(np.ones((2, 20)), 100.0, callback=1)


def test_ssa_requires_sfreq_for_array(drift_data):
    """Array input without sfreq raises a clear error."""
    X, _sfreq = drift_data
    with pytest.raises(ValueError, match="sfreq is required"):
        SingularSpectrumAnalysis().fit_transform(X)


def test_ssa_positional_sfreq(drift_data):
    """The canonical estimator accepts sampling frequency positionally."""
    X, sfreq = drift_data
    cleaned = SingularSpectrumAnalysis(sfreq).fit_transform(X)
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
        ({"sfreq": 250, "window_length": 101}, ValueError, "n_times"),
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
    valid, _ = compute_basic_ssa(np.ones((2, 7)), 250.0)
    assert valid.shape == (2, 7)
    with pytest.raises(ValueError, match="at least 3"):
        compute_basic_ssa(np.ones((2, 2)), 250.0)
    nonfinite = np.ones((2, 100))
    nonfinite[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        compute_basic_ssa(nonfinite, 250.0)


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
    low_before = band_power(X, sfreq, 0.0, 3.0)
    low_after = band_power(cleaned.get_data(), sfreq, 0.0, 3.0)
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


def test_ssa_rejects_changed_mne_channel_order(drift_data):
    """Transform requires the exact fitted MNE channel names and order."""
    mne = pytest.importorskip("mne")
    X, sfreq = drift_data
    names = [f"EEG{i:02d}" for i in range(X.shape[0])]
    raw = mne.io.RawArray(X, mne.create_info(names, sfreq, "eeg"), verbose=False)
    estimator = SingularSpectrumAnalysis(window_length=40).fit(raw)
    reordered = raw.copy().reorder_channels(names[::-1])
    with pytest.raises(ValueError, match="names/order"):
        estimator.transform(reordered)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"sfreq": True}, TypeError, "sfreq"),
        ({"sfreq": 250.0, "drop_freq_max": True}, TypeError, "drop_freq_max"),
        ({"sfreq": 250.0, "drop_freq_max": np.nan}, ValueError, "finite"),
        ({"sfreq": 250.0, "drop_band": [1.0, 2.0]}, TypeError, "drop_band"),
        ({"sfreq": 250.0, "drop_band": (True, 2.0)}, TypeError, "bounds"),
        ({"sfreq": 250.0, "drop_band": (1.0, np.nan)}, ValueError, "bounds"),
    ],
)
def test_ssa_scalar_contracts_reject_ambiguous_values(kwargs, error, match):
    """Boolean, non-finite, and structurally ambiguous parameters fail clearly."""
    with pytest.raises(error, match=match):
        SingularSpectrumAnalysis(**kwargs).fit_transform(np.ones((2, 200)))


def test_single_channel_primitive_validates_shape_and_finiteness():
    """The single-channel API rejects multidimensional and non-finite series."""
    with pytest.raises(TypeError, match="sfreq"):
        compute_basic_ssa(np.ones((2, 20)), True)
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
        compute_basic_ssa(np.empty((0, 100)), 100.0)
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
    with caplog.at_level(logging.INFO, logger="mne_denoise"):
        SingularSpectrumAnalysis(sfreq=sfreq, verbose=True).fit_transform(X[:2])
    summaries = [
        record for record in caplog.records if record.message.startswith("Basic SSA:")
    ]
    assert len(summaries) == 1
    for token in ("window=", "channels=", "dropped="):
        assert token in summaries[0].message
