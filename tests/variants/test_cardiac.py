"""Contract tests for the experimental cardiac DSS recipe."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import mne
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone

from mne_denoise.dss import (
    DSS,
    CardiacDSS,
    CardiacDSSDiagnostics,
    CardiacDSSStatus,
    CycleAverageBias,
)


def _cardiac_data(seed=0, n_times=1_200):
    """Return deterministic multichannel data with a cycle-locked source."""
    rng = np.random.default_rng(seed)
    events = np.arange(100, n_times - 60, 200)
    source = np.zeros(n_times)
    template = np.hanning(40)
    for event in events:
        source[event - 15 : event + 25] += template
    mixing = np.array([2.0, -1.0, 0.5, 1.25])
    data = np.outer(mixing, source) + 0.05 * rng.standard_normal((4, n_times))
    return data, events


def _estimator(events, **kwargs):
    """Construct the canonical sample-valued subtraction recipe."""
    params = {
        "qrs_events": events,
        "event_unit": "samples",
        "event_origin": "data",
        "window": (-20, 30),
        "window_unit": "samples",
        "component_action": "subtract",
        "component_selection": 1,
        "n_components": 3,
        "normalize_input": False,
    }
    params.update(kwargs)
    return CardiacDSS(**params)


def test_cardiac_dss_matches_explicit_cycle_average_dss_recipe():
    """The wrapper is exactly the declared CycleAverageBias plus DSS recipe."""
    data, events = _cardiac_data()
    observed_est = _estimator(events).fit(data)
    observed = observed_est.transform(data)

    bias = CycleAverageBias(
        event_samples=events,
        window=(-20, 30),
        window_unit="samples",
        event_origin="data",
    )
    reference_est = DSS(
        bias=bias,
        n_components=3,
        component_action="subtract",
        component_selection=1,
        normalize_input=False,
    ).fit(data)
    reference = reference_est.transform(data)

    assert_allclose(observed_est.filters_, reference_est.filters_, atol=0, rtol=0)
    assert_allclose(observed_est.patterns_, reference_est.patterns_, atol=0, rtol=0)
    assert_allclose(observed, reference, atol=0, rtol=0)
    assert observed_est.diagnostics_.status is CardiacDSSStatus.APPLIED
    assert observed_est.diagnostics_.valid_event_count == len(events)


def test_cardiac_dss_synthetic_cycle_locked_energy_is_reduced():
    """A dominant one-source synthetic artifact exercises subtraction polarity."""
    data, events = _cardiac_data(seed=4)
    cleaned = _estimator(events).fit_transform(data)

    before = np.stack([data[:, event - 15 : event + 25] for event in events]).mean(0)
    after = np.stack([cleaned[:, event - 15 : event + 25] for event in events]).mean(0)
    assert np.linalg.norm(after) < 0.25 * np.linalg.norm(before)


def test_fit_transform_is_fit_then_transform_and_events_do_not_leak():
    """Transform is inductive and never calls the fit-only event bias."""
    data, events = _cardiac_data()
    direct = _estimator(events).fit_transform(data)
    fitted = _estimator(events).fit(data)

    def fail_if_reused(_data):
        raise AssertionError("QRS bias leaked into transform")

    fitted.bias_.apply = fail_if_reused
    separate = fitted.transform(data)
    assert_allclose(direct, separate, atol=0, rtol=0)


def test_second_and_sample_coordinates_are_equivalent():
    """Event and window seconds use the documented nearest-sample conversion."""
    data, events = _cardiac_data(n_times=1_000)
    events = events[:4]
    samples = _estimator(events).fit_transform(data)
    seconds_est = CardiacDSS(
        qrs_events=events / 100.0,
        event_unit="seconds",
        sfreq=100.0,
        event_origin="data",
        window=(-0.2, 0.3),
        window_unit="seconds",
        component_action="subtract",
        component_selection=1,
        n_components=3,
        normalize_input=False,
    )
    seconds = seconds_est.fit_transform(data)

    assert seconds_est.diagnostics_.window_samples == (-20, 30)
    assert seconds_est.valid_event_samples_ == tuple(events)
    assert_allclose(seconds, samples, atol=0, rtol=0)


def test_raw_and_data_event_origins_are_equivalent():
    """Raw acquisition numbering subtracts first_samp exactly once."""
    data, events = _cardiac_data()
    first_samp = 10_000
    data_origin = _estimator(events).fit_transform(data)
    raw_origin = _estimator(
        events + first_samp,
        event_origin="raw",
        first_samp=first_samp,
    ).fit_transform(data)

    assert_allclose(raw_origin, data_origin, atol=0, rtol=0)


def test_epoch_events_are_boundary_safe_and_epoch_major():
    """Incomplete windows are excluded before epoch-major concatenation."""
    rng = np.random.default_rng(2)
    data = rng.standard_normal((3, 20, 2))
    events = np.array([[0, 1], [0, 10], [1, 18], [1, 10]])
    est = _estimator(
        events,
        window=(-2, 3),
        n_components=2,
        min_valid_events=2,
    ).fit(data)

    assert est.valid_event_samples_ == (10, 30)
    assert est.bias_.event_samples_data_.tolist() == [10, 30]
    assert est.diagnostics_.input_event_count == 4
    assert est.diagnostics_.valid_event_count == 2
    assert est.diagnostics_.excluded_event_count == 2
    assert est.transform(data).shape == data.shape


def test_insufficient_subtraction_abstains_with_exact_copy():
    """Subtraction safely passes through when no complete event window remains."""
    data, _ = _cardiac_data(n_times=300)
    est = _estimator([1], window=(-20, 30), min_valid_events=1)
    out = est.fit_transform(data)

    assert est.diagnostics_.status is CardiacDSSStatus.ABSTAINED
    assert "0 complete QRS windows" in est.diagnostics_.reason
    assert_array_equal(out, data)
    assert out is not data
    assert est.filters_ is None


@pytest.mark.parametrize("action", ["extract", "retain"])
def test_insufficient_non_subtraction_is_inadmissible(action):
    """An unfitted source or retained-sensor output has no valid shape contract."""
    data, _ = _cardiac_data(n_times=300)
    est = _estimator([1], component_action=action).fit(data)

    assert est.diagnostics_.status is CardiacDSSStatus.INADMISSIBLE
    with pytest.raises(RuntimeError, match="inadmissible"):
        est.transform(data)


def test_explicit_zero_selection_is_exact_noop_without_events():
    """An explicit zero-component subtraction never fabricates a DSS fit."""
    data, _ = _cardiac_data(n_times=300)
    est = _estimator([], component_selection=0)
    out = est.fit_transform(data)

    assert est.diagnostics_.status is CardiacDSSStatus.NO_OP
    assert est.diagnostics_.n_selected == 0
    assert_array_equal(out, data)
    assert out is not data


def test_zero_variance_and_zero_bias_energy_abstain():
    """Exact mathematical degeneracies become described safe abstentions."""
    constant = np.ones((3, 100))
    constant_est = _estimator([50], window=(-5, 5)).fit(constant)
    assert constant_est.diagnostics_.status is CardiacDSSStatus.ABSTAINED
    assert "centered variance" in constant_est.diagnostics_.reason

    data = np.zeros((3, 100))
    data[:, :20] = np.arange(20)
    zero_bias_est = _estimator([50], window=(-5, 5)).fit(data)
    assert zero_bias_est.diagnostics_.status is CardiacDSSStatus.ABSTAINED
    assert "zero energy" in zero_bias_est.diagnostics_.reason


def test_diagnostics_are_frozen_and_json_safe():
    """Diagnostics can enter provenance records without mutable arrays."""
    data, events = _cardiac_data()
    diagnostics = _estimator(events).fit(data).get_diagnostics()

    assert isinstance(diagnostics, CardiacDSSDiagnostics)
    assert json.loads(json.dumps(diagnostics.to_dict()))["status"] == "applied"
    with pytest.raises(FrozenInstanceError):
        diagnostics.valid_event_count = 0


def test_sklearn_clone_preserves_constructor_operating_point():
    """The wrapper follows the estimator constructor identity convention."""
    _, events = _cardiac_data()
    original = _estimator(events)
    copied = clone(original)

    assert_array_equal(copied.qrs_events, events)
    assert copied.component_action == "subtract"
    assert copied.component_selection == 1
    assert copied.window_unit == "samples"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"component_action": "remove"}, "component_action"),
        ({"component_selection": -1}, "component_selection"),
        ({"component_selection": "auto"}, "component_selection"),
        ({"min_valid_events": 0}, "min_valid_events"),
        ({"event_unit": "minutes"}, "event_unit"),
        ({"window_unit": "minutes"}, "window_unit"),
        ({"event_origin": "epoch"}, "event_origin"),
        ({"event_origin": "raw", "event_unit": "seconds"}, "sample numbers"),
        ({"n_components": 0}, "n_components"),
    ],
)
def test_invalid_operating_points_raise(kwargs, match):
    """Ambiguous or structurally invalid operating points fail before fitting."""
    data, events = _cardiac_data(n_times=300)
    with pytest.raises(ValueError, match=match):
        _estimator(events[:1], **kwargs).fit(data)


@pytest.mark.parametrize(
    ("events", "match"),
    [
        ([[0, 20]], "one-dimensional"),
        ([10.5], "integers"),
        ([np.nan], "finite"),
        ([20, 20], "duplicate"),
    ],
)
def test_invalid_continuous_event_coordinates_raise(events, match):
    """Continuous coordinates are explicit, finite, integral, and unique."""
    data, _ = _cardiac_data(n_times=300)
    with pytest.raises(ValueError, match=match):
        _estimator(events, window=(-5, 5)).fit(data)


@pytest.mark.parametrize(
    ("events", "match"),
    [
        ([20], "shape"),
        ([[0.5, 20]], "epoch indices"),
        ([[0, 20.5]], "integers"),
        ([[0, 20], [0, 20]], "duplicate"),
    ],
)
def test_invalid_epoched_event_coordinates_raise(events, match):
    """Epoched coordinates require unique integer epoch/sample pairs."""
    data = np.ones((3, 50, 2))
    with pytest.raises(ValueError, match=match):
        _estimator(events, window=(-5, 5)).fit(data)


def _mixed_info(sfreq=100.0):
    return mne.create_info(
        ["EEG001", "EEG002", "EEG003", "EEG004", "EOG001"],
        sfreq,
        ["eeg", "eeg", "eeg", "eeg", "eog"],
    )


def test_mne_raw_preserves_time_channel_type_unit_and_annotations():
    """Raw subtraction changes selected data only and retains all metadata."""
    data, events = _cardiac_data()
    full_data = np.vstack([data, np.linspace(-1, 1, data.shape[1])]) * 1e-6
    raw = mne.io.RawArray(
        full_data,
        _mixed_info(),
        first_samp=2_000,
        verbose=False,
    )
    raw.info["bads"] = ["EEG004"]
    raw.set_annotations(mne.Annotations([0.5], [0.1], ["bad_motion"]))
    units = [channel["unit"] for channel in raw.info["chs"]]

    est = _estimator(
        events + raw.first_samp,
        event_origin="raw",
        first_samp=raw.first_samp,
    )
    out = est.fit_transform(raw)

    assert isinstance(out, mne.io.BaseRaw)
    assert out.first_samp == raw.first_samp
    assert out.ch_names == raw.ch_names
    assert out.get_channel_types() == raw.get_channel_types()
    assert [channel["unit"] for channel in out.info["chs"]] == units
    assert out.info["bads"] == raw.info["bads"]
    assert list(out.annotations.description) == ["bad_motion"]
    assert_array_equal(out.get_data(picks="eog"), raw.get_data(picks="eog"))
    assert est.diagnostics_.sfreq == raw.info["sfreq"]


def test_mne_raw_rejects_mismatched_first_samp_and_sfreq():
    """Metadata disagreements cannot silently shift event coordinates."""
    data, events = _cardiac_data()
    info = mne.create_info(
        ["EEG001", "EEG002", "EEG003", "EEG004"],
        100.0,
        "eeg",
    )
    raw = mne.io.RawArray(data * 1e-6, info, verbose=False)

    with pytest.raises(ValueError, match="first_samp does not match"):
        _estimator(events + 1, event_origin="raw", first_samp=1).fit(raw)
    with pytest.raises(ValueError, match="sfreq does not match"):
        _estimator(events, sfreq=200.0).fit(raw)


def test_mne_epochs_preserve_layout_metadata_and_untouched_channel():
    """Epoch-local QRS pairs preserve the MNE Epochs container contract."""
    pd = pytest.importorskip("pandas")
    data, _ = _cardiac_data(n_times=400)
    rng = np.random.default_rng(12)
    epochs_data = (
        np.stack(
            [
                np.vstack([data[:, :200], rng.standard_normal(200)]),
                np.vstack([data[:, 200:], rng.standard_normal(200)]),
            ]
        )
        * 1e-6
    )
    events = np.array([[100, 0, 1], [400, 0, 2]])
    epochs = mne.EpochsArray(
        epochs_data,
        _mixed_info(),
        events=events,
        event_id={"a": 1, "b": 2},
        tmin=-0.25,
        baseline=(None, 0),
        verbose=False,
    )
    epochs.metadata = pd.DataFrame({"trial": ["a", "b"]})
    qrs_pairs = np.array([[0, 100], [1, 100]])

    est = _estimator(
        qrs_pairs,
        window=(-20, 30),
        min_valid_events=2,
    )
    out = est.fit_transform(epochs)

    assert isinstance(out, mne.BaseEpochs)
    assert out.ch_names == epochs.ch_names
    assert out.get_channel_types() == epochs.get_channel_types()
    assert_array_equal(out.events, epochs.events)
    assert out.event_id == epochs.event_id
    assert out.tmin == epochs.tmin
    assert out.baseline == epochs.baseline
    assert out.metadata.equals(epochs.metadata)
    assert_array_equal(out.get_data(picks="eog"), epochs.get_data(picks="eog"))
    assert est.diagnostics_.input_layout == "epochs"


def test_raw_origin_is_inadmissible_for_epoched_input():
    """A single Raw acquisition offset cannot represent epoch-local events."""
    data = np.ones((3, 50, 2))
    with pytest.raises(ValueError, match="continuous arrays"):
        _estimator(
            [[0, 20]],
            event_origin="raw",
            first_samp=0,
            window=(-5, 5),
        ).fit(data)


def test_transform_before_fit_raises():
    """Diagnostics make the fitted-state gate explicit."""
    data, events = _cardiac_data(n_times=300)
    with pytest.raises(RuntimeError, match="not fitted"):
        _estimator(events[:1]).transform(data)
