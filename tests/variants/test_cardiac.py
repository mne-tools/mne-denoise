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


def _diagnostic_record(**overrides):
    """Construct one valid public diagnostic record with selected overrides."""
    values = {
        "status": CardiacDSSStatus.APPLIED,
        "reason": None,
        "input_layout": "array-2d",
        "component_action": "subtract",
        "component_selection": 1,
        "input_event_count": 3,
        "valid_event_count": 2,
        "excluded_event_count": 1,
        "n_channels": 4,
        "fitted_channel_names": None,
        "n_times": 100,
        "n_epochs": None,
        "sfreq": 100.0,
        "window_samples": (-10, 20),
        "event_origin": "data",
        "first_samp": None,
        "n_selected": 1,
        "eigenvalues": (2.0, 1.0),
    }
    values.update(overrides)
    return CardiacDSSDiagnostics(**values)


@pytest.mark.parametrize(
    ("overrides", "error", "match"),
    [
        ({"status": "applied"}, TypeError, "CardiacDSSStatus"),
        ({"reason": "unexpected"}, ValueError, "cannot include a reason"),
        (
            {"status": CardiacDSSStatus.ABSTAINED},
            ValueError,
            "require a reason",
        ),
        ({"component_action": "remove"}, ValueError, "component_action"),
        ({"component_selection": -1}, ValueError, "component_selection"),
        ({"input_event_count": np.int64(3)}, ValueError, "Event counts"),
        ({"valid_event_count": -1}, ValueError, "Event counts"),
        ({"valid_event_count": 1}, ValueError, "must sum"),
        ({"n_channels": 0}, ValueError, "channel and time"),
        (
            {"fitted_channel_names": ("EEG001",)},
            ValueError,
            "matching n_channels",
        ),
        (
            {
                "fitted_channel_names": (
                    "EEG001",
                    "EEG002",
                    "EEG003",
                    "EEG004",
                )
            },
            ValueError,
            "Array diagnostics cannot include",
        ),
        (
            {
                "input_layout": "raw",
                "fitted_channel_names": None,
            },
            ValueError,
            "MNE diagnostics require",
        ),
        ({"n_times": 0}, ValueError, "channel and time"),
        ({"n_epochs": 0}, ValueError, "n_epochs"),
        ({"sfreq": np.nan}, ValueError, "sfreq"),
        ({"sfreq": 0.0}, ValueError, "sfreq"),
        ({"window_samples": (-1.0, 2)}, ValueError, "window_samples"),
        ({"window_samples": (-1, 2.0)}, ValueError, "window_samples"),
        ({"window_samples": (2, 2)}, ValueError, "window_samples"),
        ({"event_origin": "epoch"}, ValueError, "event_origin"),
        (
            {"event_origin": "raw", "first_samp": None},
            ValueError,
            "require first_samp",
        ),
        ({"first_samp": 10}, ValueError, "cannot include first_samp"),
        ({"n_selected": -1}, ValueError, "n_selected"),
        ({"eigenvalues": (np.nan,)}, ValueError, "eigenvalues"),
    ],
)
def test_diagnostics_reject_inconsistent_records(overrides, error, match):
    """Every public diagnostic record enforces its cross-field invariants."""
    with pytest.raises(error, match=match):
        _diagnostic_record(**overrides)


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


@pytest.mark.parametrize("terminal", ["no_op", "abstained"])
@pytest.mark.parametrize(
    ("mutation", "error", "match"),
    [
        (lambda data: data[:, :-1], ValueError, "shape"),
        (lambda data: data[:, :, np.newaxis], TypeError, "container/layout"),
        (lambda data: data.tolist(), TypeError, "NumPy ndarray"),
        (
            lambda data: data.astype(np.complex128),
            TypeError,
            "real-valued, not complex",
        ),
        (
            lambda data: np.where(
                np.arange(data.size).reshape(data.shape) == 0, np.nan, data
            ),
            ValueError,
            "finite values",
        ),
    ],
)
def test_passthrough_transform_enforces_frozen_input_contract(
    terminal, mutation, error, match
):
    """No-op and abstained paths cannot bypass transform hard checks."""
    data, _ = _cardiac_data(n_times=300)
    if terminal == "no_op":
        est = _estimator([], component_selection=0).fit(data)
    else:
        est = _estimator([1]).fit(data)
    with pytest.raises(error, match=match):
        est.transform(mutation(data.copy()))


def test_passthrough_transform_never_reconsults_fit_events():
    """Contract validation is independent of mutable fit-only event input."""

    class EventsMustNotBeRead:
        def __array__(self):
            raise AssertionError("QRS events leaked into transform")

    data, _ = _cardiac_data(n_times=300)
    est = _estimator([], component_selection=0).fit(data)
    est.qrs_events = EventsMustNotBeRead()

    assert_array_equal(est.transform(data), data)


def test_applied_transform_is_inductive_but_rejects_complex_and_nonfinite_data():
    """Applied spatial filters accept new lengths under the fitted channel contract."""
    data, events = _cardiac_data()
    est = _estimator(events).fit(data)

    assert est.transform(data[:, :600]).shape == (data.shape[0], 600)
    with pytest.raises(TypeError, match="real-valued, not complex"):
        est.transform(data[:, :600].astype(np.complex128))
    nonfinite = data[:, :600].copy()
    nonfinite[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite values"):
        est.transform(nonfinite)
    with pytest.raises(ValueError, match="channel count"):
        est.transform(data[:-1])


def test_missing_internal_input_contract_cannot_passthrough():
    """A corrupted fitted state cannot fall back to unchecked copying."""
    data, _ = _cardiac_data(n_times=300)
    est = _estimator([], component_selection=0).fit(data)
    est._transform_contract_ = None

    with pytest.raises(RuntimeError, match="no fitted input contract"):
        est.transform(data)


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
    ("data", "error", "match"),
    [
        (np.full((2, 10), "not-a-number"), TypeError, "must be numeric"),
        (
            np.ones((2, 10), dtype=np.complex128),
            TypeError,
            "real-valued, not complex",
        ),
        (
            np.ones((2, 10)) + 1j,
            TypeError,
            "real-valued, not complex",
        ),
        (np.array([[0.0, np.nan], [1.0, 2.0]]), ValueError, "finite values"),
        (np.empty((0, 10)), ValueError, "channels and at least two samples"),
        (np.empty((2, 1)), ValueError, "channels and at least two samples"),
        (np.empty((2, 10, 0)), ValueError, "channels and at least two samples"),
    ],
)
def test_invalid_fit_data_raise_before_event_processing(data, error, match):
    """Numeric, finite, non-empty sensor/time layouts are hard preconditions."""
    with pytest.raises(error, match=match):
        _estimator([], component_selection=0).fit(data)


@pytest.mark.parametrize("sfreq", [True, 0.0, np.nan])
def test_invalid_sampling_frequency_raises(sfreq):
    """Boolean, non-positive, and non-finite sampling rates are inadmissible."""
    data, _ = _cardiac_data(n_times=300)
    with pytest.raises(ValueError, match="finite positive"):
        _estimator([], component_selection=0, sfreq=sfreq).fit(data)


def test_second_coordinates_require_sampling_frequency():
    """An array cannot infer the scale of second-valued coordinates."""
    data, _ = _cardiac_data(n_times=300)
    est = _estimator(
        [1.0],
        event_unit="seconds",
        window=(-5, 5),
        window_unit="samples",
    )
    with pytest.raises(ValueError, match="sfreq is required"):
        est.fit(data)


def test_integer_second_coordinate_conversion_is_range_safe():
    """Integer seconds cover normal, multiplication-overflow, and cast overflow."""
    data, _ = _cardiac_data(n_times=300)
    valid = _estimator(
        [1],
        event_unit="seconds",
        sfreq=100.0,
        component_selection=0,
        window=(-5, 5),
    ).fit(data)
    assert valid.valid_event_samples_ == (100,)

    with pytest.raises(ValueError, match="64-bit sample indices"):
        _estimator(
            np.array([10**10_000], dtype=object),
            event_unit="seconds",
            sfreq=100.0,
            window=(-5, 5),
        ).fit(data)
    with pytest.raises(ValueError, match="64-bit sample indices"):
        _estimator(
            [np.iinfo(np.int64).max],
            event_unit="seconds",
            sfreq=1e300,
            window=(-5, 5),
        ).fit(data)


def test_float_second_coordinate_product_overflow_is_rejected():
    """Finite seconds whose scaled value is infinite never reach round()."""
    data, _ = _cardiac_data(n_times=300)
    with pytest.raises(ValueError, match="64-bit sample indices"):
        _estimator(
            [1e300],
            event_unit="seconds",
            sfreq=1e10,
            window=(-5, 5),
        ).fit(data)


@pytest.mark.parametrize(
    ("window", "match"),
    [
        ((-1, 0, 1), "exactly two"),
        ((1, 1), "empty or reversed"),
        ((2, 1), "empty or reversed"),
    ],
)
def test_invalid_window_contract_is_rejected(window, match):
    """Cardiac-owned window conversion validates shape and resolved ordering."""
    data, _ = _cardiac_data(n_times=300)
    with pytest.raises(ValueError, match=match):
        _estimator([100], window=window).fit(data)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"event_origin": "raw"}, ValueError, "first_samp is required"),
        (
            {"event_origin": "raw", "first_samp": True},
            TypeError,
            "first_samp must be an integer",
        ),
        ({"first_samp": 10}, ValueError, "must be omitted"),
    ],
)
def test_first_sample_contract_rejects_ambiguous_offsets(kwargs, error, match):
    """Raw offsets are explicit integers and data-relative input forbids them."""
    data, _ = _cardiac_data(n_times=300)
    with pytest.raises(error, match=match):
        _estimator([], component_selection=0, **kwargs).fit(data)


@pytest.mark.parametrize(
    "event",
    [
        np.uint64(2**63),
        np.uint64(np.iinfo(np.uint64).max),
        float(np.iinfo(np.int64).max),
        1e300,
    ],
)
def test_event_coordinate_overflow_is_rejected_before_integer_cast(event):
    """Unsigned and floating overflow cannot wrap into valid int64 events."""
    data, _ = _cardiac_data(n_times=300)
    with pytest.raises(ValueError, match="64-bit sample indices"):
        _estimator([event], window=(-5, 5)).fit(data)


def test_int64_extreme_events_are_safely_excluded_without_arithmetic_overflow():
    """Representable extremes remain exact while window bounds use Python ints."""
    data, _ = _cardiac_data(n_times=300)
    events = np.array([np.iinfo(np.int64).min, np.iinfo(np.int64).max], dtype=np.int64)
    est = _estimator(events, component_selection=0, window=(-5, 5)).fit(data)

    assert est.diagnostics_.input_event_count == 2
    assert est.diagnostics_.valid_event_count == 0
    assert est.diagnostics_.excluded_event_count == 2


def test_raw_origin_extreme_offset_uses_exact_python_integer_subtraction():
    """A large acquisition offset is subtracted without signed wraparound."""
    data, _ = _cardiac_data(n_times=300)
    maximum = int(np.iinfo(np.int64).max)
    est = _estimator(
        [maximum],
        event_origin="raw",
        first_samp=maximum,
        component_selection=0,
        window=(0, 1),
    ).fit(data)

    assert est.valid_event_samples_ == (0,)
    assert est.bias_.event_samples_data_.tolist() == [0]


def test_half_open_window_boundaries_are_valid_without_pre_post_overflow():
    """Events whose windows exactly meet either data edge remain admissible."""
    data, _ = _cardiac_data(n_times=300)
    est = _estimator(
        [4, 5, 295, 296],
        component_selection=0,
        window=(-5, 5),
    ).fit(data)

    assert est.valid_event_samples_ == (5, 295)
    assert est.diagnostics_.valid_event_count == 2
    assert est.diagnostics_.excluded_event_count == 2


@pytest.mark.parametrize("event", [np.inf, -np.inf, np.nan])
def test_nonfinite_event_coordinates_are_rejected(event):
    """Nonfinite coordinates fail before rounding or integer conversion."""
    data, _ = _cardiac_data(n_times=300)
    with pytest.raises(ValueError, match="finite numeric coordinates"):
        _estimator([event], window=(-5, 5)).fit(data)


def test_extreme_epoch_coordinates_do_not_overflow_bounds_or_concatenation():
    """Epoched int64 extremes are excluded before local-window arithmetic."""
    data = np.ones((2, 10, 2))
    events = np.array(
        [
            [np.iinfo(np.int64).min, np.iinfo(np.int64).max],
            [np.iinfo(np.int64).max, np.iinfo(np.int64).min],
        ],
        dtype=np.int64,
    )
    est = _estimator(
        events,
        component_selection=0,
        window=(-2, 2),
    ).fit(data)

    assert est.diagnostics_.valid_event_count == 0
    assert est.diagnostics_.excluded_event_count == 2


def test_unrepresentable_epoch_index_is_rejected_before_cast():
    """Unsigned epoch identifiers above int64 cannot wrap to negative indices."""
    data = np.ones((2, 10, 2))
    events = np.array([[np.uint64(2**63), np.uint64(5)]], dtype=np.uint64)
    with pytest.raises(ValueError, match="64-bit sample indices"):
        _estimator(events, window=(-2, 2)).fit(data)


def test_empty_epoched_events_are_an_explicit_noop():
    """An empty, well-shaped epoch-event table is preserved in diagnostics."""
    data = np.ones((2, 10, 3))
    est = _estimator(
        np.empty((0, 2)),
        component_selection=0,
        window=(-2, 2),
    )
    out = est.fit_transform(data)

    assert_array_equal(out, data)
    assert est.diagnostics_.status is CardiacDSSStatus.NO_OP
    assert est.diagnostics_.input_event_count == 0
    assert est.diagnostics_.n_epochs == 3


def test_epoched_event_indices_reject_non_numeric_values():
    """Epoch identifiers cannot be parsed implicitly from strings."""
    data = np.ones((2, 10, 3))
    with pytest.raises(ValueError, match="epoch indices must be finite integers"):
        _estimator([["first", "5"]], window=(-2, 2)).fit(data)


def test_array_compatible_container_is_not_a_numpy_input_contract():
    """Lists and tuples cannot silently change fitted container semantics."""
    data = ((1.0, 2.0, 3.0), (3.0, 2.0, 1.0))
    with pytest.raises(TypeError, match="NumPy ndarray"):
        _estimator(
            [],
            component_selection=0,
            window=(-1, 1),
        ).fit(data)


def test_fit_transform_rejects_unimplemented_fit_parameters():
    """Unknown metadata cannot be silently discarded by the recipe."""
    data, events = _cardiac_data(n_times=300)
    with pytest.raises(TypeError, match="Unexpected fit parameters: sample_weight"):
        _estimator(events[:1]).fit_transform(data, sample_weight=np.ones(300))


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
        sfreq=raw.info["sfreq"],
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
    assert_array_equal(out.get_data(picks=["EEG004"]), raw.get_data(picks=["EEG004"]))
    assert est.diagnostics_.sfreq == raw.info["sfreq"]
    assert est.diagnostics_.n_channels == 3
    assert est.diagnostics_.fitted_channel_names == (
        "EEG001",
        "EEG002",
        "EEG003",
    )
    assert est.fitted_channel_names_ == est.diagnostics_.fitted_channel_names
    assert est.filters_.shape[1] == est.diagnostics_.n_channels


def test_mne_bad_channels_are_excluded_from_cardiac_prechecks():
    """A bad-only signal cannot make a constant good-channel fit look admissible."""
    n_times = 300
    data = np.ones((5, n_times))
    data[3] = np.sin(np.linspace(0, 8 * np.pi, n_times))
    data[4] = np.linspace(-1, 1, n_times)
    raw = mne.io.RawArray(data, _mixed_info(), verbose=False)
    raw.info["bads"] = ["EEG004"]

    est = _estimator([100], window=(-10, 10)).fit(raw)
    out = est.transform(raw)

    assert est.diagnostics_.status is CardiacDSSStatus.ABSTAINED
    assert "centered variance" in est.diagnostics_.reason
    assert est.diagnostics_.fitted_channel_names == (
        "EEG001",
        "EEG002",
        "EEG003",
    )
    assert_array_equal(out.get_data(), raw.get_data())


def test_mne_fit_rejects_when_every_selected_channel_is_bad():
    """The Cardiac precheck and DSS fit share the same non-empty good set."""
    data, _ = _cardiac_data(n_times=300)
    info = mne.create_info(["EEG001", "EEG002", "EEG003", "EEG004"], 100.0, "eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.info["bads"] = list(raw.ch_names)

    with pytest.raises(ValueError, match="No good channels remain"):
        _estimator([], component_selection=0).fit(raw)


def test_mne_noop_passthrough_enforces_shape_time_and_channel_metadata():
    """MNE passthrough requires the exact frozen structural fit contract."""
    data, _ = _cardiac_data(n_times=300)
    info = mne.create_info(["EEG001", "EEG002", "EEG003", "EEG004"], 100.0, "eeg")
    raw = mne.io.RawArray(data * 1e-6, info, verbose=False)
    est = _estimator([], component_selection=0).fit(raw)

    assert_array_equal(est.transform(raw).get_data(), raw.get_data())

    shorter = raw.copy().crop(tmax=raw.times[-2])
    with pytest.raises(ValueError, match="shape"):
        est.transform(shorter)

    changed_bads = raw.copy()
    changed_bads.info["bads"] = ["EEG004"]
    with pytest.raises(ValueError, match="bads"):
        est.transform(changed_bads)

    shifted = mne.io.RawArray(
        raw.get_data(), raw.info.copy(), first_samp=1, verbose=False
    )
    with pytest.raises(ValueError, match="time_origin"):
        est.transform(shifted)

    renamed = raw.copy().rename_channels({"EEG004": "EEG099"})
    with pytest.raises(ValueError, match="channel_names"):
        est.transform(renamed)


def test_whitened_diagnostics_follow_dss_all_data_channel_pathway():
    """Joint whitening retains bad data channels exactly as underlying DSS does."""
    data, events = _cardiac_data()
    info = mne.create_info(["EEG001", "EEG002", "EEG003", "EEG004"], 100.0, "eeg")
    raw = mne.io.RawArray(data * 1e-6, info, verbose=False)
    raw.info["bads"] = ["EEG004"]
    est = _estimator(events, whiten=True).fit(raw)

    assert est.diagnostics_.fitted_channel_names == tuple(raw.ch_names)
    assert est.diagnostics_.n_channels == 4
    assert est.filters_.shape[1] == 4


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
