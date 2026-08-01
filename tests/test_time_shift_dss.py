"""Invariant tests for true data-side time-shift DSS."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import mne
import numpy as np
import pytest
from sklearn.base import clone

from mne_denoise.dss import (
    AverageBias,
    SmoothingBias,
    TimeShiftDSS,
    TimeShiftDSSDiagnostics,
    compute_time_shift_dss,
)
from mne_denoise.dss.time_shift import _lag_augment, _resolve_lags


def _identity(data):
    return data


def _array_data(seed=11, n_channels=3, n_times=240):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_channels, n_times))


def test_lag_embedding_has_explicit_sign_and_valid_region():
    data = np.arange(2 * 9.0).reshape(2, 9)
    augmented, start, stop = _lag_augment(data, (-1, 0, 2))

    assert (start, stop) == (2, 8)
    np.testing.assert_array_equal(augmented[0:2], data[:, 3:9])
    np.testing.assert_array_equal(augmented[2:4], data[:, 2:8])
    np.testing.assert_array_equal(augmented[4:6], data[:, 0:6])


def test_lag_embedding_never_wraps_or_crosses_epochs():
    epoch_a = np.arange(8.0)[None, :]
    epoch_b = (100.0 + np.arange(8.0))[None, :]
    data = np.stack([epoch_a, epoch_b], axis=-1).reshape(1, 8, 2)

    augmented, start, stop = _lag_augment(data, (-2, 0, 1))

    assert (start, stop) == (1, 6)
    assert np.all(augmented[:, :, 0] < 100.0)
    assert np.all(augmented[:, :, 1] >= 100.0)
    np.testing.assert_array_equal(augmented[0, :, 0], data[0, 3:8, 0])
    np.testing.assert_array_equal(augmented[2, :, 1], data[0, 0:5, 1])


@pytest.mark.parametrize(
    "kwargs, error, match",
    [
        ({}, ValueError, "exactly one"),
        (
            {"lag_samples": [0, 1], "lag_times": [0.0, 0.01], "sfreq": 100.0},
            ValueError,
            "exactly one",
        ),
        ({"lag_samples": 1}, TypeError, "sequence"),
        ({"lag_samples": [0, 1.5]}, ValueError, "whole sample"),
        ({"lag_samples": [1, 2]}, ValueError, "include zero"),
        ({"lag_samples": [0]}, ValueError, "non-zero"),
        ({"lag_samples": [0, 1, 1]}, ValueError, "unique"),
        ({"lag_times": [0.0, 0.01]}, ValueError, "sfreq is required"),
        (
            {"lag_times": [0.0, 0.015], "sfreq": 100.0},
            ValueError,
            "sampling grid",
        ),
    ],
)
def test_lag_contract_rejects_ambiguous_or_unaligned_values(kwargs, error, match):
    params = {"lag_samples": None, "lag_times": None, "sfreq": None}
    params.update(kwargs)
    with pytest.raises(error, match=match):
        _resolve_lags(**params)


@pytest.mark.parametrize(
    "lag_samples, match",
    [
        ([0, np.iinfo(np.intp).max + 1], "platform integer range"),
        ([np.iinfo(np.intp).min - 1, 0], "platform integer range"),
        ([0, float(2**53)], "precision loss"),
        ([0, np.uint64(np.iinfo(np.uint64).max)], "platform integer range"),
    ],
)
def test_sample_lags_reject_overflow_and_float_precision_loss(lag_samples, match):
    with pytest.raises(ValueError, match=match):
        _resolve_lags(lag_samples=lag_samples, lag_times=None, sfreq=None)


def test_exact_large_integer_sample_lag_is_not_first_coerced_to_float():
    # This value is larger than float's consecutive-integer range but remains
    # inside a 64-bit platform index. The exact integer survives parsing and is
    # then rejected only because no realistic input can span it.
    if np.iinfo(np.intp).max <= 2**53:
        pytest.skip("Platform integer range does not exceed float exact integers.")
    lag = 2**53 + 1
    samples, _, _, _ = _resolve_lags(lag_samples=[0, lag], lag_times=None, sfreq=None)
    assert samples == (0, lag)


@pytest.mark.parametrize(
    "lag_times, sfreq, match",
    [
        ([0.0, np.finfo(float).max], 2.0, "overflowed"),
        ([0.0, float(2**53)], 1.0, "exact floating-point integer range"),
        ([0, 10**1000], 1.0, "representable as finite"),
    ],
)
def test_physical_lags_reject_multiplication_overflow_and_precision_loss(
    lag_times, sfreq, match
):
    with pytest.raises(ValueError, match=match):
        _resolve_lags(lag_samples=None, lag_times=lag_times, sfreq=sfreq)


def test_sample_and_physical_lag_declarations_are_equivalent():
    data = _array_data()
    sample_result = compute_time_shift_dss(
        data,
        bias=_identity,
        lag_samples=[-2, 0, 3],
        sfreq=100.0,
        n_components=4,
    )
    time_result = compute_time_shift_dss(
        data,
        bias=_identity,
        lag_times=[-0.02, 0.0, 0.03],
        sfreq=100.0,
        n_components=4,
    )

    for sample_value, time_value in zip(sample_result[:3], time_result[:3]):
        np.testing.assert_allclose(sample_value, time_value)
    assert sample_result[3].lag_input_unit == "samples"
    assert time_result[3].lag_input_unit == "seconds"
    assert time_result[3].lag_samples == (-2, 0, 3)


def test_function_returns_fir_geometry_and_immutable_json_diagnostics():
    data = _array_data(n_channels=4)
    filters, patterns, eigenvalues, diagnostics = compute_time_shift_dss(
        data,
        bias=_identity,
        lag_samples=[-2, 0, 4],
        n_components=5,
        whitening_rank=8,
    )

    assert filters.shape == (5, 3, 4)
    assert patterns.shape == (3, 4, 5)
    assert eigenvalues.shape == (5,)
    assert isinstance(diagnostics, TimeShiftDSSDiagnostics)
    assert diagnostics.augmented_feature_count == 12
    assert diagnostics.valid_start == 4
    assert diagnostics.valid_stop == data.shape[1] - 2
    assert diagnostics.edge_policy == "preserve_input"
    json.dumps(diagnostics.to_dict())
    with pytest.raises(FrozenInstanceError):
        diagnostics.valid_start = 0


@pytest.mark.parametrize(
    "bad_bias, error, match",
    [
        (lambda data: data[:, :-1], ValueError, "preserve augmented data shape"),
        (lambda data: np.full(data.shape, np.nan), ValueError, "finite"),
        (object(), TypeError, "callable"),
    ],
)
def test_function_validates_bias_output(bad_bias, error, match):
    with pytest.raises(error, match=match):
        compute_time_shift_dss(_array_data(), bias=bad_bias, lag_samples=[0, 1])


def test_function_rejects_lags_that_leave_no_covariance_samples():
    with pytest.raises(ValueError, match="fewer than two valid samples"):
        compute_time_shift_dss(
            _array_data(n_times=5), bias=_identity, lag_samples=[0, 4]
        )


def test_estimator_sources_equal_direct_fir_application():
    data = _array_data()
    estimator = TimeShiftDSS(
        _identity,
        lag_samples=[-2, 0, 3],
        n_components=4,
        component_action="extract",
    ).fit(data)

    sources = estimator.transform(data)
    augmented, _, _ = _lag_augment(data, estimator.lag_samples_)
    augmented_2d = augmented.reshape(augmented.shape[0], -1)
    expected = estimator.filters_.reshape(4, -1) @ (
        augmented_2d - augmented_2d.mean(axis=1, keepdims=True)
    )

    np.testing.assert_allclose(sources, expected)
    assert sources.shape == (4, 235)
    assert estimator.valid_slice(data.shape[1]) == slice(3, 238)


def test_sensor_normalization_is_equivariant_to_input_units():
    data = _array_data(n_channels=3, n_times=300)
    scales = np.array([1e-6, 1.0, 1e6])
    scaled = data * scales[:, np.newaxis]
    params = {
        "bias": SmoothingBias(window=7),
        "lag_samples": [-2, 0, 3],
        "n_components": 4,
        "component_action": "extract",
        "normalize_input": True,
    }
    reference = TimeShiftDSS(**params).fit(data)
    rescaled = TimeShiftDSS(**params).fit(scaled)

    np.testing.assert_allclose(rescaled.eigenvalues_, reference.eigenvalues_, rtol=1e-9)
    np.testing.assert_allclose(
        np.abs(rescaled.transform(scaled)),
        np.abs(reference.transform(data)),
        rtol=1e-8,
        atol=1e-10,
    )


def test_estimator_epoched_source_shape_and_fit_transform_contract():
    rng = np.random.default_rng(4)
    data = rng.normal(size=(3, 80, 7))
    params = {
        "bias": AverageBias(axis="epochs"),
        "lag_samples": [-1, 0, 2],
        "n_components": 4,
        "component_action": "extract",
    }
    expected_estimator = TimeShiftDSS(**params).fit(data)
    expected = expected_estimator.transform(data)
    observed_estimator = TimeShiftDSS(**params)
    observed = observed_estimator.fit_transform(data)

    assert expected.shape == (4, 77, 7)
    np.testing.assert_allclose(observed, expected)
    np.testing.assert_allclose(
        observed_estimator.eigenvalues_, expected_estimator.eigenvalues_
    )


def test_full_rank_epoched_retain_preserves_array_orientation_and_edges():
    rng = np.random.default_rng(6)
    data = rng.normal(size=(2, 100, 4))
    estimator = TimeShiftDSS(
        _identity,
        lag_samples=[-3, 0, 2],
        component_action="retain",
    )
    retained = estimator.fit_transform(data)

    assert retained.shape == data.shape
    np.testing.assert_allclose(retained, data, atol=1e-10)
    valid = estimator.valid_slice(data.shape[1])
    np.testing.assert_array_equal(retained[:, : valid.start], data[:, : valid.start])
    np.testing.assert_array_equal(retained[:, valid.stop :], data[:, valid.stop :])


def test_full_rank_retain_reconstructs_valid_data_and_preserves_edges():
    data = _array_data(n_channels=2, n_times=320)
    estimator = TimeShiftDSS(
        _identity,
        lag_samples=[-2, 0, 3],
        component_action="retain",
    )
    retained = estimator.fit_transform(data)

    np.testing.assert_allclose(retained, data, atol=1e-10)
    valid = estimator.valid_slice(data.shape[1])
    np.testing.assert_array_equal(retained[:, : valid.start], data[:, : valid.start])
    np.testing.assert_array_equal(retained[:, valid.stop :], data[:, valid.stop :])


def test_subtract_none_is_no_op_copy_and_selected_preserves_edges():
    data = _array_data(n_times=160)
    no_op = TimeShiftDSS(
        _identity,
        lag_samples=[-2, 0, 3],
        n_components=4,
        component_action="subtract",
        component_selection=None,
    ).fit(data)
    unchanged = no_op.transform(data)
    np.testing.assert_array_equal(unchanged, data)
    assert unchanged is not data

    selected = TimeShiftDSS(
        _identity,
        lag_samples=[-2, 0, 3],
        n_components=4,
        component_action="subtract",
        component_selection=2,
    ).fit(data)
    cleaned = selected.transform(data)
    valid = selected.valid_slice(data.shape[1])
    np.testing.assert_array_equal(cleaned[:, : valid.start], data[:, : valid.start])
    np.testing.assert_array_equal(cleaned[:, valid.stop :], data[:, valid.stop :])
    assert not np.allclose(cleaned[:, valid], data[:, valid])


def test_estimator_is_cloneable_and_requires_explicit_operation_values():
    estimator = TimeShiftDSS(_identity, lag_samples=[0, 2], component_action="extract")
    cloned = clone(estimator)
    assert cloned.lag_samples == [0, 2]

    with pytest.raises(ValueError, match="component_action"):
        TimeShiftDSS(_identity, lag_samples=[0, 2], component_action="clean").fit(
            _array_data()
        )
    with pytest.raises(ValueError, match="component_selection"):
        TimeShiftDSS(_identity, lag_samples=[0, 2], component_selection=-1).fit(
            _array_data()
        )


def _raw_fixture():
    rng = np.random.default_rng(21)
    info = mne.create_info(
        ["EEG 001", "EEG 002", "STI 014"],
        sfreq=100.0,
        ch_types=["eeg", "eeg", "stim"],
    )
    data = np.vstack([rng.normal(size=(2, 180)), np.arange(180, dtype=float)[None, :]])
    raw = mne.io.RawArray(data, info, first_samp=123, verbose=False)
    raw.set_annotations(mne.Annotations([0.2], [0.1], ["marker"]))
    return raw


def test_mne_raw_sensor_action_preserves_container_units_timeline_and_extra_channels():
    raw = _raw_fixture()
    original = raw.get_data().copy()
    estimator = TimeShiftDSS(
        _identity,
        lag_times=[-0.02, 0.0, 0.03],
        component_action="retain",
    )
    out = estimator.fit_transform(raw)

    assert isinstance(out, mne.io.BaseRaw)
    assert out is not raw
    assert out.ch_names == raw.ch_names
    assert out.get_channel_types() == raw.get_channel_types()
    assert out.first_samp == raw.first_samp
    assert out.info["sfreq"] == raw.info["sfreq"]
    assert out.annotations.description.tolist() == raw.annotations.description.tolist()
    np.testing.assert_allclose(out.get_data()[:2], original[:2], atol=1e-10)
    np.testing.assert_array_equal(out.get_data()[2], original[2])
    np.testing.assert_allclose(estimator.valid_times_, raw.times[3:-2])
    np.testing.assert_allclose(estimator.get_valid_times(raw), raw.times[3:-2])
    assert estimator.fit_first_samp_ == 123


def test_mne_transform_aligns_by_channel_name_after_reordering():
    raw = _raw_fixture()
    estimator = TimeShiftDSS(
        _identity, lag_samples=[0, 2], component_action="retain"
    ).fit(raw)
    reordered = raw.copy().reorder_channels(["STI 014", "EEG 002", "EEG 001"])
    out = estimator.transform(reordered)

    for name in reordered.ch_names:
        np.testing.assert_allclose(
            out.get_data(picks=[name]), reordered.get_data(picks=[name]), atol=1e-10
        )


def test_mne_transform_rejects_sampling_and_channel_type_mismatches():
    raw = _raw_fixture()
    estimator = TimeShiftDSS(
        _identity, lag_samples=[0, 2], component_action="extract"
    ).fit(raw)

    resampled = raw.copy().resample(80.0)
    with pytest.raises(ValueError, match="sfreq"):
        estimator.transform(resampled)

    changed_type = raw.copy()
    changed_type.set_channel_types({"EEG 001": "eog"})
    with pytest.raises(ValueError, match="names, types, or physical units"):
        estimator.transform(changed_type)


def test_mne_epochs_sources_and_sensor_outputs_keep_time_and_event_alignment():
    rng = np.random.default_rng(44)
    info = mne.create_info(["EEG 001", "EEG 002"], 200.0, "eeg")
    data = rng.normal(size=(5, 2, 90))
    events = np.column_stack(
        [np.arange(5) * 200 + 100, np.zeros(5, dtype=int), np.ones(5, dtype=int)]
    )
    epochs = mne.EpochsArray(
        data, info, events=events, event_id={"target": 1}, tmin=-0.1, verbose=False
    )

    extractor = TimeShiftDSS(
        AverageBias(axis="epochs"),
        lag_samples=[-2, 0, 3],
        n_components=3,
        component_action="extract",
    ).fit(epochs)
    sources = extractor.transform(epochs)
    assert sources.shape == (5, 3, 85)
    np.testing.assert_allclose(extractor.get_valid_times(epochs), epochs.times[3:-2])

    retainer = TimeShiftDSS(
        _identity, lag_samples=[-2, 0, 3], component_action="retain"
    )
    out = retainer.fit_transform(epochs)
    assert isinstance(out, mne.BaseEpochs)
    np.testing.assert_array_equal(out.events, epochs.events)
    assert out.event_id == epochs.event_id
    assert out.tmin == epochs.tmin
    assert out.drop_log == epochs.drop_log
    np.testing.assert_allclose(out.get_data(), epochs.get_data(), atol=1e-10)


def test_mne_evoked_output_preserves_time_comment_nave_and_units():
    rng = np.random.default_rng(71)
    info = mne.create_info(["MEG 001", "MEG 002"], 250.0, ["mag", "mag"])
    evoked = mne.EvokedArray(
        rng.normal(scale=1e-12, size=(2, 140)),
        info,
        tmin=-0.2,
        comment="condition-a",
        nave=17,
        verbose=False,
    )
    estimator = TimeShiftDSS(
        _identity,
        lag_times=[-0.008, 0.0, 0.012],
        component_action="retain",
    )
    out = estimator.fit_transform(evoked)

    assert isinstance(out, mne.Evoked)
    assert out.comment == evoked.comment
    assert out.nave == evoked.nave
    assert out.tmin == evoked.tmin
    assert out.get_channel_types() == evoked.get_channel_types()
    np.testing.assert_allclose(out.times, evoked.times)
    np.testing.assert_allclose(out.data, evoked.data, rtol=1e-10, atol=1e-24)
    np.testing.assert_allclose(estimator.get_valid_times(evoked), evoked.times[3:-2])


def test_mne_and_numpy_container_families_cannot_be_mixed():
    data = _array_data(n_channels=2)
    numpy_estimator = TimeShiftDSS(_identity, lag_samples=[0, 1]).fit(data)
    with pytest.raises(TypeError, match="container families"):
        numpy_estimator.transform(_raw_fixture().pick(["EEG 001", "EEG 002"]))

    raw = _raw_fixture()
    mne_estimator = TimeShiftDSS(_identity, lag_samples=[0, 1]).fit(raw)
    with pytest.raises(TypeError, match="container families"):
        mne_estimator.transform(raw.get_data(picks=["EEG 001", "EEG 002"]))
