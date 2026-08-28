"""Shared MNE container contracts for public estimators."""

from __future__ import annotations

import numpy as np
import pytest

from tests._contract_cases import (
    ESTIMATOR_CASES,
    FITTED_CHANNEL_NAMES,
    FITTED_CHANNEL_ORDER,
    MNE_EPOCHS,
    MNE_EVOKED,
    MNE_RAW,
    SFREQ_AWARE,
)

MNE_KINDS = (MNE_RAW, MNE_EPOCHS, MNE_EVOKED)


def _mne_parameters():
    return tuple(
        pytest.param(case, kind, id=f"{case.name}-{kind.removeprefix('mne_')}")
        for case in ESTIMATOR_CASES
        for kind in MNE_KINDS
        if kind in case.capabilities
    )


def _mne_estimator(case, names):
    if case.mne_factory is not None:
        return case.mne_factory(tuple(names))
    return case.make_estimator()


@pytest.fixture
def rich_mne_inputs():
    """Create Raw, Epochs, and Evoked objects with meaningful metadata."""
    mne = pytest.importorskip("mne")
    pd = pytest.importorskip("pandas")

    sfreq = 200.0
    names = ("EEG0", "EEG1", "EEG2", "EEG3", "EOG0", "STI 014")
    types = ("eeg", "eeg", "eeg", "eeg", "eog", "stim")
    rng = np.random.default_rng(20260829)
    n_times = 800
    time = np.arange(n_times) / sfreq
    artifact = rng.standard_normal(n_times) * 1e-6
    raw_data = rng.standard_normal((len(names), n_times)) * 1e-7
    raw_data[:4] += np.array([1.0, 0.8, 1.2, 0.6])[:, None] * artifact
    raw_data[4] = artifact + 0.1 * rng.standard_normal(n_times) * 1e-6
    raw_data[5] = (np.arange(n_times) % 5 == 0).astype(float)
    raw_data[:4] += 0.2e-6 * np.sin(2 * np.pi * 10.0 * time)

    def make_info():
        info = mne.create_info(names, sfreq, types)
        info["bads"] = ["EEG1"]
        return info

    raw = mne.io.RawArray(
        raw_data,
        make_info(),
        first_samp=123,
        verbose=False,
    )
    raw.set_annotations(
        mne.Annotations(
            onset=[0.2, 2.5],
            duration=[0.1, 0.2],
            description=["BAD_contract", "edge_marker"],
        )
    )

    n_epochs, n_epoch_times = 4, 240
    epoch_data = rng.standard_normal((n_epochs, len(names), n_epoch_times)) * 1e-7
    epoch_artifact = rng.standard_normal((n_epochs, n_epoch_times)) * 1e-6
    epoch_data[:, :4] += (
        np.array([1.0, 0.8, 1.2, 0.6])[None, :, None] * epoch_artifact[:, None, :]
    )
    epoch_data[:, 4] = epoch_artifact
    epoch_data[:, 5] = (np.arange(n_epoch_times) % 7 == 0).astype(float)
    events = np.column_stack(
        (
            np.arange(n_epochs) * 300,
            np.zeros(n_epochs, dtype=int),
            np.array([1, 2, 1, 2], dtype=int),
        )
    )
    epochs = mne.EpochsArray(
        epoch_data,
        make_info(),
        events=events,
        event_id={"left": 1, "right": 2},
        tmin=-0.1,
        baseline=(-0.1, 0.0),
        metadata=pd.DataFrame({"trial": ["a", "b", "c", "d"], "quality": [1, 2, 1, 3]}),
        verbose=False,
    )
    epochs.drop([1])

    evoked = mne.EvokedArray(
        raw_data,
        make_info(),
        tmin=-0.1,
        nave=17,
        comment="contract-condition",
        verbose=False,
    )
    return {MNE_RAW: raw, MNE_EPOCHS: epochs, MNE_EVOKED: evoked, "names": names}


def _data(inst, kind):
    return inst.data if kind == MNE_EVOKED else inst.get_data()


def _channel_data(data, kind, channel):
    return data[channel] if kind != MNE_EPOCHS else data[:, channel, :]


def _assert_fitted_raw_layout_safety(case, estimator, raw, names):
    """Check the fitted Raw layout properties declared by one estimator."""
    if FITTED_CHANNEL_NAMES in case.capabilities:
        renamed = raw.copy()
        renamed.rename_channels({names[0]: "renamed-channel"})
        with pytest.raises(ValueError):
            estimator.transform(renamed)

    if FITTED_CHANNEL_ORDER in case.capabilities:
        reordered = raw.copy().reorder_channels([names[1], names[0], *names[2:]])
        with pytest.raises(ValueError):
            estimator.transform(reordered)

    if SFREQ_AWARE in case.capabilities:
        mne = pytest.importorskip("mne")
        mismatch_info = mne.create_info(names, 220.0, raw.get_channel_types())
        mismatch = mne.io.RawArray(
            raw.get_data(), mismatch_info, first_samp=raw.first_samp, verbose=False
        )
        with pytest.raises(ValueError, match="(?i)sfreq|sampling"):
            estimator.transform(mismatch)


@pytest.mark.parametrize("case, kind", _mne_parameters())
def test_mne_container_contracts_preserve_identity_and_metadata(
    case, kind, rich_mne_inputs
):
    """Public cleaning returns a copy while preserving shared MNE identity."""
    inst = rich_mne_inputs[kind]
    names = rich_mne_inputs["names"]
    before = _data(inst, kind).copy()
    before_bads = list(inst.info["bads"])

    if kind == MNE_RAW:
        before_annotations = inst.annotations.copy()
        before_first_samp = inst.first_samp
    elif kind == MNE_EPOCHS:
        before_events = inst.events.copy()
        before_event_id = dict(inst.event_id)
        before_tmin = inst.tmin
        before_baseline = inst.baseline
        before_metadata = inst.metadata.copy(deep=True)
        before_selection = inst.selection.copy()
        before_drop_log = inst.drop_log
    else:
        before_nave = inst.nave
        before_comment = inst.comment
        before_tmin = inst.tmin
        before_first = inst.first
        before_last = inst.last
        before_times = inst.times.copy()

    estimator = _mne_estimator(case, names)
    cleaned = estimator.fit_transform(inst)
    cleaned_data = _data(cleaned, kind)

    mne = pytest.importorskip("mne")
    if kind == MNE_RAW:
        assert isinstance(cleaned, mne.io.BaseRaw)
    elif kind == MNE_EPOCHS:
        assert isinstance(cleaned, mne.BaseEpochs)
    else:
        assert isinstance(cleaned, mne.Evoked)
    assert cleaned is not inst
    assert cleaned.ch_names == list(names)
    assert list(cleaned.info["bads"]) == before_bads
    assert cleaned_data.shape == before.shape
    assert inst.ch_names == list(names)
    assert list(inst.info["bads"]) == before_bads

    for channel in (4, 5):
        np.testing.assert_array_equal(
            _channel_data(cleaned_data, kind, channel),
            _channel_data(before, kind, channel),
        )
    np.testing.assert_array_equal(_data(inst, kind), before)

    if kind == MNE_RAW:
        assert cleaned.first_samp == before_first_samp
        assert cleaned.annotations == before_annotations
        assert inst.first_samp == before_first_samp
        assert inst.annotations == before_annotations
    elif kind == MNE_EPOCHS:
        np.testing.assert_array_equal(cleaned.events, before_events)
        assert cleaned.event_id == before_event_id
        assert cleaned.tmin == before_tmin
        assert cleaned.baseline == before_baseline
        pd = pytest.importorskip("pandas")
        pd.testing.assert_frame_equal(cleaned.metadata, before_metadata)
        np.testing.assert_array_equal(cleaned.selection, before_selection)
        assert cleaned.drop_log == before_drop_log
        np.testing.assert_array_equal(inst.events, before_events)
        assert inst.event_id == before_event_id
        assert inst.tmin == before_tmin
        assert inst.baseline == before_baseline
        pd.testing.assert_frame_equal(inst.metadata, before_metadata)
        np.testing.assert_array_equal(inst.selection, before_selection)
        assert inst.drop_log == before_drop_log
    else:
        assert cleaned.nave == before_nave
        assert cleaned.comment == before_comment
        assert cleaned.tmin == before_tmin
        assert cleaned.first == before_first
        assert cleaned.last == before_last
        np.testing.assert_array_equal(cleaned.times, before_times)
        assert inst.nave == before_nave
        assert inst.comment == before_comment
        assert inst.tmin == before_tmin
        assert inst.first == before_first
        assert inst.last == before_last
        np.testing.assert_array_equal(inst.times, before_times)

    if kind == MNE_RAW:
        _assert_fitted_raw_layout_safety(case, estimator, inst, names)
