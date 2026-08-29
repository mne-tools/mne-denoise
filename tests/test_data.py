"""Tests for MNE and array data extraction helpers."""

import mne
import numpy as np
import pytest

from mne_denoise._data import (
    _get_homogeneous_picks,
    continuous_to_epochs,
    epochs_to_continuous,
    extract_data_from_mne,
    reconstruct_mne_object,
)


def test_extract_data_supported_inputs():
    """Raw, Epochs, Evoked, arrays, and lists expose coherent tuple semantics."""
    rng = np.random.default_rng(0)
    info = mne.create_info(["C1", "C2"], 100.0, "eeg")
    raw_data = rng.standard_normal((2, 200))
    raw = mne.io.RawArray(raw_data, info, verbose=False)
    data, sfreq, mne_type, original, picks, names = extract_data_from_mne(raw)
    assert data.shape == raw_data.shape
    assert sfreq == 100.0 and mne_type == "raw"
    assert original is raw and picks is None and names == ["C1", "C2"]
    np.testing.assert_array_equal(data, raw_data)

    epochs_data = rng.standard_normal((5, 2, 100))
    events = np.column_stack([np.arange(5) * 100, np.zeros(5, int), np.ones(5, int)])
    epochs = mne.EpochsArray(epochs_data, info, events=events, verbose=False)
    data, sfreq, mne_type, original, picks, names = extract_data_from_mne(epochs)
    assert data.shape == epochs_data.shape
    assert sfreq == 100.0 and mne_type == "epochs"
    assert original is epochs and picks is None and names == ["C1", "C2"]
    continuous, _, epoch_type, _, _, _ = extract_data_from_mne(
        epochs, concatenate_epochs=True
    )
    assert epoch_type == "epochs" and continuous.shape == (2, 500)
    np.testing.assert_array_equal(
        continuous, epochs_data.transpose(1, 0, 2).reshape(2, -1)
    )
    channel_first, *_ = extract_data_from_mne(epochs, channel_first_epochs=True)
    np.testing.assert_array_equal(channel_first, epochs_data.transpose(1, 2, 0))

    evoked_data = rng.standard_normal((2, 100))
    evoked = mne.EvokedArray(evoked_data, info, tmin=0.0)
    data, sfreq, mne_type, original, picks, names = extract_data_from_mne(evoked)
    assert data.shape == evoked_data.shape
    assert sfreq == 100.0 and mne_type == "evoked"
    assert original is evoked and picks is None and names == ["C1", "C2"]
    np.testing.assert_array_equal(data, evoked_data)

    array = np.arange(30.0).reshape(3, 10)
    data, sfreq, mne_type, original, picks, names = extract_data_from_mne(array)
    assert data is array and sfreq is None and mne_type == "array"
    assert original is None and picks is None and names is None
    listed, sfreq, mne_type, original, picks, names = extract_data_from_mne(
        [[1, 2], [3, 4]]
    )
    assert isinstance(listed, np.ndarray)
    assert sfreq is None and mne_type == "array"
    assert original is None and picks is None and names is None


def test_extract_data_channel_selection_policy():
    """Data-channel policy excludes unsupported types and applies bad timing deliberately."""
    info = mne.create_info(
        ["stim1", "grad1", "eog1", "grad2", "misc1"],
        100.0,
        ["stim", "grad", "eog", "grad", "misc"],
    )
    raw = mne.io.RawArray(np.random.default_rng(1).standard_normal((5, 100)), info)
    data, _, _, _, picks, names = extract_data_from_mne(raw, auto_pick="data")
    assert data.shape == (2, 100)
    np.testing.assert_array_equal(picks, [1, 3])
    assert names == ["grad1", "grad2"]

    info = mne.create_info(
        ["eeg1", "mag1", "grad1", "eeg2"],
        100.0,
        ["eeg", "mag", "grad", "eeg"],
    )
    raw = mne.io.RawArray(np.random.default_rng(2).standard_normal((4, 100)), info)
    with pytest.warns(UserWarning, match="Found multiple data channel types"):
        picks = _get_homogeneous_picks(raw)
    np.testing.assert_array_equal(picks, [1])

    bad_info = mne.create_info(["eeg1", "mag1", "eeg2"], 100.0, ["eeg", "mag", "eeg"])
    bad_raw = mne.io.RawArray(
        np.random.default_rng(3).standard_normal((3, 100)), bad_info
    )
    bad_raw.info["bads"] = ["mag1"]
    with pytest.warns(UserWarning, match="Found multiple data channel types"):
        np.testing.assert_array_equal(_get_homogeneous_picks(bad_raw), [1])
    with pytest.warns(UserWarning, match="Found multiple data channel types"):
        with pytest.raises(
            ValueError, match="No good data channels remain after excluding bads"
        ):
            extract_data_from_mne(bad_raw, exclude_bads=True)

    csd_info = mne.create_info(["CSD", "EEG"], 100.0, ["csd", "eeg"])
    csd_raw = mne.io.RawArray(
        np.random.default_rng(4).standard_normal((2, 100)), csd_info
    )
    np.testing.assert_array_equal(_get_homogeneous_picks(csd_raw), [1])

    explicit_info = mne.create_info(["C1", "C2"], 100.0, "eeg")
    explicit_raw = mne.io.RawArray(np.ones((2, 100)), explicit_info, verbose=False)
    explicit_raw.info["bads"] = ["C2"]
    data, _, _, _, picks, names = extract_data_from_mne(
        explicit_raw, ch_names=["C2", "C1"], exclude_bads=True
    )
    assert data.shape == (2, 100)
    np.testing.assert_array_equal(picks, [1, 0])
    assert names == ["C2", "C1"]
    with pytest.raises(ValueError, match="Found multiple data channel types"):
        _get_homogeneous_picks(
            mne.io.RawArray(
                np.ones((2, 100)),
                mne.create_info(["mag1", "grad1"], 100.0, ["mag", "grad"]),
                verbose=False,
            ),
            auto_pick="raise",
        )
    all_bad = explicit_raw.copy()
    all_bad.info["bads"] = ["C1", "C2"]
    with pytest.raises(
        ValueError, match="No good data channels remain after excluding bads"
    ):
        extract_data_from_mne(all_bad, exclude_bads=True)
    with pytest.raises(ValueError, match="cannot both be True"):
        extract_data_from_mne(
            np.ones((2, 3, 4)), concatenate_epochs=True, channel_first_epochs=True
        )


def test_reconstruct_mne_containers():
    """Raw, Epochs, and Evoked reconstruction keeps their essential bookkeeping."""
    rng = np.random.default_rng(5)
    info = mne.create_info(["C1", "C2", "STIM"], 100.0, ["eeg", "eeg", "stim"])
    original = rng.standard_normal((3, 200))
    raw = mne.io.RawArray(original, info, first_samp=41, verbose=False)
    raw.set_annotations(mne.Annotations([0.5], [0.1], ["bad"]))
    replacement = rng.standard_normal((2, 200))
    out = reconstruct_mne_object(replacement, raw, "raw", picks=np.array([0, 1]))
    assert isinstance(out, mne.io.RawArray)
    assert out.first_samp == 41 and len(out.annotations) == 1
    np.testing.assert_allclose(out.get_data()[:2], replacement)
    np.testing.assert_array_equal(out.get_data()[2], original[2])
    np.testing.assert_array_equal(raw.get_data(), original)

    epoch_info = mne.create_info(["C1", "C2", "STIM"], 100.0, ["eeg", "eeg", "stim"])
    epoch_data = rng.standard_normal((4, 3, 80))
    epoch_events = np.column_stack(
        [np.arange(4) * 100, np.zeros(4, int), np.ones(4, int)]
    )
    epochs = mne.EpochsArray(epoch_data, epoch_info, events=epoch_events, verbose=False)
    epochs.drop([1], reason="USER")
    replacement = rng.standard_normal((3, 2, 80))
    out = reconstruct_mne_object(replacement, epochs, "epochs", picks=np.array([0, 1]))
    assert isinstance(out, mne.EpochsArray)
    np.testing.assert_array_equal(out.selection, epochs.selection)
    assert out.drop_log == epochs.drop_log
    np.testing.assert_allclose(out.get_data()[:, :2], replacement)
    np.testing.assert_array_equal(out.get_data()[:, 2], epochs.get_data()[:, 2])

    evoked = mne.EvokedArray(
        rng.standard_normal((2, 100)),
        mne.create_info(2, 100.0, "eeg"),
        tmin=-0.1,
        nave=10,
        comment="test",
    )
    out = reconstruct_mne_object(rng.standard_normal((2, 100)), evoked, "evoked")
    assert isinstance(out, mne.EvokedArray)
    assert out.nave == 10 and out.comment == "test"


def test_reconstruct_passthrough_paths():
    """Array data and unsupported reconstruction types intentionally pass through."""
    array = np.random.default_rng(6).standard_normal((3, 50))
    assert reconstruct_mne_object(array, None, "array") is array
    assert reconstruct_mne_object(array, "dummy", "array") is array
    raw = mne.io.RawArray(array, mne.create_info(3, 100.0, "eeg"), verbose=False)
    assert reconstruct_mne_object(array, raw, "unknown") is array


def test_epoch_layout_conversion_contract():
    """Epoch conversion round-trips channel-first layout and continuous input."""
    rng = np.random.default_rng(7)
    epochs = rng.standard_normal((5, 4, 30))
    continuous = epochs_to_continuous(epochs)
    assert continuous.shape == (4, 150)
    np.testing.assert_array_equal(continuous, epochs.transpose(1, 0, 2).reshape(4, -1))
    np.testing.assert_array_equal(
        continuous_to_epochs(continuous, epochs.shape), epochs
    )
    array = rng.standard_normal((4, 50))
    assert epochs_to_continuous(array) is array
    np.testing.assert_array_equal(continuous_to_epochs(array, array.shape), array)


def test_epoch_layout_conversion_validation():
    """Conversion rejects unsupported dimensionality and target geometry."""
    for data in (np.ones(10), np.ones((2, 2, 2, 2))):
        with pytest.raises(ValueError, match="must be 2D or 3D"):
            epochs_to_continuous(data)
    with pytest.raises(ValueError, match="shape must have 2 or 3 entries"):
        continuous_to_epochs(np.ones((3, 10)), (2, 3, 4, 5))
    cases = [
        (
            np.ones((2, 30)),
            (3, 3, 10),
            "continuous has 2 channels but target shape expects 3",
        ),
        (
            np.ones((2, 29)),
            (3, 2, 10),
            "continuous has 29 samples but target shape expects 30",
        ),
        (np.ones((3, 2, 10)), (3, 2, 10), "continuous must be 2D, got 3D"),
    ]
    for data, shape, message in cases:
        with pytest.raises(ValueError, match=message):
            continuous_to_epochs(data, shape)
