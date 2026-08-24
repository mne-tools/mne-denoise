"""Tests for mne_denoise.utils module."""

import mne
import numpy as np
import pytest

from mne_denoise.utils import (
    _HAS_MNE,
    extract_data_from_mne,
    reconstruct_mne_object,
)

# =====================================================================
# extract_data_from_mne
# =====================================================================


def test_extract_data_from_mne_raw():
    info = mne.create_info(ch_names=["C1", "C2"], sfreq=100.0, ch_types="eeg")
    raw = mne.io.RawArray(np.random.randn(2, 200), info)
    data, sfreq, mne_type, orig, picks, ch_names = extract_data_from_mne(raw)
    assert data.shape == (2, 200)
    assert sfreq == 100.0
    assert mne_type == "raw"
    assert orig is raw


def test_extract_data_from_mne_epochs():
    info = mne.create_info(ch_names=["C1", "C2"], sfreq=100.0, ch_types="eeg")
    data_3d = np.random.randn(5, 2, 100)
    events = np.column_stack([np.arange(5) * 100, np.zeros(5, int), np.ones(5, int)])
    epochs = mne.EpochsArray(data_3d, info, events=events)
    data, sfreq, mne_type, orig, picks, ch_names = extract_data_from_mne(epochs)
    assert data.shape == (5, 2, 100)
    assert sfreq == 100.0
    assert mne_type == "epochs"
    assert orig is epochs

    continuous, _, mne_type, _, _, _ = extract_data_from_mne(
        epochs, concatenate_epochs=True
    )
    assert continuous.shape == (2, 500)
    assert mne_type == "epochs"
    np.testing.assert_array_equal(
        continuous,
        data_3d.transpose(1, 0, 2).reshape(2, -1),
    )

    channel_first, *_ = extract_data_from_mne(epochs, channel_first_epochs=True)
    np.testing.assert_array_equal(channel_first, data_3d.transpose(1, 2, 0))


def test_extract_data_from_mne_all_data_channels():
    """The data policy should jointly pick data channels and omit stim channels."""
    info = mne.create_info(
        ["MAG", "GRAD", "EEG", "STIM"],
        100.0,
        ["mag", "grad", "eeg", "stim"],
    )
    raw = mne.io.RawArray(np.ones((4, 100)), info, verbose=False)

    data, _, _, _, picks, ch_names = extract_data_from_mne(raw, auto_pick="data")

    assert data.shape == (3, 100)
    np.testing.assert_array_equal(picks, [0, 1, 2])
    assert ch_names == ["MAG", "GRAD", "EEG"]


def test_extract_data_from_mne_can_exclude_bads():
    """Automatic extraction can establish a good-channel fitted contract."""
    info = mne.create_info(["C1", "C2", "C3"], 100.0, "eeg")
    raw = mne.io.RawArray(np.arange(300).reshape(3, 100), info, verbose=False)
    raw.info["bads"] = ["C2"]

    data, _, _, _, picks, ch_names = extract_data_from_mne(raw, exclude_bads=True)

    np.testing.assert_array_equal(picks, [0, 2])
    np.testing.assert_array_equal(data, raw.get_data(picks=[0, 2]))
    assert ch_names == ["C1", "C3"]


def test_extract_data_from_mne_explicit_names_override_bad_exclusion():
    """Explicit names remain authoritative when applying a fitted contract."""
    info = mne.create_info(["C1", "C2"], 100.0, "eeg")
    raw = mne.io.RawArray(np.ones((2, 100)), info, verbose=False)
    raw.info["bads"] = ["C2"]

    data, _, _, _, picks, ch_names = extract_data_from_mne(
        raw,
        ch_names=["C2"],
        exclude_bads=True,
    )

    np.testing.assert_array_equal(picks, [1])
    assert data.shape == (1, 100)
    assert ch_names == ["C2"]


def test_extract_data_from_mne_epoch_layout_options_are_exclusive():
    """Epoch concatenation and channel-first 3D output cannot be requested together."""
    with pytest.raises(ValueError, match="cannot both be True"):
        extract_data_from_mne(
            np.ones((2, 3, 4)),
            concatenate_epochs=True,
            channel_first_epochs=True,
        )


def test_extract_data_from_mne_evoked():
    info = mne.create_info(ch_names=["C1", "C2"], sfreq=100.0, ch_types="eeg")
    data_2d = np.random.randn(2, 100)
    evoked = mne.EvokedArray(data_2d, info, tmin=0.0)
    data, sfreq, mne_type, orig, picks, ch_names = extract_data_from_mne(evoked)
    assert data.shape == (2, 100)
    assert mne_type == "evoked"


def test_extract_data_from_mne_ndarray():
    arr = np.random.randn(3, 50)
    data, sfreq, mne_type, orig, picks, ch_names = extract_data_from_mne(arr)
    assert data.shape == (3, 50)
    assert sfreq is None
    assert mne_type == "array"
    assert orig is None

    epochs = np.arange(30).reshape(3, 2, 5)
    continuous, *_ = extract_data_from_mne(epochs, concatenate_epochs=True)
    np.testing.assert_array_equal(
        continuous,
        epochs.transpose(1, 0, 2).reshape(2, -1),
    )


def test_extract_data_from_mne_list_input():
    data, sfreq, mne_type, orig, picks, ch_names = extract_data_from_mne(
        [[1, 2], [3, 4]]
    )
    assert isinstance(data, np.ndarray)
    assert mne_type == "array"


# =====================================================================
# reconstruct_mne_object
# =====================================================================


def test_reconstruct_mne_object_array_passthrough():
    arr = np.random.randn(3, 50)
    out = reconstruct_mne_object(arr, None, "array")
    assert out is arr


def test_reconstruct_mne_object_none_orig():
    arr = np.random.randn(3, 50)
    out = reconstruct_mne_object(arr, "dummy", "array")
    assert out is arr


def test_reconstruct_mne_object_raw_reconstruction():
    info = mne.create_info(ch_names=["C1", "C2"], sfreq=100.0, ch_types="eeg")
    raw = mne.io.RawArray(np.random.randn(2, 200), info)
    new_data = np.random.randn(2, 200)
    out = reconstruct_mne_object(new_data, raw, "raw")
    assert isinstance(out, mne.io.RawArray)
    np.testing.assert_array_almost_equal(out.get_data(), new_data)


def test_reconstruct_mne_object_raw_with_annotations():
    info = mne.create_info(ch_names=["C1"], sfreq=100.0, ch_types="eeg")
    raw = mne.io.RawArray(np.random.randn(1, 200), info)
    raw.set_annotations(
        mne.Annotations(onset=[0.5], duration=[0.1], description=["bad"])
    )
    new_data = np.random.randn(1, 200)
    out = reconstruct_mne_object(new_data, raw, "raw")
    assert len(out.annotations) > 0


def test_reconstruct_mne_object_raw_preserves_first_sample_and_picks():
    """Copy-based restoration retains Raw identity and untouched channels."""
    info = mne.create_info(["C1", "C2", "STIM"], 100.0, ["eeg", "eeg", "stim"])
    original = np.random.randn(3, 200)
    raw = mne.io.RawArray(original, info, first_samp=41, verbose=False)
    replacement = np.random.randn(2, 200)
    out = reconstruct_mne_object(replacement, raw, "raw", picks=np.array([0, 1]))
    assert out.first_samp == 41
    np.testing.assert_allclose(out.get_data()[:2], replacement)
    np.testing.assert_array_equal(out.get_data()[2], original[2])
    np.testing.assert_array_equal(raw.get_data(), original)


def test_reconstruct_mne_object_epochs_reconstruction():
    info = mne.create_info(ch_names=["C1", "C2"], sfreq=100.0, ch_types="eeg")
    data_3d = np.random.randn(5, 2, 100)
    events = np.column_stack([np.arange(5) * 100, np.zeros(5, int), np.ones(5, int)])
    epochs = mne.EpochsArray(data_3d, info, events=events, event_id={"stim": 1})
    new_data = np.random.randn(5, 2, 100)
    out = reconstruct_mne_object(new_data, epochs, "epochs")
    assert isinstance(out, mne.EpochsArray)
    np.testing.assert_array_almost_equal(out.get_data(), new_data)


def test_reconstruct_mne_object_epochs_preserves_selection_and_drop_log():
    """Epoch bookkeeping survives copy-based reconstruction."""
    info = mne.create_info(["C1", "C2", "STIM"], 100.0, ["eeg", "eeg", "stim"])
    data = np.random.randn(4, 3, 80)
    events = np.column_stack([np.arange(4) * 100, np.zeros(4, int), np.ones(4, int)])
    epochs = mne.EpochsArray(data, info, events=events, verbose=False)
    epochs.drop([1], reason="USER")
    replacement = np.random.randn(3, 2, 80)
    out = reconstruct_mne_object(replacement, epochs, "epochs", picks=np.array([0, 1]))
    np.testing.assert_array_equal(out.selection, epochs.selection)
    assert out.drop_log == epochs.drop_log
    np.testing.assert_allclose(out.get_data()[:, :2], replacement)
    np.testing.assert_array_equal(out.get_data()[:, 2], epochs.get_data()[:, 2])


def test_reconstruct_mne_object_evoked_reconstruction():
    info = mne.create_info(ch_names=["C1", "C2"], sfreq=100.0, ch_types="eeg")
    data_2d = np.random.randn(2, 100)
    evoked = mne.EvokedArray(data_2d, info, tmin=-0.1, nave=10, comment="test")
    new_data = np.random.randn(2, 100)
    out = reconstruct_mne_object(new_data, evoked, "evoked")
    assert isinstance(out, mne.EvokedArray)
    assert out.nave == 10
    assert out.comment == "test"


def test_reconstruct_mne_object_unknown_type_passthrough():
    arr = np.random.randn(3, 50)
    info = mne.create_info(ch_names=["C1", "C2", "C3"], sfreq=100.0, ch_types="eeg")
    raw = mne.io.RawArray(arr, info)
    out = reconstruct_mne_object(arr, raw, "unknown")
    assert out is arr


# =====================================================================
# has_mne
# =====================================================================


def test_mne_available():
    """Test that _HAS_MNE is True in test environment."""
    assert _HAS_MNE is True


# =====================================================================
# auto_pick
# =====================================================================


def test_auto_pick_single_type():
    from mne_denoise.utils import _get_homogeneous_picks

    # Only grad and eog
    info = mne.create_info(
        ch_names=["grad1", "grad2", "eog1"],
        sfreq=100.0,
        ch_types=["grad", "grad", "eog"],
    )
    raw = mne.io.RawArray(np.random.randn(3, 100), info)

    # Should return picks for the 2 grad channels, ignoring EOG
    picks = _get_homogeneous_picks(raw)
    assert len(picks) == 2
    np.testing.assert_array_equal(picks, [0, 1])


def test_auto_pick_mixed_types_warn():
    from mne_denoise.utils import _get_homogeneous_picks

    # Mixed mag and grad
    info = mne.create_info(
        ch_names=["mag1", "grad1"], sfreq=100.0, ch_types=["mag", "grad"]
    )
    raw = mne.io.RawArray(np.random.randn(2, 100), info)

    # By default (auto_pick='auto'), it should warn and pick 'mag' (the first one)
    with pytest.warns(UserWarning, match="Found multiple data channel types"):
        picks = _get_homogeneous_picks(raw)
    assert len(picks) == 1
    assert picks[0] == 0


def test_auto_pick_mixed_types_raise():
    from mne_denoise.utils import _get_homogeneous_picks

    # Mixed mag and grad
    info = mne.create_info(
        ch_names=["mag1", "grad1"], sfreq=100.0, ch_types=["mag", "grad"]
    )
    raw = mne.io.RawArray(np.random.randn(2, 100), info)

    # If auto_pick='raise', it should raise ValueError
    with pytest.raises(ValueError, match="Found multiple data channel types"):
        _get_homogeneous_picks(raw, auto_pick="raise")
