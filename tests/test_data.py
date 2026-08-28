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


def test_extract_data_from_mne_raw():
    info = mne.create_info(ch_names=["C1", "C2"], sfreq=100.0, ch_types="eeg")
    raw_data = np.random.randn(2, 200)
    raw = mne.io.RawArray(raw_data, info)
    data, sfreq, mne_type, orig, picks, ch_names = extract_data_from_mne(raw)
    assert data.shape == (2, 200)
    assert sfreq == 100.0
    assert mne_type == "raw"
    assert orig is raw
    assert picks is None
    assert ch_names == ["C1", "C2"]
    np.testing.assert_array_equal(data, raw_data)


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
    assert picks is None
    assert ch_names == ["C1", "C2"]

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
        ch_names=["C2", "C1"],
        exclude_bads=True,
    )

    np.testing.assert_array_equal(picks, [1, 0])
    assert data.shape == (2, 100)
    assert ch_names == ["C2", "C1"]


def test_extract_data_from_mne_excluding_all_bads_raises():
    info = mne.create_info(["C1", "C2"], 100.0, "eeg")
    raw = mne.io.RawArray(np.ones((2, 100)), info, verbose=False)
    raw.info["bads"] = ["C1", "C2"]

    with pytest.raises(
        ValueError, match="No good data channels remain after excluding bads"
    ):
        extract_data_from_mne(raw, exclude_bads=True)


def test_extract_data_from_mne_missing_explicit_names():
    info = mne.create_info(["C1", "C2"], 100.0, "eeg")
    raw = mne.io.RawArray(np.ones((2, 100)), info, verbose=False)

    with pytest.raises(ValueError, match="Missing channels"):
        extract_data_from_mne(raw, ch_names=["C3"])


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
    assert sfreq == 100.0
    assert mne_type == "evoked"
    assert orig is evoked
    assert picks is None
    assert ch_names == ["C1", "C2"]
    np.testing.assert_array_equal(data, data_2d)


def test_extract_data_from_mne_ndarray():
    arr = np.random.randn(3, 50)
    data, sfreq, mne_type, orig, picks, ch_names = extract_data_from_mne(arr)
    assert data.shape == (3, 50)
    assert sfreq is None
    assert mne_type == "array"
    assert orig is None
    assert picks is None
    assert ch_names is None

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
    assert sfreq is None
    assert mne_type == "array"
    assert orig is None
    assert picks is None
    assert ch_names is None


def test_auto_pick_single_type():
    # Only grad among data, with unsupported/non-target channels present.
    info = mne.create_info(
        ch_names=["stim1", "grad1", "eog1", "grad2", "misc1"],
        sfreq=100.0,
        ch_types=["stim", "grad", "eog", "grad", "misc"],
    )
    raw = mne.io.RawArray(np.random.randn(5, 100), info)

    # Should return picks for the grad channels, ignoring non-target channels.
    picks = _get_homogeneous_picks(raw)
    assert len(picks) == 2
    np.testing.assert_array_equal(picks, [1, 3])


def test_auto_pick_priority_is_mag_grad_eeg():
    info = mne.create_info(
        ch_names=["eeg1", "mag1", "grad1", "mag2", "eeg2"],
        sfreq=100.0,
        ch_types=["eeg", "mag", "grad", "mag", "eeg"],
    )
    raw = mne.io.RawArray(np.random.randn(5, 100), info)

    with pytest.warns(UserWarning, match="Found multiple data channel types"):
        picks = _get_homogeneous_picks(raw)

    np.testing.assert_array_equal(picks, [1, 3])


def test_auto_pick_bad_channels_participate_in_type_detection():
    info = mne.create_info(
        ch_names=["eeg1", "mag1", "eeg2"],
        sfreq=100.0,
        ch_types=["eeg", "mag", "eeg"],
    )
    raw = mne.io.RawArray(np.random.randn(3, 100), info)
    raw.info["bads"] = ["mag1"]

    with pytest.warns(UserWarning, match="Found multiple data channel types"):
        picks = _get_homogeneous_picks(raw)
    np.testing.assert_array_equal(picks, [1])

    with pytest.warns(UserWarning, match="Found multiple data channel types"):
        with pytest.raises(
            ValueError, match="No good data channels remain after excluding bads"
        ):
            extract_data_from_mne(raw, exclude_bads=True)


def test_auto_pick_does_not_treat_csd_as_eeg():
    info = mne.create_info(
        ch_names=["CSD", "EEG"],
        sfreq=100.0,
        ch_types=["csd", "eeg"],
    )
    raw = mne.io.RawArray(np.random.randn(2, 100), info)

    picks = _get_homogeneous_picks(raw)

    np.testing.assert_array_equal(picks, [1])


def test_auto_pick_mixed_types_warn():
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
    # Mixed mag and grad
    info = mne.create_info(
        ch_names=["mag1", "grad1"], sfreq=100.0, ch_types=["mag", "grad"]
    )
    raw = mne.io.RawArray(np.random.randn(2, 100), info)

    # If auto_pick='raise', it should raise ValueError
    with pytest.raises(ValueError, match="Found multiple data channel types"):
        _get_homogeneous_picks(raw, auto_pick="raise")


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


def test_epochs_to_continuous_concatenates_along_time():
    """Epochs are laid out channel-first and joined end to end."""
    X = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    continuous = epochs_to_continuous(X)
    assert continuous.shape == (3, 8)
    # Channel 0 of epoch 0 followed by channel 0 of epoch 1.
    np.testing.assert_array_equal(continuous[0], np.concatenate([X[0, 0], X[1, 0]]))


def test_round_trip_is_exact():
    """continuous_to_epochs inverts epochs_to_continuous."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((5, 4, 30))
    np.testing.assert_array_equal(
        continuous_to_epochs(epochs_to_continuous(X), X.shape), X
    )


def test_continuous_input_passes_through():
    """Two-dimensional data is returned unchanged by both helpers."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((4, 50))
    assert epochs_to_continuous(X) is X
    continuous = epochs_to_continuous(X)
    np.testing.assert_array_equal(continuous_to_epochs(continuous, X.shape), X)


def test_matches_the_manual_idiom():
    """The helpers reproduce the transpose/reshape idiom they replaced."""
    rng = np.random.default_rng(2)
    X = rng.standard_normal((3, 6, 20))
    n_epochs, n_channels, n_times = X.shape
    np.testing.assert_array_equal(
        epochs_to_continuous(X), X.transpose(1, 0, 2).reshape(n_channels, -1)
    )
    continuous = epochs_to_continuous(X)
    np.testing.assert_array_equal(
        continuous_to_epochs(continuous, X.shape),
        continuous.reshape(n_channels, n_epochs, n_times).transpose(1, 0, 2),
    )


@pytest.mark.parametrize("ndim", [1, 4])
def test_epochs_to_continuous_rejects_other_dimensions(ndim):
    """Only 2-D and 3-D layouts are meaningful."""
    with pytest.raises(ValueError, match="must be 2D or 3D"):
        epochs_to_continuous(np.ones((2,) * ndim))


def test_continuous_to_epochs_rejects_other_shapes():
    """The target shape must describe continuous or epoched data."""
    with pytest.raises(ValueError, match="shape must have 2 or 3 entries"):
        continuous_to_epochs(np.ones((3, 10)), (2, 3, 4, 5))


def test_continuous_to_epochs_rejects_channel_mismatch():
    """The continuous channel count must match the target epoch shape."""
    with pytest.raises(
        ValueError,
        match="continuous has 2 channels but target shape expects 3",
    ):
        continuous_to_epochs(np.ones((2, 30)), (3, 3, 10))


def test_continuous_to_epochs_rejects_sample_count_mismatch():
    """The continuous sample count must match all target epochs."""
    with pytest.raises(
        ValueError,
        match="continuous has 29 samples but target shape expects 30",
    ):
        continuous_to_epochs(np.ones((2, 29)), (3, 2, 10))


def test_continuous_to_epochs_rejects_non_2d_continuous_input():
    """A 3-D target requires continuous channel-first input."""
    with pytest.raises(ValueError, match="continuous must be 2D, got 3D"):
        continuous_to_epochs(np.ones((3, 2, 10)), (3, 2, 10))
