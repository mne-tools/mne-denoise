"""Tests for mne_denoise.utils module."""

import mne
import numpy as np

from mne_denoise.utils import (
    _HAS_MNE,
    extract_data_from_mne,
    reconstruct_mne_object,
)

# =====================================================================
# extract_data_from_mne
# =====================================================================


class TestExtractDataFromMne:
    """Tests for extract_data_from_mne."""

    def test_raw(self):
        info = mne.create_info(ch_names=["C1", "C2"], sfreq=100.0, ch_types="eeg")
        raw = mne.io.RawArray(np.random.randn(2, 200), info)
        data, sfreq, mne_type, orig = extract_data_from_mne(raw)
        assert data.shape == (2, 200)
        assert sfreq == 100.0
        assert mne_type == "raw"
        assert orig is raw

    def test_epochs(self):
        info = mne.create_info(ch_names=["C1", "C2"], sfreq=100.0, ch_types="eeg")
        data_3d = np.random.randn(5, 2, 100)
        events = np.column_stack(
            [np.arange(5) * 100, np.zeros(5, int), np.ones(5, int)]
        )
        epochs = mne.EpochsArray(data_3d, info, events=events)
        data, sfreq, mne_type, orig = extract_data_from_mne(epochs)
        assert data.shape == (5, 2, 100)
        assert sfreq == 100.0
        assert mne_type == "epochs"
        assert orig is epochs

    def test_evoked(self):
        info = mne.create_info(ch_names=["C1", "C2"], sfreq=100.0, ch_types="eeg")
        data_2d = np.random.randn(2, 100)
        evoked = mne.EvokedArray(data_2d, info, tmin=0.0)
        data, sfreq, mne_type, orig = extract_data_from_mne(evoked)
        assert data.shape == (2, 100)
        assert mne_type == "evoked"

    def test_ndarray(self):
        arr = np.random.randn(3, 50)
        data, sfreq, mne_type, orig = extract_data_from_mne(arr)
        assert data.shape == (3, 50)
        assert sfreq is None
        assert mne_type == "array"
        assert orig is None

    def test_list_input(self):
        data, sfreq, mne_type, orig = extract_data_from_mne([[1, 2], [3, 4]])
        assert isinstance(data, np.ndarray)
        assert mne_type == "array"


# =====================================================================
# reconstruct_mne_object
# =====================================================================


class TestReconstructMneObject:
    """Tests for reconstruct_mne_object."""

    def test_array_passthrough(self):
        arr = np.random.randn(3, 50)
        out = reconstruct_mne_object(arr, None, "array")
        assert out is arr

    def test_none_orig(self):
        arr = np.random.randn(3, 50)
        out = reconstruct_mne_object(arr, "dummy", "array")
        assert out is arr

    def test_raw_reconstruction(self):
        info = mne.create_info(ch_names=["C1", "C2"], sfreq=100.0, ch_types="eeg")
        raw = mne.io.RawArray(np.random.randn(2, 200), info)
        new_data = np.random.randn(2, 200)
        out = reconstruct_mne_object(new_data, raw, "raw")
        assert isinstance(out, mne.io.RawArray)
        np.testing.assert_array_almost_equal(out.get_data(), new_data)

    def test_raw_with_annotations(self):
        info = mne.create_info(ch_names=["C1"], sfreq=100.0, ch_types="eeg")
        raw = mne.io.RawArray(np.random.randn(1, 200), info)
        raw.set_annotations(
            mne.Annotations(onset=[0.5], duration=[0.1], description=["bad"])
        )
        new_data = np.random.randn(1, 200)
        out = reconstruct_mne_object(new_data, raw, "raw")
        assert len(out.annotations) > 0

    def test_epochs_reconstruction(self):
        info = mne.create_info(ch_names=["C1", "C2"], sfreq=100.0, ch_types="eeg")
        data_3d = np.random.randn(5, 2, 100)
        events = np.column_stack(
            [np.arange(5) * 100, np.zeros(5, int), np.ones(5, int)]
        )
        epochs = mne.EpochsArray(data_3d, info, events=events, event_id={"stim": 1})
        new_data = np.random.randn(5, 2, 100)
        out = reconstruct_mne_object(new_data, epochs, "epochs")
        assert isinstance(out, mne.EpochsArray)
        np.testing.assert_array_almost_equal(out.get_data(), new_data)

    def test_evoked_reconstruction(self):
        info = mne.create_info(ch_names=["C1", "C2"], sfreq=100.0, ch_types="eeg")
        data_2d = np.random.randn(2, 100)
        evoked = mne.EvokedArray(data_2d, info, tmin=-0.1, nave=10, comment="test")
        new_data = np.random.randn(2, 100)
        out = reconstruct_mne_object(new_data, evoked, "evoked")
        assert isinstance(out, mne.EvokedArray)
        assert out.nave == 10
        assert out.comment == "test"

    def test_unknown_type_passthrough(self):
        arr = np.random.randn(3, 50)
        info = mne.create_info(ch_names=["C1", "C2", "C3"], sfreq=100.0, ch_types="eeg")
        raw = mne.io.RawArray(arr, info)
        out = reconstruct_mne_object(arr, raw, "unknown")
        assert out is arr


class TestHasMne:
    """Test that _HAS_MNE is True in test environment."""

    def test_mne_available(self):
        assert _HAS_MNE is True
