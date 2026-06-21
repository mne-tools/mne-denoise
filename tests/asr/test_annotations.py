import numpy as np
import pytest

mne = pytest.importorskip("mne")

from mne_denoise.asr._annotations import (
    _calibration_annotations,
    _rejection_annotations,
    _repair_annotations,
)


def test_repair_annotations():
    diagnostics = {
        "window_starts": [0, 10, 20],
        "window_stops": [10, 20, 30],
        "n_components_reconstructed": [0, 5, 2],
    }
    sfreq = 10.0

    # Require 2 components
    annots = _repair_annotations(
        diagnostics, sfreq, min_components=2, description="REPAIR"
    )
    assert len(annots) == 1
    assert np.allclose(annots.onset, [1.0])
    assert np.allclose(
        annots.duration, [2.0]
    )  # spans from 10 to 30 (duration 20 samples -> 2s)
    assert annots.description[0] == "REPAIR"


def test_rejection_annotations():
    sfreq = 10.0
    # Mask where False means rejected
    mask = np.array([True, True, False, False, True, False])
    annots = _rejection_annotations(mask, sfreq, description="BAD")
    assert len(annots) == 2
    assert np.allclose(annots.onset, [0.2, 0.5])
    assert np.allclose(annots.duration, [0.2, 0.1])
    assert annots.description[0] == "BAD"

    # All good
    all_good = np.array([True, True, True])
    annots2 = _rejection_annotations(all_good, sfreq, description="BAD")
    assert len(annots2) == 0

    with pytest.raises(RuntimeError, match="continuous transform diagnostics"):
        _rejection_annotations(np.array([[True]]), sfreq, "BAD")


def test_calibration_annotations():
    sfreq = 10.0
    mask = np.array([False, True, True, False])
    annots = _calibration_annotations("sample", mask, sfreq, "CAL")
    assert len(annots) == 1
    assert np.allclose(annots.onset, [0.1])
    assert np.allclose(annots.duration, [0.2])

    with pytest.raises(RuntimeError, match="only available for sample-based"):
        _calibration_annotations("window", mask, sfreq, "CAL")
