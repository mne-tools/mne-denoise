"""Tests for the shared filtering helpers (mne_denoise._filtering)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import butter, sosfiltfilt

from mne_denoise._filtering import design_butter_sos

SFREQ = 250.0


def _tone(freq: float, sfreq: float = SFREQ, n: int = 5000) -> np.ndarray:
    t = np.arange(n) / sfreq
    return np.sin(2 * np.pi * freq * t)


def _kept_ratio(
    filtered: np.ndarray, original: np.ndarray, sfreq: float = SFREQ
) -> float:
    """Ratio of filtered to original amplitude, past the filtfilt edge transient."""
    edge = int(sfreq)
    return np.std(filtered[edge:-edge]) / np.std(original[edge:-edge])


@pytest.mark.parametrize(
    "order, freqs, btype",
    [
        (4, 30.0, "lowpass"),
        (2, 30.0, "highpass"),
        (4, (8.0, 12.0), "bandpass"),
        (4, (8.0, 12.0), "bandstop"),
    ],
)
def test_design_butter_sos_matches_scipy_butter(order, freqs, btype):
    """design_butter_sos is a thin fs= wrapper -- must match butter() exactly."""
    got = design_butter_sos(order, freqs, btype, SFREQ)
    expected = butter(order, freqs, btype=btype, fs=SFREQ, output="sos")
    assert np.array_equal(got, expected)


def test_design_butter_sos_lowpass_passes_low_blocks_high():
    sos = design_butter_sos(4, 30.0, "lowpass", SFREQ)
    low, high = _tone(5.0), _tone(80.0)
    assert _kept_ratio(sosfiltfilt(sos, low), low) > 0.9
    assert _kept_ratio(sosfiltfilt(sos, high), high) < 0.05


def test_design_butter_sos_highpass_blocks_low_passes_high():
    sos = design_butter_sos(4, 30.0, "highpass", SFREQ)
    low, high = _tone(5.0), _tone(80.0)
    assert _kept_ratio(sosfiltfilt(sos, low), low) < 0.05
    assert _kept_ratio(sosfiltfilt(sos, high), high) > 0.9


def test_design_butter_sos_bandpass_keeps_band_blocks_outside():
    sos = design_butter_sos(4, (8.0, 12.0), "bandpass", SFREQ)
    inside, outside = _tone(10.0), _tone(50.0)
    assert _kept_ratio(sosfiltfilt(sos, inside), inside) > 0.9
    assert _kept_ratio(sosfiltfilt(sos, outside), outside) < 0.05


def test_design_butter_sos_bandstop_blocks_band_keeps_outside():
    sos = design_butter_sos(4, (8.0, 12.0), "bandstop", SFREQ)
    inside, outside = _tone(10.0), _tone(50.0)
    assert _kept_ratio(sosfiltfilt(sos, inside), inside) < 0.05
    assert _kept_ratio(sosfiltfilt(sos, outside), outside) > 0.9


def test_design_butter_sos_freqs_are_hz_not_prenormalized():
    """``freqs`` is Hz, normalized internally via ``fs`` -- not pre-divided by
    Nyquist by the caller. Design a fixed fraction of Nyquist at several
    sampling rates and confirm the passband tracks ``sfreq`` rather than
    being fixed to whatever rate the caller's arithmetic assumed.
    """
    for sfreq in (100.0, 250.0, 1000.0):
        target = sfreq / 10.0
        sos = design_butter_sos(4, target, "lowpass", sfreq)
        below, above = (
            _tone(target * 0.3, sfreq=sfreq),
            _tone(target * 3.0, sfreq=sfreq),
        )
        assert _kept_ratio(sosfiltfilt(sos, below), below, sfreq=sfreq) > 0.9
        assert _kept_ratio(sosfiltfilt(sos, above), above, sfreq=sfreq) < 0.05
