"""Shared Butterworth SOS filter design for internal package use.

This module centralizes the zero-phase Butterworth filter step used by
several denoisers so callers pass Hz-domain edges directly via
``fs=sfreq`` instead of each independently pre-dividing by Nyquist.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def design_butter_sos(
    order: int,
    freqs: float | tuple[float, float],
    btype: str,
    sfreq: float,
) -> np.ndarray:
    """Design a Butterworth filter as second-order sections.

    ``freqs`` is in Hz; normalization by Nyquist is handled internally via
    ``fs=sfreq`` rather than by the caller.
    """
    return butter(order, freqs, btype=btype, fs=sfreq, output="sos")


def _filter_channels(
    data: np.ndarray,
    filter_spec: tuple[str, float | tuple[float, float]] | None,
    sfreq: float,
) -> np.ndarray:
    """Filter along the last axis with a zero-phase 4th-order Butterworth."""
    if filter_spec is None:
        return data
    btype, freqs = filter_spec
    sos = design_butter_sos(4, freqs, btype, sfreq)
    return sosfiltfilt(sos, data, axis=-1)
