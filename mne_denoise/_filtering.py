"""Butterworth filter-design helpers."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter


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
