"""Filter helpers shared across methods."""

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


def line_band_mask(
    fft_freqs: np.ndarray, freqs: float | np.ndarray, half_width: float
) -> np.ndarray:
    """Return the FFT bins within ``half_width`` Hz of any of ``freqs``.

    A frequency whose band is narrower than the frequency resolution, and so
    holds no bin, gets its nearest bin instead.
    """
    freqs = np.atleast_1d(freqs)
    bins = fft_freqs[:, np.newaxis]
    band = (bins >= freqs - half_width) & (bins <= freqs + half_width)
    for column in np.flatnonzero(~band.any(axis=0)):
        band[np.argmin(np.abs(fft_freqs - freqs[column])), column] = True
    return band.any(axis=1)
