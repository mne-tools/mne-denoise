"""Numerical helpers shared by SSA tests."""

import numpy as np


def band_power(x, sfreq, fmin, fmax):
    """Return total power in a half-open frequency band."""
    x = np.atleast_2d(x)
    spectrum = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    frequencies = np.fft.rfftfreq(x.shape[-1], 1.0 / sfreq)
    band = (frequencies >= fmin) & (frequencies < fmax)
    return float(spectrum[:, band].sum())
