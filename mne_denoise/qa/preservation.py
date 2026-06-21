"""Neural-preservation metrics (ERP/ERF amplitude, SME, effect size, etc.).

Used on the real-data arms to test that target neural activity survives cleaning.
Arrays are plain numpy so the functions are dependency-light and unit-testable;
MNE objects are accepted where a ``.get_data()`` exists.
"""

from __future__ import annotations

import numpy as np


def _data(x):
    return np.asarray(x.get_data() if hasattr(x, "get_data") else x, float)


def _window(times, tmin, tmax):
    times = np.asarray(times, float)
    return (times >= tmin) & (times <= tmax)


# ---------------------------------------------------------------------------
# ERP/ERF amplitude measures
# ---------------------------------------------------------------------------
def erp_mean_amplitude(evoked, times, tmin, tmax, *, picks=None) -> float:
    """Mean amplitude in ``[tmin, tmax]`` averaged over ``picks`` (ROI)."""
    data = _data(evoked)
    if picks is not None:
        data = data[np.asarray(picks, int)]
    w = _window(times, tmin, tmax)
    if not w.any():
        raise ValueError("empty time window")
    return float(np.mean(data[..., w]))


def erp_peak_latency(evoked, times, tmin, tmax, *, picks=None, mode="abs") -> float:
    data = _data(evoked)
    if picks is not None:
        data = data[np.asarray(picks, int)]
    roi = data.mean(0) if data.ndim > 1 else data
    times = np.asarray(times, float)
    w = _window(times, tmin, tmax)
    seg, tt = roi[w], times[w]
    idx = np.argmax(np.abs(seg)) if mode == "abs" else np.argmax(seg)
    return float(tt[idx])


def erp_area(evoked, times, tmin, tmax, *, picks=None) -> float:
    data = _data(evoked)
    if picks is not None:
        data = data[np.asarray(picks, int)]
    roi = data.mean(0) if data.ndim > 1 else data
    times = np.asarray(times, float)
    w = _window(times, tmin, tmax)
    return float(np.trapezoid(np.abs(roi[w]), times[w]))


# ---------------------------------------------------------------------------
# Reliability / measurement error
# ---------------------------------------------------------------------------
def analytic_sme(trial_amplitudes) -> float:
    """Analytic standardized measurement error = SD/sqrt(n) of a per-trial measure (Luck)."""
    a = np.asarray(trial_amplitudes, float).ravel()
    if a.size < 2:
        raise ValueError("need >= 2 trials for SME")
    return float(np.std(a, ddof=1) / np.sqrt(a.size))


def sme_mean_amplitude(epochs_data, times, tmin, tmax, *, picks=None) -> float:
    """aSME for the mean-amplitude measure from epoched data ``(n_trials, n_ch, n_times)``."""
    data = _data(epochs_data)
    if data.ndim != 3:
        raise ValueError("epochs_data must be (n_trials, n_ch, n_times)")
    if picks is not None:
        data = data[:, np.asarray(picks, int), :]
    w = _window(times, tmin, tmax)
    per_trial = data[..., w].mean(axis=(1, 2))
    return analytic_sme(per_trial)


def split_half_reliability(epochs_waveforms) -> float:
    """Spearman-Brown-corrected even/odd split-half correlation of trial waveforms."""
    data = _data(epochs_waveforms)
    if data.ndim == 3:
        data = data.mean(1)               # collapse channels → (n_trials, n_times)
    if data.ndim != 2 or data.shape[0] < 4:
        raise ValueError("need (n_trials, n_times) with >= 4 trials")
    a = data[0::2].mean(0); b = data[1::2].mean(0)
    a = a - a.mean(); b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    r = float(a @ b / d) if d else 0.0
    return float(2 * r / (1 + r)) if r > -1 else r


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------
def cohen_dz(x, y=None) -> float:
    """Paired Cohen's d_z = mean(diff) / sd(diff).  ``x`` may be diffs, or pass ``y``."""
    x = np.asarray(x, float).ravel()
    diff = x if y is None else x - np.asarray(y, float).ravel()
    if diff.size < 2:
        raise ValueError("need >= 2 paired observations")
    sd = np.std(diff, ddof=1)
    if sd == 0:
        raise ValueError("zero variance of differences")
    return float(np.mean(diff) / sd)


def hedges_g(x, y=None) -> float:
    """Small-sample-corrected paired effect size."""
    n = (np.asarray(x).ravel().size if y is None else np.asarray(x).ravel().size)
    dz = cohen_dz(x, y)
    j = 1.0 - 3.0 / (4.0 * (n - 1) - 1.0)   # Hedges correction
    return float(j * dz)


# ---------------------------------------------------------------------------
# Spectral / topographic preservation
# ---------------------------------------------------------------------------
def out_of_band_power(data, sfreq, *, exclude_band, nperseg=None) -> float:
    """Total PSD power OUTSIDE ``exclude_band=(lo, hi)`` (preservation proxy)."""
    from scipy.signal import welch

    d = _data(data)
    f, p = welch(d, fs=sfreq, nperseg=nperseg, axis=-1)
    p = p.mean(0) if p.ndim > 1 else p
    lo, hi = exclude_band
    keep = ~((f >= lo) & (f <= hi))
    return float(np.trapezoid(p[keep], f[keep]))


def topographic_similarity(a, b) -> float:
    """Pearson correlation between two topography vectors."""
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    if a.shape != b.shape:
        raise ValueError("topographies must share shape")
    a = a - a.mean(); b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d else 0.0


def erd_ers(power_active, power_baseline) -> float:
    """Event-related (de)synchronization = (active - baseline) / baseline."""
    pb = float(np.mean(power_baseline))
    if pb == 0:
        raise ValueError("baseline power is zero")
    return float((np.mean(power_active) - pb) / pb)


def combine_planar_rms(grad_pair_a, grad_pair_b) -> np.ndarray:
    """Combine a planar gradiometer pair by RMS (for MEG M170 outcomes)."""
    a = np.asarray(grad_pair_a, float); b = np.asarray(grad_pair_b, float)
    return np.sqrt(a ** 2 + b ** 2)
