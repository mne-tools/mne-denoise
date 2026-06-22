"""Reference-coupling and artifact-residual metrics.

For the muscle/ocular/cardiac arms: quantify how much cleaned EEG still couples to
an artifact reference (EOG/ECG/EMG/noise-layer).  **Important (muscle track):**
reduced EEG–EMG coupling is not automatically good — distinguish high-frequency
zero-lag contamination from genuine corticomuscular coupling (see
``docs/benchmarks/BENCHMARK_PLAN.md`` and the muscle config).
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid  # stable across numpy 1.x/2.x


def _data(x):
    return np.asarray(x.get_data() if hasattr(x, "get_data") else x, float)


def regress_out(data, reference):
    """Return ``data`` with the least-squares projection onto ``reference`` removed."""
    D = np.atleast_2d(_data(data))            # (n_ch, n_times)
    R = np.atleast_2d(_data(reference))       # (n_ref, n_times)
    # solve D ≈ B @ R  → residual = D - B @ R
    B, *_ = np.linalg.lstsq(R.T, D.T, rcond=None)
    resid = D - (R.T @ B).T
    return resid


def max_abs_correlation(data, reference) -> float:
    """Largest absolute Pearson correlation between any data and reference channel."""
    D = np.atleast_2d(_data(data)); R = np.atleast_2d(_data(reference))
    Dc = D - D.mean(1, keepdims=True); Rc = R - R.mean(1, keepdims=True)
    Dn = Dc / (np.linalg.norm(Dc, axis=1, keepdims=True) + 1e-12)
    Rn = Rc / (np.linalg.norm(Rc, axis=1, keepdims=True) + 1e-12)
    return float(np.max(np.abs(Dn @ Rn.T)))


def canonical_correlation(data, reference) -> float:
    """First canonical correlation between data and reference channel sets."""
    D = np.atleast_2d(_data(data)).T          # (n_times, n_ch)
    R = np.atleast_2d(_data(reference)).T      # (n_times, n_ref)
    D = D - D.mean(0); R = R - R.mean(0)
    Qd, _ = np.linalg.qr(D); Qr, _ = np.linalg.qr(R)
    s = np.linalg.svd(Qd.T @ Qr, compute_uv=False)
    return float(np.clip(s[0], 0.0, 1.0)) if s.size else 0.0


def reference_coupling(data, reference, *, method: str = "max_abs") -> float:
    """Coupling between cleaned data and an artifact reference (lower = less coupled)."""
    if method == "max_abs":
        return max_abs_correlation(data, reference)
    if method == "cca":
        return canonical_correlation(data, reference)
    raise ValueError(f"unknown method {method!r}")


def event_locked_residual(data, events, sfreq, *, tmin, tmax, picks=None) -> float:
    """Mean absolute amplitude in ``[tmin, tmax]`` around event samples (blink/QRS)."""
    D = np.atleast_2d(_data(data))
    if picks is not None:
        D = D[np.asarray(picks, int)]
    lo = int(round(tmin * sfreq)); hi = int(round(tmax * sfreq))
    segs = []
    for e in np.asarray(events, int).ravel():
        a, b = e + lo, e + hi
        if a >= 0 and b <= D.shape[1]:
            segs.append(D[:, a:b])
    if not segs:
        raise ValueError("no valid event-locked segments")
    return float(np.mean(np.abs(np.mean(segs, axis=0))))


def hf_band_power(data, sfreq, *, fmin=20.0, fmax=100.0, nperseg=None) -> float:
    """Mean PSD power in a high-frequency (EMG) band ``[fmin, fmax]``."""
    from scipy.signal import welch

    D = _data(data)
    fmax = min(fmax, sfreq / 2.0 - 1e-6)
    f, p = welch(D, fs=sfreq, nperseg=nperseg, axis=-1)
    p = p.mean(0) if p.ndim > 1 else p
    band = (f >= fmin) & (f <= fmax)
    if not band.any():
        raise ValueError("empty HF band for given sfreq")
    return float(trapezoid(p[band], f[band]))
