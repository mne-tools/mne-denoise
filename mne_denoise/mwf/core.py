"""Multi-channel Wiener filter (MWF) for EEG artifact removal.

Somers, Francart & Bertrand (2018), "A generic EEG artifact removal algorithm based on the
multi-channel Wiener filter", J. Neural Eng. 15(3):036007 (DOI 10.1088/1741-2552/aaac92).
This is the spatial-filter core of the RELAX pipeline (Bailey et al. 2023).

Idea: with the recording split into artifact-present and artifact-free segments, the clean
signal is recovered by the multi-channel Wiener filter ``clean = R_clean @ R_artifact^{-1} @ X``,
where ``R_artifact`` is the covariance over artifact segments (signal + artifact) and
``R_clean`` the covariance over artifact-free segments (signal only). During clean segments the
two covariances coincide so the filter is ~identity (no over-cleaning); during artifact segments
it projects out the artifact subspace. Requires no reference channel: artifact segments are
marked by a high-frequency-power threshold (but a caller-supplied mask is honoured).
"""

import numpy as np


def hf_power_mask(X, sfreq, hf_hz=20.0, quantile=0.6, smooth_s=0.1):
    """Mark high-frequency-power time points as artifact (broadband muscle/transient).

    X: (n_channels, n_times). Returns a boolean (n_times,) mask, True where artifact.
    """
    from scipy.signal import butter, filtfilt

    ny = 0.5 * float(sfreq)
    b, a = butter(4, min(float(hf_hz) / ny, 0.99), btype="high")
    hf = filtfilt(b, a, np.asarray(X, dtype=float), axis=-1)
    env = (hf ** 2).mean(axis=0)                      # mean HF power across channels, per sample
    w = max(1, int(smooth_s * float(sfreq)))
    env = np.convolve(env, np.ones(w) / w, mode="same")
    return env > np.quantile(env, float(quantile))


def mwf_filter(X, mask, reg=1e-6):
    """MWF clean estimate. X: (M, T); mask: bool (T,) True = artifact. Returns cleaned (M, T)."""
    X = np.asarray(X, dtype=float)
    M = X.shape[0]
    Xa, Xc = X[:, mask], X[:, ~mask]
    if Xa.shape[1] < M + 1 or Xc.shape[1] < M + 1:    # too few samples to estimate covariances
        return X.copy()
    Ryy = np.cov(Xa)
    Rnn = np.cov(Xc)
    Ryy = Ryy + reg * (np.trace(Ryy) / M) * np.eye(M)  # diagonal loading for invertibility
    return Rnn @ np.linalg.solve(Ryy, X)               # R_clean @ R_artifact^{-1} @ X


class MWF:
    """Multi-channel Wiener filter artifact remover (Somers et al. 2018)."""

    def __init__(self, sfreq, hf_hz=20.0, quantile=0.6, reg=1e-6):
        self.sfreq = float(sfreq)
        self.hf_hz = float(hf_hz)
        self.quantile = float(quantile)
        self.reg = float(reg)

    def fit(self, X=None):
        return self

    def transform(self, X, mask=None):
        X = np.asarray(X, dtype=float)
        if mask is None:
            mask = hf_power_mask(X, self.sfreq, self.hf_hz, self.quantile)
        if mask.all() or (~mask).all():                # no contrast -> nothing to estimate
            return X.copy()
        return mwf_filter(X, mask, self.reg)

    def fit_transform(self, X, mask=None):
        return self.transform(X, mask)
