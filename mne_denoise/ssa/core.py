"""Singular Spectrum Analysis (SSA) for slow-drift / quasi-periodic artifact removal.

SSA embeds each channel into a trajectory (Hankel) matrix, decomposes it by SVD into
eigentriples, and reconstructs the series from a chosen subset by diagonal averaging.
Grouping eigentriples by the dominant frequency of their reconstructed component lets us
drop slow ocular/drift structure (and, with an explicit band, quasi-periodic cardiac
structure) while preserving oscillatory neural activity. Used across the wearable-EEG
literature for ocular, cardiac (often as SSA+ICA), and instrumental artifacts.

References
----------
.. [1] Golyandina, N., & Zhigljavsky, A. (2013). Singular Spectrum Analysis for Time
       Series. Springer.
"""
from __future__ import annotations

import numpy as np


def _hankelize(M):
    """Diagonal-average an (L, K) matrix back to a length-(L+K-1) series."""
    L, K = M.shape
    out = np.zeros(L + K - 1)
    cnt = np.zeros(L + K - 1)
    for i in range(L):
        out[i : i + K] += M[i]
        cnt[i : i + K] += 1.0
    return out / cnt


def ssa_clean_channel(x, sfreq, window_length=None, drop_freq_max=3.0,
                      drop_band=None, n_check=20, max_window=100):
    """Remove slow/quasi-periodic artifact components from a 1-D series via SSA.

    Drops eigentriples (among the top ``n_check`` by variance) whose reconstructed
    component has a dominant frequency at or below ``drop_freq_max`` Hz (slow drift /
    ocular), or within ``drop_band`` if given (e.g. a cardiac heart-rate band). Returns
    the original minus the dropped components (SSA reconstructions sum to the signal).
    """
    x = np.asarray(x, dtype=np.float64)
    N = x.size
    if N < 8:
        return x.copy()
    L = int(window_length or min(max(int(sfreq * 0.5), 20), max_window, N // 2))
    L = max(2, min(L, N - 1))
    K = N - L + 1
    X = np.lib.stride_tricks.sliding_window_view(x, K)        # (L, K) trajectory matrix
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    freqs = np.fft.rfftfreq(N, 1.0 / sfreq)
    drop = np.zeros(N)
    for i in range(min(n_check, S.size)):
        if S[i] <= 0:
            continue
        comp = _hankelize(S[i] * np.outer(U[:, i], Vt[i]))
        sp = np.abs(np.fft.rfft(comp))
        fdom = float(freqs[1:][np.argmax(sp[1:])]) if sp.size > 1 else 0.0
        is_art = (fdom <= drop_freq_max) if drop_band is None else (drop_band[0] <= fdom <= drop_band[1])
        if is_art:
            drop += comp
    return x - drop


class SSA:
    """Per-channel SSA artifact remover (Golyandina). ``transform`` decomposes each
    channel, drops eigentriples whose dominant frequency falls in the artifact band, and
    reconstructs. Per-recording / unsupervised (no train-to-eval transfer)."""

    def __init__(self, sfreq, window_length=None, drop_freq_max: float = 3.0,
                 drop_band=None, n_check: int = 20) -> None:
        self.sfreq = float(sfreq)
        self.window_length = window_length
        self.drop_freq_max = float(drop_freq_max)
        self.drop_band = drop_band
        self.n_check = int(n_check)

    def fit(self, X=None) -> "SSA":
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        out = np.empty_like(X)
        for c in range(X.shape[0]):
            out[c] = ssa_clean_channel(X[c], self.sfreq, self.window_length,
                                       self.drop_freq_max, self.drop_band, self.n_check)
        return out

    def fit_transform(self, X):
        return self.transform(X)
