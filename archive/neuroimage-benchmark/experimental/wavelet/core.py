"""Wavelet thresholding and wavelet-enhanced ICA (wICA) artifact removal.

``wavelet_denoise`` (per channel) soft-thresholds detail coefficients at the universal
threshold, removing high-frequency noise/muscle while preserving the smooth EEG. wICA
(Castellanos & Makarov 2006) wavelet-thresholds each ICA source to extract its sparse
high-amplitude transients (the artifact) and subtracts them in sensor space, recovering
neural activity that leaks into artifact components -- the single most-used artifact
method in the wearable-EEG literature for ocular and muscular contamination.

References
----------
.. [1] Castellanos, N. P., & Makarov, V. A. (2006). Recovering EEG brain signals: Artifact
       suppression with wavelet enhanced independent component analysis. J. Neurosci.
       Methods, 158(2), 300-312.
.. [2] Donoho, D. L., & Johnstone, I. M. (1994). Ideal spatial adaptation by wavelet
       shrinkage. Biometrika, 81(3), 425-455. (universal threshold)
"""
from __future__ import annotations

import numpy as np


def _max_level(n, wavelet):
    import pywt

    return max(1, pywt.dwt_max_level(int(n), pywt.Wavelet(wavelet).dec_len))


def _universal_threshold(finest_detail, n):
    sigma = np.median(np.abs(finest_detail)) / 0.6745 if np.size(finest_detail) else 0.0
    return float(sigma * np.sqrt(2.0 * np.log(max(int(n), 2))))


def wavelet_denoise(x, wavelet="sym5", level=None):
    """Per-channel wavelet denoising: soft-threshold the detail coefficients at the
    universal threshold (remove high-frequency noise / muscle, keep the smooth EEG)."""
    import pywt

    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 4:
        return x.copy()
    lvl = level or _max_level(n, wavelet)
    coeffs = pywt.wavedec(x, wavelet, level=lvl)
    thr = _universal_threshold(coeffs[-1], n)
    coeffs = [coeffs[0]] + [pywt.threshold(c, thr, mode="soft") for c in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet)[:n]


def wavelet_extract_artifact(s, wavelet="coif5", level=None):
    """Extract the sparse high-amplitude transient (artifact) from a 1-D source: hard-keep
    the coefficients above the universal threshold across all bands, zero the rest, and
    reconstruct. The neural background (small coefficients) is left in the source."""
    import pywt

    s = np.asarray(s, dtype=np.float64)
    n = s.size
    if n < 4:
        return np.zeros_like(s)
    lvl = level or _max_level(n, wavelet)
    coeffs = pywt.wavedec(s, wavelet, level=lvl)
    thr = _universal_threshold(coeffs[-1], n)
    art = [np.where(np.abs(c) > thr, c, 0.0) for c in coeffs]
    return pywt.waverec(art, wavelet)[:n]


def wavelet_denoise_multichannel(X, wavelet="sym5", level=None):
    X = np.asarray(X, dtype=np.float64)
    return np.stack([wavelet_denoise(X[c], wavelet, level) for c in range(X.shape[0])])


class WaveletICA:
    """wICA (Castellanos & Makarov 2006). FastICA decomposition (fit on TRAIN) + per-source
    wavelet artifact extraction subtracted in sensor space (applied to EVAL)."""

    def __init__(self, n_components=None, wavelet: str = "coif5",
                 random_state: int = 42, max_iter: int = 500) -> None:
        self.n_components = n_components
        self.wavelet = wavelet
        self.random_state = random_state
        self.max_iter = max_iter

    def fit(self, X) -> "WaveletICA":
        from sklearn.decomposition import FastICA

        X = np.asarray(X, dtype=np.float64)            # (n_channels, n_samples)
        k = min(self.n_components or X.shape[0], X.shape[0])
        self.ica_ = FastICA(n_components=k, random_state=self.random_state,
                            max_iter=self.max_iter, whiten="unit-variance")
        self.ica_.fit(X.T)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)            # (n_channels, n_samples)
        S = self.ica_.transform(X.T)                   # (n_samples, k)
        n = S.shape[0]
        art = np.column_stack([wavelet_extract_artifact(S[:, i], self.wavelet)[:n]
                               for i in range(S.shape[1])])
        X_art = art @ self.ica_.mixing_.T              # (n_samples, n_channels)
        return X - X_art.T

    def fit_transform(self, X):
        return self.fit(X).transform(X)
