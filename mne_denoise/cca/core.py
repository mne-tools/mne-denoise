"""Reference-free BSS-CCA (auto-CCA) for muscle/EMG artifact removal.

Canonical-Correlation-Analysis blind source separation (BSS-CCA), introduced by
De Clercq et al. (2006) for muscle-artifact removal, separates a multichannel
recording into components ordered by *autocorrelation* by running CCA between the
signal and a one-sample-lagged copy of itself. Slow, highly autocorrelated
components (neural rhythms) receive high canonical correlations; broadband,
weakly autocorrelated components (EMG/muscle) receive low ones. Dropping the
low-autocorrelation components and reconstructing removes muscle activity while
preserving neural structure. This is the reference-*free* counterpart to the
reference-aware :class:`~mne_denoise.icanclean.ICanClean`, and is the canonical
muscle comparator in the wearable-EEG literature.

The CCA solver itself is reused from :func:`mne_denoise.icanclean._cca.canonical_correlation`.

References
----------
.. [1] De Clercq, W., Vergult, A., Vanrumste, B., Van Paesschen, W., & Van Huffel, S.
       (2006). Canonical correlation analysis applied to remove muscle artifacts from
       the electroencephalogram. IEEE Trans. Biomed. Eng., 53(12), 2583-2587.
"""
from __future__ import annotations

import numpy as np

from mne_denoise.icanclean._cca import canonical_correlation


def _autocca_filter(X, rho_threshold=0.9, n_keep=None):
    """Learn the BSS-CCA cleaning matrix on ``X`` (n_channels, n_samples).

    Returns ``(C, n_kept, R)`` where ``C`` is an (n_ch, n_ch) left-multiplying
    cleaning matrix such that ``X_clean = C @ (X - mean) + mean`` keeps only the
    high-autocorrelation canonical components.
    """
    X = np.asarray(X, dtype=np.float64)
    n_ch = X.shape[0]
    Xs = X.T                                    # (n_samples, n_ch)
    Ys = np.roll(Xs, 1, axis=0)
    Ys[0] = Xs[0]                               # one-sample lag (no wrap-around leakage)
    A, _B, R, _U, _V = canonical_correlation(Xs, Ys)   # A (n_ch, d), R (d,) descending
    d = A.shape[1]
    if d == 0:
        return np.eye(n_ch), n_ch, R
    if n_keep is not None:                      # keep the top-k by autocorrelation
        keep = np.zeros(d, dtype=bool)
        keep[: max(1, int(n_keep))] = True
    else:                                       # keep high-autocorrelation (neural); drop low (EMG)
        keep = R >= float(rho_threshold)
        if not keep.any():
            keep[0] = True                      # never drop everything
    Ainv = np.linalg.pinv(A)                    # (d, n_ch)
    C = (A @ np.diag(keep.astype(np.float64)) @ Ainv).T   # left-multiply form
    return C, int(keep.sum()), R


class AutoCCA:
    """Reference-free BSS-CCA muscle/EMG remover (De Clercq 2006).

    ``fit`` learns the spatial cleaning matrix on TRAIN data (n_channels,
    n_samples) by CCA against a one-sample lag; ``transform`` applies it to new
    data, preserving the train/eval leakage barrier.

    Parameters
    ----------
    rho_threshold : float
        Keep canonical components whose autocorrelation (canonical correlation
        against the lagged signal) is at least this value; drop the rest as
        muscle. Ignored when ``n_keep`` is given.
    n_keep : int | None
        If set, keep exactly the top-``n_keep`` components by autocorrelation.
    """

    def __init__(self, rho_threshold: float = 0.9, n_keep: int | None = None) -> None:
        self.rho_threshold = float(rho_threshold)
        self.n_keep = n_keep

    def fit(self, X) -> "AutoCCA":
        self.C_, self.n_kept_, self.R_ = _autocca_filter(
            np.asarray(X), self.rho_threshold, self.n_keep
        )
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        m = X.mean(axis=1, keepdims=True)
        return self.C_ @ (X - m) + m

    def fit_transform(self, X):
        return self.fit(X).transform(X)
