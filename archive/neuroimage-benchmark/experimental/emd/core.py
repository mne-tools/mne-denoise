"""Empirical Mode Decomposition (EMD/EEMD) artifact removal.

EMD adaptively decomposes a signal into Intrinsic Mode Functions (IMFs) ordered from
highest to lowest frequency. Broadband muscle/EMG energy concentrates in the first
(highest-frequency) IMFs; dropping the IMFs whose mean frequency exceeds a cutoff and
reconstructing from the remainder removes muscle while preserving lower-frequency neural
rhythms. EEMD is the noise-assisted ensemble variant (mitigates mode mixing at higher
compute cost). Common in the wearable-EEG literature for muscle and movement artifacts
(often combined with ASR, e.g. MEMD-ASR).

Note: PyEMD ships EMD/EEMD/CEEMDAN but not multivariate MEMD, so ``memd`` is not provided.

References
----------
.. [1] Huang, N. E., et al. (1998). The empirical mode decomposition and the Hilbert
       spectrum for nonlinear and non-stationary time series analysis. Proc. R. Soc. A.
"""
from __future__ import annotations

import numpy as np


def _imf_mean_freq(imf, sfreq):
    """Mean frequency of an IMF via zero-crossings."""
    n = imf.size
    if n < 2:
        return 0.0
    zc = int(np.count_nonzero(np.diff(np.signbit(imf))))
    return zc * float(sfreq) / (2.0 * n)


def emd_clean_channel(x, sfreq, method="emd", freq_cutoff=30.0, max_imf=8,
                      trials=20, noise_width=0.2):
    """Decompose ``x`` (1-D) into IMFs and drop those with mean frequency above
    ``freq_cutoff`` Hz (broadband muscle/EMG); reconstruct from the remainder."""
    x = np.asarray(x, dtype=np.float64)
    if x.size < 16:
        return x.copy()
    if method == "eemd":
        from PyEMD import EEMD

        e = EEMD(trials=int(trials), noise_width=float(noise_width))
        try:
            e.noise_seed(42)
        except Exception:  # noqa: BLE001
            pass
        imfs = e.eemd(x, max_imf=int(max_imf))
    else:
        from PyEMD import EMD

        imfs = EMD().emd(x, max_imf=int(max_imf))
    if imfs.ndim != 2 or imfs.shape[0] < 2:
        return x.copy()
    keep = np.array([_imf_mean_freq(imf, sfreq) <= freq_cutoff for imf in imfs])
    if not keep.any():
        keep[-1] = True                       # always retain the lowest-frequency residual
    return imfs[keep].sum(axis=0)


class EMDDenoiser:
    """Per-channel EMD/EEMD muscle remover. Data-adaptive / per-recording (no train transfer)."""

    def __init__(self, sfreq, method: str = "emd", freq_cutoff: float = 30.0,
                 max_imf: int = 8, trials: int = 20) -> None:
        self.sfreq = float(sfreq)
        self.method = method
        self.freq_cutoff = float(freq_cutoff)
        self.max_imf = int(max_imf)
        self.trials = int(trials)

    def fit(self, X=None) -> "EMDDenoiser":
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.stack([
            emd_clean_channel(X[c], self.sfreq, self.method, self.freq_cutoff,
                              self.max_imf, self.trials)
            for c in range(X.shape[0])
        ])

    def fit_transform(self, X):
        return self.transform(X)
