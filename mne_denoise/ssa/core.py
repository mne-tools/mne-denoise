"""Singular Spectrum Analysis (SSA) for slow-drift / quasi-periodic artifact removal.

Singular Spectrum Analysis (Golyandina & Zhigljavsky 2013) [1]_ embeds each
channel into a trajectory (Hankel) matrix, decomposes it by SVD into
eigentriples, and reconstructs the series from a chosen subset by diagonal
averaging. Grouping eigentriples by the **dominant frequency** of their
reconstructed component lets us drop slow ocular/drift structure (and, with an
explicit band, quasi-periodic cardiac structure) while preserving oscillatory
neural activity. SSA is widely used across the wearable-EEG literature for
ocular, cardiac (often as SSA+ICA), and instrumental artifacts, and has no
maintained EEG-tuned Python implementation.

This module contains:

- ``ssa_clean_channel``: SSA cleaning of a single 1-D channel.
- ``compute_ssa``: SSA cleaning of a multichannel ``(n_channels, n_samples)``
  array, plus a diagnostics dict.
- ``SSA``: the scikit-learn estimator, compatible with MNE-Python objects or
  NumPy arrays.

SSA is per-recording and unsupervised: there is no learned operator transferred
from ``fit`` to ``transform`` (``fit`` is a no-op), so it does not require a
train/evaluation split.

References
----------
.. [1] Golyandina, N., & Zhigljavsky, A. (2013). Singular Spectrum Analysis for
       Time Series. Springer. https://doi.org/10.1007/978-3-642-34913-3
.. [2] Teixeira, A. R., et al. (2006). On the use of clustering and local
       singular spectrum analysis to remove ocular artifacts from EEG. IJCNN.
       https://doi.org/10.1109/IJCNN.2006.246999
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from ..utils import extract_data_from_mne, reconstruct_mne_object

logger = logging.getLogger(__name__)


def _hankelize(M: np.ndarray) -> np.ndarray:
    """Diagonal-average an (L, K) matrix back to a length-(L+K-1) series."""
    L, K = M.shape
    out = np.zeros(L + K - 1)
    cnt = np.zeros(L + K - 1)
    for i in range(L):
        out[i : i + K] += M[i]
        cnt[i : i + K] += 1.0
    return out / cnt


def _clean_channel(
    x: np.ndarray,
    sfreq: float,
    window_length: int | None = None,
    drop_freq_max: float = 3.0,
    drop_band: tuple[float, float] | None = None,
    n_check: int = 20,
    max_window: int = 100,
) -> tuple[np.ndarray, list[float]]:
    """SSA-clean a 1-D series; return ``(cleaned, dropped_dominant_freqs)``."""
    x = np.asarray(x, dtype=np.float64)
    N = x.size
    if N < 8:
        return x.copy(), []
    L = int(window_length or min(max(int(sfreq * 0.5), 20), max_window, N // 2))
    L = max(2, min(L, N - 1))
    K = N - L + 1
    X = np.lib.stride_tricks.sliding_window_view(x, K)  # (L, K) trajectory matrix
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    freqs = np.fft.rfftfreq(N, 1.0 / sfreq)
    drop = np.zeros(N)
    dropped_freqs: list[float] = []
    for i in range(min(n_check, S.size)):
        if S[i] <= 0:
            continue
        comp = _hankelize(S[i] * np.outer(U[:, i], Vt[i]))
        sp = np.abs(np.fft.rfft(comp))
        fdom = float(freqs[1:][np.argmax(sp[1:])]) if sp.size > 1 else 0.0
        is_art = (
            (fdom <= drop_freq_max)
            if drop_band is None
            else (drop_band[0] <= fdom <= drop_band[1])
        )
        if is_art:
            drop += comp
            dropped_freqs.append(fdom)
    return x - drop, dropped_freqs


def ssa_clean_channel(
    x: np.ndarray,
    sfreq: float,
    window_length: int | None = None,
    drop_freq_max: float = 3.0,
    drop_band: tuple[float, float] | None = None,
    n_check: int = 20,
    max_window: int = 100,
) -> np.ndarray:
    """Remove slow/quasi-periodic artifact components from a 1-D series via SSA.

    Drops eigentriples (among the top ``n_check`` by variance) whose
    reconstructed component has a dominant frequency at or below
    ``drop_freq_max`` Hz (slow drift / ocular), or within ``drop_band`` if given
    (e.g. a cardiac heart-rate band), and returns the original minus the dropped
    components.

    Parameters
    ----------
    x : ndarray, shape (n_times,)
        Single-channel signal.
    sfreq : float
        Sampling frequency in Hz.
    window_length : int | None
        SSA embedding window length. Defaults to ``min(sfreq/2, max_window)``.
    drop_freq_max : float
        Drop components with dominant frequency ≤ this value (Hz).
    drop_band : tuple of float | None
        If given, drop components whose dominant frequency falls in this band
        instead of using ``drop_freq_max``.
    n_check : int
        Number of top eigentriples (by variance) to examine.
    max_window : int
        Upper bound on the embedding window length.

    Returns
    -------
    cleaned : ndarray, shape (n_times,)
        SSA-cleaned signal.
    """
    cleaned, _ = _clean_channel(
        x, sfreq, window_length, drop_freq_max, drop_band, n_check, max_window
    )
    return cleaned


def compute_ssa(
    X: np.ndarray,
    sfreq: float,
    window_length: int | None = None,
    drop_freq_max: float = 3.0,
    drop_band: tuple[float, float] | None = None,
    n_check: int = 20,
    max_window: int = 100,
) -> tuple[np.ndarray, dict[str, Any]]:
    """SSA-clean a multichannel array channel-by-channel.

    Parameters
    ----------
    X : ndarray, shape (n_channels, n_times)
        Multichannel signal.
    sfreq : float
        Sampling frequency in Hz.
    window_length, drop_freq_max, drop_band, n_check, max_window
        See :func:`ssa_clean_channel`.

    Returns
    -------
    X_clean : ndarray, shape (n_channels, n_times)
        Cleaned signal.
    info : dict
        Diagnostics: ``dropped_counts`` (per-channel number of dropped
        eigentriples) and ``dropped_freqs`` (per-channel list of dropped
        dominant frequencies).
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(
            f"Expected a 2-D (n_channels, n_samples) array, got shape {X.shape}."
        )
    out = np.empty_like(X)
    dropped_counts = np.zeros(X.shape[0], dtype=int)
    dropped_freqs: list[list[float]] = []
    for c in range(X.shape[0]):
        out[c], freqs = _clean_channel(
            X[c], sfreq, window_length, drop_freq_max, drop_band, n_check, max_window
        )
        dropped_counts[c] = len(freqs)
        dropped_freqs.append(freqs)
    info = {"dropped_counts": dropped_counts, "dropped_freqs": dropped_freqs}
    return out, info


class SSA(BaseEstimator, TransformerMixin):
    """Per-channel SSA artifact remover (Golyandina & Zhigljavsky 2013).

    Decomposes each channel with Singular Spectrum Analysis, drops eigentriples
    whose dominant frequency falls in the artifact band (slow drift / ocular by
    default, or a caller-supplied ``drop_band`` for e.g. cardiac), and
    reconstructs. Per-recording and unsupervised: ``fit`` is a no-op and does not
    learn a transferable operator. Accepts MNE ``Raw``/``Epochs`` objects or
    NumPy ``(n_channels, n_samples)`` arrays; for MNE objects the sampling
    frequency is read from ``info`` when ``sfreq`` is not given.

    Parameters
    ----------
    sfreq : float | None
        Sampling frequency in Hz. May be omitted (``None``) when transforming an
        MNE object, in which case it is read from ``info['sfreq']``; it is
        required for NumPy-array input.
    window_length : int | None
        SSA embedding window length. Defaults to ``min(sfreq/2, max_window)``.
    drop_freq_max : float
        Drop components with dominant frequency ≤ this value (Hz).
    drop_band : tuple of float | None
        If given, drop components whose dominant frequency falls in this band.
    n_check : int
        Number of top eigentriples (by variance) to examine per channel.
    max_window : int
        Upper bound on the embedding window length.
    verbose : bool | str | int | None
        Control logging verbosity (MNE-style).

    Attributes
    ----------
    dropped_counts_ : ndarray, shape (n_channels,)
        Number of eigentriples dropped per channel (populated after
        ``transform``; for Epochs, from the last epoch processed).
    n_channels_ : int
        Number of channels processed.

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.ssa import SSA
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((8, 2000))
    >>> cleaned = SSA(sfreq=250.0, drop_freq_max=3.0).fit_transform(X)
    >>> cleaned.shape
    (8, 2000)
    """

    def __init__(
        self,
        sfreq: float | None = None,
        window_length: int | None = None,
        drop_freq_max: float = 3.0,
        drop_band: tuple[float, float] | None = None,
        n_check: int = 20,
        max_window: int = 100,
        verbose: bool | str | int | None = None,
    ) -> None:
        self.sfreq = sfreq
        self.window_length = window_length
        self.drop_freq_max = drop_freq_max
        self.drop_band = drop_band
        self.n_check = n_check
        self.max_window = max_window
        self.verbose = verbose

    def fit(self, X: Any = None, y=None) -> "SSA":
        """No-op; SSA is per-recording and learns no transferable operator.

        Included for scikit-learn compatibility. The decomposition happens in
        :meth:`transform`.
        """
        return self

    def _resolve_sfreq(self, sfreq_data: float | None) -> float:
        sfreq = sfreq_data if sfreq_data is not None else self.sfreq
        if sfreq is None:
            raise ValueError(
                "sfreq is required for array input (pass SSA(sfreq=...)) or "
                "transform an MNE object carrying a sampling frequency."
            )
        return float(sfreq)

    def _clean_2d(self, data2d: np.ndarray, sfreq: float) -> np.ndarray:
        cleaned, info = compute_ssa(
            data2d,
            sfreq,
            self.window_length,
            self.drop_freq_max,
            self.drop_band,
            self.n_check,
            self.max_window,
        )
        self.dropped_counts_ = info["dropped_counts"]
        self.n_channels_ = data2d.shape[0]
        return cleaned

    def transform(self, X: Any, y=None) -> Any:
        """Apply SSA artifact removal.

        Parameters
        ----------
        X : Raw | Epochs | ndarray
            Data to clean.
        y : None
            Ignored.

        Returns
        -------
        X_clean : Raw | Epochs | ndarray
            Cleaned data in the same format as the input.
        """
        data, sfreq_data, mne_type, orig_inst, picks, _names = extract_data_from_mne(
            X, auto_pick=True
        )
        sfreq = self._resolve_sfreq(sfreq_data)
        if mne_type == "epochs":
            cleaned = np.empty_like(data, dtype=np.float64)
            for e in range(data.shape[0]):
                cleaned[e] = self._clean_2d(data[e], sfreq)
        else:
            cleaned = self._clean_2d(np.asarray(data, dtype=np.float64), sfreq)
        if self.verbose:
            logger.info(
                "SSA: dropped a mean of %.1f components/channel.",
                float(np.mean(self.dropped_counts_)),
            )
        return reconstruct_mne_object(
            cleaned, orig_inst, mne_type, picks=picks, verbose=False
        )

    def fit_transform(self, X: Any, y=None, **fit_params) -> Any:
        """Apply SSA (``fit`` is a no-op)."""
        return self.fit(X, y).transform(X)
