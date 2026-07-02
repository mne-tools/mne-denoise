"""Multi-channel Wiener filter (MWF) for EEG artifact removal.

The multi-channel Wiener filter (Somers, Francart & Bertrand 2018) [1]_ is a
generic, reference-free artifact remover and the spatial-filter core of the
RELAX pipeline (Bailey et al. 2023) [2]_. With the recording split into
artifact-present and artifact-free segments, the clean signal is recovered by

    X_clean = R_clean @ R_artifact^{-1} @ X

where ``R_artifact`` is the covariance over artifact segments (signal + artifact)
and ``R_clean`` the covariance over artifact-free segments (signal only). During
clean segments the two covariances coincide so the filter is ~identity (no
over-cleaning); during artifact segments it projects out the artifact subspace.
No reference channel is needed: artifact segments are marked by a
high-frequency-power threshold (a caller-supplied mask is honoured).

.. note::

   MWF is a *general* spatial cleaner rather than an artifact-specific method.
   Because its clean/artifact split is driven by broadband high-frequency power,
   it can attenuate genuine neural high-frequency activity when the two
   covariances are poorly separated; validate preservation of the band of
   interest on your data. In our M/EEG benchmark it was less selective than the
   reference-free CCA/SSA methods for muscle removal — it is provided here as the
   well-cited RELAX-core building block, not as a targeted muscle remover.

This module contains:

- ``hf_power_mask``: broadband high-frequency artifact-segment detector.
- ``mwf_filter`` / ``compute_mwf``: the array-based Wiener filter.
- ``MWF``: the scikit-learn estimator, compatible with MNE-Python objects or
  NumPy arrays.

References
----------
.. [1] Somers, B., Francart, T., & Bertrand, A. (2018). A generic EEG artifact
       removal algorithm based on the multi-channel Wiener filter. Journal of
       Neural Engineering, 15(3), 036007.
       https://doi.org/10.1088/1741-2552/aaac92
.. [2] Bailey, N. W., et al. (2023). RELAX: An automated pre-processing pipeline
       for cleaning EEG data - Part 1. Clinical Neurophysiology, 149, 178-201.
       https://doi.org/10.1016/j.clinph.2023.01.007
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ..utils import extract_data_from_mne, reconstruct_mne_object

logger = logging.getLogger(__name__)


def hf_power_mask(
    X: np.ndarray,
    sfreq: float,
    hf_hz: float = 20.0,
    quantile: float = 0.6,
    smooth_s: float = 0.1,
) -> np.ndarray:
    """Mark high-frequency-power time points as artifact (broadband muscle/transient).

    Parameters
    ----------
    X : ndarray, shape (n_channels, n_times)
        Multichannel signal.
    sfreq : float
        Sampling frequency in Hz.
    hf_hz : float
        High-pass cutoff (Hz) used to isolate broadband high-frequency power.
    quantile : float
        Samples whose smoothed HF power exceeds this quantile are flagged.
    smooth_s : float
        Moving-average smoothing window (seconds) applied to the HF envelope.

    Returns
    -------
    mask : ndarray of bool, shape (n_times,)
        ``True`` where the sample is flagged as artifact.
    """
    from scipy.signal import butter, filtfilt

    ny = 0.5 * float(sfreq)
    b, a = butter(4, min(float(hf_hz) / ny, 0.99), btype="high")
    hf = filtfilt(b, a, np.asarray(X, dtype=float), axis=-1)
    env = (hf**2).mean(axis=0)  # mean HF power across channels, per sample
    # Cap the smoothing window at the signal length: np.convolve(mode="same")
    # returns length max(len(env), w), so a window longer than the signal would
    # otherwise yield a mask longer than n_times (breaks short recordings/epochs).
    w = max(1, min(int(smooth_s * float(sfreq)), env.size))
    env = np.convolve(env, np.ones(w) / w, mode="same")
    return env > np.quantile(env, float(quantile))


def mwf_filter(X: np.ndarray, mask: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    """Multi-channel Wiener filter clean estimate.

    Parameters
    ----------
    X : ndarray, shape (n_channels, n_times)
        Multichannel signal.
    mask : ndarray of bool, shape (n_times,)
        ``True`` = artifact segment, ``False`` = clean segment.
    reg : float
        Diagonal-loading factor for covariance invertibility.

    Returns
    -------
    cleaned : ndarray, shape (n_channels, n_times)
        Wiener-filtered signal.
    """
    X = np.asarray(X, dtype=float)
    mask = np.asarray(mask, dtype=bool)  # tolerate int 0/1 masks (avoid ~int bitwise NOT)
    M = X.shape[0]
    Xa, Xc = X[:, mask], X[:, ~mask]
    if M < 2 or Xa.shape[1] < M + 1 or Xc.shape[1] < M + 1:
        # spatial filter is degenerate for 1 channel, or too few samples in one
        # segment to estimate covariances
        return X.copy()
    Ryy = np.cov(Xa)
    Rnn = np.cov(Xc)
    Ryy = Ryy + reg * (np.trace(Ryy) / M) * np.eye(M)  # diagonal loading
    return Rnn @ np.linalg.solve(Ryy, X)  # R_clean @ R_artifact^{-1} @ X


def compute_mwf(
    X: np.ndarray,
    sfreq: float,
    hf_hz: float = 20.0,
    quantile: float = 0.6,
    reg: float = 1e-6,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Multi-channel Wiener filter cleaning of a data array.

    Parameters
    ----------
    X : ndarray, shape (n_channels, n_times)
        Multichannel signal.
    sfreq : float
        Sampling frequency in Hz (used only when ``mask`` is None).
    hf_hz, quantile
        Passed to :func:`hf_power_mask` when ``mask`` is None.
    reg : float
        Diagonal-loading factor for covariance invertibility.
    mask : ndarray of bool | None
        Optional artifact-segment mask. If None, it is estimated from broadband
        high-frequency power.

    Returns
    -------
    X_clean : ndarray, shape (n_channels, n_times)
        Cleaned signal.
    info : dict
        Diagnostics: ``mask`` (the artifact mask used) and
        ``artifact_fraction`` (fraction of samples flagged).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(
            f"Expected a 2-D (n_channels, n_samples) array, got shape {X.shape}."
        )
    if mask is None:
        mask = hf_power_mask(X, sfreq, hf_hz, quantile)
    mask = np.asarray(mask, dtype=bool)  # tolerate int 0/1 masks
    if mask.all() or (~mask).all():  # no contrast -> nothing to estimate
        cleaned = X.copy()
    else:
        cleaned = mwf_filter(X, mask, reg)
    info = {"mask": mask, "artifact_fraction": float(np.mean(mask))}
    return cleaned, info


class MWF(BaseEstimator, TransformerMixin):
    """Multi-channel Wiener filter artifact remover (Somers et al. 2018).

    A generic, reference-free spatial cleaner (the RELAX-pipeline core). ``fit``
    estimates the artifact/clean covariances and the resulting Wiener operator on
    the training data; ``transform`` applies that fixed operator to new data
    (leakage-safe). Artifact segments are found from broadband high-frequency
    power unless a mask is supplied. Accepts MNE ``Raw``/``Epochs`` objects or
    NumPy ``(n_channels, n_samples)`` arrays; for MNE objects the sampling
    frequency is read from ``info`` when ``sfreq`` is not given.

    .. note::

       MWF is a general cleaner, not an artifact-specific method: its HF-power
       segment split can attenuate genuine neural high-frequency activity when
       the clean/artifact covariances are poorly separated. Validate preservation
       of the band of interest on your data.

    Parameters
    ----------
    sfreq : float | None
        Sampling frequency in Hz. May be omitted for MNE input (read from
        ``info['sfreq']``); required for NumPy-array input when no ``mask`` is
        passed to ``transform``.
    hf_hz : float
        High-pass cutoff (Hz) for the high-frequency artifact detector.
    quantile : float
        HF-power quantile above which samples are flagged as artifact.
    reg : float
        Diagonal-loading factor for covariance invertibility.
    verbose : bool | str | int | None
        Control logging verbosity (MNE-style).

    Attributes
    ----------
    spatial_filter_ : ndarray, shape (n_channels, n_channels)
        The learned Wiener operator ``R_clean @ R_artifact^{-1}``.
    artifact_mask_ : ndarray of bool, shape (n_train_times,)
        Artifact-segment mask used to fit the operator.
    artifact_fraction_ : float
        Fraction of training samples flagged as artifact.

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.mwf import MWF
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((16, 4000))
    >>> cleaned = MWF(sfreq=250.0).fit_transform(X)
    >>> cleaned.shape
    (16, 4000)
    """

    def __init__(
        self,
        sfreq: float | None = None,
        hf_hz: float = 20.0,
        quantile: float = 0.6,
        reg: float = 1e-6,
        verbose: bool | str | int | None = None,
    ) -> None:
        self.sfreq = sfreq
        self.hf_hz = hf_hz
        self.quantile = quantile
        self.reg = reg
        self.verbose = verbose

    def _resolve_sfreq(self, sfreq_data: float | None) -> float | None:
        return sfreq_data if sfreq_data is not None else self.sfreq

    def _to_2d(self, X: Any) -> tuple[np.ndarray, float | None, str, Any, Any]:
        data, sfreq_data, mne_type, orig_inst, picks, _names = extract_data_from_mne(
            X, auto_pick=True
        )
        sfreq = self._resolve_sfreq(sfreq_data)
        if mne_type == "epochs":
            n_ep, n_ch, n_t = data.shape
            data2d = np.transpose(data, (1, 0, 2)).reshape(n_ch, n_ep * n_t)
        else:
            data2d = np.asarray(data, dtype=float)
        return data2d, sfreq, mne_type, orig_inst, picks

    def fit(self, X: Any, y=None, mask: np.ndarray | None = None) -> "MWF":
        """Estimate the Wiener operator on the training data.

        Parameters
        ----------
        X : Raw | Epochs | ndarray
            Training data. Epochs are concatenated along time.
        y : None
            Ignored.
        mask : ndarray of bool | None
            Optional artifact-segment mask over the (concatenated) training
            samples. If None, it is estimated from broadband HF power.

        Returns
        -------
        self : MWF
        """
        data2d, sfreq, _mne_type, _orig, _picks = self._to_2d(X)
        M = data2d.shape[0]
        if mask is None:
            if sfreq is None:
                raise ValueError(
                    "sfreq is required to estimate the artifact mask for array "
                    "input (pass MWF(sfreq=...)) or supply an explicit mask, or "
                    "fit on an MNE object carrying a sampling frequency."
                )
            mask = hf_power_mask(data2d, sfreq, self.hf_hz, self.quantile)
        mask = np.asarray(mask, dtype=bool)  # tolerate int 0/1 masks
        self.artifact_mask_ = mask
        self.artifact_fraction_ = float(np.mean(mask))

        Xa, Xc = data2d[:, mask], data2d[:, ~mask]
        if (
            M < 2
            or mask.all()
            or (~mask).all()
            or Xa.shape[1] < M + 1
            or Xc.shape[1] < M + 1
        ):
            # 1 channel is degenerate for a spatial filter, or insufficient
            # contrast -> identity operator (no cleaning)
            self.spatial_filter_ = np.eye(M)
        else:
            Ryy = np.cov(Xa)
            Rnn = np.cov(Xc)
            Ryy = Ryy + self.reg * (np.trace(Ryy) / M) * np.eye(M)
            self.spatial_filter_ = Rnn @ np.linalg.inv(Ryy)
        if self.verbose:
            logger.info(
                "MWF: fit on %.1f%% artifact samples.", 100.0 * self.artifact_fraction_
            )
        return self

    def transform(self, X: Any, y=None) -> Any:
        """Apply the learned Wiener operator.

        Parameters
        ----------
        X : Raw | Epochs | ndarray
            Data to clean (same channel layout as the fitted data).
        y : None
            Ignored.

        Returns
        -------
        X_clean : Raw | Epochs | ndarray
            Cleaned data in the same format as the input.
        """
        check_is_fitted(self, "spatial_filter_")
        data, _sfreq, mne_type, orig_inst, picks, _names = extract_data_from_mne(
            X, auto_pick=True
        )
        if mne_type == "epochs":
            cleaned = np.empty_like(data, dtype=float)
            for e in range(data.shape[0]):
                cleaned[e] = self.spatial_filter_ @ np.asarray(data[e], dtype=float)
        else:
            cleaned = self.spatial_filter_ @ np.asarray(data, dtype=float)
        return reconstruct_mne_object(
            cleaned, orig_inst, mne_type, picks=picks, verbose=False
        )

    def fit_transform(self, X: Any, y=None, mask: np.ndarray | None = None, **fit_params) -> Any:
        """Fit on ``X`` and apply to ``X`` in one step.

        Parameters
        ----------
        X : Raw | Epochs | ndarray
            Input data.
        y : None
            Ignored.
        mask : ndarray of bool | None
            Optional artifact-segment mask (see :meth:`fit`).
        **fit_params
            Ignored.

        Returns
        -------
        X_clean : Raw | Epochs | ndarray
            Cleaned data.
        """
        return self.fit(X, y, mask=mask).transform(X)

