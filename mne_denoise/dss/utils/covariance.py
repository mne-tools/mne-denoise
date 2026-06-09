"""Robust covariance estimation.

Provides methods for computing covariance matrices robust to outliers
or low sample counts (shrinkage).

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np


def compute_covariance(
    data: np.ndarray,
    *,
    method: str = "empirical",
    shrinkage: float | None = None,
    weights: np.ndarray | None = None,
    assume_centered: bool = False,
) -> np.ndarray:
    """Compute covariance matrix.

    This function provides a unified interface for covariance estimation,
    supporting both standard robust methods (shrinkage, OAS, MCD) and
    weighted empirical covariance.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Input data.
    method : str
        Method for covariance estimation:
        - 'empirical': Standard empirical covariance (weighted if `weights` provided).
        - 'shrinkage': Ledoit-Wolf shrinkage (unweighted).
        - 'oas': Oracle Approximating Shrinkage (unweighted).
        - 'mcd': Minimum Covariance Determinant (unweighted).
        Default is 'empirical'.
    shrinkage : float, optional
        Shrinkage parameter (0 to 1) for 'shrinkage' method. If None,
        optimal shrinkage is estimated.
    weights : ndarray, shape (n_times,), optional
        Sample weights for covariance computation. High weights emphasize time points,
        zero weights ignore them. Currently only supported for `method='empirical'`.
    assume_centered : bool, default=False
        If True, treat ``data`` as already centered and skip mean subtraction.

    Returns
    -------
    cov : ndarray, shape (n_channels, n_channels)
        The estimated covariance matrix.
    """
    if data.ndim == 3:
        n_channels, n_times_in, n_epochs = data.shape
        data = data.reshape(n_channels, -1)

        if weights is not None and weights.shape[0] == n_times_in:
            # Tile weights across epochs
            weights = np.tile(weights, n_epochs)

    n_channels, n_times = data.shape

    if weights is not None:
        if data.shape[1] != weights.shape[0]:
            raise ValueError(
                f"Weights length {weights.shape[0]} does not match "
                f"data samples {data.shape[1]}"
            )
        total_weight = np.sum(weights)
        if total_weight == 0:
            raise ValueError("Sum of weights is zero")

        if method != "empirical":
            # Currently we only support weighted empirical.
            raise ValueError(
                f"Weighted covariance not implemented for method '{method}'"
            )
    else:
        # If no weights are provided, use equal weights; to simplify the implementation
        weights = np.ones(n_times)
        total_weight = n_times

    if assume_centered:
        data_centered = data
    else:
        mean = np.sum(data * weights, axis=1, keepdims=True) / total_weight
        data_centered = data - mean

    if method == "empirical":
        # Weighted covariance: (X * w) @ X.T / sum(w)
        # Unweighted covariance: X @ X.T / n_times
        cov = (data_centered * weights) @ data_centered.T / total_weight

    elif method == "shrinkage":
        # Ledoit-Wolf-like shrinkage
        emp_cov = data_centered @ data_centered.T / n_times

        if shrinkage is None:
            # Estimate optimal shrinkage
            shrinkage = _ledoit_wolf_shrinkage(data_centered)

        target = np.eye(n_channels) * np.trace(emp_cov) / n_channels
        cov = (1 - shrinkage) * emp_cov + shrinkage * target

    elif method == "oas":
        # Oracle Approximating Shrinkage
        from sklearn.covariance import OAS

        oas = OAS(assume_centered=assume_centered).fit(data_centered.T)
        cov = oas.covariance_

    elif method == "mcd":
        # Minimum Covariance Determinant (robust)
        from sklearn.covariance import MinCovDet

        mcd = MinCovDet(assume_centered=assume_centered).fit(data_centered.T)
        cov = mcd.covariance_
    else:
        raise ValueError(f"Unknown covariance method: {method}")

    # Ensure symmetry
    cov = (cov + cov.T) / 2

    return cov


def compute_evoked_covariance(
    evoked: Any,
    method: str = "empirical",
    **kwargs: Any,
) -> Any:
    """Compute a sensor covariance from an :class:`mne.Evoked`.

    MNE-Python's :func:`mne.compute_covariance` and
    :func:`mne.compute_raw_covariance` accept :class:`~mne.Epochs` and
    :class:`~mne.io.Raw` only. For an already-averaged :class:`~mne.Evoked` we
    treat its time samples as observations by wrapping the data in a
    single-trial :class:`~mne.EpochsArray` and deferring to
    :func:`mne.compute_covariance`. This keeps the estimation inside the MNE
    ecosystem (returning a genuine :class:`mne.Covariance`) and matches the
    Raw/Epochs branches of :meth:`mne_denoise.dss.DSS._fit_mne`, which assume
    zero-mean data and do not remove the per-channel temporal mean.

    Parameters
    ----------
    evoked : mne.Evoked
        The averaged response. Time samples are treated as observations.
    method : str
        Covariance estimator forwarded to :func:`mne.compute_covariance`
        (e.g. ``'empirical'``, ``'shrunk'``, ``'oas'``). Default ``'empirical'``.
    **kwargs
        Additional keyword arguments forwarded to
        :func:`mne.compute_covariance` (e.g. ``rank``, ``verbose``).

    Returns
    -------
    cov : mne.Covariance
        The estimated sensor covariance.
    """
    import mne

    data = np.asarray(evoked.data)
    if data.ndim != 2:
        raise ValueError(
            f"Evoked data must be 2D (n_channels, n_times), got {data.ndim}D."
        )
    if data.shape[1] < 2:
        raise ValueError(
            "Evoked must have at least 2 time samples to estimate a covariance."
        )

    # Wrap the single averaged response as a one-trial Epochs so MNE's
    # covariance machinery applies. ``keep_sample_mean=True`` stops MNE from
    # subtracting the (single) evoked response, which would otherwise zero the
    # data and yield an all-zero covariance.
    epochs = mne.EpochsArray(
        data[np.newaxis], evoked.info, tmin=float(evoked.times[0]), verbose=False
    )
    with warnings.catch_warnings():
        # Baseline correction is intentionally not applied: DSS uses the raw
        # second-moment covariance, so MNE's "not baseline corrected" notice
        # is expected here and would otherwise leak into every Evoked fit.
        warnings.filterwarnings(
            "ignore",
            message="Epochs are not baseline corrected",
            category=RuntimeWarning,
        )
        return mne.compute_covariance(
            epochs, method=method, keep_sample_mean=True, **kwargs
        )


def _ledoit_wolf_shrinkage(data: np.ndarray) -> float:
    """Estimate optimal Ledoit-Wolf shrinkage parameter."""
    n_channels, n_times = data.shape

    # Sample covariance
    S = data @ data.T / n_times

    # Target: scaled identity
    mu = np.trace(S) / n_channels

    # Compute shrinkage intensity
    delta = ((S - mu * np.eye(n_channels)) ** 2).sum() / n_channels

    # Estimate beta
    X2 = data**2
    beta = np.sum(X2 @ X2.T / n_times - S**2) / (n_channels * n_times)

    # Shrinkage
    shrinkage = min(1.0, beta / max(delta, 1e-10))

    return max(0.0, shrinkage)
