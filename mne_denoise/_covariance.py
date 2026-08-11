"""Internal covariance estimation shared by array-based algorithms.

Provides methods for computing covariance matrices robust to outliers
or low sample counts (shrinkage).

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

from __future__ import annotations

import numpy as np

from ._validation import check_chunk_size


def compute_covariance(
    data: np.ndarray,
    *,
    method: str = "empirical",
    shrinkage: float | None = None,
    weights: np.ndarray | None = None,
    assume_centered: bool = False,
    chunk_size: int | None = None,
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
    chunk_size : int | None, default=None
        Number of samples accumulated at a time for empirical covariance.
        ``None`` uses one matrix multiplication over all samples. This option is
        not supported by the shrinkage, OAS, or MCD estimators.

    Returns
    -------
    cov : ndarray, shape (n_channels, n_channels)
        The estimated covariance matrix.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim == 3:
        n_channels, n_times_in, n_epochs = data.shape
        data = data.reshape(n_channels, -1)

        if weights is not None and weights.shape[0] == n_times_in:
            # Tile weights across epochs
            weights = np.tile(weights, n_epochs)

    if data.ndim != 2:
        raise ValueError(f"data must be 2D or 3D, got shape {data.shape}")
    n_channels, n_times = data.shape

    chunk_size = check_chunk_size(chunk_size)
    if chunk_size is not None and method != "empirical":
        raise ValueError("chunk_size is only supported for empirical covariance")

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
        mean = (data @ weights / total_weight)[:, np.newaxis]
        data_centered = data - mean

    if method == "empirical":
        # Weighted covariance: (X * w) @ X.T / sum(w)
        # Unweighted covariance: X @ X.T / n_times
        if chunk_size is None:
            cov = (data_centered * weights) @ data_centered.T / total_weight
        else:
            cov = np.zeros((n_channels, n_channels), dtype=float)
            for start in range(0, n_times, chunk_size):
                stop = min(start + chunk_size, n_times)
                chunk = data_centered[:, start:stop]
                cov += (chunk * weights[start:stop]) @ chunk.T
            cov /= total_weight

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
