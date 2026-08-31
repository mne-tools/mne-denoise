"""Covariance estimation utilities."""

from __future__ import annotations

import numpy as np

from ._validation import check_chunk_size


def _flatten_observations(
    data: np.ndarray, weights: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Normalize channel-by-observation data and optional weights."""
    data = np.asarray(data, dtype=float)
    user_weights = weights is not None
    if data.ndim == 3:
        n_channels, n_times, n_epochs = data.shape
        data = data.reshape(n_channels, -1)
        if user_weights:
            weights = np.asarray(weights, dtype=float)
            if weights.shape == (n_times,):
                weights = np.repeat(weights, n_epochs)
            elif weights.shape == (n_times, n_epochs):
                weights = weights.reshape(-1)
            elif weights.shape != (n_times * n_epochs,):
                raise ValueError(
                    "For 3D data, weights must have shape "
                    f"({n_times},), ({n_times}, {n_epochs}), or "
                    f"({n_times * n_epochs},); got {weights.shape}"
                )
    if data.ndim != 2:
        raise ValueError(f"data must be 2D or 3D, got shape {data.shape}")

    if user_weights:
        weights = np.asarray(weights, dtype=float)
        if weights.ndim != 1:
            raise ValueError(
                "For 2D data, weights must be one-dimensional; "
                f"got shape {weights.shape}"
            )
        if weights.shape[0] != data.shape[1]:
            raise ValueError(
                f"Weights length {weights.shape[0]} does not match "
                f"data samples {data.shape[1]}"
            )
        if not np.all(np.isfinite(weights)):
            raise ValueError("weights must contain only finite values")
        if np.any(weights < 0):
            raise ValueError("weights must be non-negative")
        if weights.sum() <= 0:
            raise ValueError("Sum of weights must be positive")
    else:
        weights = np.ones(data.shape[1])
    return data, weights, user_weights


def compute_mean(data: np.ndarray, *, weights: np.ndarray | None = None) -> np.ndarray:
    """Compute one channel mean over 2-D or 3-D observations."""
    data, weights, _ = _flatten_observations(data, weights)
    return (data @ weights / weights.sum())[:, np.newaxis]


def compute_covariance(
    data: np.ndarray,
    *,
    method: str = "empirical",
    shrinkage: float | None = None,
    weights: np.ndarray | None = None,
    assume_centered: bool = False,
    chunk_size: int | None = None,
) -> np.ndarray:
    """Estimate a channel covariance matrix from NumPy data.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times) or (n_channels, n_times, n_epochs)
        Channel-first data. The 3-D layout is flattened over time and epochs.
    method : {"empirical", "shrinkage", "oas", "mcd"}, default="empirical"
        Covariance estimator.
    shrinkage : float or None, default=None
        Shrinkage intensity for method="shrinkage"; None estimates it.
    weights : ndarray or None, default=None
        Non-negative observation weights. 2-D data uses (n_times,); 3-D data accepts
        (n_times,), (n_times, n_epochs), or the flattened length.
    assume_centered : bool, default=False
        Skip mean subtraction when true.
    chunk_size : int or None, default=None
        Empirical-covariance chunk size.

    Returns
    -------
    ndarray, shape (n_channels, n_channels)
        NumPy covariance array, not an MNE Covariance object.

    Notes
    -----
    The empirical covariance uses a population denominator: n_times or the total
    weight. Weighted and chunked estimates are supported only for method="empirical".
    """
    data, weights, user_weights = _flatten_observations(data, weights)
    n_channels, n_times = data.shape

    chunk_size = check_chunk_size(chunk_size)
    if chunk_size is not None and method != "empirical":
        raise ValueError("chunk_size is only supported for empirical covariance")

    if user_weights:
        total_weight = np.sum(weights)
        if method != "empirical":
            # Currently we only support weighted empirical.
            raise ValueError(
                f"Weighted covariance not implemented for method '{method}'"
            )
    else:
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
