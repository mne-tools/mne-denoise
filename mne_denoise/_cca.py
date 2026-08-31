"""Internal canonical correlation utilities."""

from __future__ import annotations

import numpy as np
from scipy import linalg as la

from ._logging import logger, verbose


@verbose
def canonical_correlation(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
    rtol: float | None = None,
    verbose: bool | str | int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute canonical correlations between two observation matrices.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features_x)
        First data matrix.
    Y : ndarray, shape (n_samples, n_features_y)
        Second data matrix with the same number of observations as X.
    sample_weight : ndarray, shape (n_samples,) | None, default=None
        Optional finite, non-negative observation weights.
    rtol : float | None, default=None
        Relative QR rank threshold.
    verbose : bool, str, int, or None, default=None
        MNE-style logging level.

    Returns
    -------
    A, B : ndarray
        Canonical coefficient matrices for X and Y.
    R : ndarray, shape (d,)
        Non-negative canonical correlations in descending order.
    U, V : ndarray, shape (n_samples, d)
        Unit-variance canonical variates.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"X and Y must have the same number of samples, "
            f"got {X.shape[0]} and {Y.shape[0]}"
        )

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must both be two-dimensional")
    if rtol is not None:
        if not np.isscalar(rtol) or isinstance(rtol, bool):
            raise TypeError("rtol must be a positive finite number or None")
        rtol = float(rtol)
        if not np.isfinite(rtol) or rtol <= 0:
            raise ValueError("rtol must be a positive finite number or None")

    if sample_weight is None:
        weights = None
        Xc = X - X.mean(axis=0, keepdims=True)
        Yc = Y - Y.mean(axis=0, keepdims=True)
        X_decomposition = Xc
        Y_decomposition = Yc
    else:
        weights = np.asarray(sample_weight, dtype=np.float64)
        if weights.shape != (X.shape[0],):
            raise ValueError(
                f"sample_weight must have shape ({X.shape[0]},); got {weights.shape}"
            )
        if not np.all(np.isfinite(weights)):
            raise ValueError("sample_weight must contain only finite values")
        if np.any(weights < 0):
            raise ValueError("sample_weight must be non-negative")
        total_weight = weights.sum()
        if total_weight <= 0:
            raise ValueError("sample_weight must have a positive sum")
        Xc = X - (weights @ X / total_weight)[np.newaxis, :]
        Yc = Y - (weights @ Y / total_weight)[np.newaxis, :]
        root_weight = np.sqrt(weights)[:, np.newaxis]
        X_decomposition = Xc * root_weight
        Y_decomposition = Yc * root_weight

    Qx, Rx, Px = la.qr(X_decomposition, mode="economic", pivoting=True)
    Qy, Ry, Py = la.qr(Y_decomposition, mode="economic", pivoting=True)

    eps = np.finfo(np.float64).eps

    scale_x = np.abs(np.diag(Rx)).max() if Rx.size else 0.0
    scale_y = np.abs(np.diag(Ry)).max() if Ry.size else 0.0
    tolx = eps * max(Xc.shape) * scale_x if rtol is None else rtol * scale_x
    toly = eps * max(Yc.shape) * scale_y if rtol is None else rtol * scale_y
    rx = int(np.sum(np.abs(np.diag(Rx)) > tolx)) if Rx.size else 0
    ry = int(np.sum(np.abs(np.diag(Ry)) > toly)) if Ry.size else 0
    logger.debug(
        "CCA ranks: X=%d, Y=%d, canonical dimensions=%d.",
        rx,
        ry,
        min(rx, ry),
    )

    if rx == 0 or ry == 0:
        d = 0
        return (
            np.empty((X.shape[1], d), dtype=np.float64),
            np.empty((Y.shape[1], d), dtype=np.float64),
            np.empty(d, dtype=np.float64),
            np.empty((X.shape[0], d), dtype=np.float64),
            np.empty((Y.shape[0], d), dtype=np.float64),
        )

    Qx = Qx[:, :rx]
    Rx = Rx[:rx, :rx]
    Qy = Qy[:, :ry]
    Ry = Ry[:ry, :ry]

    Ux, s, VyT = la.svd(Qx.T @ Qy, full_matrices=False)
    Vy = VyT.T

    R = np.clip(s, 0.0, 1.0)

    Ex = np.eye(X.shape[1])[:, Px][:, :rx]
    Ey = np.eye(Y.shape[1])[:, Py][:, :ry]

    A = Ex @ la.solve(Rx, Ux)
    B = Ey @ la.solve(Ry, Vy)

    U = Xc @ A
    V = Yc @ B

    def _unit_var(Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if weights is None:
            std = Z.std(axis=0, ddof=1)
        else:
            std = np.sqrt((weights @ (Z**2)) / weights.sum())
        std[std == 0] = 1.0
        return Z / std, 1.0 / std

    U, su = _unit_var(U)
    V, sv = _unit_var(V)
    A = (A * su).astype(np.float64)
    B = (B * sv).astype(np.float64)

    return A, B, R.astype(np.float64), U.astype(np.float64), V.astype(np.float64)
