"""ASR symmetric positive-definite matrix helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

_TINY = float(np.finfo(np.float64).tiny)


def _relative_floor(scale: float) -> float:
    """Return a floating-point floor relative to a problem's numeric scale."""
    return max(abs(float(scale)) * np.finfo(np.float64).eps, _TINY)


def _geometric_median(
    covariances: np.ndarray,
    max_iter: int = 128,
    tol: float = 1e-7,
) -> np.ndarray:
    """Compute the geometric median of a set of covariance matrices."""
    current = np.mean(covariances, axis=0)
    for _ in range(max_iter):
        distances = np.linalg.norm(
            (covariances - current).reshape(covariances.shape[0], -1),
            axis=1,
        )
        distance_floor = _relative_floor(
            max(float(np.max(distances)), float(np.linalg.norm(current)))
        )
        distances = np.maximum(distances, distance_floor)
        weights = 1.0 / distances
        update = np.tensordot(weights, covariances, axes=(0, 0)) / np.sum(weights)
        if np.linalg.norm(update - current) <= tol * max(
            float(np.linalg.norm(current)), _TINY
        ):
            current = update
            break
        current = update
    return current


def _geometric_median_chunked(
    iter_factory,
    chunk_blocks: int,
    n_channels: int,
    max_iter: int = 128,
    tol: float = 1e-7,
) -> np.ndarray:
    """Compute a covariance geometric median without storing all blocks in memory."""
    total = np.zeros((n_channels, n_channels), dtype=np.float64)
    count = 0
    for chunk in iter_factory(chunk_blocks):
        total += np.sum(chunk, axis=0)
        count += chunk.shape[0]
    current = total / max(count, 1)

    for _ in range(max_iter):
        numerator = np.zeros_like(current)
        denominator = 0.0
        for chunk in iter_factory(chunk_blocks):
            distances = np.linalg.norm(
                (chunk - current).reshape(chunk.shape[0], -1),
                axis=1,
            )
            distance_floor = _relative_floor(
                max(float(np.max(distances)), float(np.linalg.norm(current)))
            )
            distances = np.maximum(distances, distance_floor)
            weights = 1.0 / distances
            numerator += np.tensordot(weights, chunk, axes=(0, 0))
            denominator += float(np.sum(weights))
        update = numerator / denominator
        if np.linalg.norm(update - current) <= tol * max(
            float(np.linalg.norm(current)), _TINY
        ):
            current = update
            break
        current = update
    return current


def _regularize_spd(C: np.ndarray, regularization: float) -> np.ndarray:
    """Ensure a matrix is symmetric positive-definite via eigenvalue thresholding."""
    C = (C + C.T) / 2
    w, V = np.linalg.eigh(C)
    scale = max(
        abs(float(np.trace(C)) / C.shape[0]),
        float(np.max(np.abs(w))),
        _TINY,
    )
    floor = max(regularization * scale, _TINY)
    w = np.maximum(w, floor)
    return (V * w) @ V.T


def _sqrtm_spd(C: np.ndarray, regularization: float) -> np.ndarray:
    """Compute the principal matrix square root of an SPD matrix."""
    C = _regularize_spd(C, regularization)
    w, V = np.linalg.eigh(C)
    return (V * np.sqrt(np.maximum(w, _TINY))) @ V.T


def _invsqrtm_spd(C: np.ndarray, regularization: float) -> np.ndarray:
    """Compute the inverse matrix square root of an SPD matrix."""
    C = _regularize_spd(C, regularization)
    w, V = np.linalg.eigh(C)
    w = np.maximum(w, _TINY)
    return (V * (1.0 / np.sqrt(w))) @ V.T


def _logm_spd(C: np.ndarray, regularization: float) -> np.ndarray:
    """Compute the principal matrix logarithm of an SPD matrix."""
    C = _regularize_spd(C, regularization)
    w, V = np.linalg.eigh(C)
    w = np.maximum(w, _TINY)
    return (V * np.log(w)) @ V.T


def _expm_sym(S: np.ndarray) -> np.ndarray:
    """Compute the matrix exponential of a symmetric matrix."""
    S = (S + S.T) / 2.0
    w, V = np.linalg.eigh(S)
    return (V * np.exp(w)) @ V.T


def _karcher_mean_spd(
    covariances: np.ndarray,
    regularization: float,
    init: np.ndarray | None = None,
    sample_weight: np.ndarray | None = None,
    max_iter: int = 32,
    tol: float = 1e-7,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute the affine-invariant Karcher mean of SPD matrices."""
    covariances = np.asarray(covariances, dtype=np.float64)
    if covariances.ndim != 3:
        raise ValueError(
            "covariances must have shape (n_matrices, n_channels, n_channels)"
        )
    n_matrices = covariances.shape[0]
    if n_matrices < 1:
        raise ValueError("At least one covariance matrix is required")

    if sample_weight is None:
        weights = np.full(n_matrices, 1.0 / n_matrices, dtype=np.float64)
    else:
        weights = np.asarray(sample_weight, dtype=np.float64)
        if weights.shape != (n_matrices,):
            raise ValueError(
                f"sample_weight must have shape ({n_matrices},), got {weights.shape}"
            )
        if np.any(weights < 0):
            raise ValueError("sample_weight must be non-negative")
        weight_sum = np.sum(weights)
        if weight_sum <= 0:
            raise ValueError("sample_weight must sum to a positive value")
        weights = weights / weight_sum

    current = (
        _regularize_spd(init, regularization)
        if init is not None
        else _regularize_spd(np.mean(covariances, axis=0), regularization)
    )
    converged = False
    update_norm = np.inf
    iteration_count = 0
    for _ in range(max_iter):
        iteration_count += 1
        sqrt_current = _sqrtm_spd(current, regularization)
        invsqrt_current = _invsqrtm_spd(current, regularization)
        tangent = np.zeros_like(current)
        for weight, cov in zip(weights, covariances):
            centered = (
                invsqrt_current @ _regularize_spd(cov, regularization) @ invsqrt_current
            )
            tangent += weight * _logm_spd(centered, regularization)
        update_norm = float(np.linalg.norm(tangent, ord="fro"))
        if update_norm <= tol:
            converged = True
            break
        current = sqrt_current @ _expm_sym(tangent) @ sqrt_current
        current = _regularize_spd(current, regularization)

    info = {
        "riemannian_mean_iterations": int(iteration_count),
        "riemannian_mean_converged": bool(converged),
        "riemannian_mean_update_norm": float(update_norm),
    }
    return current, info


def _sqrt_and_eig(
    C: np.ndarray,
    regularization: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the ordered eigendecomposition and regularized square root of an SPD matrix."""
    w, V = np.linalg.eigh(C)
    order = np.argsort(w)
    w = w[order]
    V = V[:, order]
    scale = max(float(np.max(np.abs(w))), _TINY)
    floor = max(regularization * scale, _TINY)
    w = np.maximum(w, floor)
    M = (V * np.sqrt(w)) @ V.T
    return M, w, V


def _riemannian_nonlinear_eigenspace(
    L: np.ndarray,
    regularization: float,
    alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate the Riemannian nonlinear eigenspace for the full-basis case."""
    L = _regularize_spd(L, regularization)
    ones = np.ones(L.shape[0], dtype=np.float64)
    correction = np.linalg.solve(L, ones)
    modified = L + alpha * np.diag(correction)
    modified = (modified + modified.T) / 2.0
    D, V = np.linalg.eigh(modified)
    order = np.argsort(D)
    return D[order], V[:, order]
