"""Structural / rank QA metrics (rank, subspace geometry, cleaning footprint)."""

from __future__ import annotations

import numpy as np


def _data(x):
    return np.asarray(x.get_data() if hasattr(x, "get_data") else x, float)


def _cov(x):
    d = _data(x)
    if d.ndim == 2 and d.shape[0] != d.shape[1]:
        d = d - d.mean(1, keepdims=True)
        return (d @ d.T) / d.shape[1]
    return d  # assume already a covariance


def effective_rank(x) -> float:
    """Effective rank = exp(entropy of the normalized eigen/singular spectrum).

    (Roy & Vetterli, 2007.)  Accepts data ``(n_ch, n_times)`` or a covariance.
    """
    C = _cov(x)
    ev = np.linalg.eigvalsh((C + C.T) / 2.0)
    ev = ev[ev > 0]
    if ev.size == 0:
        return 0.0
    p = ev / ev.sum()
    h = -np.sum(p * np.log(p))
    return float(np.exp(h))


def effective_rank_change(before, after) -> float:
    return float(effective_rank(after) - effective_rank(before))


def principal_angles_deg(A, B) -> np.ndarray:
    A, B = np.asarray(A, float), np.asarray(B, float)
    Qa, _ = np.linalg.qr(A); Qb, _ = np.linalg.qr(B)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s[::-1], -1.0, 1.0)))


def covariance_distance(C1, C2, *, reg: float = 1e-10) -> float:
    """Affine-invariant Riemannian distance between two SPD covariances."""
    C1 = _cov(C1); C2 = _cov(C2)
    n = C1.shape[0]
    C1 = C1 + reg * np.eye(n); C2 = C2 + reg * np.eye(n)
    w = np.linalg.eigvalsh(np.linalg.solve(C1, C2))
    w = w[w > 0]
    return float(np.sqrt(np.sum(np.log(w) ** 2)))


def rejected_window_fraction(window_mask) -> float:
    m = np.asarray(window_mask, bool).ravel()
    return float(np.mean(m)) if m.size else 0.0


def calibration_fraction(n_calibration_samples, n_total_samples) -> float:
    if n_total_samples <= 0:
        raise ValueError("n_total_samples must be > 0")
    return float(n_calibration_samples / n_total_samples)


def artifact_ic_fraction(labels, *, artifact_classes=("eye", "muscle", "heart", "line_noise", "channel_noise")) -> float:
    labels = [str(s).lower() for s in labels]
    if not labels:
        return 0.0
    art = sum(any(c in s for c in artifact_classes) for s in labels)
    return float(art / len(labels))
