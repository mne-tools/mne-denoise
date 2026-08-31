"""Adaptive ASR learning-state helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._spd import _regularize_spd


@dataclass
class _AdaptiveSimilarityMatcher:
    """Similarity-matching adaptive subspace learner."""

    variant: str
    tau: float
    learning_rate: float
    M: np.ndarray
    W: np.ndarray
    Minv: np.ndarray

    def copy(self) -> _AdaptiveSimilarityMatcher:
        """Return an independent copy for transactional state updates."""
        return _AdaptiveSimilarityMatcher(
            variant=self.variant,
            tau=float(self.tau),
            learning_rate=float(self.learning_rate),
            M=self.M.copy(),
            W=self.W.copy(),
            Minv=self.Minv.copy(),
        )

    def fit_next(self, X: np.ndarray) -> np.ndarray:
        """Update the subspace learner on one calibration chunk.

        Parameters
        ----------
        X : ndarray, shape (n_channels, n_samples)
            The new chunk of data to process.

        Returns
        -------
        y : ndarray, shape (n_components, n_samples)
            The projected components for this chunk.
        """
        y = self.Minv @ (self.W @ X)

        step = float(self.learning_rate)
        self.W = (1.0 - 2.0 * step) * self.W + (2.0 * step * y) @ X.T / X.shape[1]

        lateral_step = step / float(self.tau)
        yy = (y @ y.T) / X.shape[1]
        if self.variant == "psw":
            target = np.eye(yy.shape[0], dtype=np.float64)
        else:
            target = self.M
        self.M = self.M + lateral_step * (yy - target)
        try:
            self.Minv = np.linalg.inv(self.M)
        except np.linalg.LinAlgError:
            self.M = _regularize_spd(self.M, 1e-8)
            self.Minv = np.linalg.inv(self.M)
        return y

    def get_components(self) -> np.ndarray:
        """Return the orthonormalized adaptive basis.

        Returns
        -------
        q : ndarray, shape (n_channels, n_components)
            The orthonormalized basis vectors (columns).
        """
        components = (self.Minv @ self.W).T
        q, _ = np.linalg.qr(components, mode="reduced")
        return q.astype(np.float64, copy=False)


def _build_adaptive_learner(
    X_filtered: np.ndarray,
    V: np.ndarray,
    variant: str,
    learning_rate: float,
    tau: float,
    regularization: float,
) -> _AdaptiveSimilarityMatcher:
    """Initialize an adaptive similarity matcher on a starting data window."""
    Y0 = V.T @ X_filtered
    n_samples = X_filtered.shape[1]
    M0 = (Y0 @ Y0.T) / n_samples
    W0 = (Y0 @ X_filtered.T) / n_samples
    M0 = _regularize_spd(M0, regularization)
    return _AdaptiveSimilarityMatcher(
        variant=variant,
        tau=float(tau),
        learning_rate=float(learning_rate),
        M=M0,
        W=W0.astype(np.float64, copy=False),
        Minv=np.linalg.inv(M0),
    )
