"""DSS convergence helpers."""

from __future__ import annotations

import numpy as np


class Gamma179:
    """Reduce the DSS step size when successive weight updates oscillate."""

    def __init__(self):
        self.gamma = 1.0
        self.deltaw = None

    def __call__(self, w_new: np.ndarray, w_old: np.ndarray, iteration: int) -> float:
        """Compute adaptive gamma."""
        if iteration <= 2:
            self.gamma = 1.0
            if iteration == 2:
                self.deltaw = w_old - w_new
        elif iteration > 2:
            deltaw_old = self.deltaw
            self.deltaw = w_old - w_new

            # Check angle between consecutive deltas
            limit = 0.0  # cos(90°)
            norm_prod = np.linalg.norm(self.deltaw) * np.linalg.norm(deltaw_old)
            if norm_prod > 1e-12:
                cos_angle = np.dot(self.deltaw, deltaw_old) / norm_prod
                if cos_angle <= limit:
                    self.gamma = 0.5

        return self.gamma

    def reset(self):
        """Reset state for new component."""
        self.gamma = 1.0
        self.deltaw = None


class GammaPredictive:
    """Adjust the DSS step size from successive weight updates."""

    def __init__(self, min_gamma: float = 0.5):
        self.gamma = 1.0
        self.deltaw = None
        self.min_gamma = min_gamma

    def __call__(self, w_new: np.ndarray, w_old: np.ndarray, iteration: int) -> float:
        """Compute adaptive gamma using predictive controller."""
        if iteration <= 2:
            self.gamma = 1.0
            if iteration == 2:
                self.deltaw = w_old - w_new
        else:
            deltaw_old = self.deltaw
            self.deltaw = w_old - w_new

            # Predictive update
            norm_sq = np.dot(deltaw_old, deltaw_old)
            if norm_sq > 1e-12:
                self.gamma = self.gamma + np.dot(self.deltaw, deltaw_old) / norm_sq
                if self.gamma < self.min_gamma:
                    self.gamma = self.min_gamma

        return self.gamma

    def reset(self):
        """Reset state for new component."""
        self.gamma = 1.0
        self.deltaw = None
