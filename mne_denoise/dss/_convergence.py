"""Adaptive learning rate (gamma) optimization for DSS.

This module implements helper classes to adaptively adjust the spectral shift
(gamma) or learning rate during Iterative DSS convergence.

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)

References
----------
.. [1] Särelä & Valpola (2005). Denoising Source Separation. J. Mach. Learn. Res., 6, 233-272.
"""

from __future__ import annotations

import numpy as np


class Gamma179:
    """Detects oscillation by comparing consecutive weight deltas.

    If the angle between deltas is > 90°, reduces gamma to 0.5.



    Usage
    -----
    >>> from mne_denoise.dss._convergence import Gamma179
    >>> gamma_fn = Gamma179()
    >>> # idss = IterativeDSS(..., gamma=gamma_fn) (Illustrative)

    References
    ----------
    Särelä & Valpola (2005). Section 2.5 "Spectral Shift" and 3.3 "Spectral Shift Revisited"
    """

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
    """Adjusts gamma based on correlation between consecutive weight deltas.

    More aggressive than gamma_179.

    Usage
    -----
    >>> from mne_denoise.dss._convergence import GammaPredictive
    >>> gamma_fn = GammaPredictive()
    >>> # idss = IterativeDSS(..., gamma=gamma_fn)

    References
    ----------
    Särelä & Valpola (2005). Section 3.4 "Detection of Overfitting"
    """

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
