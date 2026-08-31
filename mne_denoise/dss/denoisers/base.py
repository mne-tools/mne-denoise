"""DSS bias interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class LinearDenoiser(ABC):
    """Base class for DSS bias transformations.

    Subclasses implement :meth:`apply` for channel-first arrays.
    """

    @abstractmethod
    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply bias transformation to data.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_times) or (n_channels, n_times, n_epochs)
            Input data.

        Returns
        -------
        biased : ndarray, same shape as input
            Biased data with signal of interest emphasized.
        """
        pass

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Apply the bias transformation."""
        return self.apply(data)


class NonlinearDenoiser(ABC):
    """Base class for nonlinear DSS denoisers.

    Subclasses implement :meth:`denoise` for one source at a time.
    """

    @abstractmethod
    def denoise(self, source: np.ndarray) -> np.ndarray:
        """Apply nonlinear denoising to source time series.

        Parameters
        ----------
        source : ndarray, shape (n_times,) or (n_times, n_epochs)
            Source time series (single component).

        Returns
        -------
        denoised : ndarray, same shape as input
            Denoised source with enhanced signal characteristics.
        """
        pass

    def __call__(self, source: np.ndarray) -> np.ndarray:
        """Apply the nonlinear transformation."""
        return self.denoise(source)
