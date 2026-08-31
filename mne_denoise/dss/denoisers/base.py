"""Base classes for DSS denoiser functions.

Provides abstract interfaces for linear and nonlinear bias functions
that can be plugged into the DSS pipeline.

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)

References
----------
.. [1] Särelä & Valpola (2005). Denoising Source Separation. J. Mach. Learn. Res., 6, 233-272.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class LinearDenoiser(ABC):
    """Base class for linear bias functions.

    Linear denoisers apply a deterministic transformation to the data
    that emphasizes a signal of interest. The DSS algorithm then finds
    spatial filters that maximize the ratio of biased to baseline variance.

    Subclasses must implement the `apply` method.

    References
    ----------
    Särelä & Valpola (2005). Section 2.2 "Linear DSS"
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
        """Apply the linear denoiser through the callable interface.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_times) or (n_channels, n_times, n_epochs)
            Data passed to :meth:`apply`.

        Returns
        -------
        biased : ndarray, same shape as ``data``
            Bias-transformed data.
        """
        return self.apply(data)


class NonlinearDenoiser(ABC):
    """Base class for nonlinear/adaptive denoiser functions.

    Nonlinear denoisers operate on source time series rather than
    sensor data. They are used in the iterative DSS algorithm where
    the denoising function is applied to the current source estimate
    at each iteration.

    Examples include variance-based masking, kurtosis maximization,
    and other adaptive transformations.

    References
    ----------
    Särelä & Valpola (2005). Denoising Source Separation. J. Mach. Learn. Res., 6, 233-272.
    Section 2.1 "One-Unit Algorithm for Source Separation"
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
        """Apply the nonlinear denoiser through the callable interface.

        Parameters
        ----------
        source : ndarray, shape (n_times,) or (n_times, n_epochs)
            Source time series passed to :meth:`denoise`.

        Returns
        -------
        denoised : ndarray, same shape as ``source``
            Nonlinearly transformed source.
        """
        return self.denoise(source)
