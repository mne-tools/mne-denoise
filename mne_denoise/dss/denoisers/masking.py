"""Adaptive masking denoisers for DSS.

This module implements denoisers that estimate a time-varying mask $m(t)$
based on the local variance of the source signal.

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)

References
----------
.. [1] Särelä & Valpola (2005). Denoising Source Separation. J. Mach. Learn. Res., 6, 233-272.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .base import NonlinearDenoiser


class WienerMaskDenoiser(NonlinearDenoiser):
    """Adaptive Wiener mask denoiser.

    The core nonlinear DSS denoiser from Särelä & Valpola (2005). Estimates time-varying
    signal variance and applies soft Wiener-style masking:

        m(t) = σ²_signal(t) / [σ²_signal(t) + σ²_noise]
        s⁺(t) = s(t) · m(t)

    This is adaptive/nonlinear because the mask is estimated from the data.
    Ideal for bursty, non-stationary signals (spindles, beta bursts,
    intermittent artifacts).

    Parameters
    ----------
    window_samples : int
        Window size for local variance estimation. Default 50.
    noise_percentile : float
        Percentile of local variance used to estimate noise floor.
        Lower values = more aggressive denoising. Default 25.
    min_gain : float
        Minimum mask value (prevents complete zeroing). Default 0.01.
    noise_variance : float, optional
        If provided, use this fixed noise variance instead of estimating.

    Examples
    --------
    >>> from mne_denoise.dss.denoisers import WienerMaskDenoiser
    >>> denoiser = WienerMaskDenoiser(window_samples=50)
    >>> denoised = denoiser.denoise(source)

    References
    ----------
    Särelä & Valpola (2005). Section 4.4 "Spectral Shift and Approximation of the Objective
    Function with Mask-Based Denoisings"
    """

    def __init__(
        self,
        window_samples: int = 50,
        noise_percentile: float = 25.0,
        *,
        min_gain: float = 0.01,
        noise_variance: float | None = None,
    ) -> None:
        self.window_samples = max(3, window_samples)
        self.noise_percentile = noise_percentile
        self.min_gain = min_gain
        self.noise_variance = noise_variance

    def denoise(self, source: np.ndarray) -> np.ndarray:
        """Apply Wiener mask denoising.

        Parameters
        ----------
        source : ndarray, shape (n_times,) or (n_times, n_epochs)
            Source time series.

        Returns
        -------
        denoised : ndarray, same shape as input
            Wiener-masked source.
        """
        if source.ndim == 1:
            return self._denoise_1d(source)
        elif source.ndim == 2:
            _, n_epochs = source.shape
            denoised = np.zeros_like(source)
            for ep in range(n_epochs):
                denoised[:, ep] = self._denoise_1d(source[:, ep])
            return denoised
        else:
            raise ValueError(f"Source must be 1D or 2D, got {source.ndim}D")

    def _denoise_1d(self, source: np.ndarray) -> np.ndarray:
        """Apply Wiener mask to 1D source."""
        n_samples = len(source)
        window = min(self.window_samples, n_samples // 2)

        # Estimate local signal variance: σ²(t) = E[s²] - E[s]²
        source_sq = source**2
        local_mean_sq = ndimage.uniform_filter1d(source_sq, size=window, mode="reflect")
        local_mean = ndimage.uniform_filter1d(source, size=window, mode="reflect")
        local_var = np.maximum(local_mean_sq - local_mean**2, 0)

        # Estimate noise variance (from quiet periods)
        if self.noise_variance is not None:
            noise_var = self.noise_variance
        else:
            # Use percentile of local variance as noise floor estimate
            noise_var = np.percentile(local_var, self.noise_percentile)
            noise_var = max(noise_var, 1e-15)  # Prevent division by zero

        # Wiener mask: m(t) = σ²_signal / (σ²_signal + σ²_noise)
        # where σ²_signal = max(0, local_var - noise_var)
        signal_var = np.maximum(local_var - noise_var, 0)
        mask = signal_var / (signal_var + noise_var + 1e-15)

        # Apply minimum gain
        mask = np.maximum(mask, self.min_gain)

        return source * mask


class VarianceMaskDenoiser(NonlinearDenoiser):
    """Nonlinear denoiser using local variance masking.

    Identifies high-variance regions in the source time series and
    weights them higher, effectively emphasizing transient activity.
    Useful for extracting non-stationary sources.

    Parameters
    ----------
    window_samples : int
        Window size for local variance computation. Default 100.
    percentile : float
        Percentile threshold for high-variance mask. Default 75.
    soft : bool
        If True, use soft weighting based on variance magnitude.
        If False, use binary mask. Default True.

    Examples
    --------
    >>> from mne_denoise.dss.denoisers import VarianceMaskDenoiser
    >>> denoiser = VarianceMaskDenoiser(window_samples=50, percentile=80)
    >>> denoised_source = denoiser.denoise(source)

    References
    ----------
    Särelä & Valpola (2005). Section 4.4 "Spectral Shift and Approximation of the Objective
    Function with Mask-Based Denoisings"
    """

    def __init__(
        self,
        window_samples: int = 100,
        percentile: float = 75.0,
        *,
        soft: bool = True,
    ) -> None:
        self.window_samples = max(3, window_samples)
        self.percentile = percentile
        self.soft = soft

    def denoise(self, source: np.ndarray) -> np.ndarray:
        """Apply variance-based masking to a source time series.

        Parameters
        ----------
        source : ndarray, shape (n_times,) or (n_times, n_epochs)
            Source time series. Two-dimensional input is processed one epoch
            at a time.

        Returns
        -------
        denoised : ndarray, same shape as ``source``
            Source weighted by the local-variance mask.
        """
        if source.ndim == 1:
            return self._denoise_1d(source)
        elif source.ndim == 2:
            _, n_epochs = source.shape
            denoised = np.zeros_like(source)
            for ep in range(n_epochs):
                denoised[:, ep] = self._denoise_1d(source[:, ep])
            return denoised
        else:
            raise ValueError(f"Source must be 1D or 2D, got {source.ndim}D")

    def _denoise_1d(self, source: np.ndarray) -> np.ndarray:
        """Process single 1D source."""
        n_samples = len(source)
        source_sq = source**2
        window = min(self.window_samples, n_samples)

        local_mean_sq = ndimage.uniform_filter1d(source_sq, size=window, mode="reflect")
        local_mean = ndimage.uniform_filter1d(source, size=window, mode="reflect")
        local_var = np.maximum(local_mean_sq - local_mean**2, 0)

        if self.soft:
            threshold = np.percentile(local_var, self.percentile)
            if threshold < 1e-15:
                threshold = np.max(local_var) * 0.5
            if threshold < 1e-15:
                return source
            weights = 1 / (1 + np.exp(-(local_var - threshold) / (threshold * 0.5)))
            denoised = source * weights
        else:
            threshold = np.percentile(local_var, self.percentile)
            mask = local_var >= threshold
            denoised = source * mask.astype(float)

        return denoised
