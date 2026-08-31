"""Local-variance masks for iterative DSS."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .base import NonlinearDenoiser


class WienerMaskDenoiser(NonlinearDenoiser):
    """Local-variance Wiener mask for iterative DSS.

    The local variance is estimated from moving averages of the source and its
    square. A percentile of that variance, or ``noise_variance``, sets the noise
    floor; the soft gain is ``signal_variance / (signal_variance + noise_variance)``
    and is bounded below by ``min_gain``.

    Parameters
    ----------
    window_samples : int, default=50
        Window length for local statistics; values below 3 are set to 3.
    noise_percentile : float, default=25.0
        Percentile used for the estimated noise floor.
    min_gain : float, default=0.01
        Lower bound on the mask gain.
    noise_variance : float or None, default=None
        Fixed noise variance. If ``None``, estimate it from the local-variance
        percentile.
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
    """Local-variance mask for iterative DSS.

    The local variance is computed from moving means of the source and its square.
    A percentile threshold produces either a sigmoid soft mask or a binary mask.

    Parameters
    ----------
    window_samples : int, default=100
        Window length for local variance; values below 3 are set to 3.
    percentile : float, default=75.0
        Local-variance percentile used as the mask threshold.
    soft : bool, default=True
        Use a sigmoid gain when true, otherwise use a binary threshold mask.
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
