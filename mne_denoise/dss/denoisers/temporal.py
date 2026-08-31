"""Temporal bias functions for DSS."""

from __future__ import annotations

from numbers import Integral

import numpy as np

from .base import LinearDenoiser, NonlinearDenoiser


class LagAverageBias(LinearDenoiser):
    """Lag-averaging bias for ordinary sensor-space DSS.

    Parameters
    ----------
    lags : int or array-like, default=10
        An integer uses samples from 1 through ``lags``; an array supplies the
        sample lags directly.
    weighting : {"uniform", "inverse_lag"}, default="uniform"
        Equal or inverse-absolute-lag weighting.
    """

    def __init__(
        self,
        lags: int | np.ndarray = 10,
        weighting: str = "uniform",
    ) -> None:
        self.lags = lags
        self.weighting = weighting

    def _resolve_lags(self) -> np.ndarray:
        """Return a validated, unique lag vector."""
        if isinstance(self.lags, bool):
            raise TypeError("lags must be a positive integer or integer array")
        if isinstance(self.lags, Integral):
            if self.lags < 1:
                raise ValueError("lags must be positive")
            return np.arange(1, int(self.lags) + 1)
        values = np.asarray(self.lags, dtype=object)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("lags must be a non-empty one-dimensional array")
        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in values.tolist()
        ):
            raise TypeError("lags must contain only integers")
        resolved = np.unique(values.astype(int))
        if not np.any(resolved != 0):
            raise ValueError("lags must contain at least one nonzero lag")
        return resolved

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply time-shift bias.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_times) or (n_channels, n_times, n_epochs)
            Input data.

        Returns
        -------
        biased : ndarray, same shape as input
            Time-shifted averaged data.
        """
        data = np.asarray(data, dtype=np.float64)
        if data.ndim not in (2, 3):
            raise ValueError("data must be 2D or 3D")
        self._lag_array = self._resolve_lags()
        if self.weighting == "uniform":
            operation = self._equal_lag_average
        elif self.weighting == "inverse_lag":
            operation = self._weighted_lag_average
        else:
            raise ValueError("weighting must be 'uniform' or 'inverse_lag'")

        if data.ndim == 2:
            return operation(data)
        return np.stack(
            [operation(data[:, :, epoch]) for epoch in range(data.shape[2])],
            axis=2,
        )

    def _equal_lag_average(self, data: np.ndarray) -> np.ndarray:
        """Average time-shifted versions with equal weights."""
        n_channels, n_samples = data.shape
        lags = self._lag_array
        max_shift = np.max(np.abs(lags))

        if max_shift >= n_samples // 2:
            raise ValueError(
                f"Max shift ({max_shift}) too large for data length ({n_samples})"
            )

        valid_start = max_shift
        valid_end = n_samples - max_shift
        valid_length = valid_end - valid_start

        accumulated = np.zeros((n_channels, valid_length))
        for shift in lags:
            shifted = data[:, valid_start + shift : valid_end + shift]
            accumulated += shifted

        biased = accumulated / len(lags)

        # Pad to original length
        biased_full = np.zeros_like(data)
        biased_full[:, valid_start:valid_end] = biased
        return biased_full

    def _weighted_lag_average(self, data: np.ndarray) -> np.ndarray:
        """Average time-shifted versions with inverse-lag weights."""
        n_channels, n_samples = data.shape
        lags = self._lag_array
        max_shift = np.max(np.abs(lags))

        valid_start = max_shift
        valid_end = n_samples - max_shift
        valid_length = valid_end - valid_start

        accumulated = np.zeros((n_channels, valid_length))
        total_weight = 0

        for shift in lags:
            weight = 1.0 / max(abs(shift), 1)
            shifted = data[:, valid_start + shift : valid_end + shift]
            accumulated += weight * shifted
            total_weight += weight

        biased = accumulated / total_weight

        biased_full = np.zeros_like(data)
        biased_full[:, valid_start:valid_end] = biased
        return biased_full


class SmoothingBias(LinearDenoiser):
    """Causal running-mean bias for DSS.

    Parameters
    ----------
    window : int, default=10
        Smoothing-window length in samples.
    iterations : int, default=1
        Number of smoothing passes.

    Notes
    -----
    For 3D channel-first input, the implementation reshapes
    ``(n_channels, n_times, n_epochs)`` to ``(n_channels, -1)`` before smoothing,
    so the time and epoch axes are concatenated rather than smoothed independently.
    The original shape is restored on return.
    """

    def __init__(self, window: int = 10, iterations: int = 1) -> None:
        self.window = window
        self.iterations = iterations

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply the causal running-mean bias.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_times) or (n_channels, n_times, n_epochs)
            Channel-first data.

        Returns
        -------
        ndarray
            Smoothed data with the input shape.
        """
        orig_shape = data.shape
        if data.ndim == 3:
            data_2d = data.reshape(data.shape[0], -1)
        else:
            data_2d = data

        W = int(self.window)
        smoothed = data_2d.copy()

        for _ in range(self.iterations):
            mean_head = np.mean(smoothed[..., : W + 1], axis=-1, keepdims=True)
            centered = smoothed - mean_head

            # Causal running mean via cumulative sums
            cs = np.cumsum(centered, axis=-1)
            out = np.empty_like(centered)
            # First W samples: expanding window
            out[..., :W] = cs[..., :W] / np.arange(1, W + 1)
            # Remaining samples: fixed-width causal window
            out[..., W:] = (cs[..., W:] - cs[..., :-W]) / W
            smoothed = out + mean_head

        if data.ndim == 3:
            return smoothed.reshape(orig_shape)
        return smoothed


class DCTDenoiser(NonlinearDenoiser):
    """DCT-domain denoiser.

    Parameters
    ----------
    mask : ndarray or None, default=None
        Optional DCT-domain mask. If its length differs from the source length,
        the implementation interpolates it.
    cutoff_fraction : float, default=0.5
        Fraction of low-frequency DCT coefficients retained when ``mask`` is
        ``None``. Ignored when an explicit mask is supplied.
    """

    def __init__(
        self, mask: np.ndarray | None = None, cutoff_fraction: float = 0.5
    ) -> None:
        self.mask = mask
        self.cutoff_fraction = cutoff_fraction
        self._cached_mask = None
        self._cached_len = None

    def denoise(self, source: np.ndarray) -> np.ndarray:
        """Apply the configured DCT-domain mask.

        Parameters
        ----------
        source : ndarray, shape (n_times,) or (n_times, n_epochs)
            Source time series; columns of 2D input are processed separately.

        Returns
        -------
        ndarray
            Reconstructed source with the input shape.
        """
        from scipy.fftpack import dct, idct

        n = len(source)

        # Create or retrieve mask
        if self.mask is not None:
            if len(self.mask) == n:
                mask = self.mask
            else:
                # Resample mask to match signal length
                mask = np.interp(
                    np.linspace(0, 1, n), np.linspace(0, 1, len(self.mask)), self.mask
                )
        else:
            # Create lowpass mask if not cached or length changed
            if self._cached_mask is None or self._cached_len != n:
                cutoff = int(n * self.cutoff_fraction)
                mask = np.zeros(n)
                mask[:cutoff] = 1.0
                self._cached_mask = mask
                self._cached_len = n
            else:
                mask = self._cached_mask

        if source.ndim == 1:
            dct_coeffs = dct(source, type=2, norm="ortho")
            dct_filtered = dct_coeffs * mask
            return idct(dct_filtered, type=2, norm="ortho")
        elif source.ndim == 2:
            _, n_epochs = source.shape
            denoised = np.zeros_like(source)
            for ep in range(n_epochs):
                denoised[:, ep] = self._denoise_1d(source[:, ep], mask)
            return denoised
        else:
            raise ValueError(f"Source must be 1D or 2D, got {source.ndim}D")

    def _denoise_1d(self, source, mask):
        from scipy.fftpack import dct, idct

        dct_coeffs = dct(source, type=2, norm="ortho")
        dct_filtered = dct_coeffs * mask
        return idct(dct_filtered, type=2, norm="ortho")
