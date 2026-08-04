"""Temporal bias functions for DSS.

Implements time-shift and smoothing biases for extracting temporally
extended structure (slow waves, autocorrelated signals).

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)

References
----------
.. [1] de Cheveigné, A. (2010). Time-shift denoising source separation.
       Journal of Neuroscience Methods, 189(1), 113-120.
.. [2] de Cheveigné, A. & Simon, J.Z. (2008). Denoising based on spatial filtering.
       Journal of Neuroscience Methods, 171(2), 331-339.
.. [3] de Cheveigné, A. (2020). ZapLine: A simple and effective method to remove
       power line artifacts. NeuroImage, 207, 116356. (Period-matched
       smooth/residual decomposition: spatially clean only the residual branch
       and add the smooth branch back.)
"""

from __future__ import annotations

import numpy as np

from .base import LinearDenoiser, NonlinearDenoiser


class TimeShiftBias(LinearDenoiser):
    """Time-shift bias for extracting autocorrelated signals.

    Creates a bias by averaging time-shifted versions of the data,
    emphasizing signals that are predictable across time lags.

    Parameters
    ----------
    shifts : int or array-like
        If int, use lags from 1 to shifts.
        If array, use specified lag values in samples.
        Default 10.
    method : str
        Method for constructing bias:
        - 'autocorrelation': Average of shifted versions (default)
        - 'prediction': Weighted average (closer lags weighted more)

    Examples
    --------
    >>> bias = TimeShiftBias(shifts=[1, 2, 5, 10], method="prediction")
    >>> biased_data = bias.apply(data)

    See Also
    --------
    SmoothingBias : Bias for low-frequency signals.
    """

    def __init__(
        self,
        shifts: int | np.ndarray = 10,
        method: str = "autocorrelation",
    ) -> None:
        self.shifts = shifts
        self.method = method

        # Resolve shifts to array
        if isinstance(shifts, int):
            self._shift_array = np.arange(1, shifts + 1)
        else:
            self._shift_array = np.asarray(shifts)

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
        # Handle 3D data
        orig_shape = data.shape
        if data.ndim == 3:
            n_ch, n_times, n_epochs = data.shape
            data_2d = data.reshape(n_ch, -1)
        else:
            data_2d = data

        if self.method == "autocorrelation":
            biased_2d = self._autocorrelation_bias(data_2d)
        elif self.method == "prediction":
            biased_2d = self._prediction_bias(data_2d)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Restore shape
        if data.ndim == 3:
            return biased_2d.reshape(orig_shape)
        return biased_2d

    def _autocorrelation_bias(self, data: np.ndarray) -> np.ndarray:
        """Average of time-shifted versions."""
        n_channels, n_samples = data.shape
        shifts = self._shift_array
        max_shift = np.max(np.abs(shifts))

        if max_shift >= n_samples // 2:
            raise ValueError(
                f"Max shift ({max_shift}) too large for data length ({n_samples})"
            )

        valid_start = max_shift
        valid_end = n_samples - max_shift
        valid_length = valid_end - valid_start

        accumulated = np.zeros((n_channels, valid_length))
        for shift in shifts:
            shifted = data[:, valid_start + shift : valid_end + shift]
            accumulated += shifted

        biased = accumulated / len(shifts)

        # Pad to original length
        biased_full = np.zeros_like(data)
        biased_full[:, valid_start:valid_end] = biased
        return biased_full

    def _prediction_bias(self, data: np.ndarray) -> np.ndarray:
        """Weighted average (closer lags weighted more)."""
        n_channels, n_samples = data.shape
        shifts = self._shift_array
        max_shift = np.max(np.abs(shifts))

        valid_start = max_shift
        valid_end = n_samples - max_shift
        valid_length = valid_end - valid_start

        accumulated = np.zeros((n_channels, valid_length))
        total_weight = 0

        for shift in shifts:
            weight = 1.0 / max(abs(shift), 1)
            shifted = data[:, valid_start + shift : valid_end + shift]
            accumulated += weight * shifted
            total_weight += weight

        biased = accumulated / total_weight

        biased_full = np.zeros_like(data)
        biased_full[:, valid_start:valid_end] = biased
        return biased_full


class SmoothingBias(LinearDenoiser):
    """Unified temporal smoothing bias (Moving Average).

    Uses a boxcar moving average filter to smooth the data. When used to split
    the signal into a smooth branch and a residual (``data - smooth``), fitting
    DSS on the residual and adding the smooth branch back follows ZapLine's
    period-matched decomposition (de Cheveigné, 2020): with
    ``window = round(sfreq / f_line)`` the smoother has zeros at ``f_line`` and
    its harmonics, so the residual concentrates the narrowband artifact.

    Parameters
    ----------
    window : int
        Smoothing window size in samples.
        Note: If you want to cancel a specific frequency (e.g. 50Hz line noise),
        set window = int(sfreq / 50).
    iterations : int
        Number of smoothing passes. Repeated smoothing approximates a Gaussian filter
        and provides sharper frequency cutoff. Default 1.

    Examples
    --------
    >>> bias = SmoothingBias(window=20)  # Simple smoothing
    >>> biased = bias.apply(data)

    >>> # To remove 50Hz line noise (Period smoothing)
    >>> bias = SmoothingBias(window=int(1000 / 50), iterations=1)
    """

    def __init__(self, window: int = 10, iterations: int = 1) -> None:
        self.window = window
        self.iterations = iterations

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply smoothing bias.

        Uses a causal running-mean filter:
        ``y[t] = mean(x[t-W+1 : t+1])`` for ``t >= W``, with an expanding
        window for the first ``W`` samples.  Repeated ``iterations`` passes
        approximate a Gaussian kernel.
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
    """DCT domain denoiser (MATLAB denoise_dct.m).

    Applies a mask in the DCT (Discrete Cosine Transform) domain.
    Useful for temporal smoothness without explicit bandpass.

    Parameters
    ----------
    mask : ndarray or None
        DCT domain mask. Must have same length as signal, or will be
        expanded/truncated. If None, creates lowpass mask.
        If mask is None, this fraction of DCT coefficients are kept.
        Default 0.5 (lowpass, keep first 50% of coefficients).

    cutoff_fraction : float
        Fraction of DCT coefficients to keep. If mask is None,
        this fraction of DCT coefficients are kept.
        Default 0.5 (lowpass, keep first 50% of coefficients).

    Examples
    --------
    >>> from mne_denoise.dss.denoisers import DCTDenoiser
    >>> # Keep only the lowest 20% of DCT coefficients (smooth signal)
    >>> denoiser = DCTDenoiser(cutoff_fraction=0.2)
    >>> smooth_source = denoiser.denoise(source)


    References
    ----------
    Särelä & Valpola (2005). Section 4.1.2 "DENOISING BASED ON FREQUENCY CONTENT"
    """

    def __init__(
        self, mask: np.ndarray | None = None, cutoff_fraction: float = 0.5
    ) -> None:
        self.mask = mask
        self.cutoff_fraction = cutoff_fraction
        self._cached_mask = None
        self._cached_len = None

    def denoise(self, source: np.ndarray) -> np.ndarray:
        """Apply DCT filtering."""
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
