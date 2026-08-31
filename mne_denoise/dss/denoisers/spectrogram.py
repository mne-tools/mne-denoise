"""Spectrogram bias functions for DSS."""

from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.ndimage import zoom

from .base import LinearDenoiser, NonlinearDenoiser


def _apply_tf_mask(
    data_1d: np.ndarray, mask: np.ndarray, nperseg: int, noverlap: int
) -> np.ndarray:
    """Apply an STFT mask and return a signal with its original length."""
    f, t, Zxx = signal.stft(data_1d, nperseg=nperseg, noverlap=noverlap)

    # Resize mask if needed
    if mask.shape != Zxx.shape:
        zoom_factors = (Zxx.shape[0] / mask.shape[0], Zxx.shape[1] / mask.shape[1])
        mask_2d = zoom(mask, zoom_factors, order=0)  # Nearest/Linear
    else:
        mask_2d = mask

    Zxx_masked = Zxx * mask_2d

    _, reconstructed = signal.istft(Zxx_masked, nperseg=nperseg, noverlap=noverlap)

    # Match length
    if len(reconstructed) > len(data_1d):
        reconstructed = reconstructed[: len(data_1d)]
    elif len(reconstructed) < len(data_1d):
        # Pad with zeros
        padded = np.zeros(len(data_1d))
        padded[: len(reconstructed)] = reconstructed
        reconstructed = padded

    return reconstructed


class SpectrogramBias(LinearDenoiser):
    """Fixed STFT-mask bias for DSS.

    Parameters
    ----------
    mask : ndarray, shape (n_freqs, n_times)
        Time-frequency mask. It is resized with nearest-neighbour interpolation
        when its shape differs from the computed STFT.
    nperseg : int, default=256
        STFT segment length.
    noverlap : int or None, default=None
        Number of overlapping samples. ``None`` uses ``nperseg // 2``.

    Notes
    -----
    Two-dimensional channel-first input is processed channel by channel. For
    three-dimensional input, each ``(n_channels, n_times)`` epoch is processed
    separately.
    """

    def __init__(
        self,
        mask: np.ndarray,
        nperseg: int = 256,
        noverlap: int | None = None,
    ) -> None:
        self.mask = mask
        self.nperseg = nperseg
        self.noverlap = noverlap if noverlap is not None else nperseg // 2

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply a fixed spectrogram mask to all channels.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_times) or (n_channels, n_times, n_epochs)
            Channel-first data. Three-dimensional input is processed one epoch
            at a time.

        Returns
        -------
        biased : ndarray, same shape as ``data``
            Data reconstructed after applying the time-frequency mask.
        """
        # Linear denoisers operate on sensor data (n_ch, n_times)
        if data.ndim == 2:
            return self._apply_2d(data)
        elif data.ndim == 3:
            # (n_ch, n_times, n_epochs)
            n_ch, n_times, n_epochs = data.shape
            biased = np.zeros_like(data)
            for ep in range(n_epochs):
                biased[:, :, ep] = self._apply_2d(data[:, :, ep])
            return biased
        else:
            raise ValueError(f"Data must be 2D or 3D, got {data.ndim}D")

    def _apply_2d(self, data: np.ndarray) -> np.ndarray:
        # Apply strict mask to each channel
        n_ch, n_times = data.shape
        biased = np.zeros_like(data)

        for ch in range(n_ch):
            biased[ch] = _apply_tf_mask(
                data[ch], self.mask, self.nperseg, self.noverlap
            )
        return biased


class SpectrogramDenoiser(NonlinearDenoiser):
    """STFT-mask denoiser for iterative DSS.

    Parameters
    ----------
    threshold_percentile : float, default=90.0
        Percentile of STFT magnitude used as the adaptive threshold; bins above
        it receive gain one.
    nperseg : int, default=256
        STFT segment length.
    noverlap : int or None, default=None
        Number of overlapping samples. ``None`` uses ``nperseg // 2``.
    mask : ndarray or None, default=None
        Fixed time-frequency mask. When supplied, it replaces adaptive mask
        calculation.
    """

    def __init__(
        self,
        threshold_percentile: float = 90.0,
        nperseg: int = 256,
        noverlap: int | None = None,
        mask: np.ndarray | None = None,
    ) -> None:
        self.threshold_percentile = threshold_percentile
        self.nperseg = nperseg
        self.noverlap = noverlap if noverlap is not None else nperseg // 2
        self.mask = mask

    def denoise(self, source: np.ndarray) -> np.ndarray:
        """Compute or apply an STFT mask to a source.

        Parameters
        ----------
        source : ndarray, shape (n_times,) or (n_times, n_epochs)
            Source time series; columns of 2D input are processed separately.

        Returns
        -------
        ndarray
            Reconstructed source with the input shape.
        """
        if source.ndim == 2:
            _, n_epochs = source.shape
            denoised = np.zeros_like(source)
            for ep in range(n_epochs):
                denoised[:, ep] = self._denoise_1d(source[:, ep])
            return denoised
        elif source.ndim == 1:
            return self._denoise_1d(source)
        else:
            raise ValueError(f"Source must be 1D or 2D, got {source.ndim}D")

    def _denoise_1d(self, source: np.ndarray) -> np.ndarray:
        """Compute or use a mask for one source and apply it."""
        # STFT just to calculate mask if adaptive
        if self.mask is None:
            f, t, Zxx = signal.stft(
                source, nperseg=self.nperseg, noverlap=self.noverlap
            )
            # Adaptive magnitude-based mask
            magnitude = np.abs(Zxx)
            threshold = np.percentile(magnitude, self.threshold_percentile)
            computed_mask = (magnitude > threshold).astype(float)
        else:
            computed_mask = self.mask

        # Apply mask using shared logic
        return _apply_tf_mask(source, computed_mask, self.nperseg, self.noverlap)
