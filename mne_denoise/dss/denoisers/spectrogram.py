"""Spectrogram-based bias functions for DSS.

Implements denoising based on time-frequency representation masking.

Includes both Linear (SpectrogramBias) and Nonlinear (SpectrogramDenoiser)
implementations.

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)

References
----------
.. [1] Särelä & Valpola (2005). Denoising Source Separation. J. Mach. Learn. Res., 6, 233-272.
"""

from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.ndimage import zoom

from .base import LinearDenoiser, NonlinearDenoiser


def _apply_tf_mask(
    data_1d: np.ndarray, mask: np.ndarray, nperseg: int, noverlap: int
) -> np.ndarray:
    """Apply TF mask to 1D signal."""
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
    """Linear spectrogram bias (Section 4.1.3).

    Applies a FIXED time-frequency mask to the data. This is a linear operation
    used to define the signal subspace in the initialization or linear DSS step.

    Parameters
    ----------
    mask : ndarray, shape (n_freqs, n_times)
        The fixed 2D mask to apply. Must be provided for linear biasing.
    nperseg : int
        Segment length for STFT. Default 256.
    noverlap : int, optional
        Overlap between segments. Default nperseg // 2.

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.dss.denoisers import SpectrogramBias
    >>> mask = np.ones((128, 1000))
    >>> bias = SpectrogramBias(mask)
    >>> data = np.random.randn(128, 1000)
    >>> biased = bias.apply(data)

    See Also
    --------
    SpectrogramDenoiser : Adaptive nonlinear version.

    References
    ----------
    Särelä & Valpola (2005). Section 4.1.3 "SPECTROGRAM DENOISING"
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
        """Apply fixed spectrogram mask to all channels."""
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
    """Adaptive/Nonlinear spectrogram denoiser (Section 4.1.3).

    Applies masking in the time-frequency domain. This version is ADAPTIVE,
    calculating the mask from the source estimate itself at each iteration.
    This makes it distinct from the Linear SpectrogramBias.

    Parameters
    ----------
    threshold_percentile : float
        For adaptive masking, threshold below this percentile. Default 90.
        Higher percentile = sparser signal (more aggressive denoising).
    nperseg : int
        Segment length for STFT. Default 256.
    noverlap : int, optional
        Overlap between segments. Default nperseg // 2.
    mask : ndarray, shape (n_freqs, n_times), optional
        Optional FIXED mask to use instead of adaptive (hybrid mode).

    Examples
    --------
    >>> from mne_denoise.dss.denoisers import SpectrogramDenoiser
    >>> # Retain only the strongest 10% of TF-bins (aggressive denoising)
    >>> denoiser = SpectrogramDenoiser(threshold_percentile=90)
    >>> denoised = denoiser.denoise(source)

    See Also
    --------
    SpectrogramBias : Fixed linear version.

    References
    ----------
    Särelä & Valpola (2005). Section 4.1.3 "SPECTROGRAM DENOISING"

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
        """Apply adaptive 2D spectrogram masking."""
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
        """Process 1D source."""
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
