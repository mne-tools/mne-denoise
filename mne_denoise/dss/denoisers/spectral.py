"""Spectral bias functions for DSS."""

from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.fft import fft, ifft

from ..._filtering import design_butter_sos
from .base import LinearDenoiser


class BandpassBias(LinearDenoiser):
    """Bandpass-filter bias for DSS.

    Parameters
    ----------
    freq_band : tuple of float
        Lower and upper passband edges in Hz.
    sfreq : float
        Sampling frequency in Hz.
    order : int, default=4
        Butterworth filter order.
    method : {"butter"}, default="butter"
        Filter design method. Only the Butterworth design is implemented.
    """

    def __init__(
        self,
        freq_band: tuple[float, float],
        sfreq: float,
        *,
        order: int = 4,
        method: str = "butter",
    ) -> None:
        self.freq_band = freq_band
        self.sfreq = sfreq
        self.order = order
        self.method = method

        # Pre-compute filter coefficients
        self._b: np.ndarray | None = None
        self._a: np.ndarray | None = None
        self._sos: np.ndarray | None = None
        self._design_filter()

    def _design_filter(self) -> None:
        """Design the bandpass filter."""
        low, high = self.freq_band
        nyq = self.sfreq / 2

        if low <= 0:
            raise ValueError(f"Low frequency must be > 0, got {low}")
        if high >= nyq:
            raise ValueError(f"High frequency ({high}) must be < Nyquist ({nyq})")

        if self.method == "butter":
            # Use second-order sections for stability
            self._sos = design_butter_sos(self.order, [low, high], "band", self.sfreq)
        else:
            raise ValueError(f"Unknown filter method: {self.method}")

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply the bandpass bias.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_times) or (n_channels, n_times, n_epochs)
            Channel-first data.

        Returns
        -------
        ndarray
            Bandpass-filtered data with the input shape.
        """
        if data.ndim not in (2, 3):
            raise ValueError(f"Data must be 2D or 3D, got {data.ndim}D")
        # sosfiltfilt filters along `axis` independently of the other axes,
        # so a 3D (n_channels, n_times, n_epochs) array needs no per-epoch loop.
        return signal.sosfiltfilt(self._sos, data, axis=1)


class LineNoiseBias(LinearDenoiser):
    """Line-frequency bias using IIR or FFT selection.

    Parameters
    ----------
    freq : float
        Fundamental line frequency in Hz.
    sfreq : float
        Sampling frequency in Hz.
    method : {"fft", "iir"}, default="fft"
        Use exact FFT-bin selection or a narrow IIR bandpass.
    n_harmonics : int or None, default=None
        Number of harmonics in FFT mode; ``None`` includes valid harmonics below
        Nyquist.
    bandwidth : float, default=1.0
        IIR bandpass width in Hz.
    order : int, default=4
        IIR bandpass order.
    nfft : int, default=1024
        FFT block length.
    overlap : float, default=0.5
        Accepted for API compatibility; the current FFT implementation does not
        use this value.

    Notes
    -----
    FFT mode retains one rounded positive-frequency bin per harmonic and its
    conjugate bin. It processes rectangular, non-overlapping blocks of length
    ``nfft``; a trailing short block is zero-padded and truncated on output.
    Three-dimensional channel-first input is processed epoch by epoch. IIR mode
    uses :class:`BandpassBias` around the fundamental frequency.
    """

    def __init__(
        self,
        freq: float,
        sfreq: float,
        *,
        method: str = "fft",
        n_harmonics: int | None = None,
        bandwidth: float = 1.0,
        order: int = 4,
        nfft: int = 1024,
        overlap: float = 0.5,
    ) -> None:
        self.freq = freq
        self.sfreq = sfreq
        self.method = method
        self.n_harmonics = n_harmonics
        self.bandwidth = bandwidth
        self.order = order
        self.nfft = nfft
        self.overlap = overlap

        if method == "iir":
            low = freq - bandwidth / 2
            high = freq + bandwidth / 2
            self._bandpass = BandpassBias(
                freq_band=(low, high), sfreq=sfreq, order=order
            )
        elif method == "fft":
            # FFT setup logic
            nyquist = sfreq / 2
            if n_harmonics is None:
                self.n_harmonics = int(np.floor(nyquist / freq))
            else:
                max_harmonics = int(np.floor(nyquist / freq))
                self.n_harmonics = min(n_harmonics, max_harmonics)

            self._harmonic_freqs = np.array(
                [freq * (h + 1) for h in range(self.n_harmonics)]
            )
            self._harmonic_freqs = self._harmonic_freqs[self._harmonic_freqs < nyquist]
        else:
            raise ValueError(f"Unknown method '{method}', must be 'fft' or 'iir'.")

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply the configured line-frequency bias.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_times) or (n_channels, n_times, n_epochs)
            Channel-first data.

        Returns
        -------
        ndarray
            Selected line-frequency content with the input shape.
        """
        if self.method == "iir":
            return self._bandpass.apply(data)
        elif self.method == "fft":
            return self._apply_fft(data)
        return data

    def _apply_fft(self, data: np.ndarray) -> np.ndarray:
        """Apply FFT-based harmonic bias."""
        if data.ndim == 3:
            n_channels, n_times, n_epochs = data.shape
            biased = np.zeros_like(data)
            for ep in range(n_epochs):
                biased[:, :, ep] = self._apply_fft_2d(data[:, :, ep])
            return biased
        elif data.ndim == 2:
            return self._apply_fft_2d(data)
        else:
            raise ValueError(f"Data must be 2D or 3D, got {data.ndim}D")

    def _get_target_indices(self, nfft: int) -> list:
        """Return positive and conjugate FFT-bin indices for the harmonics."""
        target_indices = []

        for f in self._harmonic_freqs:
            # Positive-frequency bin: round(f / sfreq * nfft)
            idx = int(round(f / self.sfreq * nfft))
            if 0 <= idx < nfft and idx not in target_indices:
                target_indices.append(idx)

            # Negative-frequency (conjugate symmetric) bin
            idx_neg = nfft - idx
            if 0 <= idx_neg < nfft and idx_neg not in target_indices:
                target_indices.append(idx_neg)

        return target_indices

    def _apply_fft_2d(self, data: np.ndarray) -> np.ndarray:
        """Apply the FFT bias to non-overlapping rectangular blocks."""
        n_channels, n_times = data.shape

        # Use data length or nfft, whichever is smaller
        actual_nfft = min(self.nfft, n_times)
        target_indices = self._get_target_indices(actual_nfft)

        biased = np.zeros_like(data)
        pos = 0

        while pos < n_times:
            end = min(pos + actual_nfft, n_times)
            block_len = end - pos

            # FFT (zero-pads short blocks automatically)
            X = fft(data[:, pos:end], n=actual_nfft, axis=1)
            X_bias = np.zeros_like(X)
            for idx in target_indices:
                X_bias[:, idx] = X[:, idx]
            y = np.real(ifft(X_bias, axis=1))

            biased[:, pos:end] = y[:, :block_len]
            pos = end

        return biased
