"""Periodic bias functions for DSS."""

from __future__ import annotations

import numpy as np
from scipy import ndimage, signal

from .base import LinearDenoiser, NonlinearDenoiser


class PeakFilterBias(LinearDenoiser):
    """Second-order IIR peak-filter bias for DSS.

    Parameters
    ----------
    freq : float
        Target frequency in Hz.
    sfreq : float
        Sampling frequency in Hz.
    q_factor : float, default=30.0
        Quality factor passed to :func:`scipy.signal.iirpeak`; approximately,
        ``bandwidth = freq / q_factor``.
    order : int, default=2
        Accepted and stored for API compatibility. The current implementation
        always designs the filter with :func:`scipy.signal.iirpeak` and this
        parameter does not change it.
    """

    def __init__(
        self,
        freq: float,
        sfreq: float,
        *,
        q_factor: float = 30.0,
        order: int = 2,
    ) -> None:
        self.freq = freq
        self.sfreq = sfreq
        self.q_factor = q_factor
        self.order = order

        # Design peak filter
        self._sos = self._design_peak_filter()

    def _design_peak_filter(self) -> np.ndarray:
        """Design IIR peak filter using second-order sections."""
        nyq = self.sfreq / 2

        if self.freq >= nyq:
            raise ValueError(
                f"Target frequency ({self.freq} Hz) must be < Nyquist ({nyq} Hz)"
            )

        # Normalized frequency
        w0 = self.freq / nyq

        # Bandwidth from Q factor
        w0 / self.q_factor

        # Design peak filter using iirpeak
        b, a = signal.iirpeak(w0, self.q_factor)
        sos = signal.tf2sos(b, a)

        return sos

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply the peak-filter bias.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_times) or (n_channels, n_times, n_epochs)
            Channel-first data.

        Returns
        -------
        ndarray
            Peak-filtered data with the input shape.
        """
        if data.ndim not in (2, 3):
            raise ValueError(f"Data must be 2D or 3D, got {data.ndim}D")
        # sosfiltfilt filters along `axis` independently of the other axes,
        # so a 3D (n_channels, n_times, n_epochs) array needs no per-epoch loop.
        return signal.sosfiltfilt(self._sos, data, axis=1)


class CombFilterBias(LinearDenoiser):
    """Comb-filter bias for harmonic frequencies.

    Parameters
    ----------
    fundamental_freq : float
        Fundamental frequency in Hz.
    sfreq : float
        Sampling frequency in Hz.
    n_harmonics : int, default=3
        Number of harmonics to consider.
    q_factor : float, default=30.0
        Quality factor passed to each peak filter.
    q_mode : {"fixed", "proportional"}, default="fixed"
        With ``"fixed"``, Q is constant and ``bandwidth = frequency / Q``
        therefore increases in absolute width at higher harmonics. With
        ``"proportional"``, Q is multiplied by the harmonic number, giving
        approximately constant absolute bandwidth.
    weights : array-like or None, default=None
        Harmonic weights. ``None`` uses inverse harmonic-number weights.

    Notes
    -----
    The implementation sums peak-filter outputs and excludes harmonics at or
    above 95 percent of Nyquist.
    """

    def __init__(
        self,
        fundamental_freq: float,
        sfreq: float,
        *,
        n_harmonics: int = 3,
        q_factor: float = 30.0,
        q_mode: str = "fixed",
        weights: np.ndarray | None = None,
    ) -> None:
        self.fundamental_freq = fundamental_freq
        self.sfreq = sfreq
        self.n_harmonics = n_harmonics
        self.q_factor = q_factor

        # Validate q_mode
        allowed_q_modes = ("fixed", "proportional")
        if q_mode not in allowed_q_modes:
            raise ValueError(f"q_mode must be one of {allowed_q_modes}, got {q_mode!r}")
        self.q_mode = q_mode

        # Set up weights
        if weights is None:
            self.weights = np.array([1.0 / h for h in range(1, n_harmonics + 1)])
        else:
            self.weights = np.asarray(weights)
            if len(self.weights) != n_harmonics:
                raise ValueError(
                    f"weights length ({len(self.weights)}) must match "
                    f"n_harmonics ({n_harmonics})"
                )

        # Create peak filters for each valid harmonic
        self._peak_filters: list[tuple[np.ndarray, float]] = []
        self._create_harmonic_filters()

    def _create_harmonic_filters(self) -> None:
        """Create peak filter for each harmonic within Nyquist."""
        nyq = self.sfreq / 2

        for h in range(1, self.n_harmonics + 1):
            freq = self.fundamental_freq * h

            if freq >= nyq * 0.95:
                continue  # Skip harmonics too close to Nyquist

            w0 = freq / nyq
            weight = self.weights[h - 1]

            # Proportional Q scales linearly with harmonic number,
            # maintaining constant absolute bandwidth across harmonics
            q = self.q_factor * h if self.q_mode == "proportional" else self.q_factor

            b, a = signal.iirpeak(w0, q)
            sos = signal.tf2sos(b, a)
            self._peak_filters.append((sos, weight))

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply the summed harmonic peak-filter bias.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_times) or (n_channels, n_times, n_epochs)
            Channel-first data.

        Returns
        -------
        ndarray
            Weighted harmonic content with the input shape.
        """
        if len(self._peak_filters) == 0:
            raise ValueError("No valid harmonics within Nyquist frequency")
        if data.ndim not in (2, 3):
            raise ValueError(f"Data must be 2D or 3D, got {data.ndim}D")

        # sosfiltfilt filters along `axis` independently of the other axes,
        # so a 3D (n_channels, n_times, n_epochs) array needs no per-epoch loop.
        biased = np.zeros_like(data)
        for sos, weight in self._peak_filters:
            biased += weight * signal.sosfiltfilt(sos, data, axis=1)

        return biased

    @property
    def harmonic_frequencies(self) -> list[float]:
        """Return list of harmonic frequencies being filtered."""
        nyq = self.sfreq / 2
        return [
            self.fundamental_freq * h
            for h in range(1, self.n_harmonics + 1)
            if self.fundamental_freq * h < nyq * 0.95
        ]


class QuasiPeriodicDenoiser(NonlinearDenoiser):
    """Cycle-template denoiser for quasi-periodic source signals.

    Peaks are detected on the absolute source using the configured percentile and
    minimum-distance rule. The resulting cycles are rescaled to a common duration,
    averaged into a template, optionally smoothed, and mapped back to each cycle
    with its original length and amplitude scaling. The reconstructed cycle
    structure is returned; fewer than three detected peaks leave the source
    unchanged.

    Parameters
    ----------
    peak_distance : int, default=100
        Minimum peak distance in samples; values below 10 are set to 10.
    peak_height_percentile : float, default=75.0
        Percentile used as the absolute-source peak threshold.
    warp_length : int or None, default=None
        Common cycle length. ``None`` uses the median detected cycle length.
    smooth_template : bool, default=True
        Smooth the averaged template with a uniform filter.
    """

    def __init__(
        self,
        peak_distance: int = 100,
        peak_height_percentile: float = 75.0,
        *,
        warp_length: int | None = None,
        smooth_template: bool = True,
    ) -> None:
        self.peak_distance = max(10, peak_distance)
        self.peak_height_percentile = peak_height_percentile
        self.warp_length = warp_length
        self.smooth_template = smooth_template

    def denoise(self, source: np.ndarray) -> np.ndarray:
        """Apply the cycle-template denoiser.

        Parameters
        ----------
        source : ndarray, shape (n_times,) or (n_times, n_epochs)
            One source or sources arranged by columns.

        Returns
        -------
        ndarray
            Reconstructed source with the input shape.
        """
        if source.ndim == 1:
            return self._denoise_1d(source)
        elif source.ndim == 2:
            n_times, n_epochs = source.shape
            denoised = np.zeros_like(source)
            for ep in range(n_epochs):
                denoised[:, ep] = self._denoise_1d(source[:, ep])
            return denoised
        else:
            raise ValueError(f"Source must be 1D or 2D, got {source.ndim}D")

    def _denoise_1d(self, source: np.ndarray) -> np.ndarray:
        """Apply quasi-periodic denoising to 1D source."""
        n_samples = len(source)

        # Step 1: Detect peaks
        height_threshold = np.percentile(np.abs(source), self.peak_height_percentile)
        peaks, _ = signal.find_peaks(
            np.abs(source),
            height=height_threshold,
            distance=self.peak_distance,
        )

        if len(peaks) < 3:
            # Not enough cycles, return original
            return source

        # Step 2: Determine cycle boundaries (midpoints between peaks)
        boundaries = np.zeros(len(peaks) + 1, dtype=int)
        boundaries[0] = 0
        boundaries[-1] = n_samples
        for i in range(1, len(peaks)):
            boundaries[i] = (peaks[i - 1] + peaks[i]) // 2

        # Step 3: Extract cycles and determine warp length
        cycles = []
        cycle_lengths = []
        for i in range(len(peaks)):
            start = boundaries[i]
            end = boundaries[i + 1]
            if end > start:
                cycles.append(source[start:end])
                cycle_lengths.append(end - start)

        if len(cycles) < 2:
            return source

        # Warp length: use provided or median
        if self.warp_length is not None:
            warp_len = self.warp_length
        else:
            warp_len = int(np.median(cycle_lengths))
        warp_len = max(10, warp_len)

        # Step 4: Time-warp all cycles to common length and average
        warped_cycles = []
        for cycle in cycles:
            if len(cycle) >= 3:
                # Resample to warp_len
                warped = np.interp(
                    np.linspace(0, 1, warp_len), np.linspace(0, 1, len(cycle)), cycle
                )
                warped_cycles.append(warped)

        if len(warped_cycles) < 2:
            return source

        # Average to create template
        template = np.mean(warped_cycles, axis=0)

        # Optional smoothing
        if self.smooth_template:
            smooth_window = max(3, warp_len // 20)
            template = ndimage.uniform_filter1d(
                template, size=smooth_window, mode="reflect"
            )

        # Step 5: Replace each cycle with time-warped template
        denoised = np.zeros_like(source)
        for i, cycle in enumerate(cycles):
            start = boundaries[i]
            end = boundaries[i + 1]
            cycle_len = end - start

            if cycle_len >= 3:
                # Warp template back to original cycle length
                warped_template = np.interp(
                    np.linspace(0, 1, cycle_len), np.linspace(0, 1, warp_len), template
                )
                # Match amplitude to original cycle
                scale = np.std(cycle) / (np.std(warped_template) + 1e-15)
                offset = np.mean(cycle) - np.mean(warped_template) * scale
                denoised[start:end] = warped_template * scale + offset

        return denoised
