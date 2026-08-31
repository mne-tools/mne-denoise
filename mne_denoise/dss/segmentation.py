"""DSS segmentation helpers."""

from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.signal import find_peaks

from .._covariance import compute_covariance
from .._filtering import design_butter_sos

__all__ = ["CovarianceSegmenter", "FixedWindowSegmenter"]

# ---------------------------------------------------------------------------
# Covariance-based segmenter (generalised from ZapLine-plus)
# ---------------------------------------------------------------------------


class CovarianceSegmenter:
    """Segment data where windowed covariance changes.

    Parameters
    ----------
    sfreq : float
        Sampling frequency in Hz.
    min_chunk_len : float, default=30.0
        Minimum segment length in seconds.
    cov_win_len : float, default=1.0
        Covariance-window length in seconds.
    bandpass : tuple of float or None, default=None
        Optional analysis band ``(low, high)`` in Hz.
    prominence : float, default=0.5
        Covariance-distance peak prominence multiplier.

    Notes
    -----
    The segmentation strategy is based on the covariance-stationarity approach
    used by ZapLine-plus :footcite:p:`klug_kloosterman2022_zapline_plus`.

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        sfreq: float,
        min_chunk_len: float = 30.0,
        cov_win_len: float = 1.0,
        bandpass: tuple[float, float] | None = None,
        prominence: float = 0.5,
    ) -> None:
        self.sfreq = float(sfreq)
        self.min_chunk_len = min_chunk_len
        self.cov_win_len = cov_win_len
        self.bandpass = bandpass
        self.prominence = float(prominence)

    def segment(self, data: np.ndarray) -> list[tuple[int, int]]:
        """Return ``(start_sample, end_sample)`` segments from channel-first data.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_times)
            Input data.

        Returns
        -------
        list of tuple of int
            Half-open sample intervals.
        """
        n_channels, n_times = data.shape

        # Optional bandpass filter to focus analysis
        if self.bandpass is not None:
            f_low, f_high = self.bandpass
            sos = design_butter_sos(4, [f_low, f_high], "bandpass", self.sfreq)
            data_filt = signal.sosfiltfilt(sos, data, axis=1)
        else:
            data_filt = data

        # Compute sliding-window covariance series
        n_win = int(self.cov_win_len * self.sfreq)
        if n_win > n_times:
            return [(0, n_times)]

        n_steps = n_times // n_win

        covs = []
        for i in range(n_steps):
            start = i * n_win
            end = start + n_win
            chunk = data_filt[:, start:end]
            cov = compute_covariance(chunk)
            tr = np.trace(cov)
            if tr > 1e-20:
                cov = cov / tr
            covs.append(cov)

        covs = np.array(covs)

        # Successive Frobenius distances
        dists = np.array(
            [
                np.linalg.norm(covs[i] - covs[i + 1], ord="fro")
                for i in range(len(covs) - 1)
            ]
        )

        if len(dists) == 0:
            return [(0, n_times)]

        # Detect peaks (boundary candidates)
        min_distance = max(1, int(self.min_chunk_len * self.sfreq / n_win))
        peak_indices, _ = find_peaks(
            dists, prominence=np.std(dists) * self.prominence, distance=min_distance
        )
        boundary_indices = (peak_indices + 1) * n_win

        # Enforce minimum segment length
        valid_boundaries = [0]
        last_boundary = 0
        min_samples = int(self.min_chunk_len * self.sfreq)

        for b in boundary_indices:
            if (b - last_boundary) >= min_samples:
                valid_boundaries.append(b)
                last_boundary = b

        if (n_times - last_boundary) < min_samples and len(valid_boundaries) > 1:
            valid_boundaries.pop()

        valid_boundaries.append(n_times)

        return [
            (valid_boundaries[i], valid_boundaries[i + 1])
            for i in range(len(valid_boundaries) - 1)
        ]


# ---------------------------------------------------------------------------
# Fixed-window segmenter
# ---------------------------------------------------------------------------


class FixedWindowSegmenter:
    """Segment data into fixed-length windows.

    Parameters
    ----------
    sfreq : float
        Sampling frequency in Hz.
    window_len : float, default=30.0
        Window length in seconds.
    """

    def __init__(self, sfreq: float, window_len: float = 30.0) -> None:
        self.sfreq = float(sfreq)
        self.window_len = window_len

    def segment(self, data: np.ndarray) -> list[tuple[int, int]]:
        """Return fixed ``(start_sample, end_sample)`` windows.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_times)
            Channel-first data.

        Returns
        -------
        list of tuple of int
            Half-open sample intervals.

        Notes
        -----
        The final window is merged into the previous window when it is shorter than
        half the requested length.
        """
        n_times = data.shape[1]
        win_samples = int(self.window_len * self.sfreq)

        if win_samples >= n_times:
            return [(0, n_times)]

        segments: list[tuple[int, int]] = []
        start = 0
        while start < n_times:
            end = min(start + win_samples, n_times)
            # Merge a tiny trailing segment (< 50 % of window) into the last
            if end - start < win_samples // 2 and segments:
                prev_start, _ = segments[-1]
                segments[-1] = (prev_start, end)
            else:
                segments.append((start, end))
            start = end

        return segments
