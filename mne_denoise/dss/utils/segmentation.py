"""Data segmentation utilities for DSS.

Provides strategies for splitting continuous data into segments based on
statistical properties.  Used by :class:`~mne_denoise.dss.linear.DSS` in
segmented mode to handle non-stationary artifacts.

Available segmenters
--------------------
- :class:`CovarianceSegmenter` – splits at covariance-stationarity boundaries.
- :class:`FixedWindowSegmenter` – splits into fixed-length windows.

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)

References
----------
.. [1] Klug, M., & Kloosterman, N. A. (2022). Zapline-plus: A Zapline
       extension for automatic and adaptive removal of frequency-specific
       noise artifacts in M/EEG. Human Brain Mapping, 43(9), 2743-2758.
"""

from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.signal import find_peaks

from ..._covariance import compute_covariance
from ..._filtering import design_butter_sos

# ---------------------------------------------------------------------------
# Covariance-based segmenter (generalised from ZapLine-plus)
# ---------------------------------------------------------------------------


class CovarianceSegmenter:
    """Segment data based on covariance stationarity.

    Identifies boundaries where the spatial covariance matrix changes
    significantly, indicating non-stationary noise characteristics.
    Based on the ZapLine-plus segmentation algorithm [1]_.

    Parameters
    ----------
    sfreq : float
        Sampling frequency in Hz.
    min_chunk_len : float, default=30.0
        Minimum segment length in seconds.
    cov_win_len : float, default=1.0
        Window length for covariance computation in seconds.
    bandpass : tuple of (float, float) | None, default=None
        Bandpass filter range ``(f_low, f_high)`` in Hz to focus the
        stationarity analysis on a specific frequency band.  Useful for
        frequency-specific artifacts.  If ``None``, uses unfiltered data.
    prominence : float, default=0.5
        Sensitivity of boundary detection, expressed as a multiple of the
        standard deviation of the successive covariance distances: a
        distance peak must stand out by ``prominence * std(distances)`` to
        be accepted as a boundary.  Lower values split more eagerly; higher
        values require a more decisive change in covariance structure.
        Because the threshold is relative to the spread of the distances,
        stationary data still produces peaks, so raise this (or use
        :class:`FixedWindowSegmenter`) if you are getting spurious splits.
        Default 0.5 reproduces ZapLine-plus.

    References
    ----------
    .. [1] Klug & Kloosterman (2022). Zapline-plus …
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
        """Segment data into stationary chunks.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_times)
            Input data.

        Returns
        -------
        segments : list of (int, int)
            List of ``(start_sample, end_sample)`` tuples.
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

    Simple segmentation strategy that divides data into equal-length chunks.
    The last chunk may be merged with the previous one if it is too short.

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
        """Segment data into fixed-length windows.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_times)
            Input data.

        Returns
        -------
        segments : list of (int, int)
            List of ``(start_sample, end_sample)`` tuples.
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
