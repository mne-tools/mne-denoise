"""Stationarity-aware and fixed-window segmentation for adaptive DSS."""

from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.signal import find_peaks

from .covariance import compute_covariance


class CovarianceSegmenter:
    """Split continuous data where normalized spatial covariance changes.

    Parameters
    ----------
    sfreq : float
        Sampling frequency in Hz.
    min_chunk_len : float
        Minimum segment duration in seconds.
    cov_win_len : float
        Covariance-window duration in seconds.
    bandpass : tuple | None
        Optional band used only to find stationarity boundaries.
    """

    def __init__(
        self,
        sfreq: float,
        min_chunk_len: float = 30.0,
        cov_win_len: float = 1.0,
        bandpass: tuple[float, float] | None = None,
    ) -> None:
        self.sfreq = float(sfreq)
        self.min_chunk_len = float(min_chunk_len)
        self.cov_win_len = float(cov_win_len)
        self.bandpass = bandpass

    def segment(self, data: np.ndarray) -> list[tuple[int, int]]:
        data = np.asarray(data, dtype=float)
        if data.ndim != 2:
            raise ValueError(f"segmenter requires 2D data, got shape {data.shape}")
        n_times = data.shape[1]
        if self.bandpass is not None:
            low, high = self.bandpass
            sos = signal.butter(
                4, [low, high], btype="bandpass", fs=self.sfreq, output="sos"
            )
            working = signal.sosfiltfilt(sos, data, axis=1)
        else:
            working = data
        window = max(2, int(round(self.cov_win_len * self.sfreq)))
        if window > n_times:
            return [(0, n_times)]
        count = n_times // window
        covariances = []
        for index in range(count):
            chunk = working[:, index * window : (index + 1) * window]
            covariance = compute_covariance(chunk)
            trace = float(np.trace(covariance))
            covariances.append(covariance / trace if trace > 1e-20 else covariance)
        distances = np.asarray(
            [
                np.linalg.norm(covariances[i] - covariances[i + 1], ord="fro")
                for i in range(len(covariances) - 1)
            ]
        )
        if distances.size == 0:
            return [(0, n_times)]
        minimum_windows = max(
            1, int(round(self.min_chunk_len * self.sfreq / window))
        )
        peaks, _ = find_peaks(
            distances,
            prominence=float(np.std(distances)) * 0.5,
            distance=minimum_windows,
        )
        candidates = (peaks + 1) * window
        minimum_samples = int(round(self.min_chunk_len * self.sfreq))
        boundaries = [0]
        for candidate in candidates:
            if candidate - boundaries[-1] >= minimum_samples:
                boundaries.append(int(candidate))
        if n_times - boundaries[-1] < minimum_samples and len(boundaries) > 1:
            boundaries.pop()
        boundaries.append(n_times)
        return list(zip(boundaries[:-1], boundaries[1:]))


class FixedWindowSegmenter:
    """Split continuous data into fixed-duration windows."""

    def __init__(self, sfreq: float, window_len: float = 30.0) -> None:
        self.sfreq = float(sfreq)
        self.window_len = float(window_len)

    def segment(self, data: np.ndarray) -> list[tuple[int, int]]:
        data = np.asarray(data)
        if data.ndim != 2:
            raise ValueError(f"segmenter requires 2D data, got shape {data.shape}")
        n_times = data.shape[1]
        window = max(1, int(round(self.window_len * self.sfreq)))
        if window >= n_times:
            return [(0, n_times)]
        segments = []
        start = 0
        while start < n_times:
            end = min(start + window, n_times)
            if end - start < window // 2 and segments:
                segments[-1] = (segments[-1][0], end)
            else:
                segments.append((start, end))
            start = end
        return segments


__all__ = ["CovarianceSegmenter", "FixedWindowSegmenter"]
