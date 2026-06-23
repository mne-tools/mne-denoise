"""Adaptive ZapLine-plus utilities for automatic noise removal.

This module implements the adaptive components of ZapLine-plus [1]_ for automatic
and adaptive removal of frequency-specific noise artifacts in M/EEG recordings.

Key components:

1. **Noise frequency detection**: Automatic detection of line noise frequencies
   using Welch PSD and outlier detection.

2. **Adaptive segmentation**: Data segmentation based on covariance stationarity
   to handle non-stationary noise characteristics.

3. **Fine-grained peak detection**: Per-segment frequency refinement for
   accurate noise targeting.

4. **Artifact presence testing**: Statistical detection of whether line noise
   is present in a given segment.

5. **QA-based parameter adaptation**: Iterative adjustment of cleaning strength
   based on spectral quality assessment.

6. **Fallback cleaning**: Notch filter fallback for cases where DSS-based
   cleaning is insufficient.

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)

References
----------
.. [1] Klug, M., & Kloosterman, N. A. (2022). Zapline-plus: A Zapline extension for
       automatic and adaptive removal of frequency-specific noise artifacts in M/EEG.
       Human Brain Mapping, 43(9), 2743-2758.
       https://doi.org/10.1002/hbm.25832

.. [2] de Cheveigné, A. (2020). ZapLine: A simple and effective method to remove
       power line artifacts. NeuroImage, 207, 116356.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import signal
from scipy.signal import find_peaks, welch

from ..dss.utils.covariance import compute_covariance

logger = logging.getLogger(__name__)


def _log_psd(psd: np.ndarray) -> np.ndarray:
    """dB PSD with a floor relative to the spectrum's own peak, so peak/outlier
    detection is scale-invariant. An absolute ``1e-20`` floor flattens MEG
    magnetometer PSDs (~1e-26) to a constant, hiding the line peak entirely."""
    floor = float(np.max(psd)) * 1e-15 + 1e-300
    return 10 * np.log10(np.clip(psd, floor, None))


# -----------------------------------------------------------------------------
# 1. CleanLine Utilities (Fallback cleanup)
# -----------------------------------------------------------------------------


def apply_cleanline_notch(
    data: np.ndarray,
    sfreq: float,
    freq: float,
    bandwidth: float = 0.5,
    order: int = 4,
) -> np.ndarray:
    """Apply a narrow notch filter to remove residual line noise.

    This is a fallback for cases where ZapLine cannot fully remove noise
    without over-cleaning. Uses a Butterworth bandstop (notch) filter.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Input data to filter.
    sfreq : float
        Sampling frequency in Hz.
    freq : float
        Center frequency of the notch in Hz.
    bandwidth : float, default=0.5
        Width of the notch filter in Hz.
    order : int, default=4
        Order of the Butterworth filter.

    Returns
    -------
    filtered : ndarray, shape (n_channels, n_times)
        Notch-filtered data. Returns input unchanged if filter is invalid.

    Examples
    --------
    >>> filtered = apply_cleanline_notch(data, sfreq=1000, freq=50.0)
    """
    # Design notch filter
    nyquist = sfreq / 2
    low = (freq - bandwidth / 2) / nyquist
    high = (freq + bandwidth / 2) / nyquist

    # Ensure within valid range
    low = max(0.001, min(low, 0.999))
    high = max(0.001, min(high, 0.999))

    if low >= high:
        return data  # Cannot filter, return unchanged

    # Use bandstop (notch) filter
    sos = signal.butter(order, [low, high], btype="bandstop", output="sos")
    filtered = signal.sosfiltfilt(sos, data, axis=1)

    return filtered


def apply_hybrid_cleanup(
    data: np.ndarray,
    sfreq: float,
    freq: float,
    bandwidth: float = 0.5,
    max_power_reduction_db: float = 3.0,
) -> np.ndarray:
    """Apply hybrid cleanup with spectral impact protection.

    Applies a notch filter only if it doesn't cause excessive power loss
    in surrounding frequencies. This prevents over-aggressive cleaning
    that could damage neural signals.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Input data to filter.
    sfreq : float
        Sampling frequency in Hz.
    freq : float
        Center frequency to filter in Hz.
    bandwidth : float, default=0.5
        Width of the notch filter in Hz.
    max_power_reduction_db : float, default=3.0
        Maximum allowed power reduction (in dB) in surrounding frequencies.
        If cleaning would cause greater reduction, the original data is returned.

    Returns
    -------
    cleaned : ndarray, shape (n_channels, n_times)
        Cleaned data, or original data if cleaning would cause excessive
        signal loss.
    """
    # Apply notch
    filtered = apply_cleanline_notch(data, sfreq, freq, bandwidth)

    # Check spectral impact
    n_times = data.shape[1]
    n_fft = min(n_times, int(sfreq * 4))

    freqs, psd_orig = welch(data, fs=sfreq, nperseg=n_fft, axis=-1)
    # Recompute only necessary parts if optimization needed, but welch is fast
    _, psd_filt = welch(filtered, fs=sfreq, nperseg=n_fft, axis=-1)

    # Check power in surrounding frequencies (excluding notch region)
    surr_low = (freqs > freq - 3) & (freqs < freq - bandwidth)
    surr_high = (freqs > freq + bandwidth) & (freqs < freq + 3)
    surr_mask = surr_low | surr_high

    if not np.any(surr_mask):
        return filtered  # Can't check, apply anyway

    mean_orig = np.mean(psd_orig[:, surr_mask])
    mean_filt = np.mean(psd_filt[:, surr_mask])

    # Power reduction in dB
    reduction_db = 10 * np.log10(mean_orig / max(mean_filt, 1e-20))

    if reduction_db > max_power_reduction_db:
        # Cleanup would cause too much collateral damage
        return data

    return filtered


# -----------------------------------------------------------------------------
# 2. Detection & Segmentation Components
# -----------------------------------------------------------------------------


def find_noise_freqs(
    data: np.ndarray,
    sfreq: float,
    fmin: float = 17.0,
    fmax: float = 99.0,
    window_length: float = 6.0,
    threshold_factor: float = 4.0,
) -> list[float]:
    """Detect noise frequencies using Welch PSD and outlier detection.

    Analyzes the power spectral density to find frequencies with abnormally
    high power, indicative of line noise artifacts.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Input data to analyze.
    sfreq : float
        Sampling frequency in Hz.
    fmin : float, default=17.0
        Minimum frequency to search (Hz).
    fmax : float, default=99.0
        Maximum frequency to search (Hz).
    window_length : float, default=6.0
        Width of the spectral window for baseline estimation (Hz).
    threshold_factor : float, default=4.0
        Number of dB above local baseline to consider a peak as noise.

    Returns
    -------
    detected_freqs : list of float
        List of detected noise frequencies in Hz.

    Examples
    --------
    >>> freqs = find_noise_freqs(data, sfreq=1000)
    >>> print(f"Detected: {freqs}")  # e.g., [50.0, 60.0]
    """
    n_channels, n_times = data.shape

    n_fft = min(n_times, int(sfreq * 4))
    if n_fft < 1024:
        n_fft = 1024

    freqs, psd = welch(
        data, fs=sfreq, window="hann", nperseg=n_fft, axis=-1, average="mean"
    )

    psd_log = _log_psd(psd)
    # Geometric mean over channels (mean in log space)
    mean_log_psd = np.mean(psd_log, axis=0)

    mask = (freqs >= fmin) & (freqs <= fmax)
    search_freqs = freqs[mask]
    search_psd = mean_log_psd[mask]

    if len(search_freqs) == 0:
        return []

    # Moving window outlier detection
    detected_freqs = []

    freq_res = freqs[1] - freqs[0]
    win_samples = int(window_length / freq_res)
    half_win = win_samples // 2

    peak_indices, _ = find_peaks(search_psd)

    for idx_rel in peak_indices:
        idx_full = np.where(freqs == search_freqs[idx_rel])[0][0]

        start_idx = max(0, idx_full - half_win)
        end_idx = min(len(freqs), idx_full + half_win)

        window_psd = mean_log_psd[start_idx:end_idx]

        n_win = len(window_psd)
        if n_win < 3:
            continue

        n_third = n_win // 3
        left_third = window_psd[:n_third]
        right_third = window_psd[-n_third:]

        center_level = np.mean(np.concatenate([left_third, right_third]))
        peak_val = mean_log_psd[idx_full]

        if peak_val > center_level + threshold_factor:
            detected_freqs.append(freqs[idx_full])

    return detected_freqs


def segment_data(
    data: np.ndarray,
    sfreq: float,
    target_freq: float,
    min_chunk_len: float = 30.0,
    cov_win_len: float = 1.0,
) -> list[tuple[int, int]]:
    """Segment data into chunks based on covariance stationarity.

    Identifies boundaries where the noise characteristics change significantly
    by tracking changes in the spatial covariance matrix over time.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Input data to segment.
    sfreq : float
        Sampling frequency in Hz.
    target_freq : float
        Target noise frequency for bandpass filtering before analysis.
    min_chunk_len : float, default=30.0
        Minimum segment length in seconds.
    cov_win_len : float, default=1.0
        Window length for covariance computation in seconds.

    Returns
    -------
    segments : list of tuple
        List of ``(start_sample, end_sample)`` tuples defining segment boundaries.
    """
    n_channels, n_times = data.shape

    # 1. Filter around target freq
    f_low = target_freq - 3
    f_high = target_freq + 3

    sos = signal.butter(4, [f_low, f_high], btype="bandpass", fs=sfreq, output="sos")
    data_filt = signal.sosfiltfilt(sos, data, axis=1)

    # 2. Compute covariance series
    n_win = int(cov_win_len * sfreq)
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

    # 3. Successive distances
    dists = []
    for i in range(len(covs) - 1):
        d = np.linalg.norm(covs[i] - covs[i + 1], ord="fro")
        dists.append(d)

    dists = np.array(dists)

    if len(dists) == 0:
        return [(0, n_times)]

    # 4. Detect peaks (boundaries) - use distance.
    min_distance = max(1, int(min_chunk_len * sfreq / n_win))
    peak_indices, _ = find_peaks(
        dists, prominence=np.std(dists) * 0.5, distance=min_distance
    )
    boundary_indices = (peak_indices + 1) * n_win

    # 5. Enforce min length
    valid_boundaries = [0]
    last_boundary = 0
    min_samples = int(min_chunk_len * sfreq)

    for b in boundary_indices:
        if (b - last_boundary) >= min_samples:
            valid_boundaries.append(b)
            last_boundary = b

    if (n_times - last_boundary) < min_samples and len(valid_boundaries) > 1:
        valid_boundaries.pop()

    valid_boundaries.append(n_times)

    segments = []
    for i in range(len(valid_boundaries) - 1):
        segments.append((valid_boundaries[i], valid_boundaries[i + 1]))

    return segments


def find_fine_peak(
    data: np.ndarray, sfreq: float, coarse_freq: float, search_width: float = 0.05
) -> float:
    """Find the exact peak frequency within a narrow band.

    Refines a coarse frequency estimate by finding the spectral peak
    within a small search window.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Input data to analyze.
    sfreq : float
        Sampling frequency in Hz.
    coarse_freq : float
        Initial frequency estimate (Hz).
    search_width : float, default=0.05
        Half-width of search window around ``coarse_freq`` (Hz).

    Returns
    -------
    fine_freq : float
        Refined frequency estimate (Hz). Returns ``coarse_freq`` if
        no peak is found in the search window.
    """
    f_low = coarse_freq - search_width
    f_high = coarse_freq + search_width
    n_times = data.shape[1]
    n_fft = max(n_times, int(sfreq / 0.01))

    freqs, psd = welch(
        data, fs=sfreq, nperseg=min(n_times, 4 * int(sfreq)), nfft=n_fft, axis=-1
    )

    psd_log = _log_psd(psd)
    mean_log_psd = np.mean(psd_log, axis=0)

    mask = (freqs >= f_low) & (freqs <= f_high)
    search_freqs = freqs[mask]
    search_psd = mean_log_psd[mask]

    if len(search_freqs) == 0:
        return coarse_freq

    idx_max = np.argmax(search_psd)
    return search_freqs[idx_max]


def check_artifact_presence(
    data: np.ndarray,
    sfreq: float,
    target_freq: float,
) -> bool:
    """Check if line noise artifact is present.

    Uses spectral thresholding to determine if there is significant
    power at the target frequency relative to surrounding frequencies.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Input data to analyze.
    sfreq : float
        Sampling frequency in Hz.
    target_freq : float
        Frequency to check for artifact presence (Hz).

    Returns
    -------
    present : bool
        ``True`` if artifact is detected, ``False`` otherwise.
    """
    n_times = data.shape[1]
    n_fft = min(n_times, int(sfreq * 4))
    freqs, psd = welch(data, fs=sfreq, nperseg=n_fft, axis=-1)

    psd_log = _log_psd(psd)
    mean_log_psd = np.mean(psd_log, axis=0)

    f_low = target_freq - 3
    f_high = target_freq + 3

    idx_start = np.argmax(freqs >= f_low)
    idx_end = np.argmax(freqs > f_high)
    if idx_end == 0:
        idx_end = len(freqs)

    window_psd = mean_log_psd[idx_start:idx_end]
    if len(window_psd) < 3:
        return False

    n_third = len(window_psd) // 3
    left_third = window_psd[:n_third]
    right_third = window_psd[-n_third:]

    flanks = np.concatenate([left_third, right_third])
    center_power = np.mean(flanks)

    q_left = np.percentile(left_third, 5)
    q_right = np.percentile(right_third, 5)

    deviation = center_power - np.mean([q_left, q_right])
    threshold = center_power + 2 * deviation

    idx_target = np.argmin(np.abs(freqs - target_freq))
    peak_val = mean_log_psd[idx_target]

    return peak_val > threshold


def detect_harmonics(
    data: np.ndarray,
    sfreq: float,
    fundamental: float,
    max_harmonics: int | None = None,
    threshold_factor: float = 4.0,
    window_length: float = 6.0,
) -> list[float]:
    """Detect harmonics of a fundamental frequency.

    Searches for spectral peaks at integer multiples of the fundamental
    frequency up to the Nyquist limit.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Input data to analyze.
    sfreq : float
        Sampling frequency in Hz.
    fundamental : float
        Fundamental frequency in Hz.
    max_harmonics : int | None, default=None
        Maximum number of harmonics to search for.
        If ``None``, searches up to Nyquist frequency.
    threshold_factor : float, default=4.0
        Number of dB above local baseline to consider a peak as harmonic.
    window_length : float, default=6.0
        Width of spectral window for baseline estimation (Hz).

    Returns
    -------
    harmonics : list of float
        List of detected harmonic frequencies in Hz.
    """
    nyquist = sfreq / 2

    if max_harmonics is None:
        max_harmonics = int(np.floor(nyquist / fundamental)) - 1

    detected = []

    n_times = data.shape[1]
    n_fft = min(n_times, int(sfreq * 4))
    if n_fft < 1024:
        n_fft = 1024

    freqs, psd = welch(data, fs=sfreq, window="hann", nperseg=n_fft, axis=-1)
    psd_log = _log_psd(psd)
    mean_log_psd = np.mean(psd_log, axis=0)

    freq_res = freqs[1] - freqs[0]
    win_samples = int(window_length / freq_res)
    half_win = win_samples // 2

    for h in range(2, max_harmonics + 2):
        harmonic_freq = fundamental * h
        if harmonic_freq >= nyquist:
            break

        idx_center = np.argmin(np.abs(freqs - harmonic_freq))
        start_idx = max(0, idx_center - half_win)
        end_idx = min(len(freqs), idx_center + half_win)

        window_psd = mean_log_psd[start_idx:end_idx]
        if len(window_psd) < 3:
            continue

        n_third = len(window_psd) // 3
        left_third = window_psd[:n_third]
        right_third = window_psd[-n_third:]

        center_level = np.mean(np.concatenate([left_third, right_third]))
        peak_val = mean_log_psd[idx_center]

        if peak_val > center_level + threshold_factor:
            detected.append(harmonic_freq)

    return detected


def check_spectral_qa(
    data: np.ndarray,
    sfreq: float,
    target_freq: float,
    max_prop_above: float = 0.005,
    max_prop_below: float = 0.005,
    freq_detect_mult: float = 2.0,
) -> str:
    """Assess quality of noise cleaning using spectral analysis.

    Analyzes the power spectrum around the target frequency to determine
    if cleaning was appropriate, too weak (residual noise), or too strong
    (over-cleaning causing spectral notch).

    This implementation uses proportion-based thresholds rather than binary
    checks.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Cleaned data to assess.
    sfreq : float
        Sampling frequency in Hz.
    target_freq : float
        Target noise frequency (Hz).
    max_prop_above : float, default=0.005
        Maximum proportion of frequency samples that may be above the upper
        threshold before cleaning is considered too weak. Default is 0.5%.
    max_prop_below : float, default=0.005
        Maximum proportion of frequency samples that may be below the lower
        threshold before cleaning is considered too strong. Default is 0.5%.
    freq_detect_mult : float, default=2.0
        Multiplier for the 5% quantile deviation detector. Default is 2.

    Returns
    -------
    status : {'ok', 'weak', 'strong'}
        Quality assessment:

        - ``'ok'``: Cleaning was appropriate
        - ``'weak'``: Residual noise detected, consider stronger cleaning
        - ``'strong'``: Over-cleaning detected (spectral notch), reduce strength
    """
    n_times = data.shape[1]
    n_fft = min(n_times, int(sfreq * 4))
    freqs, psd = welch(data, fs=sfreq, nperseg=n_fft, axis=-1)

    psd_log = _log_psd(psd)
    mean_log_psd = np.mean(psd_log, axis=0)

    f_win_low = target_freq - 3
    f_win_high = target_freq + 3

    mask_win = (freqs >= f_win_low) & (freqs <= f_win_high)
    win_psd = mean_log_psd[mask_win]
    if len(win_psd) < 3:
        return "ok"

    n_third = len(win_psd) // 3
    left = win_psd[:n_third]
    right = win_psd[-n_third:]
    center_power = np.mean(np.concatenate([left, right]))

    # Use lower quantile as indicator of variability
    q_left = np.percentile(left, 5)
    q_right = np.percentile(right, 5)
    mean_lower_quantile = np.mean([q_left, q_right])
    deviation = center_power - mean_lower_quantile

    # Threshold calculation
    thresh_upper = center_power + freq_detect_mult * deviation
    thresh_lower = center_power - freq_detect_mult * deviation

    # Weak check - detailedFreqBoundsUpper = [-0.05, 0.05]
    f_tight_low = target_freq - 0.05
    f_tight_high = target_freq + 0.05
    mask_tight = (freqs >= f_tight_low) & (freqs <= f_tight_high)
    tight_psd = mean_log_psd[mask_tight]

    # Proportion-based check
    if len(tight_psd) > 0:
        prop_above = np.sum(tight_psd > thresh_upper) / len(tight_psd)
        if prop_above > max_prop_above:
            return "weak"

    # Strong check - detailedFreqBoundsLower = [-0.4, 0.1]
    f_notch_low = target_freq - 0.4
    f_notch_high = target_freq + 0.1
    mask_notch = (freqs >= f_notch_low) & (freqs <= f_notch_high)
    notch_psd = mean_log_psd[mask_notch]

    # Proportion-based check
    if len(notch_psd) > 0:
        prop_below = np.sum(notch_psd < thresh_lower) / len(notch_psd)
        if prop_below > max_prop_below:
            return "strong"

    return "ok"
