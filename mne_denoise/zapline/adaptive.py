"""Adaptive ZapLine-plus helpers."""

from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.signal import find_peaks, welch

from .._filtering import design_butter_sos


def apply_cleanline_notch(
    data: np.ndarray,
    sfreq: float,
    freq: float,
    bandwidth: float = 0.5,
    order: int = 4,
) -> np.ndarray:
    """Apply a narrow Butterworth band-stop filter.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Input data.
    sfreq : float
        Sampling frequency in Hz.
    freq : float
        Center frequency in Hz.
    bandwidth : float, default=0.5
        Notch width in Hz.
    order : int, default=4
        Filter order.

    Returns
    -------
    filtered : ndarray, shape (n_channels, n_times)
        Filtered data, or the input when the filter cannot be designed.
    """
    # Design notch filter
    nyquist = sfreq / 2
    low = max(0.001 * nyquist, freq - bandwidth / 2)
    high = min(0.999 * nyquist, freq + bandwidth / 2)

    if low >= high:
        return data  # Cannot filter, return unchanged

    # Use bandstop (notch) filter
    sos = design_butter_sos(order, [low, high], "bandstop", sfreq)
    filtered = signal.sosfiltfilt(sos, data, axis=1)

    return filtered


def apply_hybrid_cleanup(
    data: np.ndarray,
    sfreq: float,
    freq: float,
    bandwidth: float = 0.5,
    max_power_reduction_db: float = 3.0,
) -> np.ndarray:
    """Apply a notch only when surrounding spectral loss is acceptable.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Input data.
    sfreq : float
        Sampling frequency in Hz.
    freq : float
        Center frequency in Hz.
    bandwidth : float, default=0.5
        Notch width in Hz.
    max_power_reduction_db : float, default=3.0
        Maximum allowed surrounding-band reduction.

    Returns
    -------
    cleaned : ndarray, shape (n_channels, n_times)
        Filtered data, or the original data when the reduction exceeds the limit.
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


def find_noise_freqs(
    data: np.ndarray,
    sfreq: float,
    fmin: float = 17.0,
    fmax: float = 99.0,
    window_length: float = 6.0,
    threshold_factor: float = 4.0,
) -> list[float]:
    """Detect candidate noise frequencies from a Welch PSD.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Data to analyze.
    sfreq : float
        Sampling frequency in Hz.
    fmin, fmax : float, default=17.0, 99.0
        Search range in Hz.
    window_length : float, default=6.0
        Local spectral-baseline width in Hz.
    threshold_factor : float, default=4.0
        Threshold above the local baseline.

    Returns
    -------
    detected_freqs : list of float
        Detected frequencies in Hz.
    """
    n_channels, n_times = data.shape

    n_fft = min(n_times, int(sfreq * 4))
    if n_fft < 1024:
        n_fft = 1024

    freqs, psd = welch(
        data, fs=sfreq, window="hann", nperseg=n_fft, axis=-1, average="mean"
    )

    psd_log = 10 * np.log10(np.clip(psd, 1e-20, None))
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


def find_fine_peak(
    data: np.ndarray, sfreq: float, coarse_freq: float, search_width: float = 0.05
) -> float:
    """Refine a coarse frequency estimate using a local spectral peak.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Data to analyze.
    sfreq : float
        Sampling frequency in Hz.
    coarse_freq : float
        Initial frequency in Hz.
    search_width : float, default=0.05
        Half-width of the search interval in Hz.

    Returns
    -------
    fine_freq : float
        Refined frequency, or coarse_freq when no peak is found.
    """
    f_low = coarse_freq - search_width
    f_high = coarse_freq + search_width
    n_times = data.shape[1]
    n_fft = max(n_times, int(sfreq / 0.01))

    freqs, psd = welch(
        data, fs=sfreq, nperseg=min(n_times, 4 * int(sfreq)), nfft=n_fft, axis=-1
    )

    psd_log = 10 * np.log10(np.clip(psd, 1e-20, None))
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
    """Check whether a target line-noise peak is present.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Data to analyze.
    sfreq : float
        Sampling frequency in Hz.
    target_freq : float
        Target frequency in Hz.

    Returns
    -------
    present : bool
        Whether the spectral threshold is exceeded.
    """
    n_times = data.shape[1]
    n_fft = min(n_times, int(sfreq * 4))
    freqs, psd = welch(data, fs=sfreq, nperseg=n_fft, axis=-1)

    psd_log = 10 * np.log10(np.clip(psd, 1e-20, None))
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
    """Detect spectral peaks at harmonics of a fundamental frequency.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Data to analyze.
    sfreq : float
        Sampling frequency in Hz.
    fundamental : float
        Fundamental frequency in Hz.
    max_harmonics : int or None, default=None
        Maximum number of harmonics; None searches below Nyquist.
    threshold_factor : float, default=4.0
        Threshold above the local baseline.
    window_length : float, default=6.0
        Local spectral-baseline width in Hz.

    Returns
    -------
    harmonics : list of float
        Detected harmonic frequencies in Hz.
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
    psd_log = 10 * np.log10(np.clip(psd, 1e-20, None))
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
    """Classify residual spectral impact around a target frequency.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Cleaned data.
    sfreq : float
        Sampling frequency in Hz.
    target_freq : float
        Target frequency in Hz.
    max_prop_above : float, default=0.005
        Maximum proportion above the upper threshold.
    max_prop_below : float, default=0.005
        Maximum proportion below the lower threshold.
    freq_detect_mult : float, default=2.0
        Multiplier for the local detector.

    Returns
    -------
    status : {"ok", "weak", "strong"}
        Spectral QA classification.
    """
    n_times = data.shape[1]
    n_fft = min(n_times, int(sfreq * 4))
    freqs, psd = welch(data, fs=sfreq, nperseg=n_fft, axis=-1)

    psd_log = 10 * np.log10(np.clip(psd, 1e-20, None))
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
