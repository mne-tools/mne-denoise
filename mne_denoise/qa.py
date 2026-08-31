"""Quality-assurance metrics for denoising."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import _mne

__all__ = [
    "peak_attenuation_db",
    "suppression_ratio",
    "noise_surround_ratio",
    "below_noise_distortion_db",
    "spectral_distortion",
    "overclean_proportion",
    "underclean_proportion",
    "geometric_mean_psd_ratio",
    "variance_removed",
    "compute_all_qa_metrics",
    "rms_change",
    "max_abs_change",
    "channel_variance_ratio",
]

if TYPE_CHECKING:
    import mne

_EPS = 1e-30  # floor to avoid log(0)


def peak_attenuation_db(
    freqs: np.ndarray,
    psd_before: np.ndarray,
    psd_after: np.ndarray,
    target_freq: float,
    bandwidth: float = 2.0,
) -> np.ndarray:
    """Compute peak attenuation around a target frequency.

    The metric is 10 * log10(max(psd_before) / max(psd_after)) within the selected
    band; positive values indicate attenuation.

    Parameters
    ----------
    freqs : ndarray, shape (n_freqs,)
        Frequency vector in Hz.
    psd_before, psd_after : ndarray, shape (n_freqs,) or (n_channels, n_freqs)
        PSDs before and after cleaning.
    target_freq : float
        Center frequency in Hz.
    bandwidth : float, default=2.0
        Half-width of the search band in Hz.

    Returns
    -------
    float or ndarray
        Scalar for 1-D PSD input, otherwise one value per channel. Empty bands
        return NaN; after-power is floored at 1e-30.
    """
    mask = (freqs >= target_freq - bandwidth) & (freqs <= target_freq + bandwidth)
    if not mask.any():
        return np.nan if psd_before.ndim == 1 else np.full(psd_before.shape[0], np.nan)
    if psd_before.ndim == 1:
        peak_before = psd_before[mask].max()
        peak_after = psd_after[mask].max()
    else:
        peak_before = psd_before[:, mask].max(axis=1)
        peak_after = psd_after[:, mask].max(axis=1)
    return 10.0 * np.log10(peak_before / np.maximum(peak_after, _EPS))


def suppression_ratio(
    freqs: np.ndarray,
    psd_before: np.ndarray,
    psd_after: np.ndarray,
    target_freq: float,
    bandwidth: float = 2.0,
) -> float:
    """Compute the dB ratio of mean band power before and after cleaning.

    The metric is 10 * log10(mean_before / mean_after); positive values indicate
    suppression.

    Parameters
    ----------
    freqs : ndarray, shape (n_freqs,)
        Frequency vector in Hz.
    psd_before, psd_after : ndarray, shape (n_freqs,) or (n_channels, n_freqs)
        PSDs before and after cleaning.
    target_freq : float
        Center frequency in Hz.
    bandwidth : float, default=2.0
        Half-width of the band in Hz.

    Returns
    -------
    float
        Scalar ratio after averaging channels when input is 2-D. Empty bands return
        NaN; non-positive after-power returns positive infinity.
    """
    mask = (freqs >= target_freq - bandwidth) & (freqs <= target_freq + bandwidth)
    if not mask.any():
        return np.nan

    pb = psd_before.mean(axis=0) if psd_before.ndim == 2 else psd_before
    pa = psd_after.mean(axis=0) if psd_after.ndim == 2 else psd_after

    pb_mean = pb[mask].mean()
    pa_mean = pa[mask].mean()

    if pa_mean <= 0:
        return np.inf
    return 10.0 * np.log10(pb_mean / pa_mean)


def noise_surround_ratio(
    freqs: np.ndarray,
    psd_after: np.ndarray,
    target_freq: float,
    peak_bw: float = 2.0,
    surround_bw: float = 5.0,
) -> np.ndarray:
    """Compute target-band power divided by surrounding power.

    Parameters
    ----------
    freqs : ndarray, shape (n_freqs,)
        Frequency vector in Hz.
    psd_after : ndarray, shape (n_freqs,) or (n_channels, n_freqs)
        PSD after cleaning.
    target_freq : float
        Center frequency in Hz.
    peak_bw : float, default=2.0
        Half-width of the target band in Hz.
    surround_bw : float, default=5.0
        Half-width of each surrounding region in Hz.

    Returns
    -------
    float or ndarray
        Scalar for 1-D input, otherwise one value per channel. Values near one
        indicate a flat target region; larger values indicate a residual peak.
        Missing surrounding power uses a 1e-30 denominator.
    """
    peak_mask = (freqs >= target_freq - peak_bw) & (freqs <= target_freq + peak_bw)
    surr_mask = (
        (freqs >= target_freq - surround_bw) & (freqs < target_freq - peak_bw)
    ) | ((freqs > target_freq + peak_bw) & (freqs <= target_freq + surround_bw))

    if psd_after.ndim == 1:
        peak_power = psd_after[peak_mask].mean() if peak_mask.any() else 0.0
        surr_power = psd_after[surr_mask].mean() if surr_mask.any() else _EPS
        return peak_power / max(surr_power, _EPS)

    peak_power = (
        psd_after[:, peak_mask].mean(axis=1)
        if peak_mask.any()
        else np.zeros(psd_after.shape[0])
    )
    surr_power = (
        psd_after[:, surr_mask].mean(axis=1)
        if surr_mask.any()
        else np.full(psd_after.shape[0], _EPS)
    )
    return peak_power / np.maximum(surr_power, _EPS)


def below_noise_distortion_db(
    freqs: np.ndarray,
    psd_before: np.ndarray,
    psd_after: np.ndarray,
    exclude_freq: float | None = None,
    exclude_bw: float = 5.0,
    fmin: float = 1.0,
    fmax: float = 45.0,
    n_harmonics: int = 0,
) -> np.ndarray:
    """Compute mean absolute log-power distortion outside excluded bands.

    The metric is the mean of abs(10 * log10(psd_after / psd_before)) over the
    selected frequency mask; lower values indicate less broadband change.

    Parameters
    ----------
    freqs : ndarray, shape (n_freqs,)
        Frequency vector in Hz.
    psd_before, psd_after : ndarray, shape (n_freqs,) or (n_channels, n_freqs)
        PSDs before and after cleaning.
    exclude_freq : float or None, default=None
        Fundamental frequency whose harmonics are excluded.
    exclude_bw : float, default=5.0
        Half-width of each excluded band in Hz.
    fmin, fmax : float, default=1.0, 45.0
        Inclusive frequency range in Hz.
    n_harmonics : int, default=0
        Additional harmonics to exclude.

    Returns
    -------
    float or ndarray
        Scalar for 1-D input, otherwise one value per channel. An empty mask returns
        zero; before-power is floored at 1e-30.
    """
    mask = (freqs >= fmin) & (freqs <= fmax)
    if exclude_freq is not None:
        for h in range(1, n_harmonics + 2):
            hf = exclude_freq * h
            mask &= ~((freqs >= hf - exclude_bw) & (freqs <= hf + exclude_bw))
    if not mask.any():
        return 0.0 if psd_before.ndim == 1 else np.zeros(psd_before.shape[0])
    if psd_before.ndim == 1:
        ratio = np.log10(psd_after[mask] / np.maximum(psd_before[mask], _EPS))
        return float(np.mean(np.abs(ratio)) * 10.0)
    ratio = np.log10(psd_after[:, mask] / np.maximum(psd_before[:, mask], _EPS))
    return np.mean(np.abs(ratio), axis=1) * 10.0


def spectral_distortion(
    freqs: np.ndarray,
    psd_before: np.ndarray,
    psd_after: np.ndarray,
    line_freq: float = 50.0,
    n_harmonics: int = 3,
    bandwidth: float = 2.0,
) -> float:
    """Compute RMS log-power distortion away from line-noise harmonics.

    Parameters
    ----------
    freqs : ndarray, shape (n_freqs,)
        Frequency vector in Hz.
    psd_before, psd_after : ndarray, shape (n_freqs,) or (n_channels, n_freqs)
        PSDs before and after cleaning; channels are averaged for 2-D input.
    line_freq : float, default=50.0
        Fundamental line frequency in Hz.
    n_harmonics : int, default=3
        Number of harmonics to exclude.
    bandwidth : float, default=2.0
        Base half-width; the implementation excludes line_freq * k +/- 2 * bandwidth.

    Returns
    -------
    float
        RMS distortion in dB over 2--160 Hz. An empty mask returns 0.0.
    """
    safe = np.ones(len(freqs), dtype=bool)
    for k in range(1, n_harmonics + 1):
        target = line_freq * k
        safe &= ~((freqs >= target - bandwidth * 2) & (freqs <= target + bandwidth * 2))

    # Restrict to a reasonable range for evaluation
    safe &= (freqs >= 2) & (freqs <= 160)

    if not safe.any():
        return 0.0

    pb = psd_before.mean(axis=0) if psd_before.ndim == 2 else psd_before
    pa = psd_after.mean(axis=0) if psd_after.ndim == 2 else psd_after

    ratio = pa[safe] / np.maximum(pb[safe], _EPS)
    return np.sqrt(np.mean((10.0 * np.log10(ratio)) ** 2))


def overclean_proportion(
    freqs: np.ndarray,
    psd_before: np.ndarray,
    psd_after: np.ndarray,
    target_freq: float,
    bandwidth: float = 2.0,
    threshold_db: float = 3.0,
) -> float:
    """Compute the fraction of channels whose surrounding floor is over-suppressed.

    A channel is flagged when surrounding-band attenuation exceeds threshold_db.

    Parameters
    ----------
    freqs : ndarray, shape (n_freqs,)
        Frequency vector in Hz.
    psd_before, psd_after : ndarray, shape (n_freqs,) or (n_channels, n_freqs)
        PSDs before and after cleaning.
    target_freq : float
        Center frequency in Hz.
    bandwidth : float, default=2.0
        Half-width used to define the target band.
    threshold_db : float, default=3.0
        Surrounding-floor attenuation threshold in dB.

    Returns
    -------
    float
        Indicator for 1-D input or channel fraction in [0, 1] for 2-D input.
    No surrounding frequencies returns 0.0.
    """
    surr_mask = (
        (freqs >= target_freq - bandwidth * 2) & (freqs < target_freq - bandwidth)
    ) | ((freqs > target_freq + bandwidth) & (freqs <= target_freq + bandwidth * 2))
    if not surr_mask.any():
        return 0.0
    if psd_before.ndim == 1:
        floor_before = psd_before[surr_mask].mean()
        floor_after = psd_after[surr_mask].mean()
        atten_db = 10.0 * np.log10(floor_before / max(floor_after, _EPS))
        return float(atten_db > threshold_db)
    floor_before = psd_before[:, surr_mask].mean(axis=1)
    floor_after = psd_after[:, surr_mask].mean(axis=1)
    atten_db = 10.0 * np.log10(floor_before / np.maximum(floor_after, _EPS))
    return float((atten_db > threshold_db).mean())


def underclean_proportion(
    freqs: np.ndarray,
    psd_after: np.ndarray,
    target_freq: float,
    peak_bw: float = 2.0,
    surround_bw: float = 5.0,
    threshold_ratio: float = 2.0,
) -> float:
    """Compute the fraction of channels with a residual target peak.

    A channel is flagged when noise_surround_ratio exceeds threshold_ratio.

    Parameters
    ----------
    freqs : ndarray, shape (n_freqs,)
        Frequency vector in Hz.
    psd_after : ndarray, shape (n_freqs,) or (n_channels, n_freqs)
        PSD after cleaning.
    target_freq : float
        Center frequency in Hz.
    peak_bw : float, default=2.0
        Half-width of the target band in Hz.
    surround_bw : float, default=5.0
        Half-width of the surrounding bands in Hz.
    threshold_ratio : float, default=2.0
        Residual peak ratio threshold.

    Returns
    -------
    float
        Indicator for 1-D input or channel fraction in [0, 1] for 2-D input.
    Missing peak power yields an unflagged ratio of zero.
    """
    nsr = noise_surround_ratio(freqs, psd_after, target_freq, peak_bw, surround_bw)
    if np.ndim(nsr) == 0:
        return float(nsr > threshold_ratio)
    return float((nsr > threshold_ratio).mean())


def geometric_mean_psd_ratio(
    freqs: np.ndarray,
    psd_before: np.ndarray,
    psd_after: np.ndarray,
    fmin: float = 1.0,
    fmax: float = 45.0,
) -> np.ndarray:
    """Compute the geometric mean of psd_after / psd_before over a frequency range.

    Parameters
    ----------
    freqs : ndarray, shape (n_freqs,)
        Frequency vector in Hz.
    psd_before, psd_after : ndarray, shape (n_freqs,) or (n_channels, n_freqs)
        PSDs before and after cleaning.
    fmin, fmax : float, default=1.0, 45.0
        Inclusive frequency bounds in Hz.

    Returns
    -------
    float or ndarray
        Scalar for 1-D input, otherwise one value per channel. Values below one
        indicate net attenuation; an empty mask returns one. PSDs are floored at
        1e-30 before logarithms.
    """
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not mask.any():
        return 1.0 if psd_before.ndim == 1 else np.ones(psd_before.shape[0])
    if psd_before.ndim == 1:
        ratio = psd_after[mask] / np.maximum(psd_before[mask], _EPS)
        return float(np.exp(np.mean(np.log(np.maximum(ratio, _EPS)))))
    ratio = psd_after[:, mask] / np.maximum(psd_before[:, mask], _EPS)
    return np.exp(np.mean(np.log(np.maximum(ratio, _EPS)), axis=1))


def variance_removed(data_before: np.ndarray, data_after: np.ndarray) -> float:
    """Compute percentage of total variance removed.

    The definition is 100 * (1 - var(data_after) / var(data_before)).

    Parameters
    ----------
    data_before, data_after : ndarray
        Data before and after cleaning with matching shapes.

    Returns
    -------
    float
        Percentage removed. Positive values indicate reduced variance, negative
        values increased variance, and zero input variance returns 0.0.
    """
    var_before = np.var(data_before)
    if var_before == 0:
        return 0.0
    return 100.0 * (1.0 - np.var(data_after) / var_before)


def _compute_psd_pair(
    raw_before: mne.io.BaseRaw,
    raw_after: mne.io.BaseRaw,
    fmax: float = 125.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(freqs, psd_before, psd_after)`` from two Raw objects."""
    psd_b = raw_before.compute_psd(fmax=fmax, verbose=False)
    psd_a = raw_after.compute_psd(fmax=fmax, verbose=False)
    return psd_b.freqs, psd_b.get_data(), psd_a.get_data()


def compute_all_qa_metrics(
    raw_before: mne.io.BaseRaw,
    raw_after: mne.io.BaseRaw,
    line_freq: float = 50.0,
    n_harmonics: int = 0,
    fmax: float = 125.0,
) -> dict:
    """Compute the package QA summary for two MNE Raw objects.

    Parameters
    ----------
    raw_before, raw_after : mne.io.BaseRaw
        Recordings before and after cleaning.
    line_freq : float, default=50.0
        Fundamental line frequency in Hz.
    n_harmonics : int, default=0
        Number of harmonics above the fundamental.
    fmax : float, default=125.0
        Maximum PSD frequency in Hz.

    Returns
    -------
    dict
        Median summary metrics and per-harmonic attenuation/ratio arrays. PSDs are
        computed with the Raw objects' compute_psd method.
    """
    _mne.require_mne("MNE QA metrics")
    freqs, psd_b, psd_a = _compute_psd_pair(raw_before, raw_after, fmax=fmax)

    harmonics = [line_freq * h for h in range(1, n_harmonics + 2)]

    # Per-harmonic metrics
    per_h_atten: list[float] = []
    per_h_r: list[float] = []
    for hf in harmonics:
        atten = peak_attenuation_db(freqs, psd_b, psd_a, hf)
        nsr = noise_surround_ratio(freqs, psd_a, hf)
        per_h_atten.append(float(np.nanmedian(atten)))
        per_h_r.append(float(np.nanmedian(nsr)))

    # Broadband metrics
    distort = below_noise_distortion_db(
        freqs,
        psd_b,
        psd_a,
        exclude_freq=line_freq,
        n_harmonics=n_harmonics,
    )
    oc = overclean_proportion(freqs, psd_b, psd_a, line_freq)
    uc = underclean_proportion(freqs, psd_a, line_freq)
    gmr = geometric_mean_psd_ratio(freqs, psd_b, psd_a)

    return {
        "peak_attenuation_db": per_h_atten[0],
        "R_f0": per_h_r[0],
        "below_noise_distortion_db": float(np.median(distort)),
        "overclean_proportion": float(oc),
        "underclean_proportion": float(uc),
        "geometric_mean_psd_ratio": float(np.median(gmr)),
        "harmonics_hz": harmonics,
        "per_harmonic_attenuation_db": per_h_atten,
        "per_harmonic_R": per_h_r,
    }


def rms_change(data_before: np.ndarray, data_after: np.ndarray) -> float:
    """Compute the RMS of data_before - data_after.

    Parameters
    ----------
    data_before, data_after : ndarray
        Matching data arrays.

    Returns
    -------
    float
        RMS in the input data units. Empty input produces NaN.
    """
    delta = data_before - data_after
    return float(np.sqrt(np.mean(delta**2)))


def max_abs_change(data_before: np.ndarray, data_after: np.ndarray) -> float:
    """Compute the largest absolute sample-wise change.

    Parameters
    ----------
    data_before, data_after : ndarray
        Matching data arrays.

    Returns
    -------
    float
        max(abs(data_before - data_after)) in input units. Empty input raises the
        NumPy maximum error.
    """
    return float(np.max(np.abs(data_before - data_after)))


def channel_variance_ratio(
    data_before: np.ndarray, data_after: np.ndarray
) -> np.ndarray:
    """Compute after/before variance for each channel.

    Parameters
    ----------
    data_before : ndarray, shape (n_channels, n_times) or (n_epochs, n_channels, n_times)
        Data before cleaning.
    data_after : ndarray
        Data after cleaning with the same shape.

    Returns
    -------
    ndarray, shape (n_channels,)
        Variance ratio pooled over time, or epochs and time. Zero denominators are
        replaced with machine epsilon.
    """
    axis = (0, 2) if data_before.ndim == 3 else 1
    var_before = np.var(data_before, axis=axis)
    var_after = np.var(data_after, axis=axis)
    return var_after / np.maximum(var_before, np.finfo(float).eps)
