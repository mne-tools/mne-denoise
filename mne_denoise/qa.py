"""Quality assurance metrics for denoising evaluation.

This module contains:
1. Low-level spectral metrics operating on pre-computed PSD arrays.
2. A high-level benchmark helper operating on
   :class:`~mne.io.BaseRaw` objects.

All metrics are estimator-agnostic and can be used with any denoising output
as long as before/after PSDs (or Raw objects) are available.

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

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
    """Attenuation (dB) of the dominant peak around a target frequency.

    Parameters
    ----------
    freqs : array of shape (n_freqs,)
        Frequency vector.
    psd_before : array of shape (n_channels, n_freqs) or (n_freqs,)
        PSD before cleaning.
    psd_after : array of shape (n_channels, n_freqs) or (n_freqs,)
        PSD after cleaning.
    target_freq : float
        Centre frequency of the peak (Hz).
    bandwidth : float
        Half-bandwidth (Hz) around *target_freq* to search for the peak.

    Returns
    -------
    attenuation : ndarray | float
        Per-channel attenuation in dB for 2D PSD input, or a scalar value for
        1D PSD input.

    Notes
    -----
    This metric compares the maximum PSD value in a narrow band around
    ``target_freq``:

    ``10 * log10(max_before / max_after)``

    Positive values indicate suppression of the target peak.

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.qa import peak_attenuation_db
    >>> freqs = np.arange(0, 100, 0.5)
    >>> before = np.ones_like(freqs) * 0.01
    >>> after = before.copy()
    >>> band = (freqs >= 49) & (freqs <= 51)
    >>> before[band] = 1.0
    >>> after[band] = 0.5
    >>> float(peak_attenuation_db(freqs, before, after, 50.0)) > 0
    True
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
    """Suppression ratio (dB) of mean band power around a target frequency.

    Parameters
    ----------
    freqs : array of shape (n_freqs,)
        Frequency vector.
    psd_before, psd_after : ndarray
        PSDs before and after cleaning.
    target_freq : float
        Center frequency (Hz).
    bandwidth : float
        Half-bandwidth (Hz).

    Returns
    -------
    ratio_db : float
        Suppression ratio in dB.

    Notes
    -----
    For 2D PSD input, channels are averaged first. The ratio is computed from
    mean power in the selected band:

    ``10 * log10(mean_before / mean_after)``

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.qa import suppression_ratio
    >>> freqs = np.arange(0, 100, 0.5)
    >>> before = np.ones_like(freqs)
    >>> after = before * 0.1
    >>> suppression_ratio(freqs, before, after, 50.0)
    10.0
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
    """Residual peak-to-surround power ratio around a target frequency.

    Values near ``1`` indicate the target peak is close to its surrounding
    spectral floor. Values above ``1`` indicate residual narrow-band peak power.

    Parameters
    ----------
    freqs : array of shape (n_freqs,)
        Frequency vector.
    psd_after : array of shape (n_channels, n_freqs) or (n_freqs,)
        PSD after cleaning.
    target_freq : float
        Centre frequency of the line-noise peak (Hz).
    peak_bw : float
        Half-bandwidth (Hz) of the peak region.
    surround_bw : float
        Half-bandwidth (Hz) of the surrounding region (measured from the
        outer edge of *peak_bw*).

    Returns
    -------
    ratio : ndarray | float
        Per-channel ratio for 2D PSD input, or a scalar for 1D PSD input.

    Notes
    -----
    The metric compares mean power in a peak window to mean power in two
    surrounding windows (left/right of the peak window).

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.qa import noise_surround_ratio
    >>> freqs = np.arange(0, 100, 0.5)
    >>> psd = np.ones((2, len(freqs)))
    >>> noise_surround_ratio(freqs, psd, 50.0).shape
    (2,)
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
    """Broadband spectral distortion (dB) outside excluded noise bands.

    Computed as the mean absolute log-ratio:
    ``|10 * log10(psd_after / psd_before)|`` over selected frequencies.
    Lower values indicate less collateral broadband distortion.

    Parameters
    ----------
    freqs : array of shape (n_freqs,)
        Frequency vector.
    psd_before : array of shape (n_channels, n_freqs) or (n_freqs,)
        PSD before cleaning.
    psd_after : array of shape (n_channels, n_freqs) or (n_freqs,)
        PSD after cleaning.
    exclude_freq : float | None
        Fundamental line-noise frequency to exclude (together with its
        harmonics).  If ``None`` no exclusion is applied.
    exclude_bw : float
        Half-bandwidth (Hz) to exclude around each harmonic.
    fmin, fmax : float
        Frequency range for the broadband comparison.
    n_harmonics : int
        Number of harmonics of *exclude_freq* to also exclude
        (0 = fundamental only).

    Returns
    -------
    distortion : ndarray | float
        Per-channel distortion for 2D PSD input, or a scalar for 1D PSD input.

    Notes
    -----
    This metric is useful as a signal-preservation indicator while
    line-noise-focused metrics capture artifact suppression.

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.qa import below_noise_distortion_db
    >>> freqs = np.arange(0, 100, 0.5)
    >>> before = np.ones((2, len(freqs)))
    >>> after = before.copy()
    >>> np.allclose(below_noise_distortion_db(freqs, before, after), 0.0)
    True
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
    """Spectral distortion (dB RMS) at non-harmonic frequencies.

    This measures how much the cleaning process changed the spectrum
    outside of the target line-noise frequencies.

    Parameters
    ----------
    freqs : array of shape (n_freqs,)
        Frequency vector.
    psd_before, psd_after : array
        PSDs before and after cleaning.
    line_freq : float
        Fundamental line frequency (Hz).
    n_harmonics : int
        Number of harmonics to exclude.
    bandwidth : float
        Base exclusion bandwidth (Hz).

    Returns
    -------
    distortion : float
        RMS distortion in dB.

    Notes
    -----
    This is an RMS variant of broadband distortion using channel-averaged PSDs.
    Evaluation is restricted to 2-160 Hz and excludes line-frequency harmonics.

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.qa import spectral_distortion
    >>> freqs = np.arange(0, 200, 0.5)
    >>> psd = np.ones((2, len(freqs)))
    >>> spectral_distortion(freqs, psd, psd, line_freq=50.0, n_harmonics=3)
    0.0
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
    """Fraction of channels where the spectral floor is over-suppressed.

    Parameters
    ----------
    freqs : array of shape (n_freqs,)
        Frequency vector.
    psd_before, psd_after : ndarray
        PSDs before and after cleaning.
    target_freq : float
        Centre frequency (Hz) of the line-noise peak.
    bandwidth : float
        Half-bandwidth (Hz) used for peak identification.
    threshold_db : float
        Attenuation threshold in dB.

    Returns
    -------
    proportion : float
        Value in [0, 1].

    Notes
    -----
    A channel is flagged as over-cleaned when attenuation in the surrounding
    floor region exceeds ``threshold_db``.

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.qa import overclean_proportion
    >>> freqs = np.arange(0, 100, 0.5)
    >>> psd = np.ones((4, len(freqs)))
    >>> overclean_proportion(freqs, psd, psd, 50.0)
    0.0
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
    """Fraction of channels where the line-noise peak remains prominent.

    Parameters
    ----------
    freqs : array of shape (n_freqs,)
        Frequency vector.
    psd_after : ndarray
        PSD after cleaning.
    target_freq : float
        Centre frequency (Hz).
    peak_bw, surround_bw : float
        Bandwidths for peak and surround.
    threshold_ratio : float
        Ratio above which a channel is considered under-cleaned.

    Returns
    -------
    proportion : float
        Value in [0, 1].

    Notes
    -----
    A channel is flagged as under-cleaned when
    :func:`noise_surround_ratio` exceeds ``threshold_ratio``.

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.qa import underclean_proportion
    >>> freqs = np.arange(0, 100, 0.5)
    >>> psd = np.ones((4, len(freqs)))
    >>> underclean_proportion(freqs, psd, 50.0)
    0.0
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
    """Geometric mean of ``psd_after / psd_before`` across broadband.

    Parameters
    ----------
    freqs : array of shape (n_freqs,)
        Frequency vector.
    psd_before, psd_after : ndarray
        PSDs before and after cleaning.
    fmin, fmax : float
        Frequency range.

    Returns
    -------
    gm_ratio : ndarray | float
        Per-channel geometric-mean ratio for 2D PSD input, or a scalar for 1D
        PSD input.

    Notes
    -----
    Values near ``1`` indicate small broadband spectral changes. Values below
    ``1`` indicate net broadband attenuation.

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.qa import geometric_mean_psd_ratio
    >>> freqs = np.arange(0, 100, 0.5)
    >>> before = np.ones((2, len(freqs)))
    >>> after = before * 0.5
    >>> np.allclose(geometric_mean_psd_ratio(freqs, before, after), 0.5)
    True
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
    """Percentage of total variance removed after denoising.

    Parameters
    ----------
    data_before : ndarray
        Data before denoising.
    data_after : ndarray
        Data after denoising.

    Returns
    -------
    pct_removed : float
        Percentage of variance removed:
        ``100 * (1 - var(data_after) / var(data_before))``.

    Notes
    -----
    Returns ``0.0`` when ``data_before`` has zero variance.

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.qa import variance_removed
    >>> x = np.array([1.0, -1.0, 1.0, -1.0])
    >>> variance_removed(x, 0.5 * x)
    75.0
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
    """Compute all QA metrics for a line-noise removal benchmark.

    Parameters
    ----------
    raw_before, raw_after : mne.io.BaseRaw
        Raw recordings before and after cleaning.
    line_freq : float
        Fundamental line-noise frequency (Hz).
    n_harmonics : int
        Number of harmonics above the fundamental to evaluate.
    fmax : float
        Maximum frequency for PSD computation.

    Returns
    -------
    metrics : dict
        Dictionary with scalar summary metrics and per-harmonic vectors.
        Scalar keys:
        ``peak_attenuation_db``, ``R_f0``, ``below_noise_distortion_db``,
        ``overclean_proportion``, ``underclean_proportion``,
        ``geometric_mean_psd_ratio``.

    Notes
    -----
    ``peak_attenuation_db`` and ``R_f0`` scalar outputs correspond to the
    first evaluated harmonic (fundamental line frequency).

    Examples
    --------
    >>> # metrics = compute_all_qa_metrics(raw_before, raw_after, line_freq=50.0)
    >>> # float(metrics["peak_attenuation_db"])
    """
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
