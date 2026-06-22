"""Spectral QA metrics for line-noise removal evaluation.

All metrics use the **geometric-mean PSD** (mean of log₁₀-PSDs across EEG
channels, computed with ``scipy.signal.welch``).  This matches the
Zapline-plus / paper-5 methodology and avoids the masking effect of
per-channel averaging.

Design principles
-----------------
* Every public function accepts plain NumPy arrays (``freqs``, ``psd``)
  **or** an MNE ``Raw`` object (via :func:`geometric_mean_psd`).
* ``line_freq`` and ``n_harmonics`` are explicit parameters — no module-
  level globals.
* Default bandwidths (``center_bw=0.05 Hz``, ``side_bw=1.0 Hz``) match
  the paper-5 reference implementation exactly (≈1 FFT bin at
  ``nperseg = sfreq × 4``).
* Return types are always plain Python scalars or NumPy arrays so that
  the results serialise trivially to JSON / TSV.

Authors
-------
Sina Esmaeili  — sina.esmaeili@umontreal.ca
Hamza Abdelhedi — hamza.abdelhedi@umontreal.ca
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.integrate import trapezoid  # stable across numpy 1.x/2.x (np.trapezoid is 2.0+)
from scipy.signal import welch as sp_welch

if TYPE_CHECKING:
    pass

# =====================================================================
# Geometric-mean PSD
# =====================================================================


def geometric_mean_psd(raw, nperseg=None):
    """Compute the geometric-mean PSD across channels.

    Uses ``scipy.signal.welch``, then averages log₁₀-PSDs across channels
    (geometric mean).  Matches the paper-5 reference.

    Parameters
    ----------
    raw : mne.io.Raw | numpy.ndarray
        If an MNE Raw object, ``raw.get_data()`` is used and *sfreq* is
        read from ``raw.info``.  If a 2-D array of shape
        ``(n_channels, n_times)``, *sfreq* defaults to ``1.0`` (the
        caller should interpret the returned ``freqs`` accordingly).
    nperseg : int | None
        FFT segment length passed to ``scipy.signal.welch``.
        If *None* and *raw* is an MNE Raw object, defaults to
        ``int(sfreq * 4)`` (matching paper-5).  If *None* and *raw* is
        an array, defaults to ``1024``.

    Returns
    -------
    freqs : ndarray, shape (n_freqs,)
        Frequency vector (Hz).
    gm_psd : ndarray, shape (n_freqs,)
        Geometric-mean PSD across channels.

    Notes
    -----
    The geometric mean is ``10 ** mean(log10(psd), axis=0)``.
    A floor of ``1e-30`` is applied before the log to avoid ``-inf``.
    """
    if hasattr(raw, "get_data"):
        data = raw.get_data()
        sfreq = raw.info["sfreq"]
    else:
        data = np.asarray(raw)
        sfreq = 1.0

    if nperseg is None:
        nperseg = int(sfreq * 4) if sfreq > 1.0 else 1024

    freqs, psd = sp_welch(data, fs=sfreq, nperseg=nperseg, noverlap=nperseg // 2)
    log_psd = np.log10(np.maximum(psd, 1e-30))
    gm_psd = 10 ** log_psd.mean(axis=0)
    return freqs, gm_psd


# =====================================================================
# Per-frequency metrics
# =====================================================================


def noise_surr_ratio(freqs, psd, target_freq, *, center_bw=0.05, side_bw=1.0):
    """R(f₀): ratio of peak power to surrounding spectral floor.

    Matches paper-5 metric **M0**.

    Parameters
    ----------
    freqs : ndarray
        Frequency vector.
    psd : ndarray
        PSD values (same length as *freqs*).
    target_freq : float
        Centre frequency of the line-noise harmonic (Hz).
    center_bw : float
        Half-bandwidth around *target_freq* for the peak (Hz).
        Default ``0.05`` (≈1 FFT bin at standard resolution).
    side_bw : float
        Half-bandwidth for the surrounding sideband floor (Hz).
        Default ``1.0``.

    Returns
    -------
    R : float
        Noise-surround ratio.  ``R ≈ 1`` means the peak is gone;
        ``R >> 1`` means the peak remains.
    """
    center = (freqs >= target_freq - center_bw) & (freqs <= target_freq + center_bw)
    left = (freqs >= target_freq - side_bw) & (freqs < target_freq - center_bw)
    right = (freqs > target_freq + center_bw) & (freqs <= target_freq + side_bw)
    sides = left | right

    if not center.any() or not sides.any():
        return float("nan")
    return float(psd[center].mean() / psd[sides].mean())


def peak_attenuation_db(freqs, psd_before, psd_after, target_freq, *, bw=0.05):
    """Attenuation (dB) of the spectral peak at *target_freq*.

    Matches paper-5 metric **M1**.  Uses the *mean* power in the peak
    band (not max) to be robust to single-bin fluctuations.

    Parameters
    ----------
    freqs : ndarray
        Frequency vector common to both PSDs.
    psd_before, psd_after : ndarray
        Geometric-mean PSDs before and after cleaning.
    target_freq : float
        Centre frequency (Hz).
    bw : float
        Half-bandwidth around the peak (Hz).  Default ``0.05``.

    Returns
    -------
    attenuation_db : float
        Positive means the peak was reduced.
    """
    mask = (freqs >= target_freq - bw) & (freqs <= target_freq + bw)
    if not mask.any():
        return float("nan")
    p_pre = psd_before[mask].mean()
    p_post = psd_after[mask].mean()
    if p_post <= 0:
        return float("inf")
    return float(10 * np.log10(p_pre / p_post))


def below_noise_distortion(
    freqs, psd_before, psd_after, target_freq, *, low_offset=11.0, high_offset=1.0
):
    """Percent change in integrated power below the noise frequency.

    Matches paper-5 metric **M2**::

        %ΔA = 100 × (A_post − A_pre) / A_pre

    where *A* is the integral (trapezoid rule) of the PSD in the band
    ``[f₀ − low_offset, f₀ − high_offset]``.  A value near zero
    indicates minimal spectral distortion.

    Parameters
    ----------
    freqs : ndarray
        Frequency vector.
    psd_before, psd_after : ndarray
        Geometric-mean PSDs.
    target_freq : float
        Fundamental line frequency (Hz).
    low_offset : float
        Lower edge: ``target_freq − low_offset`` (Hz).  Default ``11.0``.
    high_offset : float
        Upper edge: ``target_freq − high_offset`` (Hz).  Default ``1.0``.

    Returns
    -------
    pct_change : float
        Percent change in integrated sub-peak power.  Near zero is ideal;
        positive means power was *added*, negative means power was removed.
    """
    mask = (freqs >= target_freq - low_offset) & (freqs <= target_freq - high_offset)
    if not mask.any():
        return float("nan")
    a_pre = trapezoid(psd_before[mask], freqs[mask])
    a_post = trapezoid(psd_after[mask], freqs[mask])
    if a_pre <= 0:
        return float("nan")
    return float(100.0 * (a_post - a_pre) / a_pre)


def overclean_proportion(
    freqs, psd_after, target_freq, *, low=-0.4, high=0.1, n_sigma=2.0
):
    """Proportion of bins near *target_freq* that show notch over-cleaning.

    Matches paper-5 metric **M3**.  Returns the fraction of frequency
    bins in ``[f₀ + low, f₀ + high]`` where the post-cleaning PSD falls
    below ``sideband_mean − n_sigma × |center − sideband|``.

    Parameters
    ----------
    freqs : ndarray
        Frequency vector.
    psd_after : ndarray
        Geometric-mean PSD after cleaning.
    target_freq : float
        Centre frequency (Hz).
    low, high : float
        Offset range around *target_freq* to inspect (Hz).
    n_sigma : float
        Number of "deviations" for the threshold.

    Returns
    -------
    proportion : float
        Fraction of bins that are over-cleaned (0–1).
    """
    region = (freqs >= target_freq + low) & (freqs <= target_freq + high)
    if not region.any():
        return float("nan")
    region_psd = psd_after[region]

    # Centre: very narrow band around f₀
    center = (freqs >= target_freq - 0.05) & (freqs <= target_freq + 0.05)
    center_power = psd_after[center].mean() if center.any() else region_psd.mean()

    # Sidebands: ±1 Hz excluding the narrow centre
    sides = (freqs >= target_freq - 1.0) & (freqs <= target_freq + 1.0) & ~center
    if sides.any():
        side_power = psd_after[sides].mean()
        deviation = abs(center_power - side_power)
        threshold = side_power - n_sigma * deviation
    else:
        threshold = center_power * 0.5

    return float((region_psd < threshold).mean())


def underclean_proportion(
    freqs, psd_after, target_freq, *, bw=0.05, side_bw=1.0, threshold_frac=0.5
):
    """Proportion of bins near *target_freq* still showing a residual peak.

    Matches paper-5 metric **M4**.  Returns the fraction of bins within
    ``f₀ ± bw`` where the post-cleaning PSD exceeds
    ``surround_level × (1 + threshold_frac)``.

    Parameters
    ----------
    freqs : ndarray
        Frequency vector.
    psd_after : ndarray
        Geometric-mean PSD after cleaning.
    target_freq : float
        Centre frequency (Hz).
    bw : float
        Half-bandwidth around the peak (Hz).  Default ``0.05``.
    side_bw : float
        Half-bandwidth for sideband reference (Hz).  Default ``1.0``.
    threshold_frac : float
        Fraction above sideband level that flags under-cleaning.

    Returns
    -------
    proportion : float
        Fraction of bins that are under-cleaned (0–1).
    """
    center = (freqs >= target_freq - bw) & (freqs <= target_freq + bw)
    sides = ((freqs >= target_freq - side_bw) & (freqs < target_freq - bw)) | (
        (freqs > target_freq + bw) & (freqs <= target_freq + side_bw)
    )
    if not center.any() or not sides.any():
        return float("nan")
    surround_level = psd_after[sides].mean()
    threshold = surround_level * (1 + threshold_frac)
    return float((psd_after[center] > threshold).mean())


# =====================================================================
# Aggregate helper
# =====================================================================


def compute_all_qa_metrics(
    raw_before, raw_after, *, line_freq=50.0, n_harmonics=3, nperseg=None
):
    """Compute all QA metrics per harmonic and return a summary dict.

    Matches the paper-5 ``compute_all_qa_metrics`` function but also
    accepts MNE Raw objects (not just arrays).

    Parameters
    ----------
    raw_before, raw_after : mne.io.Raw | ndarray
        Data before and after line-noise removal.
    line_freq : float
        Fundamental mains frequency (Hz).
    n_harmonics : int
        Number of harmonics to evaluate.
    nperseg : int | None
        FFT segment length for Welch PSD.  If *None*, uses the default
        of :func:`geometric_mean_psd` (``int(sfreq * 4)``).

    Returns
    -------
    metrics : dict
        Contains both nested per-harmonic detail and flat summary keys.

        Per-harmonic (nested):

        - ``harmonics`` — ``{50.0: {R_pre, R_post, ...}, 100.0: {...}}``

        Flat summary (at fundamental f₀):

        - ``R_f0`` — noise-surround ratio at fundamental (post-cleaning)
        - ``R_f0_pre`` — noise-surround ratio at fundamental (pre-cleaning)
        - ``peak_attenuation_db`` — attenuation at fundamental (dB)
        - ``below_noise_pct`` — percent change in sub-peak power
        - ``overclean_proportion`` — mean overclean proportion
        - ``underclean_proportion`` — mean underclean proportion
        - ``harmonics_hz`` — list of harmonic frequencies evaluated
        - ``R_per_harmonic`` — list of R_post values per harmonic
        - ``attenuation_per_harmonic_db`` — list of attenuations per harmonic
    """
    f, geo_pre = geometric_mean_psd(raw_before, nperseg=nperseg)
    _, geo_post = geometric_mean_psd(raw_after, nperseg=nperseg)

    sfreq_nyq = f[-1]  # approximate Nyquist
    harmonics_nested = {}
    harmonics_hz = []
    r_values = []
    atten_values = []
    oc_values = []
    uc_values = []

    for k in range(1, n_harmonics + 1):
        fk = line_freq * k
        if fk >= sfreq_nyq:
            continue

        harmonics_hz.append(float(fk))

        r_pre = noise_surr_ratio(f, geo_pre, fk)
        r_post = noise_surr_ratio(f, geo_post, fk)
        atten = peak_attenuation_db(f, geo_pre, geo_post, fk)
        below = below_noise_distortion(f, geo_pre, geo_post, fk)
        oc = overclean_proportion(f, geo_post, fk)
        uc = underclean_proportion(f, geo_post, fk)

        harmonics_nested[fk] = {
            "R_pre": r_pre,
            "R_post": r_post,
            "attenuation_dB": atten,
            "below_noise_pct": below,
            "overclean_prop": oc,
            "underclean_prop": uc,
        }

        r_values.append(r_post)
        atten_values.append(atten)
        oc_values.append(oc)
        uc_values.append(uc)

    return {
        # ── Nested per-harmonic detail ──
        "harmonics": harmonics_nested,
        # ── Flat summary (backward-compatible keys) ──
        "R_f0": float(r_values[0]) if r_values else float("nan"),
        "R_f0_pre": (
            float(noise_surr_ratio(f, geo_pre, line_freq))
            if harmonics_hz
            else float("nan")
        ),
        "peak_attenuation_db": (
            float(atten_values[0]) if atten_values else float("nan")
        ),
        "below_noise_pct": (
            float(harmonics_nested[line_freq]["below_noise_pct"])
            if line_freq in harmonics_nested
            else float("nan")
        ),
        "overclean_proportion": (float(np.nanmean(oc_values)) if oc_values else 0.0),
        "underclean_proportion": (float(np.nanmean(uc_values)) if uc_values else 0.0),
        # ── Per-harmonic lists ──
        "harmonics_hz": harmonics_hz,
        "R_per_harmonic": [float(r) for r in r_values],
        "attenuation_per_harmonic_db": [float(a) for a in atten_values],
    }
