"""ASR calibration-window helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._distribution import fit_rms_distribution
from ._validation import (
    _check_enough_samples,
    _round_half_up,
    _validate_common_params,
)


def _get_window_starts(n_times: int, win_len: int, overlap: float) -> np.ndarray:
    """Compute sample indices for the start of overlapping sliding windows."""
    if win_len < 2:
        raise ValueError("window length must be at least 2 samples")
    if win_len > n_times:
        raise ValueError(
            f"Window length ({win_len} samples) exceeds data length ({n_times} samples)"
        )
    step = max(1, int(round(win_len * (1 - overlap))))
    starts = list(range(0, n_times - win_len + 1, step))
    last = n_times - win_len
    if starts[-1] != last:
        starts.append(last)
    return np.asarray(starts, dtype=int)


def _get_fractional_window_starts(
    n_times: int,
    win_len: int,
    overlap: float,
) -> np.ndarray:
    """Return window starts using precise fractional overlap step sizes."""
    if win_len < 2:
        raise ValueError("window length must be at least 2 samples")
    if win_len > n_times:
        raise ValueError(
            f"Window length ({win_len} samples) exceeds data length ({n_times} samples)"
        )
    step = win_len * (1.0 - overlap)
    starts_1_based = np.arange(1.0, n_times - win_len + np.finfo(float).eps, step)
    starts = np.asarray(
        [_round_half_up(start) - 1 for start in starts_1_based], dtype=int
    )
    return np.unique(starts)


def _get_window_weights(win_len: int) -> np.ndarray:
    """Generate Hanning weights for smooth cross-fading."""
    if win_len <= 2:
        return np.ones(win_len, dtype=np.float64)
    return np.hanning(win_len + 2)[1:-1].astype(np.float64)


def _compute_window_rms(X: np.ndarray, starts: np.ndarray, win_len: int) -> np.ndarray:
    """Compute per-channel RMS values over a set of windows."""
    windows = starts[:, np.newaxis] + np.arange(win_len, dtype=int)[np.newaxis, :]
    squared = X[:, windows] ** 2
    return np.sqrt(np.sum(squared, axis=2) / win_len)


def _resolve_bad_channel_count(
    max_bad_channels: float | int,
    n_channels: int,
) -> int:
    """Resolve the tolerated bad-channel fraction into an absolute count."""
    if isinstance(max_bad_channels, float) and 0 < max_bad_channels < 1:
        resolved = _round_half_up(n_channels * max_bad_channels)
    else:
        resolved = int(max_bad_channels)
    if resolved < 0:
        raise ValueError("max_bad_channels must be non-negative")
    return min(resolved, n_channels)


def _select_clean_windows(
    X: np.ndarray,
    starts: np.ndarray,
    win_len: int,
    ref_max_bad_channels: float,
    ref_tolerances: tuple[float, float],
    max_dropout_fraction: float,
    min_clean_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Identify clean calibration windows using Z-scored channel RMS distributions."""
    diagnostics = _compute_window_diagnostics(
        X,
        starts,
        win_len,
        max_bad_channels=ref_max_bad_channels,
        zthresholds=ref_tolerances,
        max_dropout_fraction=max_dropout_fraction,
        min_clean_fraction=min_clean_fraction,
    )
    clean = diagnostics["window_keep_mask"].copy()
    if not np.any(clean):
        zscores = diagnostics["window_rms_zscores"]
        penalty = np.mean(np.maximum(zscores - max(ref_tolerances), 0.0), axis=1)
        clean[np.argmin(penalty)] = True
    return clean, diagnostics["window_rms_zscores"]


def _compute_window_diagnostics(
    X: np.ndarray,
    starts: np.ndarray,
    win_len: int,
    max_bad_channels: float | int,
    zthresholds: tuple[float, float],
    max_dropout_fraction: float,
    min_clean_fraction: float,
    fit_quantiles: tuple[float, float] = (0.022, 0.6),
    beta_grid: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute detailed window retention diagnostics over a predefined grid."""
    n_channels = X.shape[0]
    rms = _compute_window_rms(X, starts, win_len)

    zscores = np.empty_like(rms)
    mu_values = np.empty(n_channels, dtype=np.float64)
    sigma_values = np.empty(n_channels, dtype=np.float64)
    beta_values = np.empty(n_channels, dtype=np.float64)
    fit_errors = np.empty(n_channels, dtype=np.float64)
    fit_intervals = np.empty((n_channels, 2), dtype=np.float64)
    fit_sample_counts = np.empty(n_channels, dtype=int)
    for ch_idx in range(n_channels):
        mu, sigma, info = fit_rms_distribution(
            rms[ch_idx],
            min_clean_fraction=min_clean_fraction,
            max_dropout_fraction=max_dropout_fraction,
            fit_quantiles=fit_quantiles,
            beta_grid=beta_grid,
            return_info=True,
        )
        mu_values[ch_idx] = mu
        sigma_values[ch_idx] = sigma
        beta_values[ch_idx] = info["beta"]
        fit_errors[ch_idx] = info["fit_error"]
        fit_intervals[ch_idx] = info["fit_interval"]
        fit_sample_counts[ch_idx] = info["n_fit_samples"]
        zscores[ch_idx] = (rms[ch_idx] - mu) / max(sigma, np.finfo(float).eps)

    tolerated_bad_channels = _resolve_bad_channel_count(max_bad_channels, n_channels)
    window_remove_mask = np.zeros(starts.size, dtype=bool)
    if tolerated_bad_channels < n_channels:
        swz = np.sort(zscores, axis=0)
        z_low, z_high = zthresholds
        if z_high > 0:
            window_remove_mask |= swz[-1 - tolerated_bad_channels] > z_high
        if z_low < 0:
            window_remove_mask |= swz[tolerated_bad_channels] < z_low
    window_keep_mask = ~window_remove_mask

    return {
        "window_starts": starts,
        "window_stops": starts + win_len,
        "window_rms": rms.T,
        "window_rms_zscores": zscores.T,
        "window_keep_mask": window_keep_mask,
        "window_remove_mask": window_remove_mask,
        "mu": mu_values,
        "sigma": sigma_values,
        "beta": beta_values,
        "fit_error": fit_errors,
        "fit_interval": fit_intervals,
        "n_fit_samples": fit_sample_counts,
        "n_windows": int(starts.size),
        "n_rejected_windows": int(np.sum(window_remove_mask)),
    }


def _concatenate_windows(
    X: np.ndarray,
    starts: np.ndarray,
    win_len: int,
) -> np.ndarray:
    """Concatenate a subset of windows into a continuous flat array."""
    out = np.empty((X.shape[0], len(starts) * win_len), dtype=np.float64)
    for idx, start in enumerate(starts):
        out[:, idx * win_len : (idx + 1) * win_len] = X[:, start : start + win_len]
    return out


def _create_sample_mask_from_windows(
    n_times: int,
    starts: np.ndarray,
    win_len: int,
    window_remove_mask: np.ndarray,
) -> np.ndarray:
    """Project window-level rejection decisions to a sample-level binary mask."""
    sample_mask = np.ones(n_times, dtype=bool)
    for start in starts[np.asarray(window_remove_mask, dtype=bool)]:
        sample_mask[start : start + win_len] = False
    return sample_mask


def _create_good_sample_mask_from_mne(
    raw: Any, prefixes: tuple[str, ...]
) -> np.ndarray:
    """Create a sample mask identifying data not marked by specific MNE annotations."""
    n_times = raw.n_times
    mask = np.ones(n_times, dtype=bool)
    if not hasattr(raw, "annotations") or len(raw.annotations) == 0:
        return mask
    sfreq = raw.info["sfreq"]
    first_time = (
        raw.first_time
        if hasattr(raw, "first_time")
        else getattr(raw, "_first_time", 0.0)
    )
    for onset, duration, description in zip(
        raw.annotations.onset,
        raw.annotations.duration,
        raw.annotations.description,
    ):
        desc = str(description).lower()
        if not any(desc.startswith(prefix.lower()) for prefix in prefixes):
            continue
        start = max(0, int(np.floor((onset - first_time) * sfreq)))
        stop = min(n_times, int(np.ceil((onset + duration - first_time) * sfreq)))
        mask[start:stop] = False
    return mask


def _merge_sample_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping or adjacent half-open sample spans."""
    if not spans:
        return []
    spans = sorted(spans)
    merged = [spans[0]]
    for start, stop in spans[1:]:
        last_start, last_stop = merged[-1]
        if start <= last_stop:
            merged[-1] = (last_start, max(last_stop, stop))
        else:
            merged.append((start, stop))
    return merged


def _mask_to_sample_spans(mask: np.ndarray) -> list[tuple[int, int]]:
    """Convert a 1-D boolean mask to half-open sample spans."""
    mask = np.asarray(mask, dtype=bool).ravel()
    if mask.size == 0:
        return []
    edges = np.diff(np.concatenate(([False], mask, [False])).astype(int))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), stops.tolist()))


def compute_clean_window_mask(
    X: np.ndarray,
    sfreq: float,
    *,
    max_bad_channels: float | int = 0.2,
    zthresholds: tuple[float, float] = (-3.5, 5.0),
    window_length: float = 1.0,
    window_overlap: float = 0.66,
    max_dropout_fraction: float = 0.1,
    min_clean_fraction: float = 0.25,
    fit_quantiles: tuple[float, float] = (0.022, 0.6),
    beta_grid: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute a retained-sample mask from ASR window statistics.

    Parameters
    ----------
    X : ndarray, shape (n_channels, n_times)
        Continuous data.
    sfreq : float
        Sampling frequency in Hz.
    max_bad_channels : float or int, default=0.2
        Maximum bad-channel fraction or count per window.
    zthresholds : tuple of float, default=(-3.5, 5.0)
        Lower and upper channel-RMS z-score limits.
    window_length : float, default=1.0
        Window length in seconds.
    window_overlap : float, default=0.66
        Window overlap fraction.
    max_dropout_fraction : float, default=0.1
        Low-tail fraction ignored during RMS fitting.
    min_clean_fraction : float, default=0.25
        Minimum clean fraction used for RMS fitting.
    fit_quantiles : tuple of float, default=(0.022, 0.6)
        Quantile interval for the RMS fit.
    beta_grid : ndarray or None, default=None
        Optional generalized-Gaussian shape grid.

    Returns
    -------
    sample_mask : ndarray, shape (n_times,)
        Boolean mask of retained samples.
    diagnostics : dict
        Window-level RMS, z-score, and mask diagnostics.
    """
    _validate_common_params(
        sfreq=sfreq,
        cutoff=1.0,
        window_length=window_length,
        window_overlap=window_overlap,
        max_dropout_fraction=max_dropout_fraction,
        min_clean_fraction=min_clean_fraction,
        regularization=1e-8,
    )
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(
            f"ASR window rejection expects a 2D array (n_channels, n_times), got {X.shape}"
        )
    if X.shape[0] < 1:
        raise ValueError("ASR window rejection requires at least one channel")
    if not np.all(np.isfinite(X)):
        X = np.nan_to_num(X, copy=True)

    n_channels, n_times = X.shape
    _check_enough_samples(n_times, sfreq, window_length)
    window_length_samples = _round_half_up(window_length * sfreq)
    starts = _get_fractional_window_starts(
        n_times, window_length_samples, window_overlap
    )
    diagnostics = _compute_window_diagnostics(
        X,
        starts,
        window_length_samples,
        max_bad_channels=max_bad_channels,
        zthresholds=zthresholds,
        max_dropout_fraction=max_dropout_fraction,
        min_clean_fraction=min_clean_fraction,
        fit_quantiles=fit_quantiles,
        beta_grid=beta_grid,
    )
    window_remove_mask = diagnostics["window_remove_mask"]

    sample_mask = np.ones(n_times, dtype=bool)
    for start in starts[window_remove_mask]:
        sample_mask[start : start + window_length_samples] = False

    diagnostics = dict(diagnostics)
    diagnostics["sample_mask"] = sample_mask
    diagnostics["fraction_retained_samples"] = float(np.mean(sample_mask))
    diagnostics["fraction_rejected_samples"] = float(1.0 - np.mean(sample_mask))
    return sample_mask, diagnostics


def _extract_clean_calibration_samples(
    X: np.ndarray,
    sfreq: float,
    window_length: float,
    window_overlap: float,
    max_bad_channels: float,
    zthresholds: tuple[float, float],
    max_dropout_fraction: float,
    min_clean_fraction: float,
    beta_grid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Extract a subset of continuous clean calibration data using robust diagnostics."""
    win_len = _round_half_up(window_length * sfreq)
    starts = _get_fractional_window_starts(X.shape[1], win_len, window_overlap)
    diagnostics = _compute_window_diagnostics(
        X,
        starts,
        win_len,
        max_bad_channels=max_bad_channels,
        zthresholds=zthresholds,
        max_dropout_fraction=max_dropout_fraction,
        min_clean_fraction=min_clean_fraction,
        fit_quantiles=(0.022, 0.6),
        beta_grid=beta_grid,
    )
    if not np.any(diagnostics["window_keep_mask"]):
        scores = diagnostics["window_rms_zscores"]
        penalty = np.mean(np.maximum(scores - max(zthresholds), 0.0), axis=1)
        diagnostics["window_keep_mask"][int(np.argmin(penalty))] = True
        diagnostics["window_remove_mask"] = ~diagnostics["window_keep_mask"]

    sample_mask = _create_sample_mask_from_windows(
        X.shape[1],
        starts,
        win_len,
        diagnostics["window_remove_mask"],
    )
    return X[:, sample_mask], sample_mask, diagnostics
