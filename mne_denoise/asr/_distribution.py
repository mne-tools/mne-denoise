"""EEG amplitude-distribution fitting for ASR calibration thresholds.

Robust fit of a generalised-Gaussian model to per-window RMS values
(``fit_rms_distribution``, the ASR calibration convention) plus the histogram and
robust location/scale helpers it relies on.
"""

from __future__ import annotations

from typing import Any, Literal, overload

import numpy as np
from scipy import special

from ..progress import _emit_progress, _ProgressCallback
from ._validation import _round_half_up

_AASR_BETA_GRID = 1.7 + 0.15 * np.arange(13, dtype=np.float64)


@overload
def fit_rms_distribution(
    values: np.ndarray,
    *,
    min_clean_fraction: float = ...,
    max_dropout_fraction: float = ...,
    fit_quantiles: tuple[float, float] = ...,
    beta_grid: np.ndarray | None = ...,
    return_info: Literal[False] = ...,
) -> tuple[float, float]: ...


@overload
def fit_rms_distribution(
    values: np.ndarray,
    *,
    min_clean_fraction: float = ...,
    max_dropout_fraction: float = ...,
    fit_quantiles: tuple[float, float] = ...,
    beta_grid: np.ndarray | None = ...,
    return_info: Literal[True],
) -> tuple[float, float, dict[str, Any]]: ...


def fit_rms_distribution(
    values: np.ndarray,
    *,
    min_clean_fraction: float = 0.25,
    max_dropout_fraction: float = 0.1,
    fit_quantiles: tuple[float, float] = (0.022, 0.6),
    beta_grid: np.ndarray | None = None,
    return_info: bool = False,
) -> tuple[float, float] | tuple[float, float, dict[str, Any]]:
    """Fit robust clean EEG RMS statistics.

    This implements the truncated generalized-Gaussian grid search used by
    the ASR calibration. The fitter sorts finite RMS values,
    searches over plausible low-tail dropout offsets and clean interval
    widths, and selects the generalized-Gaussian shape with minimum
    histogram KL divergence.

    Parameters
    ----------
    values : ndarray, shape (n_windows,)
        RMS or amplitude statistics for one component/channel.
    min_clean_fraction : float
        Minimum fraction of values assumed to be clean.
    max_dropout_fraction : float
        Maximum low-tail fraction that may be ignored as dropouts.
    fit_quantiles : tuple of float
        Lower and upper quantile span used for the clean interval search.
        The upper value also controls the preferred interval width.
    beta_grid : ndarray | None
        Generalized-Gaussian shape grid. If ``None``, use values from 1.7 to
        3.5, matching the range commonly cited for ASR ports.
    return_info : bool
        If True, return an additional diagnostics dictionary.

    Returns
    -------
    mu : float
        Robust location estimate of the clean RMS distribution.
    sigma : float
        Robust standard-deviation estimate of the clean RMS distribution.
    info : dict
        Returned only when ``return_info=True``. Contains ``beta``,
        ``fit_error``, ``fit_interval``, and ``n_fit_samples``.

    Examples
    --------
    Calculate robust statistics for a noisy array, ignoring massive outliers:

    >>> import numpy as np
    >>> from mne_denoise.asr import fit_rms_distribution
    >>> rng = np.random.default_rng(42)
    >>> clean = np.abs(rng.normal(10.0, 2.0, 5000))
    >>> artifacts = np.abs(rng.normal(30.0, 10.0, 500))
    >>> noisy_data = np.concatenate([clean, artifacts])
    >>> mu, sigma = fit_rms_distribution(noisy_data)
    >>> print(f"Robust mean: {mu:.1f}")
    Robust mean: 10.0
    """
    if not (0 <= max_dropout_fraction < 1):
        raise ValueError("max_dropout_fraction must be in [0, 1)")
    if not (0 < min_clean_fraction <= 1):
        raise ValueError("min_clean_fraction must be in (0, 1]")
    if max_dropout_fraction + min_clean_fraction >= 1:
        raise ValueError(
            "max_dropout_fraction + min_clean_fraction must be less than 1"
        )
    q_low, q_high = fit_quantiles
    if not (0 <= q_low < q_high <= 1):
        raise ValueError("fit_quantiles must satisfy 0 <= low < high <= 1")

    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Cannot fit ASR thresholds from empty RMS distribution")
    finite = np.sort(finite)

    beta_grid = (
        1.7 + 0.15 * np.arange(13, dtype=np.float64)
        if beta_grid is None
        else np.asarray(beta_grid, dtype=np.float64)
    )
    if beta_grid.size == 0:
        raise ValueError("beta_grid must contain positive values")
    if np.any(beta_grid <= 1) or np.any(beta_grid >= 7):
        raise ValueError("beta_grid values must be in the open interval (1, 7)")

    best = _fit_rms_distribution_grid_search(
        finite,
        min_clean_fraction=min_clean_fraction,
        max_dropout_fraction=max_dropout_fraction,
        fit_quantiles=(q_low, q_high),
        beta_grid=beta_grid,
    )

    if return_info:
        info = {
            "beta": float(best["beta"]),
            "fit_error": float(best["fit_error"]),
            "fit_interval": tuple(best["fit_interval"]),
            "n_fit_samples": int(best["n_fit_samples"]),
            "score": float(best["score"]),
        }
        return float(best["mu"]), float(best["sigma"]), info
    return float(best["mu"]), float(best["sigma"])


def _fit_rms_distribution_grid_search(
    values: np.ndarray,
    *,
    min_clean_fraction: float,
    max_dropout_fraction: float,
    fit_quantiles: tuple[float, float],
    beta_grid: np.ndarray,
) -> dict[str, Any]:
    """Perform a truncated generalized-Gaussian grid search for RMS distributions.

    Searches for the optimal location (mu) and scale (sigma) by sliding a
    theoretical generalized Gaussian distribution across a histogram of the
    provided data, ignoring extreme artifact tails.

    Parameters
    ----------
    values : ndarray, shape (n_windows,)
        Sorted array of valid (finite) window RMS values.
    min_clean_fraction : float
        Minimum fraction of the data assumed to be clean (non-artifact).
    max_dropout_fraction : float
        Maximum low-tail fraction that may be ignored as sensor dropouts.
    fit_quantiles : tuple of float
        Lower and upper quantiles defining the core search interval.
    beta_grid : ndarray
        Grid of generalized Gaussian shape parameters to evaluate.

    Returns
    -------
    dict
        Dictionary containing the best fit parameters: 'mu', 'sigma', 'beta',
        'fit_error', 'fit_interval', 'n_fit_samples', and 'score'.
    """
    q_low, q_high = fit_quantiles
    step_sizes = (0.01, 0.01)
    n_values = values.size

    bounds_by_beta = []
    rescale = np.empty(beta_grid.size, dtype=np.float64)
    for idx, beta in enumerate(beta_grid):
        sign = np.sign(np.asarray([q_low, q_high]) - 0.5)
        gamma_arg = sign * (2.0 * np.asarray([q_low, q_high]) - 1.0)
        bounds = sign * special.gammaincinv(1.0 / beta, gamma_arg) ** (1.0 / beta)
        bounds_by_beta.append(bounds)
        rescale[idx] = beta / (2.0 * special.gamma(1.0 / beta))

    max_width = q_high - q_low
    min_width = min_clean_fraction * max_width
    n_range = _round_half_up(n_values * max_width)
    offsets = np.asarray(
        [
            _round_half_up(n_values * offset)
            for offset in np.arange(
                q_low,
                q_low + max_dropout_fraction + np.finfo(float).eps,
                step_sizes[0],
            )
        ],
        dtype=int,
    )
    row_idx = np.arange(n_range, dtype=int)[:, np.newaxis]
    sample_idx = row_idx + offsets[np.newaxis, :]
    sample_idx = np.minimum(sample_idx, n_values - 1)
    ranges = values[sample_idx]
    range_start = ranges[0].copy()
    ranges = ranges - range_start[np.newaxis, :]

    opt_val = np.inf
    opt_beta = np.nan
    opt_bounds = None
    opt_lu = None
    opt_m = 0
    widths = np.arange(max_width, min_width - np.finfo(float).eps, -step_sizes[1])
    for width in widths:
        m = _round_half_up(n_values * width)
        if m < 2 or m > ranges.shape[0]:
            continue
        denominators = ranges[m - 1]
        valid = denominators > np.finfo(float).eps
        if not np.any(valid):
            continue
        nbins = max(1, _round_half_up(3.0 * np.log2(1.0 + m / 2.0)))
        scaled = np.empty((m, ranges.shape[1]), dtype=np.float64)
        scaled[:, valid] = ranges[:m, valid] * (nbins / denominators[valid])
        scaled[:, ~valid] = np.nan
        counts = _histc_scaled_bins(scaled, nbins)
        logq = np.log(counts + 0.01)

        for beta_idx, beta in enumerate(beta_grid):
            bounds = bounds_by_beta[beta_idx]
            x = bounds[0] + ((np.arange(nbins) + 0.5) / nbins) * np.diff(bounds)[0]
            p = np.exp(-(np.abs(x) ** beta)) * rescale[beta_idx]
            p = p / np.sum(p)
            kl = np.sum(p[:, np.newaxis] * (np.log(p)[:, np.newaxis] - logq), axis=0)
            kl = kl + np.log(m)
            kl[~valid] = np.inf
            idx = int(np.argmin(kl))
            min_val = float(kl[idx])
            if min_val < opt_val:
                opt_val = min_val
                opt_beta = float(beta)
                opt_bounds = bounds
                opt_lu = (
                    float(range_start[idx]),
                    float(range_start[idx] + ranges[m - 1, idx]),
                )
                opt_m = int(m)

    if opt_lu is None or opt_bounds is None:
        mu, sigma = _robust_location_scale(values)
        return {
            "mu": mu,
            "sigma": sigma,
            "beta": np.nan,
            "fit_error": np.nan,
            "score": np.nan,
            "fit_interval": (0.0, 1.0),
            "n_fit_samples": int(n_values),
        }

    alpha = (opt_lu[1] - opt_lu[0]) / np.diff(opt_bounds)[0]
    mu = opt_lu[0] - opt_bounds[0] * alpha
    sigma = np.sqrt(
        (alpha**2) * special.gamma(3.0 / opt_beta) / special.gamma(1.0 / opt_beta)
    )
    return {
        "mu": float(mu),
        "sigma": float(sigma),
        "beta": float(opt_beta),
        "fit_error": float(opt_val),
        "score": float(opt_val),
        "fit_interval": (float(opt_lu[0]), float(opt_lu[1])),
        "n_fit_samples": int(opt_m),
    }


def _histc_scaled_bins(values: np.ndarray, nbins: int) -> np.ndarray:
    """Histogram columns into discrete scale bins.

    This function mimics the strict non-standard behavior of MATLAB's ``histc``
    for precise compatibility with the legacy ASR calibration grid search.

    Parameters
    ----------
    values : ndarray, shape (n_samples, n_columns)
        The scaled standard normal values to be binned.
    nbins : int
        The total number of bins to calculate.

    Returns
    -------
    counts : ndarray, shape (nbins, n_columns)
        The resulting histogram counts per column.
    """
    counts = np.zeros((nbins, values.shape[1]), dtype=np.float64)
    for col in range(values.shape[1]):
        finite = values[:, col]
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            continue
        bins = np.floor(finite).astype(int)
        bins = np.clip(bins, 0, nbins - 1)
        counts[:, col] = np.bincount(bins, minlength=nbins)
    return counts


def _robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    """Estimate robust location and scale using median absolute deviation (MAD).

    Parameters
    ----------
    values : ndarray
        The input data array.

    Returns
    -------
    mu : float
        The median of the data.
    sigma : float
        The scaled MAD of the data (1.4826 * MAD), falling back to standard
        deviation if MAD is strictly zero.
    """
    values = np.asarray(values, dtype=np.float64)
    mu = float(np.median(values))
    mad = float(np.median(np.abs(values - mu)))
    sigma = 1.4826 * mad
    if sigma <= np.finfo(float).eps:
        sigma = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    if sigma <= np.finfo(float).eps:
        sigma = max(abs(mu) * 1e-6, np.finfo(float).eps)
    return mu, sigma


def _fit_adaptive_thresholds(
    X: np.ndarray,
    V: np.ndarray,
    sfreq: float,
    window_length: float,
    window_overlap: float,
    cutoff: float,
    min_clean_fraction: float,
    max_dropout_fraction: float,
    callback: _ProgressCallback | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Fit adaptive ASR thresholds for each component via its RMS distribution.

    Parameters
    ----------
    X : ndarray, shape (n_channels, n_times)
        The filtered continuous data.
    V : ndarray, shape (n_channels, n_components)
        The unmixing matrix (eigenvectors of the calibration covariance).
    sfreq : float
        The sampling frequency of the data.
    window_length : float
        The length of the moving window in seconds.
    window_overlap : float
        The overlap fraction between adjacent windows.
    cutoff : float
        The cutoff multiplier for the standard deviation (Z-score).
    min_clean_fraction : float
        The minimum fraction of clean data expected.
    max_dropout_fraction : float
        The maximum fraction of dropout data expected.
    callback : callable | None
        Called synchronously after each component threshold is fitted. Callback
        return values are ignored and callback exceptions propagate unchanged.

    Returns
    -------
    thresholds : ndarray, shape (n_components,)
        The computed adaptive threshold for each component.
    info_out : dict
        A dictionary containing diagnostic arrays for mu, sigma, beta,
        fit error, fit intervals, window starts, and window length in samples.
    """
    from ._windowing import _get_fractional_window_starts

    n_times = X.shape[1]
    win_len = _round_half_up(window_length * sfreq)
    starts = _get_fractional_window_starts(n_times, win_len, window_overlap)
    projected = np.abs(X.T @ V)

    thresholds = np.empty(projected.shape[1], dtype=np.float64)
    mu_values = np.empty(projected.shape[1], dtype=np.float64)
    sigma_values = np.empty(projected.shape[1], dtype=np.float64)
    beta_values = np.empty(projected.shape[1], dtype=np.float64)
    fit_errors = np.empty(projected.shape[1], dtype=np.float64)
    fit_intervals = np.empty((projected.shape[1], 2), dtype=np.float64)
    for comp_idx in range(projected.shape[1]):
        rms = np.empty(len(starts), dtype=np.float64)
        comp = projected[:, comp_idx]
        for idx, start in enumerate(starts):
            segment = comp[start : start + win_len]
            rms[idx] = np.sqrt(np.mean(segment**2))
        mu, sigma, info = fit_rms_distribution(
            rms,
            min_clean_fraction=min_clean_fraction,
            max_dropout_fraction=max_dropout_fraction,
            return_info=True,
        )
        mu_values[comp_idx] = mu
        sigma_values[comp_idx] = sigma
        beta_values[comp_idx] = info["beta"]
        fit_errors[comp_idx] = info["fit_error"]
        fit_intervals[comp_idx] = info["fit_interval"]
        thresholds[comp_idx] = mu + cutoff * sigma
        _emit_progress(
            callback,
            method="adaptive_asr",
            stage="calibration",
            current=comp_idx + 1,
            total=projected.shape[1],
            component=comp_idx + 1,
            metric=float(thresholds[comp_idx]),
        )

    info_out: dict[str, Any] = {
        "mu": mu_values,
        "sigma": sigma_values,
        "beta": beta_values,
        "fit_error": fit_errors,
        "fit_interval": fit_intervals,
        "window_starts": starts,
        "window_length_samples": int(win_len),
    }
    return thresholds, info_out
