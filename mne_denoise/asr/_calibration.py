"""ASR Calibration Module.

This module is responsible for fitting the core Artifact Subspace Reconstruction
(ASR) statistical model from continuous reference data.

The primary entry point, ``calibrate_asr``, executes a multi-step pipeline:
1. Automatically identifies "clean" spatial covariance windows.
2. Aggregates these windows using robust geometry (Standard or Riemannian).
3. Derives the mixing square-root matrix ``M`` to map data into principal space.
4. Fits a generalized Gaussian distribution to windowed RMS values to calculate
   the direction-dependent threshold cutoff matrix ``T``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .._logging import logger, verbose
from ..progress import _emit_progress, _ProgressCallback, _validate_callback
from ._covariance import _aggregate_block_covariances
from ._distribution import fit_rms_distribution
from ._filters import _design_statistics_filter, _lfilter_channels
from ._spd import (
    _regularize_spd,
    _riemannian_nonlinear_eigenspace,
    _sqrt_and_eig,
    _sqrtm_spd,
)
from ._types import ASRState
from ._validation import (
    _check_enough_samples,
    _round_half_up,
    _validate_array_2d,
    _validate_common_params,
)
from ._windowing import (
    _create_sample_mask_from_windows,
    _get_fractional_window_starts,
    _select_clean_windows,
)


@verbose
def calibrate_asr(
    X: np.ndarray,
    sfreq: float,
    cutoff: float = 20.0,
    window_length: float = 0.5,
    window_overlap: float = 0.66,
    calibration: str = "auto",
    calibration_window_length: float = 1.0,
    calibration_window_overlap: float = 0.66,
    ref_max_bad_channels: float = 0.075,
    ref_tolerances: tuple[float, float] = (-np.inf, 5.5),
    blocksize: int = 10,
    max_dropout_fraction: float = 0.1,
    min_clean_fraction: float = 0.25,
    cov_estimator: str = "geometric_median",
    regularization: float = 1e-8,
    filter_kind: str = "none",
    method: str = "standard",
    max_mem_mb: int | None = 512,
    callback=None,
    verbose: bool | str | int | None = None,
) -> tuple[ASRState, dict[str, Any]]:
    """Calibrate a standard ASR model from continuous data.

    Parameters
    ----------
    X : ndarray, shape (n_channels, n_times)
        Continuous calibration data.
    sfreq : float
        Sampling frequency in Hz.
    cutoff : float
        ASR threshold multiplier. Lower values clean more aggressively.
    window_length : float
        Processing/statistics window length in seconds.
    window_overlap : float
        Overlap fraction for threshold-fitting windows.
    calibration : {'auto', 'manual'}
        Whether to select clean calibration windows automatically or use all
        supplied samples.
    calibration_window_length : float
        Window length in seconds for automatic clean-window selection.
    calibration_window_overlap : float
        Overlap fraction for automatic clean-window selection.
    ref_max_bad_channels : float
        Maximum fraction of channels that may exceed ``ref_tolerances`` for a
        calibration window to be retained.
    ref_tolerances : tuple of float
        Lower and upper robust z-score tolerances for clean-window selection.
    blocksize : int
        Number of successive samples averaged into each covariance block for
        robust calibration covariance estimation.
    max_dropout_fraction : float
        Fraction of the lowest RMS values excluded while fitting thresholds.
    min_clean_fraction : float
        Minimum central fraction used to estimate clean RMS statistics.
    cov_estimator : {'geometric_median', 'mean', 'median'}
        Robust aggregation rule for calibration-window covariance matrices.
    regularization : float
        Relative eigenvalue floor used for SPD regularization.
    filter_kind : {'none', 'asr', 'highpass'}
        Statistics-only filter. ``'asr'`` applies the original inverse-EEG
        Yule-Walker pre-emphasis filter, ``'highpass'`` applies a lightweight
        high-pass filter, and ``'none'`` avoids implicit filtering.
    max_mem_mb : int | None
        Reserved memory limit for future chunking. Present for API stability.
    callback : callable | None
        Called synchronously after each principal-component threshold is fitted
        with an ASR calibration progress event. Callback return values are
        ignored and callback exceptions propagate unchanged.
    verbose : bool | str | int | None
        MNE-style logging level. Calibration details are emitted at DEBUG;
        the owning estimator reports the user-facing calibration result.

    Returns
    -------
    state : ASRState
        Fitted ASR state containing the threshold matrix T and mixing matrix M.
    diagnostics : dict
        Calibration diagnostics, including filter state and geometry info.

    Examples
    --------
    Calibrate an ASR model from a 10-channel, 1000-sample array:

    >>> import numpy as np
    >>> from mne_denoise.asr import calibrate_asr
    >>> rng = np.random.default_rng(42)
    >>> data = rng.standard_normal((10, 1000))
    >>> state, diagnostics = calibrate_asr(data, sfreq=250.0, cutoff=20.0)
    >>> print(f"Threshold matrix shape: {state.T.shape}")
    Threshold matrix shape: (10, 10)
    """
    callback = _validate_callback(callback)
    _validate_common_params(
        sfreq=sfreq,
        cutoff=cutoff,
        window_length=window_length,
        window_overlap=window_overlap,
        max_dropout_fraction=max_dropout_fraction,
        min_clean_fraction=min_clean_fraction,
        regularization=regularization,
    )
    if calibration not in ("auto", "manual"):
        raise ValueError("calibration must be 'auto' or 'manual'")
    if cov_estimator not in ("geometric_median", "mean", "median"):
        raise ValueError(
            "cov_estimator must be 'geometric_median', 'mean', or 'median'"
        )
    if method not in ("standard", "riemannian", "riemannian_windowed"):
        raise ValueError(
            "method must be 'standard', 'riemannian', or 'riemannian_windowed'"
        )
    if blocksize < 1:
        raise ValueError("blocksize must be at least 1")

    X = _validate_array_2d(X)
    n_channels, n_times = X.shape
    _check_enough_samples(n_times, sfreq, min(window_length, calibration_window_length))

    cal_len = _round_half_up(calibration_window_length * sfreq)

    if calibration == "auto":
        cal_starts = _get_fractional_window_starts(
            n_times,
            cal_len,
            calibration_window_overlap,
        )
        clean_window_mask, clean_window_scores = _select_clean_windows(
            X,
            cal_starts,
            cal_len,
            ref_max_bad_channels=ref_max_bad_channels,
            ref_tolerances=ref_tolerances,
            max_dropout_fraction=max_dropout_fraction,
            min_clean_fraction=min_clean_fraction,
        )
        clean_sample_mask = _create_sample_mask_from_windows(
            n_times,
            cal_starts,
            cal_len,
            ~clean_window_mask,
        )
        X_calibration = X[:, clean_sample_mask]
    else:
        # Manual calibration consumes all supplied samples directly. Do not
        # impose the longer automatic-selection window on pointwise backends
        # such as Juggler; only the threshold window must fit.
        cal_starts = np.array([], dtype=int)
        clean_window_mask = np.ones(len(cal_starts), dtype=bool)
        clean_window_scores = np.zeros((len(cal_starts), n_channels), dtype=np.float64)
        clean_sample_mask = np.ones(n_times, dtype=bool)
        X_calibration = X

    filter_b, filter_a = _design_statistics_filter(sfreq, filter_kind)
    X_clean, filter_zi = _lfilter_channels(X_calibration, filter_b, filter_a)
    riemannian_info: dict[str, Any] = {}
    # Both Riemannian variants aggregate block covariances with Riemannian primitives
    # (geometric median + Karcher-style block reduction). The difference is the
    # eigenspace family used for V (and downstream T):
    #   - "riemannian"           : tangent-space V (standard reference one-shot processing)
    #   - "riemannian_windowed"  : standard eigh on the Riemannian-aggregated C
    #                              (cutoff-sensitive per-window processing)
    use_riemannian_aggregation = method in ("riemannian", "riemannian_windowed")
    C, memory_info = _aggregate_block_covariances(
        X_clean,
        blocksize,
        cov_estimator,
        covariance_kind="standard" if use_riemannian_aggregation else "padded",
        max_mem_mb=max_mem_mb,
    )
    C = _regularize_spd(C, regularization)
    if method == "riemannian":
        M = _sqrtm_spd(C, regularization)
        eigvals = np.linalg.eigvalsh(C)
        eigvals = np.sort(eigvals)
        _, V = _riemannian_nonlinear_eigenspace(M, regularization)
    else:
        # Both "standard" and "riemannian_windowed" use standard eigh on C.
        # The Riemannian-windowed variant gets robustness from the geometric-
        # median aggregation above; cutoff sensitivity comes from the matching
        # V family at calibration and per-window processing time.
        M, eigvals, V = _sqrt_and_eig(C, regularization)
    rank = int(np.sum(eigvals > regularization * np.max(eigvals)))

    thresholds, threshold_info = _fit_component_thresholds(
        X_clean,
        V,
        sfreq=sfreq,
        window_length=window_length,
        window_overlap=window_overlap,
        cutoff=cutoff,
        min_clean_fraction=min_clean_fraction,
        max_dropout_fraction=max_dropout_fraction,
        callback=callback,
    )
    T = np.diag(thresholds) @ V.T

    state = ASRState(
        M=M,
        T=T,
        thresholds=thresholds,
        calibration_patterns=V,
        filter_b=filter_b,
        filter_a=filter_a,
        filter_zi=filter_zi,
        cov=C,
        rank=rank,
        method=method,
        riemannian_solver=(
            "nonlinear_eigenspace"
            if method in ("riemannian", "riemannian_windowed")
            else None
        ),
    )
    diagnostics = {
        "clean_window_mask": clean_window_mask,
        "clean_window_scores": clean_window_scores,
        "clean_sample_mask": clean_sample_mask,
        "calibration_window_starts": cal_starts,
        "calibration_window_length_samples": cal_len,
        "blocksize": int(blocksize),
        "n_clean_windows": int(clean_window_mask.sum()),
        "n_calibration_windows": int(len(cal_starts)),
        "calibration_samples": int(X_clean.shape[1]),
        "rank": rank,
        "thresholds": thresholds.copy(),
        "threshold_mu": threshold_info["mu"].copy(),
        "threshold_sigma": threshold_info["sigma"].copy(),
        "threshold_beta": threshold_info["beta"].copy(),
        "threshold_fit_error": threshold_info["fit_error"].copy(),
        "threshold_fit_interval": threshold_info["fit_interval"].copy(),
        "cov_condition": float(np.linalg.cond(C)),
        "covariance_geometry": method,
        "filter_kind": filter_kind,
    }
    diagnostics.update(memory_info)
    diagnostics.update(riemannian_info)
    logger.debug(
        "ASR calibration details: method=%s, clean windows=%d/%d, "
        "calibration samples=%d, rank=%d, filter=%s.",
        method,
        diagnostics["n_clean_windows"],
        diagnostics["n_calibration_windows"],
        diagnostics["calibration_samples"],
        rank,
        filter_kind,
    )
    return state, diagnostics


def _fit_component_thresholds(
    X: np.ndarray,
    V: np.ndarray,
    *,
    sfreq: float,
    window_length: float,
    window_overlap: float,
    cutoff: float,
    min_clean_fraction: float,
    max_dropout_fraction: float,
    callback: _ProgressCallback | None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Fit threshold statistics for all principal components.

    Projects the continuous clean data into the principal subspace defined by V,
    calculates the windowed RMS for each component, and fits the generalized
    Gaussian distribution to determine robust cutoff thresholds.

    Parameters
    ----------
    X : ndarray, shape (n_channels, n_times)
        The clean calibration data array.
    V : ndarray, shape (n_channels, n_components)
        The mixing matrix defining the principal subspace.
    sfreq : float
        The sampling frequency in Hz.
    window_length : float
        Length of the sliding window for RMS calculation, in seconds.
    window_overlap : float
        Overlap fraction of the sliding windows.
    cutoff : float
        Multiplier for the robust standard deviation to determine the threshold.
    min_clean_fraction : float
        Minimum central fraction used to estimate clean RMS statistics.
    max_dropout_fraction : float
        Fraction of the lowest RMS values excluded as dropouts.

    Returns
    -------
    thresholds : ndarray, shape (n_components,)
        The final calculated upper RMS thresholds for each component.
    info : dict
        A dictionary containing the full set of diagnostic arrays for each
        component, including 'mu', 'sigma', 'beta', 'fit_error', and
        'fit_interval'.
    """
    win_len = _round_half_up(window_length * sfreq)
    starts = _get_fractional_window_starts(X.shape[1], win_len, window_overlap)
    projected = V.T @ X
    thresholds = np.empty(projected.shape[0], dtype=np.float64)
    mu_values = np.empty(projected.shape[0], dtype=np.float64)
    sigma_values = np.empty(projected.shape[0], dtype=np.float64)
    beta_values = np.empty(projected.shape[0], dtype=np.float64)
    fit_errors = np.empty(projected.shape[0], dtype=np.float64)
    fit_intervals = np.empty((projected.shape[0], 2), dtype=np.float64)
    for comp_idx, comp in enumerate(projected):
        rms = np.empty(len(starts), dtype=np.float64)
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
            method="asr",
            stage="calibration",
            current=comp_idx + 1,
            total=projected.shape[0],
            component=comp_idx + 1,
            metric=float(thresholds[comp_idx]),
        )
    info = {
        "mu": mu_values,
        "sigma": sigma_values,
        "beta": beta_values,
        "fit_error": fit_errors,
        "fit_interval": fit_intervals,
    }
    return thresholds, info
