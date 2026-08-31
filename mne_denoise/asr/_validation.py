"""ASR validation helpers."""

from __future__ import annotations

import numpy as np


def _validate_common_params(
    *,
    sfreq: float,
    cutoff: float,
    window_length: float,
    window_overlap: float,
    max_dropout_fraction: float,
    min_clean_fraction: float,
    regularization: float,
) -> None:
    """Validate core ASR parameters."""
    if sfreq <= 0:
        raise ValueError("sfreq must be positive")
    if cutoff <= 0:
        raise ValueError("cutoff must be positive")
    if window_length <= 0:
        raise ValueError("window_length must be positive")
    if not (0 <= window_overlap < 1):
        raise ValueError("window_overlap must be in [0, 1)")
    if not (0 <= max_dropout_fraction < 1):
        raise ValueError("max_dropout_fraction must be in [0, 1)")
    if not (0 < min_clean_fraction <= 1):
        raise ValueError("min_clean_fraction must be in (0, 1]")
    if max_dropout_fraction + min_clean_fraction >= 1:
        raise ValueError(
            "max_dropout_fraction + min_clean_fraction must be less than 1"
        )
    if regularization <= 0:
        raise ValueError("regularization must be positive")


def _validate_array_2d(X: np.ndarray) -> np.ndarray:
    """Validate and sanitize 2D EEG arrays."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"ASR expects a 2D array (n_channels, n_times), got {X.shape}")
    if X.shape[0] < 2:
        raise ValueError("ASR requires at least two channels")
    finite_fraction = np.isfinite(X).mean(axis=1)
    if np.any(finite_fraction < 0.99):
        bad = np.where(finite_fraction < 0.99)[0].tolist()
        raise ValueError(f"Channels contain too many non-finite samples: {bad}")
    X = np.nan_to_num(X, copy=True)
    variances = np.var(X, axis=1)
    max_var = float(np.max(variances))
    # Relative floor so legitimately small-amplitude data (e.g. MEG in Tesla,
    # variance ~1e-26) is not rejected, while genuinely flat/dead channels
    # (variance ~0 relative to the rest) still are.
    if max_var <= 0.0:
        raise ValueError("All channels have zero or near-zero variance")
    bad = np.where(variances <= max_var * 1e-12)[0].tolist()
    if bad:
        raise ValueError(f"Channels with zero or near-zero variance: {bad}")
    return X


def _validate_covariance_matrix(
    covariance: np.ndarray | None,
    *,
    n_channels: int,
    name: str,
) -> np.ndarray | None:
    """Validate and symmetrize an optional channel covariance matrix."""
    if covariance is None:
        return None
    covariance = np.asarray(covariance, dtype=np.float64)
    expected_shape = (n_channels, n_channels)
    if covariance.shape != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}, got {covariance.shape}"
        )
    if not np.all(np.isfinite(covariance)):
        raise ValueError(f"{name} must contain only finite values")
    covariance = (covariance + covariance.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(covariance)
    scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(np.float64).tiny)
    if float(eigenvalues[0]) < -1e-10 * scale:
        raise ValueError(f"{name} must be positive semidefinite")
    return covariance


def _check_enough_samples(n_times: int, sfreq: float, window_length: float) -> None:
    """Ensure data is long enough to compute at least one window."""
    n_win = int(round(window_length * sfreq))
    if n_win < 2:
        raise ValueError("window_length is too short for the sampling frequency")
    if n_times < n_win:
        raise ValueError(
            f"Window length ({n_win} samples) exceeds data length ({n_times} samples)"
        )


def _round_half_up(value: float) -> int:
    """Round non-negative values to the nearest integer, ties away from zero."""
    return int(np.floor(float(value) + 0.5))


def _resolve_max_dims_padded(max_dims: float | int, n_channels: int) -> int:
    """Resolve ASR principal component retention limit (padded variant)."""
    if isinstance(max_dims, float) and max_dims < 1:
        if max_dims < 0:
            raise ValueError("max_dims must be non-negative")
        return _round_half_up(n_channels * max_dims)
    max_dims_int = int(max_dims)
    if max_dims_int < 0:
        raise ValueError("max_dims must be non-negative")
    return min(max_dims_int, n_channels)


def _resolve_max_dims(max_dims: float | int, n_channels: int) -> int:
    """Resolve ASR principal component retention limit (standard variant)."""
    if isinstance(max_dims, float):
        if not (0 <= max_dims <= 1):
            raise ValueError("float max_dims must be in [0, 1]")
        return int(np.floor(max_dims * n_channels))
    max_dims = int(max_dims)
    if not (0 <= max_dims <= n_channels):
        raise ValueError("integer max_dims must be in [0, n_channels]")
    return max_dims


def _validate_backend_params(
    *,
    method: str | None,
    experimental: bool,
    lookahead: float | None,
    stepsize: int | None,
    window_criterion: float | int | None,
) -> None:
    """Validate ASR backend and processing parameters."""
    if method not in ("standard", "riemannian", "riemannian_windowed"):
        raise NotImplementedError(
            "Supported methods are 'standard', 'riemannian_windowed', and "
            "experimental 'riemannian'."
        )
    # 'riemannian_windowed' is promoted to a first-class backend: its
    # processing is numerically identical to standard ASR
    # and its calibration covariance matches the riemannian backend, with a
    # strict integration test validation at relerr < 1e-13. See
    # tests/parity/test_riemannian_windowed_parity.py. Only the legacy
    # 'riemannian' backend (cutoff-invariant on real EEG) stays gated.
    if method == "riemannian" and not experimental:
        raise ValueError(
            "'riemannian' is experimental (cutoff-invariant on real EEG; "
            "see reports/paper_validation/rasr/). Set experimental=True to "
            "enable it, or use method='riemannian_windowed' for a "
            "cutoff-sensitive Riemannian backend."
        )
    if lookahead is not None and lookahead < 0:
        raise ValueError("lookahead must be non-negative")
    if stepsize is not None and stepsize < 1:
        raise ValueError("stepsize must be at least 1 sample")
    if window_criterion is not None and not isinstance(window_criterion, int | float):
        raise ValueError("window_criterion must be numeric or None")


def _validate_adaptive_params(
    variant: str,
    update_window_length: float,
    calibration_window_length: float,
    calibration_window_overlap: float,
    ref_max_bad_channels: float,
    learning_rate: float,
    tau: float | None,
    mw_window_length: float,
    mw_mode: str,
) -> None:
    """Validate parameters specific to the adaptive ASR implementation."""
    if variant not in ("psp", "psw", "mw"):
        raise ValueError("variant must be 'psp', 'psw', or 'mw'")
    if update_window_length <= 0:
        raise ValueError("update_window_length must be positive")
    if calibration_window_length <= 0:
        raise ValueError("clean_window_length must be positive")
    if not (0 <= calibration_window_overlap < 1):
        raise ValueError("clean_window_overlap must be in [0, 1)")
    if ref_max_bad_channels < 0:
        raise ValueError("clean_max_bad_channels must be non-negative")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if tau is not None and tau <= 0:
        raise ValueError("tau must be positive")
    if variant == "mw" and mw_window_length <= 0:
        raise ValueError("mw_window_length must be positive for variant='mw'")
    if variant == "mw" and mw_mode not in ("final_state", "sliding"):
        raise ValueError("mw_mode must be 'final_state' or 'sliding' for variant='mw'")


def _validate_juggler_params(
    strategy: str,
    dbscan_top_k: int,
    gev_grid_size: int,
    min_reference_fraction: float,
) -> None:
    """Validate Juggler-specific reference selection parameters."""
    if strategy not in ("dbscan", "gev"):
        raise ValueError("strategy must be 'dbscan' or 'gev'")
    if dbscan_top_k < 1:
        raise ValueError("dbscan_top_k must be at least 1")
    if gev_grid_size < 32:
        raise ValueError("gev_grid_size must be at least 32")
    if not (0.0 < min_reference_fraction < 1.0):
        raise ValueError("min_reference_fraction must be in (0, 1)")
