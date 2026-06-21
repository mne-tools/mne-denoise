"""Parameter validation and dimension-resolution helpers for ASR.

Low-level, dependency-free checks and resolvers shared across the ASR
calibration, processing, and estimator layers, kept here so higher-level
modules can import them without pulling in the full pipeline.
"""

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
    """Validate core ASR parameters.

    Ensures that standard numerical parameters (sampling frequency, cutoff,
    window sizes, and thresholds) are within valid mathematical bounds.

    Parameters
    ----------
    sfreq : float
        Sampling frequency in Hz. Must be positive.
    cutoff : float
        Filter cutoff frequency in Hz. Must be positive.
    window_length : float
        Length of the moving window in seconds. Must be positive.
    window_overlap : float
        Fraction of overlap between consecutive windows. Must be in [0, 1).
    max_dropout_fraction : float
        Maximum allowed dropout fraction for EEG statistics. Must be in [0, 1).
    min_clean_fraction : float
        Minimum required clean data fraction. Must be in (0, 1].
    regularization : float
        Regularization parameter for covariance matrices. Must be positive.
    """
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
    """Validate and sanitize 2D EEG arrays.

    Ensures the array has exactly two dimensions and at least two channels.
    Checks for excessive non-finite values and rejects channels with zero
    or near-zero variance relative to the global maximum variance. Finally,
    converts any remaining NaNs/Infs to zeros.

    Parameters
    ----------
    X : ndarray
        The input data array.

    Returns
    -------
    X_safe : ndarray, shape (n_channels, n_times)
        A validated, 2D float64 array with non-finite values replaced.
    """
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


def _check_enough_samples(n_times: int, sfreq: float, window_length: float) -> None:
    """Ensure data is long enough to compute at least one window.

    Parameters
    ----------
    n_times : int
        Number of samples in the data.
    sfreq : float
        Sampling frequency in Hz.
    window_length : float
        Length of the window in seconds.
    """
    n_win = int(round(window_length * sfreq))
    if n_win < 2:
        raise ValueError("window_length is too short for the sampling frequency")
    if n_times < n_win:
        raise ValueError(
            f"Window length ({n_win} samples) exceeds data length ({n_times} samples)"
        )


def _round_half_up(value: float) -> int:
    """Round non-negative values to the nearest integer, ties away from zero.

    This avoids NumPy's round-to-even behavior which shifts ASR windows.

    Parameters
    ----------
    value : float
        The non-negative value to round.

    Returns
    -------
    rounded : int
        The nearest integer.
    """
    return int(np.floor(float(value) + 0.5))


def _resolve_max_dims_padded(max_dims: float | int, n_channels: int) -> int:
    """Resolve ASR principal component retention limit (padded variant).

    Calculates the maximum number of dimensions to retain during subspace
    rejection using the padded convention, which allows float values
    greater than 1 and rounds ties away from zero.

    Parameters
    ----------
    max_dims : float | int
        Fraction (if < 1) or absolute number of dimensions to retain.
    n_channels : int
        Total number of channels in the data.

    Returns
    -------
    n_dims : int
        The absolute number of dimensions to retain.
    """
    if isinstance(max_dims, float) and max_dims < 1:
        if max_dims < 0:
            raise ValueError("max_dims must be non-negative")
        return _round_half_up(n_channels * max_dims)
    max_dims_int = int(max_dims)
    if max_dims_int < 0:
        raise ValueError("max_dims must be non-negative")
    return min(max_dims_int, n_channels)


def _resolve_max_dims(max_dims: float | int, n_channels: int) -> int:
    """Resolve ASR principal component retention limit (standard variant).

    Calculates the maximum number of dimensions to retain using the standard
    scikit-learn convention, which strictly requires fractions to be in [0, 1]
    and uses standard floor division.

    Parameters
    ----------
    max_dims : float | int
        Fraction (if <= 1) or absolute number of dimensions to retain.
    n_channels : int
        Total number of channels in the data.

    Returns
    -------
    n_dims : int
        The absolute number of dimensions to retain.
    """
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
    """Validate ASR backend and processing parameters.

    Ensures that the selected backend is supported and that its configuration
    (lookahead, stepsize, and window criteria) falls within mathematically
    and logically sound bounds.

    Parameters
    ----------
    method : {'standard', 'riemannian_windowed', 'riemannian'} | None
        The selected reconstruction backend.
    experimental : bool
        Must be True to explicitly enable the legacy 'riemannian' backend.
    lookahead : float | None
        Processing lookahead in seconds. Must be non-negative.
    stepsize : int | None
        Processing step size in samples. Must be >= 1.
    window_criterion : float | int | None
        Clean window rejection criterion.
    """
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


def _check_transform_channels(
    fitted_n_channels: int,
    fitted_ch_names: list[str] | None,
    input_n_channels: int,
    input_ch_names: list[str] | None,
) -> None:
    """Verify input data channels match the fitted ASR state.

    Parameters
    ----------
    fitted_n_channels : int
        The number of channels the ASR object was calibrated on.
    fitted_ch_names : list of str | None
        The names of the channels the ASR object was calibrated on.
    input_n_channels : int
        The number of channels in the incoming data.
    input_ch_names : list of str | None
        The names of the channels in the incoming data.

    Raises
    ------
    ValueError
        If the channel counts or names do not match.
    """
    if input_n_channels != fitted_n_channels:
        raise ValueError(
            "Input channel count does not match fitted ASR state: "
            f"{input_n_channels} vs {fitted_n_channels}"
        )
    if fitted_ch_names is not None and input_ch_names != fitted_ch_names:
        raise ValueError("Input channel names/order do not match fitted ASR state")


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
    """Validate parameters specific to the adaptive ASR implementation.

    Parameters
    ----------
    variant : str
        The adaptive variant. Must be 'psp', 'psw', or 'mw'.
    update_window_length : float
        Length of the update window in seconds. Must be > 0.
    calibration_window_length : float
        Length of the clean calibration window in seconds. Must be > 0.
    calibration_window_overlap : float
        Overlap fraction for the calibration window. Must be in [0, 1).
    ref_max_bad_channels : float
        Maximum allowed fraction of bad channels in a window to be considered
        clean. Must be >= 0.
    learning_rate : float
        Learning rate for the adaptive update. Must be > 0.
    tau : float | None
        Time constant for the exponential decay. If not None, must be > 0.
    mw_window_length : float
        Length of the moving window in seconds. Must be > 0 if variant='mw'.
    mw_mode : str
        The moving window mode. Must be 'final_state' or 'sliding' if variant='mw'.

    Raises
    ------
    ValueError
        If any of the parameters are out of their expected ranges or take
        invalid values.
    """
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
    """Validate Juggler-specific reference selection parameters.

    Ensures that the DBSCAN and GEV parameters fall within mathematically valid
    ranges for instantaneous amplitude processing.

    Parameters
    ----------
    strategy : str
        The Juggler selection strategy. Must be 'dbscan' or 'gev'.
    dbscan_top_k : int
        Number of largest per-sample channel amplitudes. Must be >= 1.
    gev_grid_size : int
        Number of grid points for GEV fitting. Must be >= 32.
    min_reference_fraction : float
        Minimum accepted retained fraction. Must be in (0, 1).
    """
    if strategy not in ("dbscan", "gev"):
        raise ValueError("strategy must be 'dbscan' or 'gev'")
    if dbscan_top_k < 1:
        raise ValueError("dbscan_top_k must be at least 1")
    if gev_grid_size < 32:
        raise ValueError("gev_grid_size must be at least 32")
    if not (0.0 < min_reference_fraction < 1.0):
        raise ValueError("min_reference_fraction must be in (0, 1)")
