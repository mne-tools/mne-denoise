"""ASR processing: reconstruct artifact subspaces in incoming data.

``process_asr`` applies the fitted calibration window by window, rejecting and
reconstructing the principal subspaces whose variance exceeds the
per-direction thresholds (raised-cosine blended, with lookahead). The
Riemannian variants implement the experimental SPD-geometry backends.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .._blending import raised_cosine_ramp
from .._logging import logger, verbose
from ..progress import _emit_progress, _ProgressCallback, _validate_callback
from ._covariance import (
    _ChunkedMovingCovariances,
    _covariance_chunk_blocks,
    _covariance_stack_bytes,
    _iter_moving_covariances_at,
    _max_mem_bytes,
    _moving_average_padded,
    _process_memory_info,
)
from ._filters import (
    _append_streaming_tail,
    _apply_statistics_filter_streaming,
    _lfilter_channels,
    _prepend_streaming_carry,
)
from ._spd import _regularize_spd, _riemannian_nonlinear_eigenspace
from ._types import ASRState, _copy_process_state
from ._validation import (
    _resolve_max_dims_padded,
    _round_half_up,
    _validate_array_2d,
    _validate_common_params,
)


@dataclass(frozen=True)
class _PreparedASRStream:
    """Validated arrays and indices shared by windowed ASR processors."""

    data: np.ndarray
    n_channels: int
    n_times: int
    win_len: int
    lookahead_samples: int
    stepsize: int
    max_bad: int
    data_stream: np.ndarray | None
    n_stream_input: int
    statistics: np.ndarray | None
    update_at: np.ndarray | None
    use_rolling_covariance: bool


def _prepare_asr_stream(
    X: np.ndarray,
    sfreq: float,
    state: ASRState,
    *,
    window_length: float,
    window_overlap: float,
    max_dims: float | int,
    regularization: float,
    max_mem_mb: int | None,
    lookahead: float | None,
    stepsize: int | None,
) -> _PreparedASRStream:
    """Validate input and prepare the common streaming ASR representation."""
    _validate_common_params(
        sfreq=sfreq,
        cutoff=1.0,
        window_length=window_length,
        window_overlap=window_overlap,
        max_dropout_fraction=0.1,
        min_clean_fraction=0.25,
        regularization=regularization,
    )
    X = _validate_array_2d(X)
    n_channels, n_times = X.shape
    if state.M.shape != (n_channels, n_channels):
        raise ValueError(
            "ASR state channel count does not match data: "
            f"{state.M.shape[0]} vs {n_channels}"
        )

    win_len = max(
        _round_half_up(window_length * sfreq), _round_half_up(1.5 * n_channels)
    )
    if n_times < win_len:
        raise ValueError(
            f"Window length ({win_len} samples) exceeds data length ({n_times} samples)"
        )
    lookahead = (win_len / sfreq) / 2.0 if lookahead is None else float(lookahead)
    if lookahead < 0:
        raise ValueError("lookahead must be non-negative")
    lookahead_samples = _round_half_up(lookahead * sfreq)
    if lookahead_samples >= n_times:
        raise ValueError("lookahead is too long for the data length")
    if stepsize is None:
        stepsize = max(1, win_len // 2)
    else:
        stepsize = int(stepsize)
    if stepsize < 1:
        raise ValueError("stepsize must be at least 1 sample")
    if stepsize > win_len:
        raise ValueError("stepsize must not exceed window_length in samples")

    max_bad = _resolve_max_dims_padded(max_dims, n_channels)
    if max_bad <= 0:
        return _PreparedASRStream(
            data=X,
            n_channels=n_channels,
            n_times=n_times,
            win_len=win_len,
            lookahead_samples=lookahead_samples,
            stepsize=stepsize,
            max_bad=max_bad,
            data_stream=None,
            n_stream_input=n_times,
            statistics=None,
            update_at=None,
            use_rolling_covariance=False,
        )

    X_proc = _append_streaming_tail(X, lookahead_samples)
    data_stream = _prepend_streaming_carry(X_proc, lookahead_samples)
    data_stream[~np.isfinite(data_stream)] = 0.0
    n_stream_input = X_proc.shape[1]
    statistics_input = data_stream[
        :, lookahead_samples : lookahead_samples + n_stream_input
    ]
    if state.filter_zi is None:
        statistics = _apply_statistics_filter_streaming(
            statistics_input,
            state.filter_b,
            state.filter_a,
        )
    else:
        statistics, _ = _lfilter_channels(
            statistics_input,
            state.filter_b,
            state.filter_a,
            zi=state.filter_zi,
        )
    update_at = np.minimum(
        np.arange(stepsize, n_stream_input + stepsize, stepsize, dtype=int),
        n_stream_input,
    )
    if update_at.size == 0 or update_at[-1] != n_stream_input:
        update_at = np.append(update_at, n_stream_input)
    update_at = np.concatenate(([1], np.unique(update_at)))
    max_mem_bytes = _max_mem_bytes(max_mem_mb)
    use_rolling_covariance = max_mem_bytes is not None and (
        _covariance_stack_bytes(n_stream_input, n_channels) > max_mem_bytes
    )
    return _PreparedASRStream(
        data=X,
        n_channels=n_channels,
        n_times=n_times,
        win_len=win_len,
        lookahead_samples=lookahead_samples,
        stepsize=stepsize,
        max_bad=max_bad,
        data_stream=data_stream,
        n_stream_input=n_stream_input,
        statistics=statistics,
        update_at=update_at,
        use_rolling_covariance=use_rolling_covariance,
    )


@verbose
def process_asr(
    X: np.ndarray,
    sfreq: float,
    state: ASRState,
    *,
    window_length: float = 0.5,
    window_overlap: float = 0.66,
    max_dims: float | int = 0.66,
    regularization: float = 1e-8,
    store_reconstruction_matrices: bool = False,
    max_mem_mb: int | None = 512,
    lookahead: float | None = None,
    stepsize: int | None = None,
    method: str | None = None,
    callback=None,
    verbose: bool | str | int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply a calibrated ASR model to continuous data.

    Parameters
    ----------
    X : ndarray, shape (n_channels, n_times)
        Continuous data in the same channel order and units used for
        calibration.
    sfreq : float
        Sampling frequency in Hz.
    state : ASRState
        Fitted calibration state from :func:`calibrate_asr`.
    window_length : float
        Processing window length in seconds.
    window_overlap : float
        Calibration threshold-window overlap. Processing follows the
        standard streaming ASR algorithm and uses ``stepsize`` for
        reconstruction-matrix updates.
    max_dims : float | int
        Maximum number of dimensions reconstructed per window. Floats in
        ``[0, 1]`` are interpreted as a fraction of channels.
    regularization : float
        Relative eigenvalue floor for window covariances.
    store_reconstruction_matrices : bool
        If True, store all window reconstruction matrices in diagnostics.
    max_mem_mb : int | None
        Reserved memory limit for future chunking. Present for API stability.
    lookahead : float | None
        Processing lookahead in seconds. If None, use ``window_length / 2``.
    stepsize : int | None
        Number of samples between reconstruction-matrix updates. If None, use
        ``floor(sfreq * window_length / 2)``, matching the standard algorithm defaults.
    method : {'standard', 'riemannian'} | None
        Covariance geometry for processing. If ``None``, use ``state.method``.
    callback : callable | None
        Called synchronously after each completed reconstruction-matrix update
        with an ASR window progress event. Callback return values are ignored
        and callback exceptions propagate unchanged.
    verbose : bool | str | int | None
        MNE-style logging level. Processing details are emitted at DEBUG; the
        owning estimator reports the user-facing result at INFO.

    Returns
    -------
    X_clean : ndarray, shape (n_channels, n_times)
        Cleaned data.
    diagnostics : dict
        Processing diagnostics.

    Examples
    --------
    Process a new array of task data using a previously calibrated ASR state:

    >>> import numpy as np
    >>> from mne_denoise.asr import process_asr
    >>> rng = np.random.default_rng(42)
    >>> task_data = rng.standard_normal((10, 2000))
    >>> # Assuming 'state' is an ASRState returned by calibrate_asr
    >>> cleaned_data, diagnostics = process_asr(task_data, sfreq=250.0, state=state)
    >>> print(f"Cleaned data shape: {cleaned_data.shape}")
    Cleaned data shape: (10, 2000)
    """
    callback = _validate_callback(callback)
    if method is None:
        method = state.method
    if method not in ("standard", "riemannian", "riemannian_windowed"):
        raise ValueError(
            "method must be 'standard', 'riemannian', or 'riemannian_windowed'"
        )
    prepared = _prepare_asr_stream(
        X,
        sfreq,
        state,
        window_length=window_length,
        window_overlap=window_overlap,
        max_dims=max_dims,
        regularization=regularization,
        max_mem_mb=max_mem_mb,
        lookahead=lookahead,
        stepsize=stepsize,
    )
    X = prepared.data
    n_channels = prepared.n_channels
    n_times = prepared.n_times
    win_len = prepared.win_len
    lookahead_samples = prepared.lookahead_samples
    stepsize = prepared.stepsize
    max_bad = prepared.max_bad
    if max_bad <= 0:
        diagnostics = _empty_process_diagnostics(n_times)
        diagnostics.update(
            _process_memory_info(
                n_channels=n_channels,
                n_stream_input=n_times,
                max_mem_mb=max_mem_mb,
                memory_mode="identity",
                peak_cov_buffer_bytes=0,
                chunk_samples=0,
                used_memory_bound=False,
            )
        )
        logger.debug(
            "ASR reconstruction details: method=%s, identity path, %d sample(s).",
            method,
            n_times,
        )
        return X.copy(), diagnostics

    assert prepared.data_stream is not None
    assert prepared.statistics is not None
    assert prepared.update_at is not None
    data_stream = prepared.data_stream
    n_stream_input = prepared.n_stream_input
    X_stats = prepared.statistics
    update_at = prepared.update_at
    use_rolling_covariance = prepared.use_rolling_covariance

    if method == "riemannian":
        X_clean, diagnostics = _process_asr_riemannian(
            data_stream,
            X_stats,
            state,
            n_times=n_times,
            n_stream_input=n_stream_input,
            lookahead_samples=lookahead_samples,
            update_at=update_at,
            max_bad=max_bad,
            stepsize=stepsize,
            win_len=win_len,
            regularization=regularization,
            store_reconstruction_matrices=store_reconstruction_matrices,
            callback=callback,
        )
        diagnostics.update(
            _process_memory_info(
                n_channels=n_channels,
                n_stream_input=n_stream_input,
                max_mem_mb=max_mem_mb,
                memory_mode="riemannian",
                peak_cov_buffer_bytes=_covariance_stack_bytes(1, n_channels),
                chunk_samples=n_stream_input,
                used_memory_bound=False,
            )
        )
        logger.debug(
            "ASR reconstruction details: method=%s, %d window(s), "
            "%d component(s) maximum, %.1f%% samples reconstructed.",
            method,
            diagnostics["n_windows"],
            diagnostics["max_components_reconstructed"],
            100.0 * diagnostics["fraction_reconstructed_samples"],
        )
        return X_clean, diagnostics

    if method == "riemannian_windowed":
        X_clean, diagnostics = _process_asr_riemannian_windowed(
            data_stream,
            X_stats,
            state,
            n_times=n_times,
            n_stream_input=n_stream_input,
            lookahead_samples=lookahead_samples,
            update_at=update_at,
            max_bad=max_bad,
            stepsize=stepsize,
            win_len=win_len,
            store_reconstruction_matrices=store_reconstruction_matrices,
            use_rolling_covariance=use_rolling_covariance,
            callback=callback,
        )
        diagnostics.update(
            _process_memory_info(
                n_channels=n_channels,
                n_stream_input=n_stream_input,
                max_mem_mb=max_mem_mb,
                memory_mode=(
                    "riemannian_windowed_rolling"
                    if use_rolling_covariance
                    else "riemannian_windowed"
                ),
                peak_cov_buffer_bytes=_covariance_stack_bytes(1, n_channels),
                chunk_samples=win_len if use_rolling_covariance else n_stream_input,
                used_memory_bound=use_rolling_covariance,
            )
        )
        logger.debug(
            "ASR reconstruction details: method=%s, %d window(s), "
            "%d component(s) maximum, %.1f%% samples reconstructed.",
            method,
            diagnostics["n_windows"],
            diagnostics["max_components_reconstructed"],
            100.0 * diagnostics["fraction_reconstructed_samples"],
        )
        return X_clean, diagnostics

    if use_rolling_covariance:
        covariance_iter = _iter_moving_covariances_at(X_stats, update_at, win_len)
        Xcov_flat = None
    else:
        outer = np.einsum("it,jt->ijt", X_stats, X_stats, optimize=True)
        Xcov_flat = outer.reshape(n_channels * n_channels, n_stream_input, order="F")
        Xcov_flat, _ = _moving_average_padded(win_len, Xcov_flat)
        covariance_iter = None

    sample_mask = np.zeros(n_times, dtype=bool)
    n_reconstructed: list[int] = []
    component_variances: list[np.ndarray] = []
    component_thresholds: list[np.ndarray] = []
    reconstruction_matrices: list[np.ndarray] = []
    window_starts: list[int] = []
    window_stops: list[int] = []

    eye = np.eye(n_channels)
    last_R = eye
    last_trivial = True
    last_n = 0
    for progress_idx, n in enumerate(update_at, start=1):
        if covariance_iter is None:
            assert Xcov_flat is not None
            Cw = Xcov_flat[:, n - 1].reshape(n_channels, n_channels, order="F")
        else:
            Cw = next(covariance_iter)
        Cw = (Cw + Cw.T) / 2.0
        D, V = np.linalg.eigh(Cw)
        order = np.argsort(D)
        D = D[order]
        V = V[:, order]

        theta2 = np.sum((state.T @ V) ** 2, axis=0)
        keep = (theta2 > D) | (np.arange(1, n_channels + 1) < (n_channels - max_bad))
        trivial = bool(np.all(keep))

        n_bad = int(n_channels - np.count_nonzero(keep))
        if trivial:
            R = eye
        else:
            basis = keep[:, np.newaxis].astype(np.float64) * (V.T @ state.M)
            R = state.M @ np.linalg.pinv(basis) @ V.T
            R = np.real_if_close(R).astype(np.float64)

        applied = (not trivial) or (not last_trivial)
        if applied and n > last_n:
            subrange = slice(last_n, n)
            width = n - last_n
            blend = raised_cosine_ramp(width)
            segment = data_stream[:, subrange]
            data_stream[:, subrange] = (R @ segment) * blend[np.newaxis, :] + (
                last_R @ segment
            ) * (1.0 - blend[np.newaxis, :])

        start_out = max(last_n, lookahead_samples) - lookahead_samples
        stop_out = min(n, lookahead_samples + n_times) - lookahead_samples
        if stop_out > start_out:
            window_starts.append(int(start_out))
            window_stops.append(int(stop_out))
            n_reconstructed.append(n_bad)
            component_variances.append(D.copy())
            component_thresholds.append(theta2.copy())
            if applied:
                sample_mask[start_out:stop_out] = True
            if store_reconstruction_matrices:
                reconstruction_matrices.append(R.copy())

        last_n = int(n)
        last_R = R
        last_trivial = trivial
        _emit_progress(
            callback,
            method="asr",
            stage="window",
            current=progress_idx,
            total=len(update_at),
            component=None,
            metric=float(n_bad),
        )

    X_clean = data_stream[:, lookahead_samples : lookahead_samples + n_times].copy()

    n_reconstructed_arr = np.asarray(n_reconstructed, dtype=int)
    diagnostics = {
        "window_starts": np.asarray(window_starts, dtype=int),
        "window_stops": np.asarray(window_stops, dtype=int),
        "sample_mask": sample_mask,
        "n_components_reconstructed": n_reconstructed_arr,
        "component_variances": np.asarray(component_variances, dtype=np.float64),
        "component_thresholds": np.asarray(component_thresholds, dtype=np.float64),
        "n_windows": int(len(n_reconstructed_arr)),
        "fraction_reconstructed_windows": float(
            np.mean(n_reconstructed_arr > 0) if n_reconstructed_arr.size else 0.0
        ),
        "fraction_reconstructed_samples": float(np.mean(sample_mask)),
        "max_components_reconstructed": int(n_reconstructed_arr.max(initial=0)),
        "lookahead_samples": int(lookahead_samples),
        "stepsize_samples": int(stepsize),
        "window_length_samples": int(win_len),
        "covariance_geometry": method,
    }
    if use_rolling_covariance:
        diagnostics.update(
            _process_memory_info(
                n_channels=n_channels,
                n_stream_input=n_stream_input,
                max_mem_mb=max_mem_mb,
                memory_mode="rolling",
                peak_cov_buffer_bytes=_covariance_stack_bytes(1, n_channels),
                chunk_samples=win_len,
                used_memory_bound=True,
            )
        )
    else:
        diagnostics.update(
            _process_memory_info(
                n_channels=n_channels,
                n_stream_input=n_stream_input,
                max_mem_mb=max_mem_mb,
                memory_mode="full",
                peak_cov_buffer_bytes=_covariance_stack_bytes(
                    n_stream_input, n_channels
                ),
                chunk_samples=n_stream_input,
                used_memory_bound=False,
            )
        )
    if store_reconstruction_matrices:
        diagnostics["reconstruction_matrices"] = np.asarray(reconstruction_matrices)
    logger.debug(
        "ASR reconstruction details: method=%s, %d window(s), "
        "%d component(s) maximum, %.1f%% samples reconstructed.",
        method,
        diagnostics["n_windows"],
        diagnostics["max_components_reconstructed"],
        100.0 * diagnostics["fraction_reconstructed_samples"],
    )
    return X_clean, diagnostics


def _empty_process_diagnostics(n_times: int) -> dict[str, Any]:
    """Return identity-processing diagnostics.

    Parameters
    ----------
    n_times : int
        Number of time samples in the processed block.

    Returns
    -------
    diagnostics : dict
        A diagnostic dictionary describing a trivial identity reconstruction
        where no components were removed.
    """
    return {
        "window_starts": np.array([0], dtype=int),
        "window_stops": np.array([n_times], dtype=int),
        "sample_mask": np.zeros(n_times, dtype=bool),
        "n_components_reconstructed": np.array([0], dtype=int),
        "component_variances": np.empty((1, 0), dtype=np.float64),
        "component_thresholds": np.empty((1, 0), dtype=np.float64),
        "n_windows": 1,
        "fraction_reconstructed_windows": 0.0,
        "fraction_reconstructed_samples": 0.0,
        "max_components_reconstructed": 0,
        "lookahead_samples": 0,
        "stepsize_samples": 0,
        "window_length_samples": 0,
    }


def _process_asr_riemannian(
    data_stream: np.ndarray,
    X_stats: np.ndarray,
    state: ASRState,
    *,
    n_times: int,
    n_stream_input: int,
    lookahead_samples: int,
    update_at: np.ndarray,
    max_bad: int,
    stepsize: int,
    win_len: int,
    regularization: float,
    store_reconstruction_matrices: bool,
    callback: _ProgressCallback | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the standard Riemannian chunk covariance backend.

    Parameters
    ----------
    data_stream : ndarray
        The padded, lookahead-applied data stream.
    X_stats : ndarray
        The filtered subset of the data stream used for covariance calculations.
    state : ASRState
        Fitted calibration state containing mixing/threshold matrices.
    n_times : int
        Original number of time samples before padding.
    n_stream_input : int
        Number of input samples in the data stream.
    lookahead_samples : int
        Number of samples used for window lookahead padding.
    update_at : ndarray
        Array of sample indices at which to update reconstruction matrices.
    max_bad : int
        Maximum allowable number of rejected components per window.
    stepsize : int
        Sample step size between consecutive window matrix updates.
    win_len : int
        Length of the running processing window in samples.
    regularization : float
        Regularization applied to the symmetric positive definite covariance.
    store_reconstruction_matrices : bool
        If True, diagnostic dictionaries will contain per-window reconstruction matrices.

    Returns
    -------
    X_clean : ndarray
        The cleaned original data block.
    diagnostics : dict
        A dictionary containing processing diagnostic metadata.
    """
    n_channels = data_stream.shape[0]
    eye = np.eye(n_channels)
    Cw = (X_stats @ X_stats.T) / n_stream_input
    Cw = _regularize_spd(Cw, regularization)

    D, V = _riemannian_nonlinear_eigenspace(Cw, regularization)
    theta2 = np.sum((state.T @ V) ** 2, axis=0)
    keep = (theta2 > D) | (np.arange(1, n_channels + 1) < (n_channels - max_bad))
    trivial = bool(np.all(keep))
    n_bad = int(n_channels - np.count_nonzero(keep))

    if trivial:
        R = eye
    else:
        basis = keep[:, np.newaxis].astype(np.float64) * (V.T @ state.M)
        R = state.M @ np.linalg.pinv(basis) @ V.T
        R = np.real_if_close(R).astype(np.float64)

    sample_mask = np.zeros(n_times, dtype=bool)
    n_reconstructed: list[int] = []
    component_variances: list[np.ndarray] = []
    component_thresholds: list[np.ndarray] = []
    reconstruction_matrices: list[np.ndarray] = []
    window_starts: list[int] = []
    window_stops: list[int] = []

    last_R = eye
    last_trivial = True
    last_n = 0
    for progress_idx, n in enumerate(update_at, start=1):
        applied = (not trivial) or (not last_trivial)
        if applied and n > last_n:
            subrange = slice(last_n, n)
            width = n - last_n
            blend = raised_cosine_ramp(width)
            segment = data_stream[:, subrange]
            data_stream[:, subrange] = (R @ segment) * blend[np.newaxis, :] + (
                last_R @ segment
            ) * (1.0 - blend[np.newaxis, :])

        start_out = max(last_n, lookahead_samples) - lookahead_samples
        stop_out = min(n, lookahead_samples + n_times) - lookahead_samples
        if stop_out > start_out:
            window_starts.append(int(start_out))
            window_stops.append(int(stop_out))
            n_reconstructed.append(n_bad)
            component_variances.append(D.copy())
            component_thresholds.append(theta2.copy())
            if applied:
                sample_mask[start_out:stop_out] = True
            if store_reconstruction_matrices:
                reconstruction_matrices.append(R.copy())
        last_n = int(n)
        last_R = R
        last_trivial = trivial
        _emit_progress(
            callback,
            method="asr",
            stage="window",
            current=progress_idx,
            total=len(update_at),
            component=None,
            metric=float(n_bad),
        )

    X_clean = data_stream[:, lookahead_samples : lookahead_samples + n_times].copy()
    n_reconstructed_arr = np.asarray(n_reconstructed, dtype=int)
    diagnostics = {
        "window_starts": np.asarray(window_starts, dtype=int),
        "window_stops": np.asarray(window_stops, dtype=int),
        "sample_mask": sample_mask,
        "n_components_reconstructed": n_reconstructed_arr,
        "component_variances": np.asarray(component_variances, dtype=np.float64),
        "component_thresholds": np.asarray(component_thresholds, dtype=np.float64),
        "n_windows": int(len(n_reconstructed_arr)),
        "fraction_reconstructed_windows": float(
            np.mean(n_reconstructed_arr > 0) if n_reconstructed_arr.size else 0.0
        ),
        "fraction_reconstructed_samples": float(np.mean(sample_mask)),
        "max_components_reconstructed": int(n_reconstructed_arr.max(initial=0)),
        "lookahead_samples": int(lookahead_samples),
        "stepsize_samples": int(stepsize),
        "window_length_samples": int(win_len),
        "covariance_geometry": "riemannian",
        "riemannian_solver": "nonlinear_eigenspace",
        "riemannian_mean_iterations": np.zeros(
            len(n_reconstructed_arr),
            dtype=int,
        ),
        "riemannian_mean_converged": np.ones(
            len(n_reconstructed_arr),
            dtype=bool,
        ),
        "riemannian_mean_update_norm": np.zeros(
            len(n_reconstructed_arr),
            dtype=np.float64,
        ),
    }
    if store_reconstruction_matrices:
        diagnostics["reconstruction_matrices"] = np.asarray(reconstruction_matrices)
    return X_clean, diagnostics


def _process_asr_windowed(
    data_stream: np.ndarray,
    X_stats: np.ndarray,
    state: ASRState,
    *,
    n_times: int,
    n_stream_input: int,
    lookahead_samples: int,
    update_at: np.ndarray,
    max_bad: int,
    stepsize: int,
    win_len: int,
    store_reconstruction_matrices: bool,
    use_rolling_covariance: bool,
    callback: _ProgressCallback | None = None,
    progress_method: str = "asr",
    component_weight_function: Callable[
        [np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray
    ]
    | None = None,
    return_component_weights: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Process windowed ASR with binary or custom soft component weights.

    Standard ``riemannian_windowed`` ASR uses the binary keep mask when
    ``component_weight_function`` is None. Experimental extensions can provide
    a callable that returns continuous keep weights without duplicating the
    covariance, eigendecomposition, blending, and diagnostics loop.

    Parameters
    ----------
    data_stream : ndarray
        The padded, lookahead-applied data stream.
    X_stats : ndarray
        The filtered subset of the data stream used for covariance calculations.
    state : ASRState
        Fitted calibration state containing mixing/threshold matrices.
    n_times : int
        Original number of time samples before padding.
    n_stream_input : int
        Number of input samples in the data stream.
    lookahead_samples : int
        Number of samples used for window lookahead padding.
    update_at : ndarray
        Array of sample indices at which to update reconstruction matrices.
    max_bad : int
        Maximum allowable number of rejected components per window.
    stepsize : int
        Sample step size between consecutive window matrix updates.
    win_len : int
        Length of the running processing window in samples.
    store_reconstruction_matrices : bool
        If True, diagnostic dictionaries will contain per-window reconstruction matrices.
    use_rolling_covariance : bool
        Whether to optimize performance by iteratively updating the window covariance.
    component_weight_function : callable | None
        Optional callable receiving ``(variances, eigenvectors, thresholds,
        forced_keep)`` and returning one keep weight in ``[0, 1]`` per
        component. ``None`` selects standard binary ASR reconstruction.
    return_component_weights : bool
        If True, include binary or soft component weights in diagnostics.

    Returns
    -------
    X_clean : ndarray
        The cleaned original data block.
    diagnostics : dict
        A dictionary containing processing diagnostic metadata.
    """
    n_channels = data_stream.shape[0]

    if use_rolling_covariance:
        covariance_iter = _iter_moving_covariances_at(X_stats, update_at, win_len)
        Xcov_flat = None
    else:
        outer = np.einsum("it,jt->ijt", X_stats, X_stats, optimize=True)
        Xcov_flat = outer.reshape(n_channels * n_channels, n_stream_input, order="F")
        Xcov_flat, _ = _moving_average_padded(win_len, Xcov_flat)
        covariance_iter = None

    sample_mask = np.zeros(n_times, dtype=bool)
    n_reconstructed: list[int] = []
    component_variances: list[np.ndarray] = []
    component_thresholds: list[np.ndarray] = []
    component_weights: list[np.ndarray] = []
    reconstruction_matrices: list[np.ndarray] = []
    window_starts: list[int] = []
    window_stops: list[int] = []

    eye = np.eye(n_channels)
    last_R = eye
    last_trivial = True
    last_n = 0
    for progress_idx, n in enumerate(update_at, start=1):
        if covariance_iter is None:
            assert Xcov_flat is not None
            Cw = Xcov_flat[:, n - 1].reshape(n_channels, n_channels, order="F")
        else:
            Cw = next(covariance_iter)
        Cw = (Cw + Cw.T) / 2.0
        # Standard SPD eigendecomposition (matches _process_asr_standard).
        # See the docstring for why we do NOT use _riemannian_nonlinear_eigenspace
        # at processing time.
        D, V = np.linalg.eigh(Cw)
        order = np.argsort(D)
        D = D[order]
        V = V[:, order]

        theta2 = np.sum((state.T @ V) ** 2, axis=0)
        forced_keep = np.arange(1, n_channels + 1) < (n_channels - max_bad)
        if component_weight_function is None:
            keep = (theta2 > D) | forced_keep
            weights = keep.astype(np.float64)
            trivial = bool(np.all(keep))
            n_bad = int(n_channels - np.count_nonzero(keep))
            if trivial:
                R = eye
            else:
                basis = keep[:, np.newaxis].astype(np.float64) * (V.T @ state.M)
                R = state.M @ np.linalg.pinv(basis) @ V.T
                R = np.real_if_close(R).astype(np.float64)
        else:
            weights = np.asarray(
                component_weight_function(D, V, theta2, forced_keep),
                dtype=np.float64,
            )
            if weights.shape != (n_channels,):
                raise ValueError(
                    "component_weight_function must return shape "
                    f"({n_channels},), got {weights.shape}"
                )
            weights = np.clip(weights, 0.0, 1.0)
            trivial = bool(np.all(weights >= 1.0 - 1e-12))
            n_bad = int(np.count_nonzero(weights < 0.5))
            keep = (weights >= 0.5) | forced_keep
            if np.all(keep):
                hard_reconstruction = eye
            else:
                basis = keep[:, np.newaxis].astype(np.float64) * (V.T @ state.M)
                hard_reconstruction = state.M @ np.linalg.pinv(basis) @ V.T
            R = (V * weights) @ V.T + (
                (V * (1.0 - weights)) @ V.T @ hard_reconstruction
            )
            R = np.real_if_close(R).astype(np.float64)

        applied = (not trivial) or (not last_trivial)
        if applied and n > last_n:
            subrange = slice(last_n, n)
            width = n - last_n
            blend = raised_cosine_ramp(width)
            segment = data_stream[:, subrange]
            data_stream[:, subrange] = (R @ segment) * blend[np.newaxis, :] + (
                last_R @ segment
            ) * (1.0 - blend[np.newaxis, :])

        start_out = max(last_n, lookahead_samples) - lookahead_samples
        stop_out = min(n, lookahead_samples + n_times) - lookahead_samples
        if stop_out > start_out:
            window_starts.append(int(start_out))
            window_stops.append(int(stop_out))
            n_reconstructed.append(n_bad)
            component_variances.append(D.copy())
            component_thresholds.append(theta2.copy())
            if component_weight_function is not None or return_component_weights:
                component_weights.append(weights.copy())
            if applied:
                sample_mask[start_out:stop_out] = True
            if store_reconstruction_matrices:
                reconstruction_matrices.append(R.copy())

        last_n = int(n)
        last_R = R
        last_trivial = trivial
        _emit_progress(
            callback,
            method=progress_method,
            stage="window",
            current=progress_idx,
            total=len(update_at),
            component=None,
            metric=float(n_bad),
        )

    X_clean = data_stream[:, lookahead_samples : lookahead_samples + n_times].copy()
    n_reconstructed_arr = np.asarray(n_reconstructed, dtype=int)
    diagnostics = {
        "window_starts": np.asarray(window_starts, dtype=int),
        "window_stops": np.asarray(window_stops, dtype=int),
        "sample_mask": sample_mask,
        "n_components_reconstructed": n_reconstructed_arr,
        "component_variances": np.asarray(component_variances, dtype=np.float64),
        "component_thresholds": np.asarray(component_thresholds, dtype=np.float64),
        "n_windows": int(len(n_reconstructed_arr)),
        "fraction_reconstructed_windows": float(
            np.mean(n_reconstructed_arr > 0) if n_reconstructed_arr.size else 0.0
        ),
        "fraction_reconstructed_samples": float(np.mean(sample_mask)),
        "max_components_reconstructed": int(n_reconstructed_arr.max(initial=0)),
        "lookahead_samples": int(lookahead_samples),
        "stepsize_samples": int(stepsize),
        "window_length_samples": int(win_len),
        "covariance_geometry": "riemannian_windowed",
        "riemannian_solver": "nonlinear_eigenspace",
    }
    if component_weight_function is not None or return_component_weights:
        diagnostics["component_weights"] = np.asarray(
            component_weights, dtype=np.float64
        )
    if store_reconstruction_matrices:
        diagnostics["reconstruction_matrices"] = np.asarray(reconstruction_matrices)
    return X_clean, diagnostics


def _process_asr_riemannian_windowed(
    data_stream: np.ndarray,
    X_stats: np.ndarray,
    state: ASRState,
    *,
    n_times: int,
    n_stream_input: int,
    lookahead_samples: int,
    update_at: np.ndarray,
    max_bad: int,
    stepsize: int,
    win_len: int,
    store_reconstruction_matrices: bool,
    use_rolling_covariance: bool,
    callback: _ProgressCallback | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run standard cutoff-sensitive per-window Riemannian ASR."""
    return _process_asr_windowed(
        data_stream,
        X_stats,
        state,
        n_times=n_times,
        n_stream_input=n_stream_input,
        lookahead_samples=lookahead_samples,
        update_at=update_at,
        max_bad=max_bad,
        stepsize=stepsize,
        win_len=win_len,
        store_reconstruction_matrices=store_reconstruction_matrices,
        use_rolling_covariance=use_rolling_covariance,
        callback=callback,
    )


def _process_adaptive_chunk(
    X: np.ndarray,
    sfreq: float,
    state: ASRState,
    process_state: dict[str, Any],
    window_length: float,
    lookahead: float | None,
    stepsize: int | None,
    max_dims: float | int,
    store_reconstruction_matrices: bool,
    adaptive_variant: str,
    max_mem_mb: int | float | None,
    callback: _ProgressCallback | None = None,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    """Mirror the AASR ``reconstruct()`` wrapper around ``asr_process``.

    Parameters
    ----------
    X : ndarray, shape (n_channels, n_times)
        The input data chunk to be reconstructed.
    sfreq : float
        The sampling frequency of the data.
    state : ASRState
        The current calibration state of the ASR algorithm.
    process_state : dict
        A dictionary containing stateful values carried over from the previous
        chunk (e.g., streaming carry, filter states, and the adaptive learner).
    window_length : float
        The length of the moving window in seconds.
    lookahead : float | None
        The lookahead duration in seconds. If None, defaults to half the window length.
    stepsize : int | None
        The processing step size in samples.
    max_dims : float | int
        The maximum number of dimensions to reconstruct.
    store_reconstruction_matrices : bool
        If True, returns the mixing matrices used for reconstruction.
    adaptive_variant : str
        The variant of the adaptive updating rule used (e.g., 'psw').
    max_mem_mb : int | float | None
        The maximum memory allowed for block computations.
    callback : callable | None
        Called synchronously after each completed reconstruction-matrix update.

    Returns
    -------
    X_clean : ndarray
        The cleaned (reconstructed) data chunk.
    diagnostics : dict
        Diagnostic information about the chunk processing.
    next_process_state : dict
        The updated process state dictionary to carry forward to the next chunk.
    """
    X = _validate_array_2d(X)
    n_channels, n_times = X.shape
    lookahead_samples = (
        _round_half_up((window_length / 2.0) * sfreq)
        if lookahead is None
        else _round_half_up(lookahead * sfreq)
    )
    if lookahead_samples >= n_times:
        raise ValueError("lookahead is too long for the data length")
    win_len = max(
        _round_half_up(window_length * sfreq), _round_half_up(1.5 * n_channels)
    )
    stepsize = 32 if stepsize is None else int(stepsize)
    if stepsize < 1:
        raise ValueError("stepsize must be at least 1 sample")

    sig = _append_streaming_tail(X, lookahead_samples)
    # Convert parametric max_dims limits into fixed rank tolerances
    max_bad = _resolve_max_dims_padded(max_dims, n_channels)
    win_len = int(_round_half_up(window_length * sfreq))
    n_stream_input = sig.shape[1]
    if max_bad <= 0:
        diagnostics = _empty_process_diagnostics(n_times)
        diagnostics.update(
            _process_memory_info(
                n_channels=n_channels,
                n_stream_input=n_stream_input,
                max_mem_mb=max_mem_mb,
                memory_mode="identity",
                peak_cov_buffer_bytes=0,
                chunk_samples=0,
                used_memory_bound=False,
            )
        )
        return X.copy(), diagnostics, _copy_process_state(process_state)

    carry = process_state.get("carry")
    if carry is None:
        data_stream = np.concatenate(
            [
                2.0 * sig[:, [0]] - sig[:, lookahead_samples:0:-1],
                sig,
            ],
            axis=1,
        )
    else:
        data_stream = np.concatenate([carry, sig], axis=1)
    data_stream = np.asarray(data_stream, dtype=np.float64)
    data_stream[~np.isfinite(data_stream)] = 0.0

    X_stats, iir_state = _lfilter_channels(
        data_stream[:, lookahead_samples : lookahead_samples + n_stream_input],
        state.filter_b,
        state.filter_a,
        zi=process_state.get("iir"),
    )

    update_at = np.minimum(
        np.arange(stepsize, n_stream_input + stepsize, stepsize, dtype=int),
        n_stream_input,
    )
    if update_at.size == 0 or update_at[-1] != n_stream_input:
        update_at = np.append(update_at, n_stream_input)
    update_at = np.unique(update_at)

    last_R = process_state.get("last_R")
    last_trivial = bool(process_state.get("last_trivial", True))
    if last_R is None:
        last_R = np.eye(n_channels, dtype=np.float64)
        last_trivial = True
        update_at = np.concatenate(([1], update_at))

    estimated_cov_bytes = _covariance_stack_bytes(n_stream_input, n_channels)
    max_mem_bytes = _max_mem_bytes(max_mem_mb)
    use_chunked_covariance = (
        max_mem_bytes is not None and estimated_cov_bytes > max_mem_bytes
    )
    cov_state_in = process_state.get("cov")
    cov_source = None
    if use_chunked_covariance:
        chunk_samples = _covariance_chunk_blocks(n_channels, max_mem_bytes)
        cov_source = _ChunkedMovingCovariances(
            X_stats,
            update_at,
            win_len,
            chunk_samples=chunk_samples,
            zi=cov_state_in,
        )
        covariance_iter = iter(cov_source)
        Xcov_flat = None
        cov_state = None
        peak_cov_buffer_bytes = _covariance_stack_bytes(chunk_samples, n_channels)
    else:
        chunk_samples = n_stream_input
        outer = np.einsum("it,jt->ijt", X_stats, X_stats, optimize=True)
        Xcov_flat = outer.reshape(n_channels * n_channels, n_stream_input, order="F")
        Xcov_flat, cov_state = _moving_average_padded(
            win_len,
            Xcov_flat,
            zi=cov_state_in,
        )
        covariance_iter = None
        peak_cov_buffer_bytes = estimated_cov_bytes

    sample_mask = np.zeros(n_times, dtype=bool)
    n_reconstructed: list[int] = []
    component_variances: list[np.ndarray] = []
    component_thresholds: list[np.ndarray] = []
    reconstruction_matrices: list[np.ndarray] = []
    window_starts: list[int] = []
    window_stops: list[int] = []

    last_n = 0
    for progress_idx, n in enumerate(update_at, start=1):
        if covariance_iter is None:
            assert Xcov_flat is not None
            Cw = Xcov_flat[:, n - 1].reshape(n_channels, n_channels, order="F")
        else:
            Cw = next(covariance_iter)
        Cw = (Cw + Cw.T) / 2.0
        D, V = np.linalg.eigh(Cw)
        order = np.argsort(D)
        D = D[order]
        V = V[:, order]
        theta2 = np.sum((state.T @ V) ** 2, axis=0)
        keep = (theta2 > D) | (np.arange(1, n_channels + 1) < (n_channels - max_bad))
        trivial = bool(np.all(keep))
        n_bad = int(n_channels - np.count_nonzero(keep))
        if trivial:
            R = np.eye(n_channels, dtype=np.float64)
        else:
            basis = keep[:, np.newaxis].astype(np.float64) * (V.T @ state.M)
            R = state.M @ np.linalg.pinv(basis) @ V.T
            R = np.real_if_close(R).astype(np.float64)

        applied = (not trivial) or (not last_trivial)
        if applied and n > last_n:
            subrange = slice(last_n, n)
            width = n - last_n
            blend = raised_cosine_ramp(width)
            segment = data_stream[:, subrange]
            data_stream[:, subrange] = (R @ segment) * blend[np.newaxis, :] + (
                last_R @ segment
            ) * (1.0 - blend[np.newaxis, :])

        start_out = max(last_n, lookahead_samples) - lookahead_samples
        stop_out = min(n, lookahead_samples + n_times) - lookahead_samples
        if stop_out > start_out:
            window_starts.append(int(start_out))
            window_stops.append(int(stop_out))
            n_reconstructed.append(n_bad)
            component_variances.append(D.copy())
            component_thresholds.append(theta2.copy())
            if applied:
                sample_mask[start_out:stop_out] = True
            if store_reconstruction_matrices:
                reconstruction_matrices.append(R.copy())

        last_n = int(n)
        last_R = R
        last_trivial = trivial
        _emit_progress(
            callback,
            method="adaptive_asr",
            stage="window",
            current=progress_idx,
            total=len(update_at),
            component=None,
            metric=float(n_bad),
        )

    carry_out = (
        data_stream[:, -lookahead_samples:].copy() if lookahead_samples > 0 else None
    )
    delayed = data_stream[:, :n_stream_input].copy()
    X_clean = delayed[:, lookahead_samples:]
    if cov_source is not None:
        cov_state = cov_source.cov_state

    n_reconstructed_arr = np.asarray(n_reconstructed, dtype=int)
    diagnostics = {
        "window_starts": np.asarray(window_starts, dtype=int),
        "window_stops": np.asarray(window_stops, dtype=int),
        "sample_mask": sample_mask,
        "n_components_reconstructed": n_reconstructed_arr,
        "component_variances": np.asarray(component_variances, dtype=np.float64),
        "component_thresholds": np.asarray(component_thresholds, dtype=np.float64),
        "n_windows": int(len(n_reconstructed_arr)),
        "fraction_reconstructed_windows": float(
            np.mean(n_reconstructed_arr > 0) if n_reconstructed_arr.size else 0.0
        ),
        "fraction_reconstructed_samples": float(np.mean(sample_mask)),
        "max_components_reconstructed": int(n_reconstructed_arr.max(initial=0)),
        "lookahead_samples": int(lookahead_samples),
        "stepsize_samples": int(stepsize),
        "window_length_samples": int(win_len),
        "covariance_geometry": "standard",
        "adaptive_variant": adaptive_variant,
    }
    diagnostics.update(
        _process_memory_info(
            n_channels=n_channels,
            n_stream_input=n_stream_input,
            max_mem_mb=max_mem_mb,
            memory_mode="chunked" if use_chunked_covariance else "full",
            peak_cov_buffer_bytes=peak_cov_buffer_bytes,
            chunk_samples=chunk_samples,
            used_memory_bound=use_chunked_covariance,
        )
    )
    if store_reconstruction_matrices:
        diagnostics["reconstruction_matrices"] = np.asarray(reconstruction_matrices)

    logger.debug(
        "AdaptiveASR reconstruction details: variant=%s, %d window(s), "
        "%d component(s) maximum, %.1f%% samples reconstructed.",
        adaptive_variant,
        diagnostics["n_windows"],
        diagnostics["max_components_reconstructed"],
        100.0 * diagnostics["fraction_reconstructed_samples"],
    )

    next_state = {
        "cov": cov_state.copy() if cov_state is not None else None,
        "carry": carry_out,
        "iir": iir_state.copy() if iir_state is not None else None,
        "last_R": last_R.copy(),
        "last_trivial": bool(last_trivial),
    }
    return X_clean, diagnostics, next_state
