"""Juggler Artifact Subspace Reconstruction (JugglerASR) module.

This module implements the ``JugglerASR`` class and its associated helper functions,
providing a scikit-learn and MNE-compatible estimator for extreme-motion EEG.

Unlike standard ASR which uses sliding-window thresholding to extract clean
calibration blocks, Juggler relies on instantaneous amplitude thresholding via
DBSCAN or Generalized Extreme Value (GEV) distribution fitting to isolate clean
reference samples.

This module exposes the following core components:
- :class:`JugglerASR`: The primary estimator class matching the ASR API.
- :func:`select_juggler_reference_samples`: The standalone sample selection routine
  (exposing the ``'dbscan'`` and ``'gev'`` strategies) which can be used independently
  of subspace reconstruction.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import spatial, stats

from .._data import extract_data_from_mne
from .._logging import logger, verbose
from ..progress import _validate_callback
from ._calibration import calibrate_asr
from ._filters import _design_statistics_filter, _lfilter_channels
from ._validation import (
    _validate_array_2d,
    _validate_backend_params,
    _validate_common_params,
    _validate_juggler_params,
)
from ._windowing import _create_good_sample_mask_from_mne
from .core import ASR

if TYPE_CHECKING:
    from mne.epochs import BaseEpochs
    from mne.io import BaseRaw


@verbose
def select_juggler_reference_samples(
    X: np.ndarray,
    sfreq: float,
    strategy: str = "dbscan",
    selection_filter_kind: str = "asr",
    dbscan_top_k: int = 5,
    dbscan_eps: float | str = "auto",
    dbscan_min_samples: int | float | str = "auto",
    gev_grid_size: int = 2048,
    min_reference_fraction: float = 0.05,
    verbose: bool | str | int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Select calibration samples using Juggler's ASR rules.

    Parameters
    ----------
    X : ndarray, shape (n_channels, n_times)
        Continuous candidate calibration data.
    sfreq : float
        Sampling frequency in Hz.
    strategy : {'dbscan', 'gev'}
        Juggler reference-selection strategy.
    selection_filter_kind : {'asr', 'highpass', 'none'}
        Statistics-only filter applied before amplitude ranking. The paper
        uses the ASR pre-emphasis filter, so ``'asr'`` is the default.
    dbscan_top_k : int
        Number of largest per-sample channel amplitudes to keep as the DBSCAN
        feature vector. The paper uses five channels.
    dbscan_eps : float | {'auto', 'paper'}
        DBSCAN neighborhood radius. ``'auto'`` and ``'paper'`` use one tenth
        of the modal maximum amplitude, matching the paper description.
    dbscan_min_samples : int | float | {'auto', 'paper'}
        DBSCAN core-neighborhood count. ``'auto'`` and ``'paper'`` use ten
        percent of the mode-derived clean-sample count.
    gev_grid_size : int
        Number of grid points used when locating the fitted GEV mode.
    min_reference_fraction : float
        Minimum acceptable retained fraction. Smaller retained sets are treated
        as calibration failures.
    verbose : bool | str | int | None
        MNE-style logging level. Selection diagnostics are emitted at DEBUG;
        the owning estimator reports the fitted result at INFO.

    Returns
    -------
    X_ref : ndarray, shape (n_channels, n_selected_times)
        Selected samples from the causally pre-emphasized data. Juggler applies
        the ASR IIR filter before pointwise selection and calibrates from the
        resulting filtered reference samples.
    sample_mask : ndarray, shape (n_times,)
        Boolean mask of the retained reference samples.
    diagnostics : dict
        Selection diagnostics including fitted modes, DBSCAN labels, and the
        retained fraction.

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.asr.juggler import select_juggler_reference_samples
    >>> rng = np.random.default_rng(42)
    >>> sfreq = 100.0
    >>> # Generate 10 seconds of 4-channel noise
    >>> data = rng.standard_normal((4, 1000))
    >>> # Inject a large artifact in the middle
    >>> data[:, 450:550] *= 20.0
    >>> X_ref, sample_mask, diagnostics = select_juggler_reference_samples(
    ...     data, sfreq, strategy="dbscan"
    ... )
    >>> # The reference dataset should be smaller than the original
    >>> X_ref.shape[1] < data.shape[1]
    True
    """
    X = _validate_array_2d(X)
    _validate_juggler_params(
        strategy=strategy,
        dbscan_top_k=dbscan_top_k,
        gev_grid_size=gev_grid_size,
        min_reference_fraction=min_reference_fraction,
    )

    filter_b, filter_a = _design_statistics_filter(sfreq, selection_filter_kind)
    X_stats, filter_zi = _lfilter_channels(X, filter_b, filter_a)

    amplitude = np.abs(X_stats)
    sorted_amplitude = np.sort(amplitude, axis=0)[::-1]
    top_k = min(int(dbscan_top_k), X.shape[0])
    features = sorted_amplitude[:top_k].T
    leading_amplitude = features[:, 0]

    diagnostics: dict[str, Any] = {
        "reference_selection_strategy": strategy,
        "selection_filter_kind": selection_filter_kind,
        "selection_filter_b": filter_b.copy(),
        "selection_filter_a": filter_a.copy(),
        "selection_filter_zi": filter_zi.copy(),
        "leading_amplitude": leading_amplitude.copy(),
        "dbscan_top_k": int(top_k),
    }

    if strategy == "dbscan":
        sample_mask, dbscan_info = _select_dbscan_reference_mask(
            features,
            leading_amplitude,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
        )
        diagnostics.update(dbscan_info)
    else:
        sample_mask, gev_info = _select_gev_reference_mask(
            leading_amplitude,
            grid_size=gev_grid_size,
        )
        diagnostics.update(gev_info)

    selected_samples = int(np.sum(sample_mask))
    keep_fraction = float(selected_samples / sample_mask.size)
    minimum_samples = max(1, int(np.floor(min_reference_fraction * sample_mask.size)))
    if selected_samples < minimum_samples:
        raise RuntimeError(
            "Juggler reference selection retained too little data: "
            f"{keep_fraction * 100:.1f}% < {min_reference_fraction * 100:.1f}%."
        )

    X_ref = X_stats[:, sample_mask]
    diagnostics.update(
        {
            "reference_sample_mask": sample_mask.copy(),
            "reference_selected_samples": selected_samples,
            "reference_candidate_samples": int(X.shape[1]),
            "reference_selected_fraction": keep_fraction,
        }
    )
    logger.debug(
        "Juggler reference selection: strategy=%s, retained %d/%d samples (%.1f%%).",
        strategy,
        selected_samples,
        sample_mask.size,
        100.0 * keep_fraction,
    )
    return X_ref, sample_mask, diagnostics


class JugglerASR(ASR):
    """Juggler Artifact Subspace Reconstruction (JugglerASR).

    This estimator replaces standard ASR's sliding-window calibration logic
    with a pointwise amplitude procedure to select clean reference samples (Kim et al. 2025).
    The burst-repair (reconstruction) stage remains identical to :class:`mne_denoise.asr.ASR`.

    Two strategies for reference selection are available:
    - ``strategy='dbscan'``: Density-based spatial clustering (ASRDBSCAN)
    - ``strategy='gev'``: Generalized Extreme Value distribution fitting (ASRGEV)

    Parameters
    ----------
    sfreq : float | None, default=None
        Sampling frequency in Hz.
    cutoff : float, default=20.0
        ASR threshold multiplier.
    strategy : {'dbscan', 'gev'}, default='dbscan'
        The reference sample selection strategy.
    window_length : float, default=0.5
        Length of the burst-repair reconstruction window in seconds.
    window_overlap : float, default=0.66
        Overlap fraction of the reconstruction window.
    max_dropout_fraction : float, default=0.1
        Maximum allowed fraction of dropped out (zeroed) samples per window.
    min_clean_fraction : float, default=0.25
        Minimum allowed fraction of clean components per window.
    picks : str | list[str] | list[int] | None, default="eeg"
        Channels to process.
    calibration_window_length : float, default=1.0
        Fallback calibration window parameter (rarely used in Juggler, which selects point-by-point).
    calibration_window_overlap : float, default=0.66
        Fallback calibration overlap.
    ref_max_bad_channels : float, default=0.075
        Maximum fraction of bad channels in a window.
    ref_tolerances : tuple[float, float], default=(-np.inf, 5.5)
        Z-score tolerances for fallback rejection.
    blocksize : int, default=10
        Processing blocksize for covariance matrix memory.
    max_dims : float | int, default=0.66
        Maximum retained variance/dimensions for reconstruction.
    reject_by_annotation : bool, default=True
        Whether to reject bad segments annotated in MNE.
    skip_by_annotation : tuple[str, ...], default=("bad", "bad_acq_skip")
        Annotation descriptions to skip.
    cov_estimator : str, default="geometric_median"
        Covariance estimator ('geometric_median' or 'euclidean').
    regularization : float, default=1e-8
        Covariance regularization.
    filter_kind : str, default="asr"
        Pre-emphasis filter applied during calibration and reconstruction.
    window_criterion : float | int | str | None, default=None
        Additional criterion for dropping dirty windows.
    window_criterion_tolerances : tuple[float, float], default=(-np.inf, 7.0)
        Tolerances for window dropping.
    lookahead : float | None, default=None
        State tracking parameter.
    stepsize : int | None, default=None
        Step size.
    max_mem_mb : int | None, default=512
        Maximum memory allowed for block processing.
    copy : bool, default=True
        Whether to copy data.
    store_reconstruction_matrices : bool, default=False
        Whether to store all per-window reconstruction matrices.
    selection_filter_kind : str, default="asr"
        Filter applied *before* DBSCAN/GEV sample selection.
    dbscan_top_k : int, default=5
        Number of largest per-sample channel amplitudes to keep as DBSCAN features.
    dbscan_eps : float | str, default="auto"
        DBSCAN neighborhood radius.
    dbscan_min_samples : int | float | str, default="auto"
        DBSCAN core-neighborhood count.
    gev_grid_size : int, default=2048
        Number of grid points for GEV fitting.
    min_reference_fraction : float, default=0.05
        Minimum fraction of samples that must be retained as clean reference.
    random_state : int | None, default=None
        Random state for reproducibility.
    n_jobs : int | None, default=None
        Number of parallel jobs.
    verbose : bool | str | int | None, default=None
        Logging verbosity.
    """

    def __init__(
        self,
        sfreq: float | None = None,
        cutoff: float = 20.0,
        strategy: str = "dbscan",
        window_length: float = 0.5,
        window_overlap: float = 0.66,
        max_dropout_fraction: float = 0.1,
        min_clean_fraction: float = 0.25,
        picks: str | list[str] | list[int] | None = "eeg",
        calibration_window_length: float = 1.0,
        calibration_window_overlap: float = 0.66,
        ref_max_bad_channels: float = 0.075,
        ref_tolerances: tuple[float, float] = (-np.inf, 5.5),
        blocksize: int = 10,
        max_dims: float | int = 0.66,
        reject_by_annotation: bool = True,
        skip_by_annotation: tuple[str, ...] = ("bad", "bad_acq_skip"),
        cov_estimator: str = "geometric_median",
        regularization: float = 1e-8,
        filter_kind: str = "asr",
        window_criterion: float | int | str | None = None,
        window_criterion_tolerances: tuple[float, float] = (-np.inf, 7.0),
        lookahead: float | None = None,
        stepsize: int | None = None,
        max_mem_mb: int | None = 512,
        copy: bool = True,
        store_reconstruction_matrices: bool = False,
        selection_filter_kind: str = "asr",
        dbscan_top_k: int = 5,
        dbscan_eps: float | str = "auto",
        dbscan_min_samples: int | float | str = "auto",
        gev_grid_size: int = 2048,
        min_reference_fraction: float = 0.05,
        random_state: int | None = None,
        n_jobs: int | None = None,
        verbose: bool | str | int | None = None,
    ) -> None:
        super().__init__(
            sfreq=sfreq,
            cutoff=cutoff,
            window_length=window_length,
            window_overlap=window_overlap,
            max_dropout_fraction=max_dropout_fraction,
            min_clean_fraction=min_clean_fraction,
            method="standard",
            experimental=False,
            calibration="manual",
            picks=picks,
            calibration_window_length=calibration_window_length,
            calibration_window_overlap=calibration_window_overlap,
            ref_max_bad_channels=ref_max_bad_channels,
            ref_tolerances=ref_tolerances,
            blocksize=blocksize,
            max_dims=max_dims,
            reject_by_annotation=reject_by_annotation,
            skip_by_annotation=skip_by_annotation,
            cov_estimator=cov_estimator,
            regularization=regularization,
            filter_kind=filter_kind,
            window_criterion=window_criterion,
            window_criterion_tolerances=window_criterion_tolerances,
            lookahead=lookahead,
            stepsize=stepsize,
            max_mem_mb=max_mem_mb,
            copy=copy,
            store_reconstruction_matrices=store_reconstruction_matrices,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        self.strategy = strategy
        self.selection_filter_kind = selection_filter_kind
        self.dbscan_top_k = dbscan_top_k
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.gev_grid_size = gev_grid_size
        self.min_reference_fraction = min_reference_fraction

    @verbose
    def fit(
        self,
        X: BaseRaw | BaseEpochs | np.ndarray,
        y=None,
        calibration: BaseRaw | BaseEpochs | np.ndarray | None = None,
        calibration_mask: np.ndarray | None = None,
        *,
        callback=None,
        verbose: bool | str | int | None = None,
    ) -> JugglerASR:
        """Fit JugglerASR from a contaminated or clean calibration stream.

        Unlike standard ASR, this step runs DBSCAN or GEV on instantaneous
        amplitudes to isolate clean reference samples, then estimates the
        clean signal subspace covariance.

        Parameters
        ----------
        X : mne.io.Raw | mne.Epochs | np.ndarray
            The primary data stream.
        y : None
            Ignored. Included for scikit-learn compatibility.
        calibration : mne.io.Raw | mne.Epochs | np.ndarray | None, default=None
            Separate calibration dataset. If None, `X` is used.
        calibration_mask : np.ndarray | None, default=None
            Optional boolean mask shape `(n_times,)` to pre-select samples.
        callback : callable | None
            Called synchronously after each shared ASR threshold calibration
            component completes. Callback return values are ignored and
            callback exceptions propagate unchanged.

        Returns
        -------
        self : JugglerASR
            The fitted instance.
        """
        del y
        callback = _validate_callback(callback)
        _validate_backend_params(
            method=self.method,
            experimental=self.experimental,
            lookahead=self.lookahead,
            stepsize=self.stepsize,
            window_criterion=self.window_criterion,
        )
        _validate_common_params(
            sfreq=self.sfreq if self.sfreq is not None else 1.0,
            cutoff=self.cutoff,
            window_length=self.window_length,
            window_overlap=self.window_overlap,
            max_dropout_fraction=self.max_dropout_fraction,
            min_clean_fraction=self.min_clean_fraction,
            regularization=self.regularization,
        )
        _validate_juggler_params(
            strategy=self.strategy,
            dbscan_top_k=self.dbscan_top_k,
            gev_grid_size=self.gev_grid_size,
            min_reference_fraction=self.min_reference_fraction,
        )
        if self.filter_kind != self.selection_filter_kind:
            raise ValueError(
                "JugglerASR requires filter_kind and selection_filter_kind to "
                "match so calibration and reconstruction use the same "
                "statistics filter"
            )

        fit_input = X if calibration is None else calibration
        data, sfreq, mne_type, orig_inst, picks, ch_names = extract_data_from_mne(
            fit_input,
            auto_pick=True,
            concatenate_epochs=True,
        )
        if mne_type == "evoked":
            raise ValueError(
                "JugglerASR.fit() does not support Evoked calibration data"
            )
        sfreq = self._resolve_sfreq(sfreq)
        data_2d = np.asarray(data, dtype=np.float64)
        if calibration_mask is not None:
            calibration_mask = np.asarray(calibration_mask, dtype=bool)
            if calibration_mask.shape != (data_2d.shape[1],):
                raise ValueError(
                    "calibration_mask must have shape (n_times,), got "
                    f"{calibration_mask.shape}"
                )
            data_2d = data_2d[:, calibration_mask]
        if mne_type == "raw" and self.reject_by_annotation:
            good_mask = _create_good_sample_mask_from_mne(
                orig_inst, self.skip_by_annotation
            )
            data_2d = data_2d[:, good_mask]

        self._warn_preprocessing_state(orig_inst, mne_type)
        reference_data, reference_mask, reference_info = (
            select_juggler_reference_samples(
                data_2d,
                sfreq,
                strategy=self.strategy,
                selection_filter_kind=self.selection_filter_kind,
                dbscan_top_k=self.dbscan_top_k,
                dbscan_eps=self.dbscan_eps,
                dbscan_min_samples=self.dbscan_min_samples,
                gev_grid_size=self.gev_grid_size,
                min_reference_fraction=self.min_reference_fraction,
            )
        )
        state, cal_info = calibrate_asr(
            reference_data,
            sfreq,
            cutoff=self.cutoff,
            window_length=self.window_length,
            window_overlap=self.window_overlap,
            calibration="manual",
            calibration_window_length=self.calibration_window_length,
            calibration_window_overlap=self.calibration_window_overlap,
            ref_max_bad_channels=self.ref_max_bad_channels,
            ref_tolerances=self.ref_tolerances,
            blocksize=self.blocksize,
            max_dropout_fraction=self.max_dropout_fraction,
            min_clean_fraction=self.min_clean_fraction,
            cov_estimator=self.cov_estimator,
            regularization=self.regularization,
            # Reference samples have already been filtered continuously before
            # pointwise selection. Filtering their concatenation again would
            # introduce discontinuity transients and alter the calibration.
            filter_kind="none",
            method="standard",
            max_mem_mb=self.max_mem_mb,
            callback=callback,
        )
        state.filter_b = np.asarray(reference_info["selection_filter_b"]).copy()
        state.filter_a = np.asarray(reference_info["selection_filter_a"]).copy()
        state.filter_zi = np.asarray(reference_info["selection_filter_zi"]).copy()
        cal_info.update(reference_info)
        cal_info["filter_kind"] = self.filter_kind
        cal_info["calibration_input_filtering"] = "continuous_before_selection"
        cal_info["clean_window_mask"] = np.array([], dtype=bool)
        cal_info["clean_window_scores"] = np.empty(
            (0, data_2d.shape[0]), dtype=np.float64
        )
        cal_info["n_clean_windows"] = 0
        cal_info["n_calibration_windows"] = 0
        cal_info["reference_selection_strategy"] = self.strategy
        cal_info["reference_mask_kind"] = "sample"

        self.state_ = state
        self.sfreq_ = float(sfreq)
        self.picks_ = picks
        self.ch_names_ = ch_names
        self.n_channels_ = data_2d.shape[0]
        self.M_ = state.M
        self.mixing_ = state.M
        self.T_ = state.T
        self.threshold_matrix_ = state.T
        self.thresholds_ = state.thresholds
        self.calibration_patterns_ = state.calibration_patterns
        self.patterns_ = state.calibration_patterns
        self.rank_ = state.rank
        self.reference_sample_mask_ = reference_mask
        self.clean_window_mask_ = np.array([], dtype=bool)
        self.clean_window_scores_ = np.empty((0, data_2d.shape[0]), dtype=np.float64)
        # JugglerASR selects calibration data sample-by-sample, not by windows.
        self.calibration_mask_kind_ = "sample"
        self.calibration_info_ = cal_info
        self.history_ = {
            "method": "juggler",
            "strategy": self.strategy,
            "source_type": mne_type,
            "n_channels": self.n_channels_,
            "sfreq": self.sfreq_,
        }
        logger.info(
            "JugglerASR: strategy=%s, method=%s, channels=%d, sfreq=%.3g Hz, "
            "cutoff=%.3g, rank=%d, retained %.1f%% reference samples.",
            self.strategy,
            self.method,
            self.n_channels_,
            self.sfreq_,
            self.cutoff,
            self.rank_,
            100.0 * reference_info["reference_selected_fraction"],
        )
        return self

    def get_calibration_mask(self) -> np.ndarray:
        """Return the sample-wise reference mask chosen during calibration.

        JugglerASR selects calibration data point-by-point (Kim et al. 2025),
        so the mask is **sample-based** (``calibration_mask_kind_ == "sample"``),
        unlike the window-based mask of the standard ASR backends.

        Returns
        -------
        mask : ndarray of bool, shape (n_times,)
            Boolean array where ``True`` indicates the sample was retained as a
            clean calibration reference by the DBSCAN/GEV selector.
        """
        self._check_is_fitted()
        return np.asarray(self.reference_sample_mask_, dtype=bool).copy()


def _select_dbscan_reference_mask(
    features: np.ndarray,
    leading_amplitude: np.ndarray,
    dbscan_eps: float | str,
    dbscan_min_samples: int | float | str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select reference samples using the DBSCAN clustering strategy.

    Parameters
    ----------
    features : np.ndarray
        2D array of shape (n_samples, top_k) representing the sorted top-K channel
        amplitudes for each sample.
    leading_amplitude : np.ndarray
        1D array of the maximum amplitude per sample.
    dbscan_eps : float | str
        The DBSCAN neighborhood radius parameter or 'auto'/'paper' for automatic sizing.
    dbscan_min_samples : int | float | str
        The DBSCAN core samples parameter or 'auto'/'paper' for automatic sizing.

    Returns
    -------
    sample_mask : np.ndarray
        Boolean array where ``True`` indicates the sample is selected as clean.
    diagnostics : dict
        A dictionary containing the resolved DBSCAN parameters, cluster scores, and labels.
    """
    del leading_amplitude
    feature_scale = np.linalg.norm(features, axis=1)
    mode = _histogram_mode(feature_scale)
    estimated_clean_count = int(np.sum(feature_scale <= mode))
    eps = _resolve_dbscan_eps(dbscan_eps, mode, feature_scale)
    min_samples = _resolve_dbscan_min_samples(
        dbscan_min_samples,
        estimated_clean_count,
        features.shape[0],
    )
    labels, dbscan_memory_info = _dbscan_chebyshev_memory_bounded(
        features,
        eps=eps,
        min_samples=min_samples,
    )
    candidate_labels = np.unique(labels[labels >= 0])
    if candidate_labels.size == 0:
        raise RuntimeError(
            "DBSCAN found no non-noise cluster. Increase eps or provide a "
            "longer calibration stream."
        )

    cluster_scores = []
    cluster_sizes = []
    for label in candidate_labels:
        label_points = features[labels == label]
        cluster_scores.append(
            float(np.median(np.linalg.norm(label_points, ord=np.inf, axis=1)))
        )
        cluster_sizes.append(int(label_points.shape[0]))

    score_order = np.lexsort(
        (-np.asarray(cluster_sizes, dtype=int), np.asarray(cluster_scores, dtype=float))
    )
    selected_label = int(candidate_labels[score_order[0]])
    sample_mask = labels == selected_label

    diagnostics = {
        "juggler_dbscan_mode": float(mode),
        "juggler_dbscan_scale": "l2_norm",
        "juggler_dbscan_eps": float(eps),
        "juggler_dbscan_min_samples": int(min_samples),
        "juggler_dbscan_labels": labels.copy(),
        "juggler_dbscan_selected_label": selected_label,
        "juggler_dbscan_cluster_sizes": np.asarray(cluster_sizes, dtype=int),
        "juggler_dbscan_cluster_scores": np.asarray(cluster_scores, dtype=np.float64),
        "juggler_dbscan_estimated_clean_count": int(estimated_clean_count),
        **dbscan_memory_info,
    }
    return sample_mask, diagnostics


def _dbscan_chebyshev_memory_bounded(
    features: np.ndarray,
    *,
    eps: float,
    min_samples: int,
    count_batch_size: int = 4096,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run exact Chebyshev DBSCAN without materializing all neighborhoods.

    Scikit-learn's DBSCAN stores the complete radius-neighborhood graph. Dense
    reference clusters can therefore require quadratic memory, which is
    prohibitive for the 100,000-sample Juggler simulation. This implementation
    preserves the DBSCAN definition while keeping only batched neighbor counts,
    occupied grid cells, and one border-point neighborhood in memory.

    Parameters
    ----------
    features : ndarray, shape (n_samples, n_features)
        Feature vectors to cluster.
    eps : float
        Chebyshev neighborhood radius.
    min_samples : int
        Minimum neighborhood size, including the point itself, for a core
        sample.
    count_batch_size : int
        Number of points used per radius-count query.

    Returns
    -------
    labels : ndarray, shape (n_samples,)
        DBSCAN cluster labels, with ``-1`` denoting noise.
    diagnostics : dict
        Memory-backend and core-sample diagnostics.
    """
    features = np.ascontiguousarray(features, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("features must be a non-empty 2D array")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be positive and finite")
    if min_samples < 1:
        raise ValueError("min_samples must be at least 1")
    count_batch_size = max(1, int(count_batch_size))

    n_samples = features.shape[0]
    cell_coordinates = np.floor(features / eps).astype(np.int64)
    sample_cells: dict[tuple[int, ...], list[int]] = {}
    for index in range(n_samples):
        key = tuple(int(value) for value in cell_coordinates[index])
        sample_cells.setdefault(key, []).append(index)
    sample_cell_keys = sorted(sample_cells, key=lambda key: sample_cells[key][0])
    sample_cell_positions = {
        key: position for position, key in enumerate(sample_cell_keys)
    }
    sample_cell_trees = {
        key: spatial.cKDTree(features[np.asarray(indices, dtype=int)])
        for key, indices in sample_cells.items()
    }
    neighbor_offsets = tuple(itertools.product((-1, 0, 1), repeat=features.shape[1]))

    # Points in the same eps-wide Chebyshev cell are all mutual neighbors.
    # Accumulate exact counts only across occupied adjacent cells. This avoids
    # a dense all-point radius query, whose running time can become quadratic
    # even when ``return_length=True`` keeps its memory bounded.
    neighbor_counts = np.empty(n_samples, dtype=np.int64)
    for indices in sample_cells.values():
        neighbor_counts[np.asarray(indices, dtype=int)] = len(indices)
    for left_position, left_key in enumerate(sample_cell_keys):
        left_indices = np.asarray(sample_cells[left_key], dtype=int)
        left_points = features[left_indices]
        for offset in neighbor_offsets:
            right_key = tuple(
                coordinate + delta for coordinate, delta in zip(left_key, offset)
            )
            right_position = sample_cell_positions.get(right_key)
            if right_position is None or right_position <= left_position:
                continue
            right_indices = np.asarray(sample_cells[right_key], dtype=int)
            right_points = features[right_indices]
            left_needed = neighbor_counts[left_indices] < int(min_samples)
            if np.any(left_needed):
                neighbor_counts[left_indices[left_needed]] += sample_cell_trees[
                    right_key
                ].query_ball_point(
                    left_points[left_needed],
                    r=eps,
                    p=np.inf,
                    return_length=True,
                    workers=1,
                )
            right_needed = neighbor_counts[right_indices] < int(min_samples)
            if np.any(right_needed):
                neighbor_counts[right_indices[right_needed]] += sample_cell_trees[
                    left_key
                ].query_ball_point(
                    right_points[right_needed],
                    r=eps,
                    p=np.inf,
                    return_length=True,
                    workers=1,
                )
    core_mask = neighbor_counts >= int(min_samples)
    core_indices = np.flatnonzero(core_mask)
    labels = np.full(n_samples, -1, dtype=np.int64)
    if core_indices.size == 0:
        return labels, {
            "juggler_dbscan_backend": "ckdtree_grid_memory_bounded",
            "juggler_dbscan_core_samples": 0,
            "juggler_dbscan_core_cells": 0,
            "juggler_dbscan_count_batch_size": count_batch_size,
        }

    # Every pair of points in the same eps-wide Chebyshev grid cell is a
    # neighbor. Connected components can therefore be found at cell level,
    # avoiding a radius query for every point in a dense core cluster.
    core_cells: dict[tuple[int, ...], list[int]] = {}
    for index in core_indices:
        key = tuple(int(value) for value in cell_coordinates[index])
        core_cells.setdefault(key, []).append(int(index))
    cell_keys = sorted(core_cells, key=lambda key: core_cells[key][0])
    parent = np.arange(len(cell_keys), dtype=np.int64)
    cell_min_index = np.asarray(
        [core_cells[key][0] for key in cell_keys], dtype=np.int64
    )

    def find(position: int) -> int:
        while parent[position] != position:
            parent[position] = parent[parent[position]]
            position = int(parent[position])
        return position

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if cell_min_index[left_root] <= cell_min_index[right_root]:
            parent[right_root] = left_root
            cell_min_index[left_root] = min(
                cell_min_index[left_root], cell_min_index[right_root]
            )
        else:
            parent[left_root] = right_root
            cell_min_index[right_root] = min(
                cell_min_index[left_root], cell_min_index[right_root]
            )

    cell_trees = {
        key: spatial.cKDTree(features[np.asarray(indices, dtype=int)])
        for key, indices in core_cells.items()
    }
    cell_positions = {key: position for position, key in enumerate(cell_keys)}
    for left_position, left_key in enumerate(cell_keys):
        left_points = features[np.asarray(core_cells[left_key], dtype=int)]
        for offset in neighbor_offsets:
            right_key = tuple(
                coordinate + delta for coordinate, delta in zip(left_key, offset)
            )
            right_position = cell_positions.get(right_key)
            if right_position is None:
                continue
            if right_position <= left_position:
                continue
            right_points = features[np.asarray(core_cells[right_key], dtype=int)]
            if left_points.shape[0] <= right_points.shape[0]:
                distances, _ = cell_trees[right_key].query(
                    left_points,
                    k=1,
                    p=np.inf,
                    distance_upper_bound=eps,
                    workers=1,
                )
            else:
                distances, _ = cell_trees[left_key].query(
                    right_points,
                    k=1,
                    p=np.inf,
                    distance_upper_bound=eps,
                    workers=1,
                )
            if np.any(np.isfinite(distances)):
                union(left_position, right_position)

    roots = np.asarray([find(position) for position in range(len(cell_keys))])
    unique_roots = sorted(set(roots.tolist()), key=lambda root: cell_min_index[root])
    root_to_label = {root: label for label, root in enumerate(unique_roots)}
    for position, key in enumerate(cell_keys):
        label = root_to_label[int(roots[position])]
        labels[np.asarray(core_cells[key], dtype=int)] = label

    # A non-core point is a DBSCAN border point when it neighbors a core
    # sample. Grouping border points by grid cell avoids one tree query per
    # point. All core points within one cell share a component label, and only
    # the 3**n_features adjacent cells can contain Chebyshev neighbors.
    border_cells: dict[tuple[int, ...], list[int]] = {}
    for index in np.flatnonzero(~core_mask):
        key = tuple(int(value) for value in cell_coordinates[index])
        border_cells.setdefault(key, []).append(int(index))
    for border_key, indices in border_cells.items():
        border_indices = np.asarray(indices, dtype=int)
        border_points = features[border_indices]
        for offset in neighbor_offsets:
            core_key = tuple(
                coordinate + delta for coordinate, delta in zip(border_key, offset)
            )
            core_indices_in_cell = core_cells.get(core_key)
            if core_indices_in_cell is None:
                continue
            core_label = int(labels[core_indices_in_cell[0]])
            if core_key == border_key:
                matched = np.ones(border_indices.size, dtype=bool)
            else:
                distances, _ = cell_trees[core_key].query(
                    border_points,
                    k=1,
                    p=np.inf,
                    distance_upper_bound=eps,
                    workers=1,
                )
                matched = np.isfinite(distances)
            matched_indices = border_indices[matched]
            current = labels[matched_indices]
            labels[matched_indices] = np.where(
                current < 0, core_label, np.minimum(current, core_label)
            )

    return labels, {
        "juggler_dbscan_backend": "ckdtree_grid_memory_bounded",
        "juggler_dbscan_core_samples": int(core_indices.size),
        "juggler_dbscan_core_cells": int(len(core_cells)),
        "juggler_dbscan_count_batch_size": count_batch_size,
    }


def _select_gev_reference_mask(
    leading_amplitude: np.ndarray,
    grid_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select reference samples using the Generalized Extreme Value (GEV) strategy.

    Parameters
    ----------
    leading_amplitude : np.ndarray
        1D array containing the maximum absolute amplitude across channels for each sample.
    grid_size : int
        Number of grid points to use when finding the mode of the fitted GEV distribution.

    Returns
    -------
    sample_mask : np.ndarray
        Boolean array where ``True`` indicates the sample is selected as clean.
    diagnostics : dict
        A dictionary containing the fitted GEV parameters and mode.
    """
    amplitude_scale = float(np.median(leading_amplitude))
    if not np.isfinite(amplitude_scale) or amplitude_scale <= 0:
        amplitude_scale = float(np.max(leading_amplitude))
    if not np.isfinite(amplitude_scale) or amplitude_scale <= 0:
        raise RuntimeError("GEV fitting requires at least one positive amplitude")
    normalized = leading_amplitude / amplitude_scale
    try:
        shape, loc_normalized, scale_normalized = stats.genextreme.fit(normalized)
    except Exception as exc:
        raise RuntimeError(f"GEV fitting failed: {exc}") from exc
    if not np.isfinite(scale_normalized) or scale_normalized <= 0:
        raise RuntimeError("GEV fitting returned a non-positive scale")

    distribution = stats.genextreme(
        shape,
        loc=loc_normalized,
        scale=scale_normalized,
    )
    if shape < 1.0 and abs(shape) > 1e-10:
        standardized_mode = (1.0 - (1.0 - shape) ** shape) / shape
        mode_normalized = loc_normalized + scale_normalized * standardized_mode
    elif abs(shape) <= 1e-10:
        mode_normalized = float(loc_normalized)
    else:
        # For boundary-mode shapes, evaluate a fixed probability grid. Working
        # in normalized coordinates keeps this fallback invariant to EEG units.
        probabilities = np.linspace(1e-6, 1.0 - 1e-6, int(grid_size))
        grid = np.asarray(distribution.ppf(probabilities), dtype=np.float64)
        logpdf = np.asarray(distribution.logpdf(grid), dtype=np.float64)
        valid = np.isfinite(grid) & np.isfinite(logpdf)
        if grid.shape == probabilities.shape and np.any(valid):
            valid_grid = grid[valid]
            valid_logpdf = logpdf[valid]
            mode_normalized = float(valid_grid[int(np.argmax(valid_logpdf))])
        else:
            mode_normalized = _histogram_mode(normalized)
    if not np.isfinite(mode_normalized):
        mode_normalized = _histogram_mode(normalized)
    mode = float(mode_normalized * amplitude_scale)
    sample_mask = normalized <= mode_normalized
    loc = float(loc_normalized * amplitude_scale)
    scale = float(scale_normalized * amplitude_scale)
    diagnostics = {
        "juggler_gev_shape": float(shape),
        "juggler_gev_loc": loc,
        "juggler_gev_scale": scale,
        "juggler_gev_mode": mode,
        "juggler_gev_grid_size": int(grid_size),
        "juggler_gev_normalization_scale": amplitude_scale,
    }
    return sample_mask, diagnostics


def _histogram_mode(values: np.ndarray) -> float:
    """Calculate the mode of a distribution using a histogram approach.

    Parameters
    ----------
    values : np.ndarray
        1D array of numerical values.

    Returns
    -------
    mode : float
        The estimated mode of the distribution.
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Cannot estimate a mode from empty values")
    scale = max(float(np.max(np.abs(values))), np.finfo(float).tiny)
    if float(np.ptp(values)) <= np.finfo(float).eps * scale:
        return float(values[0])
    edges = np.histogram_bin_edges(values, bins="fd")
    if edges.size < 2:
        return float(np.median(values))
    counts, edges = np.histogram(values, bins=edges)
    idx = int(np.argmax(counts))
    return float(0.5 * (edges[idx] + edges[idx + 1]))


def _resolve_dbscan_eps(
    value: float | str,
    mode: float,
    feature_scale: np.ndarray,
) -> float:
    """Resolve the DBSCAN ``eps`` neighborhood radius.

    When ``value`` is ``'auto'`` or ``'paper'``, eps is set to one tenth of the
    histogram mode of the feature scale. If the resulting eps is non-positive or
    non-finite, a fallback based on the median of positive feature scales is used.

    Parameters
    ----------
    value : float | str
        User-specified eps or ``'auto'``/``'paper'`` for automatic derivation.
    mode : float
        The histogram mode of the L2-normed feature scale.
    feature_scale : np.ndarray
        1D array of per-sample L2 norms used as a fallback anchor.

    Returns
    -------
    eps : float
        The resolved positive neighborhood radius.
    """
    if isinstance(value, str):
        if value not in ("auto", "paper"):
            raise ValueError("dbscan_eps must be a positive float, 'auto', or 'paper'")
        eps = mode / 10.0
    else:
        eps = float(value)
    if not np.isfinite(eps) or eps <= np.finfo(float).eps:
        positive = feature_scale[feature_scale > 0]
        if positive.size == 0:
            raise RuntimeError(
                "Cannot derive a positive DBSCAN eps from zero-amplitude data"
            )
        eps = max(float(np.median(positive)) / 10.0, np.finfo(float).eps)
    return float(eps)


def _resolve_dbscan_min_samples(
    value: int | float | str,
    estimated_clean_count: int,
    n_times: int,
) -> int:
    """Resolve the DBSCAN ``min_samples`` core-neighborhood count.

    When ``value`` is ``'auto'`` or ``'paper'``, min_samples is set to 10% of the
    estimated clean sample count. A float in (0, 1] is treated as a fraction of
    the estimated clean count. The result is clamped to [2, n_times].

    Parameters
    ----------
    value : int | float | str
        User-specified count, a fraction, or ``'auto'``/``'paper'``.
    estimated_clean_count : int
        Number of samples estimated to be clean (amplitude <= mode).
    n_times : int
        Total number of time samples in the data.

    Returns
    -------
    min_samples : int
        The resolved core-neighborhood count, at least 2.
    """
    if isinstance(value, str):
        if value not in ("auto", "paper"):
            raise ValueError(
                "dbscan_min_samples must be an int, a float fraction, 'auto', or 'paper'"
            )
        min_samples = int(np.ceil(0.10 * max(estimated_clean_count, 1)))
    elif isinstance(value, float) and 0.0 < value <= 1.0:
        min_samples = int(np.ceil(float(value) * max(estimated_clean_count, 1)))
    else:
        min_samples = int(value)
    min_samples = max(2, min_samples)
    min_samples = min(min_samples, n_times)
    return int(min_samples)
