"""Guided soft-reconstruction variant of ASR."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from .._data import extract_data_from_mne
from .._logging import logger, verbose
from ..progress import _ProgressCallback, _validate_callback
from ._covariance import (
    _covariance_stack_bytes,
    _process_memory_info,
)
from ._guidance import (
    _compute_guidance_covariance,
    _guided_component_weights,
)
from ._reconstruction import (
    _empty_process_diagnostics,
    _prepare_asr_stream,
    _process_asr_windowed,
)
from ._types import ASRState
from ._validation import _validate_covariance_matrix
from .core import ASR

_EXPERIMENTAL_DISCLAIMER = (
    "GuidedASR soft reconstruction is an unpublished, unvalidated experimental "
    "research prototype. Validate neural-signal preservation and artifact "
    "attenuation independently before using it in scientific analyses."
)


@verbose
def process_guided_asr(
    X: np.ndarray,
    sfreq: float,
    state: ASRState,
    *,
    artifact_cov: np.ndarray | None = None,
    preserve_cov: np.ndarray | None = None,
    reconstruction: str = "soft",
    guidance_strength: float = 1.0,
    window_length: float = 0.5,
    window_overlap: float = 0.66,
    max_dims: float | int = 0.66,
    regularization: float = 1e-8,
    store_reconstruction_matrices: bool = False,
    max_mem_mb: int | None = 512,
    lookahead: float | None = None,
    stepsize: int | None = None,
    callback=None,
    verbose: bool | str | int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply a calibrated ASR state with guided reconstruction.

    Parameters
    ----------
    X : ndarray, shape (n_channels, n_times)
        Continuous data in the calibrated channel order.
    sfreq : float
        Sampling frequency in Hz.
    state : ASRState
        Calibration state returned by calibrate_asr.
    artifact_cov : ndarray, shape (n_channels, n_channels), or None, default=None
        Covariance describing directions to attenuate.
    preserve_cov : ndarray, shape (n_channels, n_channels), or None, default=None
        Covariance describing directions to preserve.
    reconstruction : {"soft", "hard"}, default="soft"
        Use guidance-aware continuous weights or standard binary ASR weights.
    guidance_strength : float, default=1.0
        Guidance contribution in [0, 1].
    window_length : float, default=0.5
        Processing window length in seconds.
    window_overlap : float, default=0.66
        Processing-window overlap.
    max_dims : float or int, default=0.66
        Maximum fraction or number of reconstructed components.
    regularization : float, default=1e-8
        Relative covariance eigenvalue floor.
    store_reconstruction_matrices : bool, default=False
        Include reconstruction matrices in diagnostics.
    max_mem_mb : int or None, default=512
        Memory bound for covariance processing.
    lookahead : float or None, default=None
        Processing lookahead in seconds.
    stepsize : int or None, default=None
        Samples between reconstruction updates.
    callback : callable or None, default=None
        Synchronous progress callback.
    verbose : bool, str, int, or None, default=None
        Logging level.

    Returns
    -------
    X_clean : ndarray, shape (n_channels, n_times)
        Reconstructed data.
    diagnostics : dict
        ASR diagnostics with guided weights and reconstruction mode.

    Notes
    -----
    Hard reconstruction requires both guidance covariances to be None. Soft
    reconstruction requires the GuidedASR experimental opt-in when used through the
    estimator.
    """
    callback = _validate_callback(callback)
    if reconstruction not in ("soft", "hard"):
        raise ValueError("reconstruction must be 'soft' or 'hard'")
    if not np.isfinite(guidance_strength) or not 0 <= guidance_strength <= 1:
        raise ValueError("guidance_strength must be a finite number in [0, 1]")
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
    artifact_cov = _validate_covariance_matrix(
        artifact_cov,
        name="artifact_cov",
        n_channels=n_channels,
    )
    preserve_cov = _validate_covariance_matrix(
        preserve_cov,
        name="preserve_cov",
        n_channels=n_channels,
    )
    if reconstruction == "hard" and (
        artifact_cov is not None or preserve_cov is not None
    ):
        raise ValueError(
            "artifact_cov and preserve_cov only affect reconstruction='soft'"
        )

    win_len = prepared.win_len
    lookahead_samples = prepared.lookahead_samples
    stepsize = prepared.stepsize
    max_bad = prepared.max_bad
    if max_bad <= 0:
        diagnostics = _empty_process_diagnostics(n_times)
        diagnostics.update(
            {
                "soft_weights": np.ones((1, n_channels), dtype=np.float64),
                "mean_soft_weight": 1.0,
                "covariance_geometry": "guided",
                "reconstruction": reconstruction,
            }
        )
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
            "GuidedASR reconstruction details: mode=%s, identity path, %d sample(s).",
            reconstruction,
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

    component_weight_function = None
    if reconstruction == "soft":

        def component_weight_function(
            variances: np.ndarray,
            eigenvectors: np.ndarray,
            thresholds: np.ndarray,
            forced_keep: np.ndarray,
        ) -> np.ndarray:
            return _guided_component_weights(
                variances,
                eigenvectors,
                thresholds,
                forced_keep=forced_keep,
                artifact_covariance=artifact_cov,
                preserve_covariance=preserve_cov,
                strength=guidance_strength,
            )

    X_clean, diagnostics = _process_asr_windowed(
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
        component_weight_function=component_weight_function,
        return_component_weights=True,
        callback=callback,
        progress_method="guided_asr",
    )
    weights = diagnostics.pop("component_weights")
    diagnostics.update(
        {
            "soft_weights": weights,
            "mean_soft_weight": float(weights.mean()) if weights.size else 1.0,
            "covariance_geometry": "guided",
            "reconstruction": reconstruction,
        }
    )
    diagnostics.update(
        _process_memory_info(
            n_channels=n_channels,
            n_stream_input=n_stream_input,
            max_mem_mb=max_mem_mb,
            memory_mode=("guided_rolling" if use_rolling_covariance else "guided"),
            peak_cov_buffer_bytes=_covariance_stack_bytes(1, n_channels),
            chunk_samples=win_len if use_rolling_covariance else n_stream_input,
            used_memory_bound=use_rolling_covariance,
        )
    )
    logger.debug(
        "GuidedASR reconstruction details: mode=%s, %d window(s), "
        "mean component keep weight=%.3f.",
        reconstruction,
        diagnostics["n_windows"],
        diagnostics["mean_soft_weight"],
    )
    return X_clean, diagnostics


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------


class GuidedASR(ASR):
    """Guided soft-reconstruction variant of ASR.

    GuidedASR uses artifact and preserve bias covariances to modify ASR component
    weights. Soft reconstruction requires experimental=True.

    Parameters
    ----------
    sfreq : float or None, default=None
        Sampling frequency in Hz; inferred from MNE metadata when available.
    cutoff : float, default=20.0
        ASR threshold multiplier.
    window_length : float, default=0.5
        Processing window length in seconds.
    window_overlap : float, default=0.66
        Processing-window overlap.
    max_dropout_fraction : float, default=0.1
        Fraction of low-RMS values excluded from threshold estimation.
    min_clean_fraction : float, default=0.25
        Minimum central fraction used for clean RMS statistics.
    picks : str, list of str, list of int, or None, default="eeg"
        MNE channels to process; NumPy input uses all rows.
    calibration : {"auto", "manual"}, default="auto"
        Calibration mode.
    calibration_window_length : float, default=1.0
        Automatic calibration-window length in seconds.
    calibration_window_overlap : float, default=0.66
        Automatic calibration-window overlap.
    ref_max_bad_channels : float, default=0.075
        Maximum bad-channel fraction in a calibration window.
    ref_tolerances : tuple of float, default=(-np.inf, 5.5)
        Robust z-score bounds for calibration-window selection.
    blocksize : int, default=10
        Samples per calibration covariance block.
    max_dims : float or int, default=0.66
        Maximum fraction or number of reconstructed dimensions.
    reject_by_annotation : bool, default=True
        Exclude bad annotated samples during calibration.
    skip_by_annotation : tuple of str, default=("bad", "bad_acq_skip")
        Annotation prefixes treated as bad.
    cov_estimator : {"geometric_median", "mean", "median"}, default="geometric_median"
        Calibration-covariance aggregation rule.
    regularization : float, default=1e-8
        Relative covariance eigenvalue floor.
    filter_kind : {"none", "asr", "highpass"}, default="asr"
        Filter used for ASR statistics.
    window_criterion : float, int, or None, default=None
        Optional final retained-sample criterion.
    window_criterion_tolerances : tuple of float, default=(-np.inf, 7.0)
        Robust z-score bounds for the final criterion.
    lookahead : float or None, default=None
        Processing lookahead in seconds.
    stepsize : int or None, default=None
        Samples between reconstruction updates.
    max_mem_mb : int or None, default=512
        Memory bound for covariance processing.
    copy : bool, default=True
        Reserved compatibility parameter; transformations return new outputs.
    store_reconstruction_matrices : bool, default=False
        Store per-window reconstruction matrices in diagnostics.
    artifact_biases : sequence or None, default=None
        DSS bias operators defining artifact-like covariance directions.
    preserve_biases : sequence or None, default=None
        DSS bias operators defining directions to preserve.
    reconstruction : {"soft", "hard"}, default="soft"
        Guided continuous weights or binary ASR reconstruction.
    guidance_strength : float, default=1.0
        Guidance contribution in [0, 1].
    experimental : bool, default=False
        Must be true for soft reconstruction.
    random_state : int or None, default=None
        Reserved for stochastic calibration.
    n_jobs : int or None, default=None
        Reserved for future parallel processing.
    verbose : bool, str, int, or None, default=None
        Logging level.

    See Also
    --------
    ASR
        Standard ASR without guidance covariances.
    process_guided_asr
        Low-level array processing with a calibrated ASR state.

    Notes
    -----
    With reconstruction="hard" and no bias operators, this uses the
    riemannian_windowed ASR backend. The soft path is an unpublished, unvalidated
    experimental research API and requires independent evaluation of artifact
    attenuation and signal preservation.
    """

    _progress_method = "guided_asr"

    def __init__(
        self,
        sfreq: float | None = None,
        cutoff: float = 20.0,
        window_length: float = 0.5,
        window_overlap: float = 0.66,
        max_dropout_fraction: float = 0.1,
        min_clean_fraction: float = 0.25,
        picks: str | list[str] | list[int] | None = "eeg",
        calibration: str = "auto",
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
        window_criterion: float | int | None = None,
        window_criterion_tolerances: tuple[float, float] = (-np.inf, 7.0),
        lookahead: float | None = None,
        stepsize: int | None = None,
        max_mem_mb: int | None = 512,
        copy: bool = True,
        store_reconstruction_matrices: bool = False,
        artifact_biases: list | tuple | None = None,
        preserve_biases: list | tuple | None = None,
        reconstruction: str = "soft",
        guidance_strength: float = 1.0,
        experimental: bool = False,
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
            method="riemannian_windowed",
            experimental=experimental,
            calibration=calibration,
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
        self.artifact_biases = artifact_biases
        self.preserve_biases = preserve_biases
        self.reconstruction = reconstruction
        self.guidance_strength = guidance_strength

    # -- fit ---------------------------------------------------------------

    @verbose
    def fit(
        self,
        X,
        y=None,
        *,
        calibration=None,
        calibration_mask=None,
        callback=None,
        verbose: bool | str | int | None = None,
    ) -> GuidedASR:
        """Fit ASR calibration and optional guidance covariances.

        Parameters
        ----------
        X : Raw, Epochs, or ndarray
            Target data; it also supplies calibration when calibration is None.
        y : None, default=None
            Ignored for scikit-learn compatibility.
        calibration : Raw, Epochs, or ndarray, default=None
            Optional separate ASR calibration data.
        calibration_mask : ndarray of bool or None, default=None
            Optional mask for calibration samples.
        callback : callable or None, default=None
            Synchronous calibration progress callback.
        verbose : bool, str, int, or None, default=None
            Logging level for this call.

        Returns
        -------
        GuidedASR
            The fitted estimator.
        """
        callback = _validate_callback(callback)
        if self.reconstruction not in ("soft", "hard"):
            raise ValueError("reconstruction must be 'soft' or 'hard'")
        if (
            not np.isfinite(self.guidance_strength)
            or not 0 <= self.guidance_strength <= 1
        ):
            raise ValueError("guidance_strength must be a finite number in [0, 1]")
        if self.reconstruction == "soft" and not self.experimental:
            raise ValueError(
                "GuidedASR soft reconstruction is experimental; pass "
                "experimental=True to use it (reconstruction='hard' reproduces "
                "standard ASR and needs no opt-in)."
            )
        if self.reconstruction == "hard" and (
            self.artifact_biases or self.preserve_biases
        ):
            raise ValueError(
                "artifact_biases and preserve_biases only affect reconstruction='soft'"
            )
        if self.reconstruction == "soft":
            warnings.warn(_EXPERIMENTAL_DISCLAIMER, UserWarning, stacklevel=2)

        super().fit(
            X,
            y=y,
            calibration=calibration,
            calibration_mask=calibration_mask,
            callback=callback,
        )

        # Bias operators define artifact / brain *subspaces*, so they are
        # estimated from the primary recording ``X`` (which contains those
        # phenomena), whereas the ASR threshold model above uses ``calibration``
        # when provided.
        self.artifact_cov_ = None
        self.preserve_cov_ = None
        if self.artifact_biases or self.preserve_biases:
            data_2d, _, _, _, _, _ = extract_data_from_mne(
                X,
                ch_names=self.ch_names_,
                auto_pick=True,
                concatenate_epochs=True,
            )
            data_2d = np.asarray(data_2d, dtype=np.float64)
            if data_2d.shape[0] != self.n_channels_:
                raise ValueError(
                    "GuidedASR bias data channel count does not match calibration: "
                    f"{data_2d.shape[0]} vs {self.n_channels_}"
                )
            self.artifact_cov_ = _compute_guidance_covariance(
                data_2d,
                self.artifact_biases,
                name="artifact_biases",
            )
            self.preserve_cov_ = _compute_guidance_covariance(
                data_2d,
                self.preserve_biases,
                name="preserve_biases",
            )
        self.history_.update(
            {
                "estimator": "GuidedASR",
                "reconstruction": self.reconstruction,
                "guidance_strength": self.guidance_strength,
                "experimental": self.reconstruction == "soft",
            }
        )
        logger.info(
            "GuidedASR: reconstruction=%s, guidance strength=%.3g, "
            "artifact biases=%d, preserve biases=%d.",
            self.reconstruction,
            self.guidance_strength,
            len(self.artifact_biases or ()),
            len(self.preserve_biases or ()),
        )
        return self

    # -- transform ---------------------------------------------------------

    def _process(
        self,
        selected: np.ndarray,
        sfreq: float,
        *,
        callback: _ProgressCallback | None = None,
    ):
        return process_guided_asr(
            selected,
            sfreq,
            self.state_,
            artifact_cov=getattr(self, "artifact_cov_", None),
            preserve_cov=getattr(self, "preserve_cov_", None),
            reconstruction=self.reconstruction,
            guidance_strength=self.guidance_strength,
            window_length=self.window_length,
            window_overlap=self.window_overlap,
            max_dims=self.max_dims,
            regularization=self.regularization,
            store_reconstruction_matrices=self.store_reconstruction_matrices,
            max_mem_mb=self.max_mem_mb,
            lookahead=self.lookahead,
            stepsize=self.stepsize,
            callback=callback,
        )
