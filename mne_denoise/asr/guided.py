"""Guided ASR: DSS-biased soft Artifact Subspace Reconstruction (experimental).

Standard ASR detects *when/where* an EEG subspace is statistically abnormal
(its variance exceeds a clean-calibration threshold) and removes it with a
**binary** keep/reject decision. Its documented weakness, acute in mobile/MoBI
EEG, is over-cleaning: because the decision is variance-only, it can reconstruct
away real high-variance neural activity (task ERPs, SSVEP, gait-locked rhythms).

``GuidedASR`` keeps ASR's abnormality detection but adds two things:

1. **Bias operators** (reused from the DSS machinery) score *what kind* each
   flagged component direction is -- artifact-like vs brain-like -- via the
   quadratic form of the direction against bank covariances ``C_artifact`` and
   ``C_preserve``.
2. **Soft reconstruction** replaces binary rejection of an ASR-flagged
   component with a continuous keep weight ``w in [0, 1]`` (1 = keep, 0 =
   suppress, intermediate = partial attenuation).

The soft weight rescues components ASR would wrongly remove when they are
brain-like, while leaving artifact-like abnormal components suppressed. The
estimator is built on the ``method="riemannian_windowed"`` backbone, so with
``reconstruction="hard"`` and no bias operators it is mathematically identical
to :class:`mne_denoise.asr.ASR` with ``method="riemannian_windowed"``.

This is an **experimental proof-of-concept** and must be opted into with
``experimental=True``.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from ..utils import extract_data_from_mne
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
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Apply a calibrated ASR state with guided soft reconstruction.

    This function follows the same streaming contract as
    :func:`mne_denoise.asr.process_asr`. It uses the shared windowed ASR
    processor and changes only the component keep weights.

    Parameters
    ----------
    X : ndarray, shape (n_channels, n_times)
        Continuous data in the channel order used for calibration.
    sfreq : float
        Sampling frequency in Hz.
    state : ASRState
        Fitted calibration state returned by
        :func:`mne_denoise.asr.calibrate_asr`.
    artifact_cov : ndarray, shape (n_channels, n_channels) | None
        Covariance describing directions that should be attenuated. The
        covariance is validated and symmetrized. Component scores are divided
        by its trace so only spatial structure affects guidance.
    preserve_cov : ndarray, shape (n_channels, n_channels) | None
        Covariance describing directions that should be preserved. The
        covariance is validated and symmetrized. Component scores are divided
        by its trace so only spatial structure affects guidance.
    reconstruction : {'soft', 'hard'}
        Reconstruction rule. ``'soft'`` applies guidance-aware continuous
        weights. ``'hard'`` exactly follows windowed ASR and requires both
        guidance covariances to be ``None``.
    guidance_strength : float
        Guidance contribution in ``[0, 1]``. Zero returns baseline soft-ASR
        weights; one applies the full artifact/preserve adjustment.
    window_length : float
        Processing window length in seconds.
    window_overlap : float
        Accepted for parity with :func:`mne_denoise.asr.process_asr`.
    max_dims : float | int
        Maximum number or fraction of components reconstructed per window.
    regularization : float
        Relative eigenvalue floor for covariance calculations.
    store_reconstruction_matrices : bool
        If True, include per-window reconstruction matrices in diagnostics.
    max_mem_mb : int | None
        Memory bound controlling full-stack versus rolling covariance updates.
    lookahead : float | None
        Processing lookahead in seconds. ``None`` uses half a window.
    stepsize : int | None
        Samples between reconstruction-matrix updates. ``None`` uses half a
        window.

    Returns
    -------
    X_clean : ndarray, shape (n_channels, n_times)
        Reconstructed data.
    diagnostics : dict
        Standard ASR processing diagnostics plus ``soft_weights``,
        ``mean_soft_weight``, and ``reconstruction``.

    See Also
    --------
    process_asr : Apply a calibrated standard ASR model.
    GuidedASR : MNE- and scikit-learn-compatible guided estimator.

    Notes
    -----
    Soft GuidedASR is an unpublished and unvalidated research prototype. Its
    output must be evaluated independently for signal preservation and
    artifact attenuation.

    Guidance changes only components for which ``variance >= threshold``.
    Unflagged components retain weight one. For a flagged component, baseline
    soft-ASR weight is ``threshold / variance``. Only a covariance score above
    the isotropic reference ``1 / n_channels`` counts as directional evidence.
    The normalized preserve-minus-artifact evidence moves the weight toward
    one or zero under the linear control of ``guidance_strength``.

    Examples
    --------
    Apply soft reconstruction after calibrating a state:

    >>> clean, diagnostics = process_guided_asr(
    ...     data, sfreq=250.0, state=state, reconstruction="soft"
    ... )
    """
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
    return X_clean, diagnostics


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------


class GuidedASR(ASR):
    """
    DSS-biased soft Artifact Subspace Reconstruction (experimental).

    Extends :class:`mne_denoise.asr.ASR` (``method="riemannian_windowed"``
    backbone) with soft, structure-aware reconstruction. See the module
    docstring for the algorithm.

    Parameters
    ----------
    sfreq : float | None, default=None
        Sampling frequency in Hz. Required for NumPy input and inferred from
        MNE objects otherwise.
    cutoff : float, default=20.0
        ASR threshold multiplier.
    window_length : float, default=0.5
        Processing window length in seconds.
    window_overlap : float, default=0.66
        Overlap used for processing and threshold-fitting windows.
    max_dropout_fraction : float, default=0.1
        Fraction of low-RMS values ignored during threshold estimation.
    min_clean_fraction : float, default=0.25
        Minimum central fraction used to estimate clean RMS statistics.
    picks : str | list of str | list of int | None, default='eeg'
        Channels processed for MNE inputs. ``None`` processes every channel.
    calibration : {'auto', 'manual'}, default='auto'
        Whether calibration selects clean windows or uses all samples.
    calibration_window_length : float, default=1.0
        Window length in seconds for automatic clean-window selection.
    calibration_window_overlap : float, default=0.66
        Overlap for automatic clean-window selection.
    ref_max_bad_channels : float, default=0.075
        Maximum bad-channel fraction in a clean calibration window.
    ref_tolerances : tuple of float, default=(-inf, 5.5)
        Robust z-score limits for clean calibration windows.
    blocksize : int, default=10
        Samples averaged into each robust covariance block.
    max_dims : float | int, default=0.66
        Maximum number or fraction of reconstructed components per window.
    reject_by_annotation : bool, default=True
        Exclude bad annotations during Raw calibration and preserve annotated
        samples during transform.
    skip_by_annotation : tuple of str, default=('bad', 'bad_acq_skip')
        Annotation prefixes treated as bad.
    cov_estimator : {'geometric_median', 'mean', 'median'}, default='geometric_median'
        Calibration covariance aggregation rule.
    regularization : float, default=1e-8
        Relative eigenvalue floor for covariance calculations.
    filter_kind : {'none', 'asr', 'highpass'}, default='none'
        Filter used only for ASR statistics.
    window_criterion : float | int | None, default=None
        Optional final retained-sample rejection criterion.
    window_criterion_tolerances : tuple of float, default=(-inf, 7.0)
        Robust z-score limits for final window rejection.
    lookahead : float | None, default=None
        Processing lookahead in seconds. ``None`` uses half a window.
    stepsize : int | None, default=None
        Samples between reconstruction updates. ``None`` uses half a window.
    max_mem_mb : int | None, default=512
        Bound selecting full-stack or rolling covariance updates.
    copy : bool, default=True
        Reserved for API compatibility. Transform returns a new object.
    store_reconstruction_matrices : bool, default=False
        Store every reconstruction matrix in processing diagnostics.
    artifact_biases : sequence of DSS bias operators | None, default=None
        Operators (e.g. :class:`mne_denoise.dss.denoisers.LineNoiseBias`,
        ``BandpassBias``) whose biased covariance defines the artifact-like
        subspace ``C_artifact``. Each must accept ``(n_channels, n_times)`` and
        return the same shape (the ``LinearDenoiser`` ``.apply`` contract).
        Multiple operators contribute equally after trace normalization.
    preserve_biases : sequence of DSS bias operators | None, default=None
        Operators defining the brain-like subspace ``C_preserve`` to protect
        (e.g. ``PeakFilterBias`` for SSVEP, ``BandpassBias`` for a target band).
        Multiple operators contribute equally after trace normalization.
    reconstruction : {'soft', 'hard'}, default='soft'
        ``'soft'`` uses continuous weights only for ASR-flagged components;
        ``'hard'`` reproduces standard ASR's binary keep/reject.
    guidance_strength : float, default=1.0
        Guidance contribution in ``[0, 1]``. Zero gives baseline soft-ASR
        weights and one applies the full artifact/preserve adjustment.
    experimental : bool, default=False
        Must be ``True`` to use the guided soft reconstruction.
    random_state : int | None, default=None
        Reserved for future stochastic calibration strategies.
    n_jobs : int | None, default=None
        Reserved for future parallel processing.
    verbose : bool | str | int | None, default=None
        Controls ASR progress logging.

    See Also
    --------
    ASR : Standard Artifact Subspace Reconstruction estimator.
    process_guided_asr : Low-level guided processing function.

    Notes
    -----
    ``GuidedASR`` soft reconstruction is an unpublished, unvalidated research
    prototype. Its current evidence is limited to unit tests and synthetic
    benchmarks. It must not be treated as a validated EEG preprocessing method
    without independent checks on the target data and scientific endpoints.

    With ``reconstruction="hard"`` and no bias operators, ``GuidedASR`` is
    identical to ``ASR(method="riemannian_windowed")``.

    Equal bias weighting, the isotropic evidence reference, and the linear
    guidance equation are explicit experimental modeling choices.

    Examples
    --------
    Create an explicitly opted-in estimator:

    >>> from mne_denoise.asr import GuidedASR
    >>> guided = GuidedASR(sfreq=250.0, experimental=True)
    """

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

    def fit(
        self,
        X,
        y=None,
        *,
        calibration=None,
        calibration_mask=None,
    ) -> GuidedASR:
        """
        Calibrate ASR and fit the optional guidance covariances.

        Standard ASR calibration is delegated to :meth:`ASR.fit`. Bias-bank
        covariances are then estimated from the target data only when guidance
        operators are configured.

        Parameters
        ----------
        X : Raw | Epochs | ndarray
            Target data. It supplies ASR calibration when ``calibration`` is
            ``None`` and always supplies the data used to fit bias operators.
        y : None
            Ignored.
        calibration : Raw | Epochs | ndarray | None, default=None
            Optional separate data used only for standard ASR calibration.
        calibration_mask : ndarray | None, default=None
            Optional boolean sample mask for two-dimensional calibration data
            or Raw input after annotation exclusion.

        Returns
        -------
        GuidedASR
            Fitted estimator.

        See Also
        --------
        ASR.fit : Fit standard ASR calibration.

        Notes
        -----
        The base :class:`ASR` fit performs threshold calibration. GuidedASR
        then performs one additional pass over ``X`` only when bias operators
        need artifact or preserve covariances.

        Examples
        --------
        Fit the estimator using a separate clean calibration recording:

        >>> guided.fit(target_raw, calibration=clean_raw)
        """
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

        super().fit(X, y=y, calibration=calibration, calibration_mask=calibration_mask)

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
        return self

    # -- transform ---------------------------------------------------------

    def _process(self, selected: np.ndarray, sfreq: float):
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
        )
