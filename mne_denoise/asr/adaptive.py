"""Adaptive Artifact Subspace Reconstruction (AASR) module.

This module implements the ``AdaptiveASR`` class, an experimental scikit-learn
and MNE-compatible estimator that extends standard ASR with continuous tracking
of the clean signal subspace using Hebbian or anti-Hebbian similarity-matching learning.

Unlike standard ASR which relies on a fixed static calibration, Adaptive ASR
operates in three stages:
1. **Initial Calibration**: Computes a starting robust covariance matrix and baseline
   thresholds using an initial clean data window (via ``fit``).
2. **Adaptive Tracking**: Continuously updates the clean-subspace model across incoming
   data chunks using moving-window aggregations and similarity-matching rules
   (via ``partial_fit`` or internally during streaming).
3. **Reconstruction**: Repairs burst artifacts using standard ASR logic, but applies
   the dynamically tracked covariance state instead of a fixed baseline (via ``transform``).

This module exposes three distinct adaptive tracking variants:
- ``variant="psp"`` -- plasticity-stabilized (Hebbian) similarity matching.
- ``variant="psw"`` -- plasticity-stabilized whitening (anti-Hebbian).
- ``variant="mw"`` -- moving-window calibration (where ``mw_mode`` selects either
  ``"final_state"`` or per-segment ``"sliding"`` semantics).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .._logging import set_log_level_from_verbose
from .._spatial import continuous_to_epochs
from ..utils import extract_data_from_mne, reconstruct_mne_object
from ._covariance import (
    _adaptive_covariance_sqrt,
)
from ._distribution import (
    _AASR_BETA_GRID,
    _fit_adaptive_thresholds,
)
from ._filters import _design_asr_filter, _lfilter_channels
from ._learner import _AdaptiveSimilarityMatcher, _build_adaptive_learner
from ._reconstruction import _process_adaptive_chunk
from ._types import ASRState, _copy_asr_state, _copy_process_state
from ._validation import (
    _check_transform_channels,
    _round_half_up,
    _validate_adaptive_params,
    _validate_array_2d,
    _validate_backend_params,
    _validate_common_params,
)
from ._windowing import (
    _create_good_sample_mask_from_mne,
    _extract_clean_calibration_samples,
    compute_clean_window_mask,
)
from .core import ASR

try:
    from mne.epochs import BaseEpochs
    from mne.evoked import Evoked
    from mne.io import BaseRaw
except ImportError:  # pragma: no cover
    mne = None
    BaseEpochs = Any
    Evoked = Any
    BaseRaw = Any


class AdaptiveASR(ASR):
    """Adaptive Artifact Subspace Reconstruction (AASR) estimator.

    This estimator extends the standard ASR algorithm by dynamically tracking
    the clean signal subspace over time. Three adaptive variants are exposed:

    - ``variant='psp'``: principal subspace projection updates (Hebbian)
    - ``variant='psw'``: principal subspace whitening updates (anti-Hebbian)
    - ``variant='mw'``: moving-window calibration updates

    Parameters
    ----------
    sfreq : float | None, default=None
        Sampling frequency in Hz. Required for NumPy arrays. For MNE objects,
        this may be ``None`` and is inferred from ``info['sfreq']``.
    cutoff : float, default=20.0
        ASR threshold multiplier. Values around 20 are conservative; lower
        values clean more aggressively.
    variant : {'psw', 'psp'}, default='psw'
        The adaptive update rule to use.
    window_length : float, default=0.5
        Processing/statistics window length in seconds.
    update_window_length : float, default=0.1
        RMS-statistics window length within each adaptive update segment. This is
        not the duration of the segment passed to ``partial_fit``.
    calibration_window_length : float, default=1.0
        Window length in seconds for automatic clean-window selection.
    calibration_window_overlap : float, default=0.66
        Overlap fraction for automatic clean-window selection.
    ref_max_bad_channels : float, default=0.2
        Maximum fraction of channels exceeding robust tolerances in a clean
        calibration window.
    ref_tolerances : tuple of float, default=(-3.5, 5.0)
        Lower and upper robust z-score bounds for clean-window selection.
    blocksize : int, default=10
        Number of successive samples averaged into each covariance block for
        eigendecomposition.
    max_dims : float | int, default=0.66
        Maximum fraction or absolute number of spatial dimensions to retain
        during reconstruction.
    max_dropout_fraction : float, default=0.1
        Fraction of lowest RMS values ignored while estimating thresholds.
    min_clean_fraction : float, default=0.25
        Minimum central fraction used to estimate clean RMS statistics.
    picks : str | list of str | list of int | slice | None, default='eeg'
        Channels to include. Slices and lists of integers will be interpreted
        as channel indices.
    reject_by_annotation : bool, default=True
        Whether to reject bad segments based on annotations during calibration.
    skip_by_annotation : tuple of str, default=('bad', 'bad_acq_skip')
        If a string in this tuple is a prefix of an annotation description,
        that segment is ignored during calibration.
    regularization : float, default=1e-8
        Ridge regularization added to covariance matrices to prevent singular
        inversions.
    window_criterion : float | int | str | None, default=None
        Pre-rejection criterion for entirely bad windows. If a float, acts
        as a threshold multiplier.
    window_criterion_tolerances : tuple of float, default=(-np.inf, 7.0)
        Tolerances for window_criterion testing.
    lookahead : float | None, default=None
        Lookahead time in seconds for the sliding window reconstruction.
        Defaults to half the window length.
    stepsize : int | None, default=None
        Stepsize in samples for the sliding window. Defaults to 32.
    max_mem_mb : int | None, default=512
        Maximum memory (in megabytes) allowed for internal chunking operations.
    copy : bool, default=True
        If True, data is copied before processing. If False, processing
        happens in place.
    store_reconstruction_matrices : bool, default=False
        If True, the diagnostic dictionaries will contain the applied
        reconstruction mixing matrices.
    learning_rate : float, default=0.2
        Step size parameter controlling how fast the subspace projection matrix
        incorporates new samples.
    tau : float | None, default=None
        Time constant for the lateral connections (similarity tracking). If None,
        it defaults to `10.0 / learning_rate`.
    mw_window_length : float, default=20.0
        The length of the moving window (in seconds) over which covariance is
        aggregated before triggering adaptive updates.
    mw_mode : {'final_state', 'cumulative'}, default='final_state'
        Mode for moving-window aggregation.
    random_state : int | None, default=None
        Random state for reproducibility in stochastic internal steps.
    n_jobs : int | None, default=None
        Number of jobs to run in parallel.
    verbose : bool | str | int | None, default=None
        Verbosity level.
    """

    def __init__(
        self,
        sfreq: float | None = None,
        cutoff: float = 20.0,
        variant: str = "psw",
        window_length: float = 0.5,
        update_window_length: float = 0.1,
        calibration_window_length: float = 1.0,
        calibration_window_overlap: float = 0.66,
        ref_max_bad_channels: float = 0.2,
        ref_tolerances: tuple[float, float] = (-3.5, 5.0),
        blocksize: int = 10,
        max_dims: float | int = 0.66,
        max_dropout_fraction: float = 0.1,
        min_clean_fraction: float = 0.25,
        picks: str | list[str] | list[int] | None = "eeg",
        reject_by_annotation: bool = True,
        skip_by_annotation: tuple[str, ...] = ("bad", "bad_acq_skip"),
        regularization: float = 1e-8,
        window_criterion: float | int | str | None = None,
        window_criterion_tolerances: tuple[float, float] = (-np.inf, 7.0),
        lookahead: float | None = None,
        stepsize: int | None = None,
        max_mem_mb: int | None = 512,
        copy: bool = True,
        store_reconstruction_matrices: bool = False,
        learning_rate: float = 0.2,
        tau: float | None = None,
        mw_window_length: float = 20.0,
        mw_mode: str = "final_state",
        random_state: int | None = None,
        n_jobs: int | None = None,
        verbose: bool | str | int | None = None,
    ) -> None:
        super().__init__(
            sfreq=sfreq,
            cutoff=cutoff,
            window_length=window_length,
            window_overlap=calibration_window_overlap,
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
            cov_estimator="geometric_median",
            regularization=regularization,
            filter_kind="none",
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
        self.variant = variant
        self.update_window_length = update_window_length
        self.calibration_window_length = calibration_window_length
        self.calibration_window_overlap = calibration_window_overlap
        self.ref_max_bad_channels = ref_max_bad_channels
        self.ref_tolerances = ref_tolerances
        self.learning_rate = learning_rate
        self.tau = tau
        self.mw_window_length = mw_window_length
        self.mw_mode = mw_mode

    def fit(
        self,
        X: BaseRaw | BaseEpochs | np.ndarray,
        y=None,
        calibration: BaseRaw | BaseEpochs | np.ndarray | None = None,
        calibration_mask: np.ndarray | None = None,
    ) -> AdaptiveASR:
        """Fit the initial adaptive ASR state from calibration data.

        This method estimates the initial robust spatial covariance matrix and
        computes the variance thresholds that define the clean signal subspace.

        Parameters
        ----------
        X : mne.io.Raw | mne.Epochs | np.ndarray
            The input data to be processed. If ``calibration`` is None, this data
            is used to compute the initial calibration state. For ``np.ndarray``,
            the shape should be (n_channels, n_times) for continuous data or
            (n_epochs, n_channels, n_times) for epoched data.
        y : None
            Ignored. Present for scikit-learn compatibility.
        calibration : mne.io.Raw | mne.Epochs | np.ndarray | None, default=None
            Optional separate calibration dataset. If provided, the initial baseline
            is learned from this dataset instead of ``X``.
        calibration_mask : np.ndarray | None, default=None
            Optional boolean mask of shape (n_times,) denoting which samples in the
            calibration data are clean and should be used to estimate the initial state.
            If None, the mask is estimated automatically.

        Returns
        -------
        self : AdaptiveASR
            The fitted estimator instance.
        """
        del y
        set_log_level_from_verbose(self.verbose)
        _validate_adaptive_params(
            variant=self.variant,
            update_window_length=self.update_window_length,
            calibration_window_length=self.calibration_window_length,
            calibration_window_overlap=self.calibration_window_overlap,
            ref_max_bad_channels=self.ref_max_bad_channels,
            learning_rate=self.learning_rate,
            tau=self.tau,
            mw_window_length=self.mw_window_length,
            mw_mode=self.mw_mode,
        )
        fit_input = X if calibration is None else calibration
        data, sfreq, mne_type, orig_inst, picks, ch_names = extract_data_from_mne(
            fit_input,
            auto_pick=True,
            concatenate_epochs=True,
        )
        if mne_type == "evoked":
            raise ValueError(
                "AdaptiveASR.fit() does not support Evoked calibration data"
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

        if self.variant == "mw":
            # MW-ASR (sliding-window subspace, no Hebbian carry-over).
            # Semantics: per-window subspace calibration; final state is the
            # last window's calibration. A single reconstruction pass over the
            # whole stream then uses that state.
            (
                state,
                cal_info,
                learner,
                process_state,
                mw_diagnostics,
            ) = self._fit_mw_state(data_2d, sfreq)
            self.mw_diagnostics_ = mw_diagnostics
        else:
            state, cal_info, learner, process_state = self._fit_adaptive_state(
                data_2d, sfreq
            )
            # Ensure attribute is always defined post-fit for consumer code.
            self.mw_diagnostics_ = []

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
        self.clean_window_mask_ = cal_info["clean_window_mask"]
        self.calibration_mask_kind_ = "window"
        self.clean_window_scores_ = cal_info["clean_window_scores"]
        self.calibration_info_ = cal_info
        self.adaptive_learner_ = learner
        self._initial_process_state_template_ = _copy_process_state(process_state)
        self.process_state_ = _copy_process_state(process_state)
        self.adaptive_update_history_ = [dict(cal_info)]
        self.history_ = {
            "method": "adaptive",
            "variant": self.variant,
            "source_type": mne_type,
            "n_channels": self.n_channels_,
            "sfreq": self.sfreq_,
        }
        return self

    def partial_fit(
        self,
        X: BaseRaw | BaseEpochs | np.ndarray,
        y=None,
        calibration_mask: np.ndarray | None = None,
    ) -> AdaptiveASR:
        """Update the adaptive calibration state on a new clean chunk.

        This method applies the chosen adaptive similarity-matching rule to gently
        update the underlying clean subspace model using incoming data.
        It is designed for online, streaming workflows.

        Parameters
        ----------
        X : mne.io.Raw | mne.Epochs | np.ndarray
            The incoming data chunk to update the state with. For ``np.ndarray``,
            shape must be (n_channels, n_times) or (n_epochs, n_channels, n_times).
            The chunk must contain more samples than ``calibration_window_length``
            so clean-window statistics can be estimated. The published AASR demo
            uses complete 20-second update segments and omits its incomplete tail.
        y : None
            Ignored. Present for scikit-learn compatibility.
        calibration_mask : np.ndarray | None, default=None
            Optional boolean mask of shape (n_times,) designating which samples in
            the incoming chunk are clean. If None, it is estimated automatically.

        Returns
        -------
        self : AdaptiveASR
            The updated estimator instance.
        """
        del y
        set_log_level_from_verbose(self.verbose)
        if self.variant == "mw":
            raise NotImplementedError(
                "AdaptiveASR(variant='mw') does not support partial_fit. "
                "MW-ASR semantics require a single fit() call over the full "
                "stream; the windowing happens internally. To re-calibrate, "
                "call fit() again."
            )
        if not hasattr(self, "state_"):
            return self.fit(X, calibration_mask=calibration_mask)

        data, sfreq, mne_type, orig_inst, picks, ch_names = extract_data_from_mne(
            X,
            auto_pick=True,
            concatenate_epochs=True,
        )
        if mne_type == "evoked":
            raise ValueError("AdaptiveASR.partial_fit() does not support Evoked data")
        sfreq = self._resolve_sfreq(sfreq, fitted=True)
        if not np.isclose(sfreq, self.sfreq_):
            raise ValueError(
                f"Input sfreq {sfreq} does not match fitted sfreq {self.sfreq_}"
            )
        _check_transform_channels(
            self.n_channels_,
            self.ch_names_,
            data.shape[0],
            ch_names,
        )
        self._warn_preprocessing_state(orig_inst, mne_type)
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

        update_info = self._update_adaptive_state(data_2d, sfreq)
        self.adaptive_update_history_.append(update_info)
        self.calibration_info_ = update_info
        self.clean_window_mask_ = update_info["clean_window_mask"]
        self.calibration_mask_kind_ = "window"
        self.clean_window_scores_ = update_info["clean_window_scores"]
        self.M_ = self.state_.M
        self.mixing_ = self.state_.M
        self.T_ = self.state_.T
        self.threshold_matrix_ = self.state_.T
        self.thresholds_ = self.state_.thresholds
        self.calibration_patterns_ = self.state_.calibration_patterns
        self.patterns_ = self.state_.calibration_patterns
        self.rank_ = self.state_.rank
        return self

    def transform(
        self,
        X: BaseRaw | BaseEpochs | Evoked | np.ndarray,
        y=None,
        copy: bool | None = None,
        return_diagnostics: bool = False,
    ) -> Any:
        """Clean data using the current adaptive ASR state.

        Detects high-variance artifact bursts and reconstructs them using the
        dynamically tracked spatial mixing matrices.

        Parameters
        ----------
        X : mne.io.Raw | mne.Epochs | np.ndarray
            The target data to be cleaned. Must match the channel count and
            sampling frequency used during ``fit``.
        y : None
            Ignored. Present for scikit-learn compatibility.
        copy : bool | None, default=None
            Ignored parameter provided for scikit-learn API compatibility.
        return_diagnostics : bool, default=False
            If True, returns a tuple ``(cleaned_data, diagnostics)`` where
            ``diagnostics`` is a dictionary detailing the reconstruction process.

        Returns
        -------
        X_clean : mne.io.Raw | mne.Epochs | np.ndarray
            The artifact-repaired data. Returns the same type as the input ``X``.
        diagnostics : dict, optional
            A dictionary containing processing metadata (e.g., sample masks,
            eigenvalues). Only returned if ``return_diagnostics=True``.
        """
        del y, copy
        set_log_level_from_verbose(self.verbose)
        self._check_is_fitted()
        data, sfreq, mne_type, orig_inst, picks, ch_names = extract_data_from_mne(
            X, auto_pick=True
        )
        if mne_type == "evoked":
            raise ValueError("AdaptiveASR.transform() does not support Evoked data")
        sfreq = self._resolve_sfreq(sfreq, fitted=True)
        if not np.isclose(sfreq, self.sfreq_):
            raise ValueError(
                f"Input sfreq {sfreq} does not match fitted sfreq {self.sfreq_}"
            )
        _check_transform_channels(
            self.n_channels_,
            self.ch_names_,
            data.shape[1] if mne_type == "epochs" else data.shape[0],
            ch_names,
        )
        self._warn_preprocessing_state(orig_inst, mne_type)

        if mne_type == "epochs":
            cleaned_data, diagnostics = self._transform_epochs_adaptive(data, sfreq)
        else:
            selected = np.asarray(data, dtype=np.float64)
            selected_clean, diagnostics, next_process_state = _process_adaptive_chunk(
                selected,
                sfreq,
                self.state_,
                self.process_state_,
                window_length=self.window_length,
                lookahead=self.lookahead,
                stepsize=self.stepsize,
                max_dims=self.max_dims,
                store_reconstruction_matrices=self.store_reconstruction_matrices,
                adaptive_variant=self.variant,
                max_mem_mb=self.max_mem_mb,
            )
            if mne_type == "raw" and self.reject_by_annotation:
                good_mask = _create_good_sample_mask_from_mne(
                    orig_inst, self.skip_by_annotation
                )
                selected_clean[:, ~good_mask] = selected[:, ~good_mask]
                diagnostics["sample_mask"] = diagnostics["sample_mask"] & good_mask
            if self.window_criterion is not None:
                rejection_mask, rejection_diag = compute_clean_window_mask(
                    selected_clean,
                    sfreq,
                    max_bad_channels=self.window_criterion,
                    zthresholds=self.window_criterion_tolerances,
                    window_length=self.calibration_window_length,
                    window_overlap=self.calibration_window_overlap,
                    max_dropout_fraction=self.max_dropout_fraction,
                    min_clean_fraction=self.min_clean_fraction,
                )
                if mne_type == "raw" and self.reject_by_annotation:
                    rejection_mask = rejection_mask & good_mask
                diagnostics.update(
                    {
                        "rejection_sample_mask": rejection_mask,
                        "rejection_window_starts": rejection_diag["window_starts"],
                        "rejection_window_stops": rejection_diag["window_stops"],
                        "rejection_window_keep_mask": rejection_diag[
                            "window_keep_mask"
                        ],
                        "rejection_window_remove_mask": rejection_diag[
                            "window_remove_mask"
                        ],
                        "fraction_retained_after_window_rejection": float(
                            np.mean(rejection_mask)
                        ),
                        "fraction_rejected_after_window_rejection": float(
                            1.0 - np.mean(rejection_mask)
                        ),
                    }
                )
            cleaned_data = selected_clean
            self.process_state_ = next_process_state

        self._store_transform_diagnostics(diagnostics)
        cleaned = reconstruct_mne_object(
            cleaned_data, orig_inst, mne_type, picks=picks, verbose=False
        )
        if return_diagnostics:
            return cleaned, diagnostics
        return cleaned

    def fit_transform(
        self,
        X: BaseRaw | BaseEpochs | np.ndarray,
        y=None,
        calibration: BaseRaw | BaseEpochs | np.ndarray | None = None,
        return_diagnostics: bool = False,
    ) -> Any:
        """Fit adaptive ASR and reconstruct ``X`` with the fitted state.

        For ``variant="mw", mw_mode="sliding"`` the work is per-window
        calibrate-AND-transform: each window is calibrated on itself and
        cleaned by that local calibration, then the cleaned slices are
        concatenated. ``fit()`` alone is still legal in sliding mode (it
        records per-window diagnostics) but ``transform()`` afterwards
        applies only the final window's state — semantically equivalent to
        ``mw_mode="final_state"``. The true per-segment behavior requires
        this ``fit_transform`` entry point.

        Parameters
        ----------
        X : mne.io.Raw | mne.Epochs | np.ndarray
            The input data to be fitted and transformed.
        y : None
            Ignored. Present for scikit-learn compatibility.
        calibration : mne.io.Raw | mne.Epochs | np.ndarray | None, default=None
            Optional separate calibration dataset.
        return_diagnostics : bool, default=False
            If True, returns a tuple ``(cleaned_data, diagnostics)``.

        Returns
        -------
        X_clean : mne.io.Raw | mne.Epochs | np.ndarray
            The artifact-repaired data.
        diagnostics : dict, optional
            A dictionary containing processing metadata. Only returned if
            ``return_diagnostics=True``.
        """
        if self.variant == "mw" and self.mw_mode == "sliding":
            return self._fit_transform_mw_sliding(
                X, calibration=calibration, return_diagnostics=return_diagnostics
            )
        self.fit(X, y=y, calibration=calibration)
        return self.transform(X, return_diagnostics=return_diagnostics)

    def _fit_transform_mw_sliding(
        self,
        X: BaseRaw | BaseEpochs | np.ndarray,
        calibration: BaseRaw | BaseEpochs | np.ndarray | None = None,
        return_diagnostics: bool = False,
    ) -> Any:
        """Per-window calibrate-AND-clean implementation for MW sliding mode.

        For each non-overlapping window of length ``mw_window_length``:

        1. Run the existing AdaptiveASR calibration on that window's data.
        2. Apply the resulting state to clean the same window's data.
        3. Concatenate the cleaned slices and return.

        Windows shorter than ``blocksize`` are skipped (data passes through
        unchanged). Calibration failures are also passed through.

        Parameters
        ----------
        X : mne.io.Raw | mne.Epochs | np.ndarray
            The input data to be processed using the sliding window approach.
        calibration : mne.io.Raw | mne.Epochs | np.ndarray | None, default=None
            Optional separate calibration dataset. For sliding MW mode, this is
            rarely used, as the calibration usually happens dynamically per window.
        return_diagnostics : bool, default=False
            If True, returns a tuple ``(cleaned_data, diagnostics)`` where
            ``diagnostics`` compiles metadata across all processed windows.

        Returns
        -------
        X_clean : mne.io.Raw | mne.Epochs | np.ndarray
            The artifact-repaired data.
        diagnostics : dict, optional
            A dictionary containing aggregated processing metadata. Only returned if
            ``return_diagnostics=True``.
        """
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
        _validate_adaptive_params(
            variant=self.variant,
            update_window_length=self.update_window_length,
            calibration_window_length=self.calibration_window_length,
            calibration_window_overlap=self.calibration_window_overlap,
            ref_max_bad_channels=self.ref_max_bad_channels,
            learning_rate=self.learning_rate,
            tau=self.tau,
            mw_window_length=self.mw_window_length,
            mw_mode=self.mw_mode,
        )
        fit_input = X if calibration is None else calibration
        data, sfreq, mne_type, orig_inst, picks, ch_names = extract_data_from_mne(
            fit_input,
            auto_pick=True,
        )
        if mne_type == "evoked":
            raise ValueError(
                "AdaptiveASR.fit_transform() does not support Evoked input "
                "with variant='mw', mw_mode='sliding'."
            )
        sfreq_val = self._resolve_sfreq(sfreq)
        if mne_type == "epochs":
            data_2d = np.transpose(data, (1, 0, 2)).reshape(data.shape[1], -1)
        else:
            data_2d = np.asarray(data, dtype=np.float64)
        if mne_type == "raw" and self.reject_by_annotation:
            good_mask = _create_good_sample_mask_from_mne(
                orig_inst, self.skip_by_annotation
            )
            data_2d_masked = data_2d[:, good_mask]
        else:
            data_2d_masked = data_2d

        self._warn_preprocessing_state(orig_inst, mne_type)

        n_times = data_2d_masked.shape[1]
        win_samples = max(1, int(round(self.mw_window_length * sfreq_val)))
        cleaned = data_2d_masked.copy()
        mw_diagnostics: list[dict[str, Any]] = []

        n_windows = (n_times + win_samples - 1) // win_samples
        last = None  # store the last successful calibration for the public state
        for window_idx in range(n_windows):
            start = window_idx * win_samples
            stop = min(start + win_samples, n_times)
            window = data_2d_masked[:, start:stop]
            entry: dict[str, Any] = {
                "window_idx": int(window_idx),
                "window_start": int(start),
                "window_stop": int(stop),
                "n_samples": int(window.shape[1]),
            }
            if window.shape[1] < self.blocksize:
                entry["status"] = "skipped_too_short"
                mw_diagnostics.append(entry)
                continue
            try:
                state, cal_info, learner, process_state = self._fit_adaptive_state(
                    window, sfreq_val
                )
                window_cleaned, _, _ = _process_adaptive_chunk(
                    window,
                    sfreq_val,
                    state,
                    process_state,
                    window_length=self.window_length,
                    lookahead=self.lookahead,
                    stepsize=self.stepsize,
                    max_dims=self.max_dims,
                    store_reconstruction_matrices=False,
                    adaptive_variant=self.variant,
                    max_mem_mb=self.max_mem_mb,
                )
                cleaned[:, start:stop] = window_cleaned
                entry.update(
                    {
                        "status": "passed",
                        "M": np.asarray(state.M, dtype=np.float64),
                        "T": np.asarray(state.T, dtype=np.float64),
                        "thresholds": np.asarray(state.thresholds, dtype=np.float64),
                        "rank": int(state.rank),
                        "clean_window_fraction": cal_info.get(
                            "calibration_clean_window_fraction"
                        ),
                    }
                )
                last = (state, cal_info, learner, process_state)
            except Exception as exc:  # noqa: BLE001
                print("CAUGHT:", repr(exc))
                entry.update(
                    {
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            mw_diagnostics.append(entry)

        if last is None:
            raise RuntimeError(
                "MW-ASR sliding-mode fit_transform() found no usable window "
                f"(n_windows={n_windows}, mw_window_length={self.mw_window_length})"
            )
        state, cal_info, learner, process_state = last
        cal_info = dict(cal_info)
        cal_info["adaptive_variant"] = "mw"
        cal_info["mw_mode"] = "sliding"
        cal_info["mw_n_windows"] = int(len(mw_diagnostics))
        cal_info["mw_window_length_s"] = float(self.mw_window_length)

        # Populate the standard fitted-state attributes from the FINAL window's
        # calibration so downstream introspection (.M_, .T_, .calibration_info_,
        # ...) behaves the same way as the existing final_state mode.
        self.state_ = state
        self.sfreq_ = float(sfreq_val)
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
        self.clean_window_mask_ = np.array([], dtype=bool)
        self.calibration_mask_kind_ = "window"
        self.clean_window_scores_ = np.empty((0, data_2d.shape[0]), dtype=np.float64)
        self.calibration_info_ = cal_info
        self.mw_diagnostics_ = mw_diagnostics
        self.process_state_ = _copy_process_state(process_state)
        self._initial_process_state_template_ = _copy_process_state(process_state)
        self._adaptive_learner_ = learner
        self.history_ = {
            "method": "adaptive",
            "variant": "mw",
            "mw_mode": "sliding",
            "source_type": mne_type,
            "n_channels": self.n_channels_,
            "sfreq": self.sfreq_,
        }
        self.diagnostics_ = {
            "adaptive_variant": "mw",
            "mw_mode": "sliding",
            "covariance_geometry": "adaptive",
            "n_components_reconstructed": np.zeros(len(mw_diagnostics), dtype=int),
            "fraction_reconstructed_samples": 0.0,
            "fraction_reconstructed_windows": 0.0,
            "n_windows": int(len(mw_diagnostics)),
        }

        full = np.asarray(data, dtype=np.float64).copy()
        idx = slice(None) if picks is None else picks
        if mne_type == "raw" and self.reject_by_annotation:
            sub = full[idx].copy()
            sub[:, good_mask] = cleaned
            sub[:, ~good_mask] = data_2d[:, ~good_mask]
            full[idx] = sub
        elif mne_type == "epochs":
            n_epochs = data.shape[0]
            n_times_ep = data.shape[2]
            full[:, idx, :] = continuous_to_epochs(
                cleaned, (n_epochs, cleaned.shape[0], n_times_ep)
            )
        else:
            full[idx, :] = cleaned
        result = reconstruct_mne_object(full, orig_inst, mne_type, verbose=False)
        if return_diagnostics:
            return result, self.diagnostics_
        return result

    def reset_process_state(self) -> None:
        """Reset the streaming reconstruction state to the fitted baseline."""
        self._check_is_fitted()
        self.process_state_ = _copy_process_state(self._initial_process_state_template_)

    def _fit_mw_state(
        self,
        X: np.ndarray,
        sfreq: float,
    ) -> tuple[
        ASRState,
        dict[str, Any],
        _AdaptiveSimilarityMatcher,
        dict[str, Any],
        list[dict[str, Any]],
    ]:
        """MW-ASR: per-window subspace calibration, final-window state.

        Splits the input into non-overlapping windows of length
        ``mw_window_length`` seconds, runs the standard subspace calibration
        on each window, records per-window diagnostics, and returns only the
        final window's state for the subsequent reconstruction pass.

        Windows shorter than ``blocksize`` samples are skipped because they
        are too short for robust calibration.

        Parameters
        ----------
        X : np.ndarray
            The input data array of shape (n_channels, n_times).
        sfreq : float
            The sampling frequency of the data in Hz.

        Returns
        -------
        state : ASRState
            The standard ASR calibration state from the final valid window.
        cal_info : dict
            Calibration diagnostics dictionary from the final valid window,
            updated with MW-specific metadata (number of windows, length).
        learner : _AdaptiveSimilarityMatcher
            The adaptive learner instantiated for the final window.
        process_state : dict
            The process state dictionary for streaming reconstruction.
        diagnostics_list : list of dict
            A list containing calibration diagnostic dictionaries for every
            processed window.
        """
        X = _validate_array_2d(X)
        sfreq = float(sfreq)
        n_times = X.shape[1]
        win_samples = max(1, int(round(self.mw_window_length * sfreq)))

        diagnostics_list: list[dict[str, Any]] = []
        last = None
        n_windows = (n_times + win_samples - 1) // win_samples
        for window_idx in range(n_windows):
            start = window_idx * win_samples
            stop = min(start + win_samples, n_times)
            window = X[:, start:stop]
            if window.shape[1] < self.blocksize:
                continue
            try:
                state, cal_info, learner, process_state = self._fit_adaptive_state(
                    window, sfreq
                )
            except Exception as exc:  # noqa: BLE001
                print("CAUGHT:", repr(exc))
                diagnostics_list.append(
                    {
                        "window_idx": int(window_idx),
                        "window_start": int(start),
                        "window_stop": int(stop),
                        "n_samples": int(window.shape[1]),
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                continue
            diagnostics_list.append(
                {
                    "window_idx": int(window_idx),
                    "window_start": int(start),
                    "window_stop": int(stop),
                    "n_samples": int(window.shape[1]),
                    "status": "passed",
                    "M": np.asarray(state.M, dtype=np.float64),
                    "T": np.asarray(state.T, dtype=np.float64),
                    "thresholds": np.asarray(state.thresholds, dtype=np.float64),
                    "rank": int(state.rank),
                    "clean_window_fraction": cal_info.get(
                        "calibration_clean_window_fraction"
                    ),
                }
            )
            last = (state, cal_info, learner, process_state)

        if last is None:
            raise RuntimeError(
                "MW-ASR fit() found no usable window for calibration "
                f"(n_windows={n_windows}, mw_window_length={self.mw_window_length})"
            )
        state, cal_info, learner, process_state = last
        cal_info = dict(cal_info)
        cal_info["adaptive_variant"] = "mw"
        cal_info["mw_n_windows"] = int(len(diagnostics_list))
        cal_info["mw_window_length_s"] = float(self.mw_window_length)
        return state, cal_info, learner, process_state, diagnostics_list

    def _fit_adaptive_state(
        self,
        X: np.ndarray,
        sfreq: float,
    ) -> tuple[ASRState, dict[str, Any], _AdaptiveSimilarityMatcher, dict[str, Any]]:
        X = _validate_array_2d(X)
        self._check_adaptive_segment_length(X, sfreq, operation="fit")
        X_clean, clean_sample_mask, clean_diag = _extract_clean_calibration_samples(
            X,
            sfreq,
            window_length=self.calibration_window_length,
            window_overlap=self.calibration_window_overlap,
            max_bad_channels=self.ref_max_bad_channels,
            zthresholds=self.ref_tolerances,
            max_dropout_fraction=self.max_dropout_fraction,
            min_clean_fraction=self.min_clean_fraction,
            beta_grid=_AASR_BETA_GRID,
        )
        filter_b, filter_a = _design_asr_filter(sfreq)
        X_filtered, iir_state = _lfilter_channels(X_clean, filter_b, filter_a)
        M, C, eigvals, V, covariance_memory_info = _adaptive_covariance_sqrt(
            X_filtered,
            blocksize=self.blocksize,
            regularization=self.regularization,
            max_mem_mb=self.max_mem_mb,
        )
        thresholds, threshold_info = _fit_adaptive_thresholds(
            X_filtered,
            V,
            sfreq=sfreq,
            window_length=self.window_length,
            window_overlap=self.calibration_window_overlap,
            cutoff=self.cutoff,
            min_clean_fraction=self.min_clean_fraction,
            max_dropout_fraction=self.max_dropout_fraction,
        )
        state = ASRState(
            M=M,
            T=np.diag(thresholds) @ V.T,
            thresholds=thresholds,
            calibration_patterns=V,
            filter_b=filter_b,
            filter_a=filter_a,
            filter_zi=iir_state,
            cov=C,
            rank=int(np.sum(eigvals > self.regularization * np.max(eigvals))),
            method="standard",
            riemannian_solver=None,
        )
        learner = _build_adaptive_learner(
            X_filtered,
            V,
            variant="psp" if self.variant == "mw" else self.variant,
            learning_rate=self.learning_rate,
            tau=self._resolved_tau(),
            regularization=self.regularization,
        )
        process_state = {
            "cov": None,
            "carry": None,
            "iir": iir_state.copy(),
            "last_R": None,
            "last_trivial": True,
        }
        diagnostics = self._adaptive_calibration_info(
            clean_diag,
            clean_sample_mask,
            thresholds,
            threshold_info,
            event="fit",
        )
        diagnostics.update(covariance_memory_info)
        diagnostics["rank"] = int(state.rank)
        return state, diagnostics, learner, process_state

    def _update_adaptive_state(self, X: np.ndarray, sfreq: float) -> dict[str, Any]:
        """Update the adaptive tracking state on a new chunk of data.

        Extracts clean calibration samples from the new chunk, runs them through the
        similarity-matching learner to update the principal components (V), re-estimates
        the covariance metric (M), and updates the statistical thresholds. The resulting
        matrices are saved back into the estimator's ``state_``.

        Parameters
        ----------
        X : np.ndarray
            The incoming data chunk of shape (n_channels, n_times).
        sfreq : float
            The sampling frequency in Hz.

        Returns
        -------
        diagnostics : dict
            A dictionary containing update diagnostics, threshold fit metrics, and
            covariance memory usage info.
        """
        X = _validate_array_2d(X)
        self._check_adaptive_segment_length(X, sfreq, operation="partial_fit")
        X_clean, clean_sample_mask, clean_diag = _extract_clean_calibration_samples(
            X,
            sfreq,
            window_length=self.calibration_window_length,
            window_overlap=self.calibration_window_overlap,
            max_bad_channels=self.ref_max_bad_channels,
            zthresholds=self.ref_tolerances,
            max_dropout_fraction=self.max_dropout_fraction,
            min_clean_fraction=self.min_clean_fraction,
            beta_grid=_AASR_BETA_GRID,
        )
        X_filtered, _ = _lfilter_channels(
            X_clean, self.state_.filter_b, self.state_.filter_a
        )
        # Work on a private learner copy. If covariance or threshold estimation
        # fails, the fitted estimator remains at its last valid state.
        updated_learner = self.adaptive_learner_.copy()
        updated_learner.fit_next(X_filtered)
        V = updated_learner.get_components()
        M, C, eigvals, _, covariance_memory_info = _adaptive_covariance_sqrt(
            X_filtered,
            blocksize=self.blocksize,
            regularization=self.regularization,
            max_mem_mb=self.max_mem_mb,
        )
        thresholds, threshold_info = _fit_adaptive_thresholds(
            X_filtered,
            V,
            sfreq=sfreq,
            window_length=self.update_window_length,
            window_overlap=self.calibration_window_overlap,
            cutoff=self.cutoff,
            min_clean_fraction=self.min_clean_fraction,
            max_dropout_fraction=self.max_dropout_fraction,
        )

        self.state_.M = M
        self.state_.T = np.diag(thresholds) @ V.T
        self.state_.thresholds = thresholds
        self.state_.calibration_patterns = V
        self.state_.cov = C
        self.state_.rank = int(np.sum(eigvals > self.regularization * np.max(eigvals)))
        self.adaptive_learner_ = updated_learner

        diagnostics = self._adaptive_calibration_info(
            clean_diag,
            clean_sample_mask,
            thresholds,
            threshold_info,
            event="update",
        )
        diagnostics.update(covariance_memory_info)
        return diagnostics

    def _check_adaptive_segment_length(
        self, X: np.ndarray, sfreq: float, *, operation: str
    ) -> None:
        """Reject segments that cannot form a clean-selection window."""
        minimum_samples = _round_half_up(self.calibration_window_length * sfreq) + 1
        if X.shape[1] >= minimum_samples:
            return
        minimum_seconds = minimum_samples / float(sfreq)
        raise ValueError(
            f"AdaptiveASR.{operation}() requires at least {minimum_samples} samples "
            f"({minimum_seconds:.6g} s at {sfreq:g} Hz) for clean-window "
            f"estimation; received {X.shape[1]} samples. Accumulate a longer "
            "update segment or omit the incomplete trailing segment."
        )

    def _adaptive_calibration_info(
        self,
        clean_diag: dict[str, Any],
        clean_sample_mask: np.ndarray,
        thresholds: np.ndarray,
        threshold_info: dict[str, np.ndarray],
        event: str,
    ) -> dict[str, Any]:
        """Compile calibration and thresholding diagnostics into a unified dictionary.

        Parameters
        ----------
        clean_diag : dict
            Diagnostics from the window cleaning step.
        clean_sample_mask : np.ndarray
            Boolean array of shape (n_times,) indicating samples kept for calibration.
        thresholds : np.ndarray
            The fitted cutoff thresholds for each principal component.
        threshold_info : dict
            Metadata from the threshold fitting process (e.g., mu, sigma, beta).
        event : str
            The name of the event triggering the calibration (e.g., 'fit', 'update').

        Returns
        -------
        info : dict
            A comprehensive diagnostics dictionary containing all calibration parameters.
        """
        return {
            "event": event,
            "clean_window_mask": np.asarray(
                clean_diag["window_keep_mask"], dtype=bool
            ).copy(),
            "clean_window_scores": np.asarray(
                clean_diag["window_rms_zscores"], dtype=np.float64
            ).copy(),
            "clean_sample_mask": np.asarray(clean_sample_mask, dtype=bool).copy(),
            "calibration_window_starts": np.asarray(
                clean_diag["window_starts"], dtype=int
            ).copy(),
            "calibration_window_length_samples": int(
                clean_diag["window_stops"][0] - clean_diag["window_starts"][0]
            ),
            "blocksize": int(self.blocksize),
            "n_clean_windows": int(np.sum(clean_diag["window_keep_mask"])),
            "n_calibration_windows": int(clean_diag["n_windows"]),
            "calibration_samples": int(np.sum(clean_sample_mask)),
            "rank": int(self.state_.rank) if hasattr(self, "state_") else 0,
            "thresholds": thresholds.copy(),
            "threshold_mu": threshold_info["mu"].copy(),
            "threshold_sigma": threshold_info["sigma"].copy(),
            "threshold_beta": threshold_info["beta"].copy(),
            "threshold_fit_error": threshold_info["fit_error"].copy(),
            "threshold_fit_interval": threshold_info["fit_interval"].copy(),
            "threshold_window_starts": threshold_info["window_starts"].copy(),
            "threshold_window_length_samples": int(
                threshold_info["window_length_samples"]
            ),
            "covariance_geometry": "standard",
            "adaptive_variant": self.variant,
            "statistics_filter": "yulewalk",
        }

    def _resolved_tau(self) -> float:
        if self.tau is not None:
            return float(self.tau)
        return 1e-5 if self.variant == "psw" else 0.8

    def _transform_epochs_adaptive(
        self,
        data: np.ndarray,
        sfreq: float,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        cleaned = np.asarray(data, dtype=np.float64).copy()
        epoch_diags = []
        starts_all: list[np.ndarray] = []
        stops_all: list[np.ndarray] = []
        sample_masks: list[np.ndarray] = []
        rejection_masks: list[np.ndarray] = []
        rejection_starts_all: list[np.ndarray] = []
        rejection_stops_all: list[np.ndarray] = []
        rejection_keep_masks: list[np.ndarray] = []
        rejection_remove_masks: list[np.ndarray] = []
        counts: list[np.ndarray] = []
        for epoch_idx in range(cleaned.shape[0]):
            selected = cleaned[epoch_idx, :, :]
            selected_clean, diag, _ = _process_adaptive_chunk(
                selected,
                sfreq,
                _copy_asr_state(self.state_),
                _copy_process_state(self._initial_process_state_template_),
                window_length=self.window_length,
                lookahead=self.lookahead,
                stepsize=self.stepsize,
                max_dims=self.max_dims,
                store_reconstruction_matrices=self.store_reconstruction_matrices,
                adaptive_variant=self.variant,
                max_mem_mb=self.max_mem_mb,
            )
            cleaned[epoch_idx, :, :] = selected_clean
            if self.window_criterion is not None:
                rejection_mask, rejection_diag = compute_clean_window_mask(
                    selected_clean,
                    sfreq,
                    max_bad_channels=self.window_criterion,
                    zthresholds=self.window_criterion_tolerances,
                    window_length=self.calibration_window_length,
                    window_overlap=self.calibration_window_overlap,
                    max_dropout_fraction=self.max_dropout_fraction,
                    min_clean_fraction=self.min_clean_fraction,
                )
                diag["rejection_sample_mask"] = rejection_mask
                diag["rejection_window_starts"] = rejection_diag["window_starts"]
                diag["rejection_window_stops"] = rejection_diag["window_stops"]
                diag["rejection_window_keep_mask"] = rejection_diag["window_keep_mask"]
                diag["rejection_window_remove_mask"] = rejection_diag[
                    "window_remove_mask"
                ]
                diag["fraction_retained_after_window_rejection"] = float(
                    np.mean(rejection_mask)
                )
                diag["fraction_rejected_after_window_rejection"] = float(
                    1.0 - np.mean(rejection_mask)
                )
            epoch_diags.append(diag)
            starts_all.append(diag["window_starts"])
            stops_all.append(diag["window_stops"])
            sample_masks.append(diag["sample_mask"])
            if "rejection_sample_mask" in diag:
                rejection_masks.append(diag["rejection_sample_mask"])
                rejection_starts_all.append(diag["rejection_window_starts"])
                rejection_stops_all.append(diag["rejection_window_stops"])
                rejection_keep_masks.append(diag["rejection_window_keep_mask"])
                rejection_remove_masks.append(diag["rejection_window_remove_mask"])
            counts.append(diag["n_components_reconstructed"])

        diagnostics: dict[str, Any] = {
            "epoch_diagnostics": epoch_diags,
            "window_starts": np.concatenate(starts_all)
            if starts_all
            else np.array([], dtype=int),
            "window_stops": np.concatenate(stops_all)
            if stops_all
            else np.array([], dtype=int),
            "sample_mask": np.vstack(sample_masks)
            if sample_masks
            else np.empty((0, 0), dtype=bool),
            "n_components_reconstructed": np.concatenate(counts)
            if counts
            else np.array([], dtype=int),
            "n_windows": int(sum(diag["n_windows"] for diag in epoch_diags)),
            "covariance_geometry": "standard",
            "adaptive_variant": self.variant,
        }
        diagnostics["fraction_reconstructed_windows"] = (
            float(np.mean(diagnostics["n_components_reconstructed"] > 0))
            if diagnostics["n_components_reconstructed"].size
            else 0.0
        )
        diagnostics["fraction_reconstructed_samples"] = (
            float(np.mean(diagnostics["sample_mask"]))
            if diagnostics["sample_mask"].size
            else 0.0
        )
        diagnostics["max_components_reconstructed"] = int(
            diagnostics["n_components_reconstructed"].max(initial=0)
        )
        if rejection_masks:
            diagnostics["rejection_sample_mask"] = np.vstack(rejection_masks)
            diagnostics["rejection_window_starts"] = (
                np.concatenate(rejection_starts_all)
                if rejection_starts_all
                else np.array([], dtype=int)
            )
            diagnostics["rejection_window_stops"] = (
                np.concatenate(rejection_stops_all)
                if rejection_stops_all
                else np.array([], dtype=int)
            )
            diagnostics["rejection_window_keep_mask"] = (
                np.concatenate(rejection_keep_masks)
                if rejection_keep_masks
                else np.array([], dtype=bool)
            )
            diagnostics["rejection_window_remove_mask"] = (
                np.concatenate(rejection_remove_masks)
                if rejection_remove_masks
                else np.array([], dtype=bool)
            )
            diagnostics["fraction_retained_after_window_rejection"] = float(
                np.mean(diagnostics["rejection_sample_mask"])
            )
            diagnostics["fraction_rejected_after_window_rejection"] = float(
                1.0 - np.mean(diagnostics["rejection_sample_mask"])
            )
        return cleaned, diagnostics
