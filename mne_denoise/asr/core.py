"""Artifact Subspace Reconstruction."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .. import _mne
from .._data import extract_data_from_mne, reconstruct_mne_object
from .._logging import logger, verbose
from .._validation import check_channel_layout
from ..progress import _emit_progress, _ProgressCallback, _validate_callback
from ._annotations import (
    _calibration_annotations,
    _rejection_annotations,
    _repair_annotations,
)
from ._calibration import calibrate_asr
from ._reconstruction import process_asr
from ._validation import (
    _validate_backend_params,
    _validate_common_params,
)
from ._windowing import _create_good_sample_mask_from_mne, compute_clean_window_mask

if TYPE_CHECKING:
    from mne.epochs import BaseEpochs
    from mne.evoked import Evoked
    from mne.io import BaseRaw


class ASR(BaseEstimator, TransformerMixin):
    """Artifact Subspace Reconstruction estimator.

    ASR calibrates a clean signal subspace and reconstructs high-amplitude windows in
    continuous EEG or MEG data. It accepts channel-first NumPy arrays and supported
    MNE containers.

    Parameters
    ----------
    sfreq : float or None, default=None
        Sampling frequency in Hz. Required for NumPy input; inferred from MNE
        metadata when available.
    cutoff : float, default=20.0
        Threshold multiplier. Lower values generally reconstruct more components;
        the numerical interpretation depends on calibration and processing settings.
    window_length : float, default=0.5
        Processing window length in seconds.
    window_overlap : float, default=0.66
        Processing-window overlap fraction.
    max_dropout_fraction : float, default=0.1
        Fraction of low-RMS values excluded from threshold estimation.
    min_clean_fraction : float, default=0.25
        Minimum central fraction used for clean RMS statistics.
    method : {"standard", "riemannian_windowed", "riemannian"}, default="standard"
        Covariance/reconstruction backend. "riemannian" requires experimental=True.
    experimental : bool, default=False
        Required for the "riemannian" backend.
    calibration : {"auto", "manual"}, default="auto"
        Whether to select clean calibration windows or use all supplied samples.
    picks : str, list of str, list of int, or None, default="eeg"
        MNE channels to process. NumPy input uses all rows.
    calibration_window_length : float, default=1.0
        Window length in seconds for automatic calibration selection.
    calibration_window_overlap : float, default=0.66
        Overlap fraction for automatic calibration selection.
    ref_max_bad_channels : float, default=0.075
        Maximum bad-channel fraction in a calibration window.
    ref_tolerances : tuple of float, default=(-np.inf, 5.5)
        Robust z-score bounds for calibration-window selection.
    blocksize : int, default=10
        Samples aggregated per calibration covariance block.
    max_dims : float or int, default=0.66
        Maximum fraction or number of dimensions reconstructed per window.
    reject_by_annotation : bool, default=True
        Exclude bad annotated samples during Raw calibration and preserve them during
        Raw transformation.
    skip_by_annotation : tuple of str, default=("bad", "bad_acq_skip")
        Annotation prefixes treated as bad.
    cov_estimator : {"geometric_median", "mean", "median"}, default="geometric_median"
        Calibration-covariance aggregation rule.
    regularization : float, default=1e-8
        Relative covariance eigenvalue floor.
    filter_kind : {"none", "asr", "highpass"}, default="asr"
        Filter used for statistics; reconstructed output uses the original data.
    window_criterion : float, int, or None, default=None
        Optional final retained-sample criterion.
    window_criterion_tolerances : tuple of float, default=(-np.inf, 7.0)
        Robust z-score bounds for the final criterion.
    lookahead : float or None, default=None
        Processing lookahead in seconds; None uses half a window.
    stepsize : int or None, default=None
        Samples between reconstruction updates; None uses half a window.
    max_mem_mb : int or None, default=512
        Memory cap for covariance processing.
    copy : bool, default=True
        Reserved compatibility parameter; transformations return new outputs.
    store_reconstruction_matrices : bool, default=False
        Store per-window reconstruction matrices in diagnostics.
    random_state : int or None, default=None
        Reserved for future stochastic calibration.
    n_jobs : int or None, default=None
        Reserved for future parallel processing.
    verbose : bool, str, int, or None, default=None
        Logging level.

    See Also
    --------
    AdaptiveASR
        Adaptive calibration variants for changing recording statistics.
    JugglerASR
        Alternative calibration-sample selection for high-motion recordings.
    GuidedASR
        Experimental guidance-aware reconstruction.

    Notes
    -----
    NumPy input uses (n_channels, n_times). MNE Raw and Epochs are supported;
    fit does not accept Evoked, while transform preserves the input container and
    metadata. Transformations do not mutate their input. Real applications should
    calibrate on representative clean data; the synthetic example only illustrates
    the estimator lifecycle.

    References
    ----------
    :footcite:p:`kothe_jung2016_asr,chang2018_asr,chang2020_asr`

    .. footbibliography::

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.asr import ASR
    >>> rng = np.random.default_rng(0)
    >>> data = rng.standard_normal((8, 2000))
    >>> asr = ASR(sfreq=250.0, cutoff=20.0)
    >>> clean = asr.fit_transform(data)
    """

    _progress_method = "asr"

    def __init__(
        self,
        sfreq: float | None = None,
        cutoff: float = 20.0,
        window_length: float = 0.5,
        window_overlap: float = 0.66,
        max_dropout_fraction: float = 0.1,
        min_clean_fraction: float = 0.25,
        method: str = "standard",
        experimental: bool = False,
        calibration: str = "auto",
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
        window_criterion: float | int | None = None,
        window_criterion_tolerances: tuple[float, float] = (-np.inf, 7.0),
        lookahead: float | None = None,
        stepsize: int | None = None,
        max_mem_mb: int | None = 512,
        copy: bool = True,
        store_reconstruction_matrices: bool = False,
        random_state: int | None = None,
        n_jobs: int | None = None,
        verbose: bool | str | int | None = None,
    ) -> None:
        self.sfreq = sfreq
        self.cutoff = cutoff
        self.window_length = window_length
        self.window_overlap = window_overlap
        self.max_dropout_fraction = max_dropout_fraction
        self.min_clean_fraction = min_clean_fraction
        self.method = method
        self.experimental = experimental
        self.calibration = calibration
        self.picks = picks
        self.calibration_window_length = calibration_window_length
        self.calibration_window_overlap = calibration_window_overlap
        self.ref_max_bad_channels = ref_max_bad_channels
        self.ref_tolerances = ref_tolerances
        self.blocksize = blocksize
        self.max_dims = max_dims
        self.reject_by_annotation = reject_by_annotation
        self.skip_by_annotation = skip_by_annotation
        self.cov_estimator = cov_estimator
        self.regularization = regularization
        self.filter_kind = filter_kind
        self.window_criterion = window_criterion
        self.window_criterion_tolerances = window_criterion_tolerances
        self.lookahead = lookahead
        self.stepsize = stepsize
        self.max_mem_mb = max_mem_mb
        self.copy = copy
        self.store_reconstruction_matrices = store_reconstruction_matrices
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

    @verbose
    def fit(
        self,
        X: BaseRaw | BaseEpochs | np.ndarray,
        y=None,
        *,
        calibration: BaseRaw | BaseEpochs | np.ndarray | None = None,
        calibration_mask: np.ndarray | None = None,
        callback=None,
        verbose: bool | str | int | None = None,
    ) -> ASR:
        """Fit the ASR calibration state.

        Parameters
        ----------
        X : Raw, Epochs, or ndarray
            Data used for calibration when calibration is None. NumPy input is
            (n_channels, n_times).
        y : None, default=None
            Ignored for scikit-learn compatibility.
        calibration : Raw, Epochs, or ndarray, default=None
            Optional separate calibration data with matching channels.
        calibration_mask : ndarray of bool, shape (n_times,), or None, default=None
            Samples to use from a 2D calibration input.
        callback : callable or None, default=None
            Synchronous calibration progress callback.
        verbose : bool, str, int, or None, default=None
            Logging level for this call.

        Returns
        -------
        ASR
            The fitted estimator.
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
        fit_input = X if calibration is None else calibration
        data, sfreq, mne_type, orig_inst, picks, ch_names = extract_data_from_mne(
            fit_input,
            auto_pick=True,
            concatenate_epochs=True,
        )
        if mne_type == "evoked":
            raise ValueError("ASR.fit() does not support Evoked calibration data")
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
        state, cal_info = calibrate_asr(
            data_2d,
            sfreq,
            cutoff=self.cutoff,
            window_length=self.window_length,
            window_overlap=self.window_overlap,
            calibration=self.calibration,
            calibration_window_length=self.calibration_window_length,
            calibration_window_overlap=self.calibration_window_overlap,
            ref_max_bad_channels=self.ref_max_bad_channels,
            ref_tolerances=self.ref_tolerances,
            blocksize=self.blocksize,
            max_dropout_fraction=self.max_dropout_fraction,
            min_clean_fraction=self.min_clean_fraction,
            cov_estimator=self.cov_estimator,
            regularization=self.regularization,
            filter_kind=self.filter_kind,
            method=self.method,
            max_mem_mb=self.max_mem_mb,
            callback=callback,
        )

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
        self.clean_window_scores_ = cal_info["clean_window_scores"]
        self.calibration_mask_kind_ = "window"
        self.calibration_info_ = cal_info
        logger.info(
            "%s calibrated: method=%s, channels=%d, sfreq=%.3g Hz, "
            "cutoff=%.3g, rank=%d, clean calibration windows=%d/%d.",
            type(self).__name__,
            self.method,
            self.n_channels_,
            self.sfreq_,
            self.cutoff,
            self.rank_,
            cal_info.get("n_clean_windows", 0),
            cal_info.get("n_calibration_windows", 0),
        )
        self.history_ = {
            "method": self.method,
            "calibration": self.calibration,
            "source_type": mne_type,
            "n_channels": self.n_channels_,
            "sfreq": self.sfreq_,
        }
        return self

    def _process(
        self,
        data: np.ndarray,
        sfreq: float,
        *,
        callback: _ProgressCallback | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Process one continuous channel-by-time array.

        Subclasses may override this hook while retaining the public container workflow.
        """
        return process_asr(
            data,
            sfreq,
            self.state_,
            window_length=self.window_length,
            window_overlap=self.window_overlap,
            max_dims=self.max_dims,
            regularization=self.regularization,
            store_reconstruction_matrices=self.store_reconstruction_matrices,
            max_mem_mb=self.max_mem_mb,
            lookahead=self.lookahead,
            stepsize=self.stepsize,
            method=self.method,
            callback=callback,
        )

    @verbose
    def transform(
        self,
        X: BaseRaw | BaseEpochs | Evoked | np.ndarray,
        y=None,
        copy: bool | None = None,
        return_diagnostics: bool = False,
        *,
        callback=None,
        verbose: bool | str | int | None = None,
    ) -> Any:
        """Apply the fitted ASR model.

        Parameters
        ----------
        X : Raw, Epochs, Evoked, or ndarray
            Data to clean.
        y : None, default=None
            Ignored for scikit-learn compatibility.
        copy : bool or None, default=None
            Reserved compatibility parameter.
        return_diagnostics : bool, default=False
            If true, return (cleaned, diagnostics).
        callback : callable or None, default=None
            Synchronous reconstruction progress callback.
        verbose : bool, str, int, or None, default=None
            Logging level for this call.

        Returns
        -------
        cleaned : Raw, Epochs, Evoked, or ndarray
            Cleaned data with the input type and layout.
        diagnostics : dict
            Returned only when return_diagnostics=True.
        """
        del y, copy
        callback = _validate_callback(callback)
        self._check_is_fitted()
        data, sfreq, mne_type, orig_inst, picks, ch_names = extract_data_from_mne(
            X, auto_pick=True
        )
        sfreq = self._resolve_sfreq(sfreq, fitted=True)
        if not np.isclose(sfreq, self.sfreq_):
            raise ValueError(
                f"Input sfreq {sfreq} does not match fitted sfreq {self.sfreq_}"
            )
        if self.ch_names_ is not None and ch_names is None:
            raise ValueError(
                "ASR was fitted with named channels; transform input must provide "
                "channel names so their order can be verified."
            )
        check_channel_layout(
            type(self).__name__,
            n_channels=data.shape[1] if mne_type == "epochs" else data.shape[0],
            fitted_n_channels=self.n_channels_,
            ch_names=ch_names,
            fitted_ch_names=self.ch_names_,
        )
        self._warn_preprocessing_state(orig_inst, mne_type)

        if mne_type == "epochs":
            cleaned_data, diagnostics = self._transform_epochs(
                data,
                sfreq,
                callback=callback,
            )
        else:
            selected = np.asarray(data, dtype=np.float64)
            selected_clean, diagnostics = self._process(
                selected,
                sfreq,
                callback=callback,
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

        self._store_transform_diagnostics(diagnostics)
        logger.info(
            "%s: processed method=%s, %d window(s), %.1f%% of samples "
            "reconstructed (max %d component(s)).",
            type(self).__name__,
            diagnostics.get("covariance_geometry", self.method),
            diagnostics["n_windows"],
            100.0 * diagnostics["fraction_reconstructed_samples"],
            diagnostics["max_components_reconstructed"],
        )
        cleaned = reconstruct_mne_object(cleaned_data, orig_inst, mne_type, picks=picks)
        if return_diagnostics:
            return cleaned, diagnostics
        return cleaned

    @verbose
    def fit_transform(
        self,
        X: BaseRaw | BaseEpochs | np.ndarray,
        y=None,
        calibration: BaseRaw | BaseEpochs | np.ndarray | None = None,
        return_diagnostics: bool = False,
        *,
        callback=None,
        verbose: bool | str | int | None = None,
    ) -> Any:
        """Fit ASR and transform the input.

        Parameters
        ----------
        X : Raw, Epochs, or ndarray
            Data to clean and, when calibration is None, to calibrate on.
        y : None, default=None
            Ignored for scikit-learn compatibility.
        calibration : Raw, Epochs, or ndarray, default=None
            Optional separate calibration data.
        return_diagnostics : bool, default=False
            If true, return (cleaned, diagnostics).
        callback : callable or None, default=None
            Callback passed to calibration and reconstruction.
        verbose : bool, str, int, or None, default=None
            Logging level for this call.

        Returns
        -------
        cleaned : Raw, Epochs, or ndarray
            Cleaned data with the input type and layout.
        diagnostics : dict
            Returned only when return_diagnostics=True.
        """
        callback = _validate_callback(callback)
        self.fit(X, y=y, calibration=calibration, callback=callback)
        return self.transform(
            X,
            return_diagnostics=return_diagnostics,
            callback=callback,
        )

    def get_diagnostics(self) -> dict[str, Any]:
        """Return diagnostics from the most recent transformation.

        Returns
        -------
        dict
            A copy of the latest diagnostics, or an empty dictionary before transform.
        """
        self._check_is_fitted()
        if not hasattr(self, "diagnostics_"):
            return {}
        return dict(self.diagnostics_)

    def get_calibration_mask(self) -> np.ndarray:
        """Return the boolean mask used for calibration.

        Returns
        -------
        ndarray of bool
            A copy of the clean-window or reference-sample mask.
        """
        self._check_is_fitted()
        return np.asarray(self.clean_window_mask_, dtype=bool).copy()

    def get_rejection_mask(self) -> np.ndarray:
        """Return the retained-sample mask from final window rejection.

        Returns
        -------
        ndarray of bool, shape (n_times,)
            True for samples retained by the optional window_criterion pass.
        """
        self._check_is_fitted()
        if not hasattr(self, "rejection_sample_mask_"):
            raise RuntimeError(
                "No final rejection mask is available. Enable window_criterion and "
                "run transform first."
            )
        return np.asarray(self.rejection_sample_mask_, dtype=bool).copy()

    def to_annotations(
        self,
        kind: str = "repair",
        min_components: int = 1,
        description: str | None = None,
    ) -> Any:
        """Convert ASR decisions to MNE annotations.

        Parameters
        ----------
        kind : {"repair", "rejection", "calibration"}, default="repair"
            Decision to annotate. "calibration" is available for JugglerASR
            reference-sample selection.
        min_components : int, default=1
            Minimum reconstructed-component count for kind="repair".
        description : str or None, default=None
            Annotation label; a kind-specific label is used when omitted.

        Returns
        -------
        mne.Annotations
            Annotation spans for the requested decision.
        """
        self._check_is_fitted()
        _mne.require_mne("ASR annotations")
        if kind == "repair":
            return _repair_annotations(
                diagnostics=self.diagnostics_,
                sfreq=self.sfreq_,
                min_components=min_components,
                description=description or "ASR_REPAIR",
            )
        if kind == "rejection":
            return _rejection_annotations(
                rejection_sample_mask=self.rejection_sample_mask_,
                sfreq=self.sfreq_,
                description=description or "ASR_REJECT",
            )
        if kind == "calibration":
            return _calibration_annotations(
                calibration_mask_kind=getattr(self, "calibration_mask_kind_", "window"),
                reference_sample_mask=getattr(self, "reference_sample_mask_", None),
                sfreq=self.sfreq_,
                description=description or "ASR_REFERENCE",
            )
        raise ValueError(
            f"kind must be 'repair', 'rejection', or 'calibration', got {kind!r}"
        )

    def _resolve_sfreq(self, sfreq: float | None, fitted: bool = False) -> float:
        if sfreq is None:
            sfreq = self.sfreq_ if fitted and hasattr(self, "sfreq_") else self.sfreq
        if sfreq is None:
            raise ValueError("sfreq must be provided for NumPy array inputs")
        if sfreq <= 0:
            raise ValueError("sfreq must be positive")
        return float(sfreq)

    def _transform_epochs(
        self,
        data: np.ndarray,
        sfreq: float,
        *,
        callback: _ProgressCallback | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reconstruct each epoch independently and aggregate diagnostics."""
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
        n_epochs = cleaned.shape[0]
        for epoch_idx in range(n_epochs):
            selected = cleaned[epoch_idx, :, :]
            selected_clean, diag = self._process(selected, sfreq, callback=None)
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
            _emit_progress(
                callback,
                method=self._progress_method,
                stage="epoch",
                current=epoch_idx + 1,
                total=n_epochs,
                component=None,
                metric=float(diag["fraction_reconstructed_samples"]),
            )
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
        soft_weights = [
            diag["soft_weights"]
            for diag in epoch_diags
            if np.asarray(diag.get("soft_weights", np.empty((0,)))).size
        ]
        if soft_weights:
            diagnostics["soft_weights"] = np.concatenate(soft_weights, axis=0)
            diagnostics["mean_soft_weight"] = float(
                np.mean(diagnostics["soft_weights"])
            )
        for key in ("covariance_geometry", "reconstruction"):
            if epoch_diags and key in epoch_diags[0]:
                diagnostics[key] = epoch_diags[0][key]
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

    def _store_transform_diagnostics(self, diagnostics: dict[str, Any]) -> None:
        """Store diagnostics from the latest transform."""
        self.diagnostics_ = diagnostics
        self.sample_mask_ = diagnostics["sample_mask"]
        self.window_starts_ = diagnostics["window_starts"]
        self.window_stops_ = diagnostics["window_stops"]
        self.n_components_reconstructed_ = diagnostics["n_components_reconstructed"]
        self.n_windows_ = diagnostics["n_windows"]
        self.fraction_reconstructed_windows_ = diagnostics[
            "fraction_reconstructed_windows"
        ]
        self.fraction_reconstructed_samples_ = diagnostics[
            "fraction_reconstructed_samples"
        ]
        self.max_components_reconstructed_ = diagnostics["max_components_reconstructed"]
        if "rejection_sample_mask" in diagnostics:
            self.rejection_sample_mask_ = diagnostics["rejection_sample_mask"]
            self.rejection_window_starts_ = diagnostics["rejection_window_starts"]
            self.rejection_window_stops_ = diagnostics["rejection_window_stops"]
            self.rejection_window_keep_mask_ = diagnostics["rejection_window_keep_mask"]
            self.rejection_window_remove_mask_ = diagnostics[
                "rejection_window_remove_mask"
            ]
            self.fraction_retained_after_window_rejection_ = diagnostics[
                "fraction_retained_after_window_rejection"
            ]
            self.fraction_rejected_after_window_rejection_ = diagnostics[
                "fraction_rejected_after_window_rejection"
            ]
        elif hasattr(self, "rejection_sample_mask_"):
            del self.rejection_sample_mask_
            del self.rejection_window_starts_
            del self.rejection_window_stops_
            del self.rejection_window_keep_mask_
            del self.rejection_window_remove_mask_
            del self.fraction_retained_after_window_rejection_
            del self.fraction_rejected_after_window_rejection_

    def _warn_preprocessing_state(self, inst: Any, mne_type: str) -> None:
        """Warn when MNE preprocessing metadata may affect ASR."""
        if mne_type == "array" or inst is None:
            return
        highpass = inst.info.get("highpass", None)
        if highpass is not None and highpass < 0.25:
            warnings.warn(
                "ASR assumes high-pass filtered data; input info reports "
                f"highpass={highpass} Hz.",
                UserWarning,
                stacklevel=3,
            )
        if len(inst.info.get("projs", [])) > 0:
            warnings.warn(
                "ASR is sensitive to data rank; active or unapplied projectors "
                "may affect covariance estimates.",
                UserWarning,
                stacklevel=3,
            )

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "state_"):
            raise RuntimeError("ASR is not fitted. Call fit() first.")
