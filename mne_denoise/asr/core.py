"""Artifact Subspace Reconstruction (ASR) core module.

This module implements the primary ``ASR`` class, which serves as a scikit-learn
and MNE-compatible estimator for removing high-variance artifacts from continuous
and epoched neurophysiological data (EEG/MEG).

ASR operates in two stages:
1. **Calibration**: Identifies segments of clean data and computes a robust
   covariance matrix. This establishes a baseline for clean signal characteristics.
2. **Reconstruction**: Uses a sliding window over the target data to detect
   segments whose variance exceeds the clean baseline (defined by a cutoff threshold).
   These segments are linearly reconstructed using a mixing matrix derived from
   the clean covariance.

This class acts as the central interface, delegating the mathematical operations
like calibration, spatial filtering, and reconstruction to focused internal submodules.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .._logging import logger, set_log_level_from_verbose
from ..utils import extract_data_from_mne, reconstruct_mne_object
from ._annotations import (
    _calibration_annotations,
    _rejection_annotations,
    _repair_annotations,
)
from ._calibration import calibrate_asr
from ._reconstruction import process_asr
from ._validation import (
    _check_transform_channels,
    _validate_backend_params,
    _validate_common_params,
)
from ._windowing import _create_good_sample_mask_from_mne, compute_clean_window_mask

try:
    import mne
    from mne.epochs import BaseEpochs
    from mne.evoked import Evoked
    from mne.io import BaseRaw
except ImportError:  # pragma: no cover
    mne = None  # pragma: no cover


class ASR(BaseEstimator, TransformerMixin):
    """Artifact Subspace Reconstruction (ASR) scikit-learn transformer.

    ASR is an automated, statistical method for removing high-amplitude, transient
    artifacts (such as eye blinks, muscle bursts, and sensor motion) from continuous
    electroencephalography (EEG) or magnetoencephalography (MEG) data.

    It operates by learning a clean signal subspace from a calibration dataset (or
    clean segments of the target dataset) and using this baseline to identify and
    reconstruct corrupted segments. This class provides a fully scikit-learn and
    MNE-compatible interface.

    Parameters
    ----------
    sfreq : float | None, default=None
        Sampling frequency in Hz. Required for NumPy arrays. For MNE objects,
        this may be ``None`` and is inferred from ``info['sfreq']``.
    cutoff : float, default=20.0
        ASR threshold multiplier. Values around 20 are conservative; lower
        values clean more aggressively.
    window_length : float, default=0.5
        Processing/statistics window length in seconds.
    window_overlap : float, default=0.66
        Overlap fraction for processing and threshold-fitting windows.
    max_dropout_fraction : float, default=0.1
        Fraction of lowest RMS values ignored while estimating thresholds.
    min_clean_fraction : float, default=0.25
        Minimum central fraction used to estimate clean RMS statistics.
    method : {'standard', 'riemannian', 'riemannian_windowed'}, default='standard'
        ASR backend.

        - ``'standard'`` — standard Euclidean ASR.
        - ``'riemannian'`` — experimental SPD-manifold covariance backend,
          NOTE: this backend computes one covariance +
          one reconstruction matrix for the entire stream, so its cleaned
          output is **cutoff-invariant on real EEG** (the ``cutoff`` knob does
          not meaningfully change the result). Use it primarily for research or benchmarking, not
          for cutoff tuning.
        - ``'riemannian_windowed'`` — per-window Riemannian backend that keeps
          the Riemannian-aggregated (geometric-median) calibration but applies
          a standard per-window eigendecomposition at processing time. Unlike
          ``'riemannian'``, its ``cutoff`` knob works: ``% data modified`` and
          ``% variance reduced`` scale monotonically with ``cutoff`` like
          ``'standard'`` does. This is a **first-class backend** (no
          ``experimental`` flag required): its processing is numerically identical to
          standard ASR while preserving robust manifold calibration. Prefer it over ``'riemannian'`` whenever you need
          cutoff control with Riemannian-robust calibration.
    experimental : bool, default=False
        Explicit opt-in for the unstable ``method='riemannian'`` research
        backend (cutoff-invariant on real EEG). Not required for
        ``'riemannian_windowed'``.
    calibration : {'auto', 'manual'}, default='auto'
        Calibration mode. ``'auto'`` selects clean windows before fitting;
        ``'manual'`` uses all supplied calibration samples.
    calibration_window_length : float, default=0.5
        Window length in seconds for automatic clean-window selection.
    calibration_window_overlap : float, default=0.66
        Overlap fraction for automatic clean-window selection.
    ref_max_bad_channels : float, default=0.2
        Maximum fraction of channels exceeding robust tolerances in a clean
        calibration window.
    ref_tolerances : tuple of float, default=(-3.5, 5.0)
        Lower and upper robust z-score bounds for clean-window selection.
    blocksize : int | 'clean_rawdata', default=10
        Number of successive samples averaged into each covariance block for
        robust calibration covariance estimation. ``'clean_rawdata'``
        reproduces the MATLAB reference implementation's memory-dependent
        block-size rule.
    max_dims : float | int, default=0.66
        Maximum number of dimensions reconstructed per processing window.
    reject_by_annotation : bool, default=True
        If True, samples under bad annotations are excluded during Raw
        calibration and preserved during Raw transform.
    skip_by_annotation : tuple of str, default=('bad', 'bad_acq_skip')
        Annotation description prefixes treated as bad when
        ``reject_by_annotation=True``.
    cov_estimator : {'geometric_median', 'mean', 'median'}, default='geometric_median'
        Aggregation rule for calibration-window covariance matrices.
    regularization : float, default=1e-8
        Relative eigenvalue floor for covariance regularization.
    filter_kind : {'none', 'asr', 'highpass'}, default='asr'
        Statistics-only filter. The cleaned output is reconstructed from the
        original unfiltered data.
    window_criterion : float | int | None, default=None
        Optional clean_windows-style final rejection criterion. If numeric,
        this is the maximum tolerated number or fraction of bad channels per
        retained window after ASR correction. ``None`` disables
        final rejection-mask computation.
    window_criterion_tolerances : tuple of float, default=(-3.5, 5.0)
        Lower and upper robust z-score thresholds for final clean_windows-style
        retained-sample masking.
    lookahead : float | None, default=None
        Processing lookahead in seconds. Defaults to ``window_length / 2``.
    stepsize : int | None, default=None
        Number of samples between reconstruction-matrix updates. If ``None``,
        use the default ``floor(sfreq * window_length / 2)``.
    max_mem_mb : int | None, default=200
        Reserved memory limit for future chunking.
    copy : bool, default=True
        Reserved API flag. Transform returns a new object/array.
    store_reconstruction_matrices : bool, default=False
        Store per-window reconstruction matrices in diagnostics.
    random_state : int | None, default=None
        Reserved for future stochastic calibration strategies.
    n_jobs : int | None, default=None
        Reserved for future parallel processing.
    verbose : bool | str | int | None, default=None
        Controls progress logging on the ``mne_denoise.asr`` logger. ``True``
        enables INFO messages (e.g. the calibration summary), ``False``
        restricts to warnings, a level name/int sets that level, and ``None``
        leaves the current logging configuration unchanged.


    Notes
    -----
    **Key Tuning Parameters**

    * **cutoff**: The primary dial for ASR aggressiveness. A value of 20 is standard and
      conservative (modifies mostly extreme artifacts), while 10 is aggressive (modifies
      more moderate activity).
    * **method**: Use ``'standard'`` for standard ASR workflows, or
      ``'riemannian_windowed'`` for a more mathematically robust manifold-based
      calibration covariance estimation that still responds monotonically to ``cutoff``.
    * **window_criterion**: Provide a numeric value (e.g., ``0.25``) to enable an
      statistical final rejection pass after ASR reconstruction, which
      drops any remaining windows that still contain too many artifactual channels.

    **Calibration vs. Reconstruction Windows**

    ASR uses two different sliding windows. ``calibration_window_length`` (default 1.0s) is
    used exclusively during ``fit()`` to find clean baseline segments. ``window_length``
    (default 0.5s) is used during ``transform()`` to detect and reconstruct artifacts.

    Attributes
    ----------
    sfreq_ : float
        Sampling frequency used during fitting.
    ch_names_ : list of str | None
        Fitted channel names for MNE inputs.
    picks_ : ndarray
        Row/channel indices cleaned in the fitted data.
    M_ : ndarray
        Calibration covariance square root.
    T_ : ndarray
        Direction-dependent threshold matrix.
    thresholds_ : ndarray
        Per-component RMS thresholds.
    clean_window_mask_ : ndarray
        Calibration windows retained as clean.
    sample_mask_ : ndarray
        Samples reconstructed during the last transform.
    rejection_sample_mask_ : ndarray
        Boolean retained-sample mask from optional clean_windows-style final
        rejection. Present after transforms when ``window_criterion`` is
        enabled.
    n_components_reconstructed_ : ndarray
        Number of reconstructed components per processing window.
    diagnostics_ : dict
        Last-transform diagnostics.
    calibration_info_ : dict
        Calibration diagnostics.

    Examples
    --------
    Clean an MNE Raw object using standard ASR:

    >>> import mne
    >>> from mne_denoise.asr import ASR
    >>> raw = mne.io.read_raw_fif("sample_audvis_raw.fif", preload=True)
    >>> asr = ASR(cutoff=20.0)
    >>> # Calibration and reconstruction happen in one pass with fit_transform
    >>> clean_raw = asr.fit_transform(raw)

    Clean a NumPy array, passing the sampling frequency explicitly:

    >>> import numpy as np
    >>> data = np.random.randn(32, 5000)
    >>> asr = ASR(sfreq=250.0, cutoff=15.0)
    >>> clean_data = asr.fit_transform(data)

    Perform independent calibration on a known clean baseline:

    >>> asr = ASR(cutoff=20.0)
    >>> asr.fit(clean_baseline_raw)
    >>> clean_target_raw = asr.transform(target_raw)
    """

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
        blocksize: int | Literal["clean_rawdata"] = 10,
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

    def fit(
        self,
        X: BaseRaw | BaseEpochs | np.ndarray,
        y=None,
        *,
        calibration: BaseRaw | BaseEpochs | np.ndarray | None = None,
        calibration_mask: np.ndarray | None = None,
    ) -> ASR:
        """Fit ASR calibration state.

        Parameters
        ----------
        X : Raw | Epochs | ndarray
            Data used for calibration when ``calibration`` is ``None``.
            NumPy arrays must have shape ``(n_channels, n_times)``.
        y : None
            Ignored.
        calibration : Raw | Epochs | ndarray | None, default=None
            Optional separate calibration data with matching channels.
        calibration_mask : ndarray | None, default=None
            Optional boolean sample mask for 2D calibration arrays or Raw
            inputs after annotation exclusion.

        Returns
        -------
        self : ASR
            Fitted estimator.

        Examples
        --------
        Fit ASR directly on the data you intend to clean:

        >>> import mne
        >>> from mne_denoise.asr import ASR
        >>> raw = mne.io.read_raw_fif("sample_audvis_raw.fif", preload=True)
        >>> asr = ASR(cutoff=20.0).fit(raw)

        Fit ASR on a dedicated clean resting-state recording, but pass the
        target data as ``X`` for scikit-learn pipeline compatibility:

        >>> asr = ASR(cutoff=20.0)
        >>> asr.fit(X=target_raw, calibration=resting_state_raw)
        >>> clean_target = asr.transform(target_raw)
        """
        del y
        set_log_level_from_verbose(self.verbose)
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
            "ASR calibrated: method=%s, %d channels, rank %d.",
            self.method,
            self.n_channels_,
            self.rank_,
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
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Process one continuous channel-by-time array.

        Subclasses can override this internal hook while retaining the current
        Raw, Epochs, Evoked, annotation, rejection-mask, and diagnostics
        workflows implemented by :meth:`transform`.
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
        )

    def transform(
        self,
        X: BaseRaw | BaseEpochs | Evoked | np.ndarray,
        y=None,
        copy: bool | None = None,
        return_diagnostics: bool = False,
    ) -> Any:
        """Apply the fitted ASR model.

        Parameters
        ----------
        X : Raw | Epochs | Evoked | ndarray
            Data to clean.
        y : None
            Ignored.
        copy : bool | None, default=None
            Reserved API flag. Transform returns a new object/array.
        return_diagnostics : bool, default=False
            If True, return ``(cleaned, diagnostics)``.

        Returns
        -------
        cleaned : Raw | Epochs | Evoked | ndarray
            Cleaned data with the same type/shape as ``X``.
        diagnostics : dict
            Returned only when ``return_diagnostics=True``.

        Examples
        --------
        Transform raw data using a previously fitted ASR model:

        >>> asr = ASR(cutoff=20.0).fit(resting_state_raw)
        >>> clean_target = asr.transform(target_raw)

        Retrieve detailed diagnostics alongside the cleaned data:

        >>> clean_data, diagnostics = asr.transform(data, return_diagnostics=True)
        """
        del y, copy
        set_log_level_from_verbose(self.verbose)
        self._check_is_fitted()
        data, sfreq, mne_type, orig_inst, picks, ch_names = extract_data_from_mne(
            X, auto_pick=True
        )
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
            cleaned_data, diagnostics = self._transform_epochs(data, sfreq)
        else:
            selected = np.asarray(data, dtype=np.float64)
            selected_clean, diagnostics = self._process(selected, sfreq)
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
        """Fit ASR and apply it to ``X``.

        Parameters
        ----------
        X : Raw | Epochs | ndarray
            Data to clean. Also used for calibration when ``calibration`` is ``None``.
        y : None
            Ignored.
        calibration : Raw | Epochs | ndarray | None, default=None
            Optional separate calibration data with matching channels.
        return_diagnostics : bool, default=False
            If True, return ``(cleaned, diagnostics)``.

        Returns
        -------
        cleaned : Raw | Epochs | ndarray
            Cleaned data with the same type/shape as ``X``.
        diagnostics : dict
            Returned only when ``return_diagnostics=True``.

        Examples
        --------
        Fit and transform in a single pass (standard workflow):

        >>> import mne
        >>> from mne_denoise.asr import ASR
        >>> raw = mne.io.read_raw_fif("sample_audvis_raw.fif", preload=True)
        >>> asr = ASR(cutoff=20.0)
        >>> clean_raw = asr.fit_transform(raw)

        Fit and transform, extracting the diagnostic dictionary:

        >>> clean_data, diagnostics = asr.fit_transform(data, return_diagnostics=True)
        """
        self.fit(X, y=y, calibration=calibration)
        return self.transform(X, return_diagnostics=return_diagnostics)

    def get_diagnostics(self) -> dict[str, Any]:
        """Return diagnostics from the last transform.

        Returns
        -------
        diagnostics : dict
            A dictionary containing detailed statistics and metrics from the
            most recent `transform` call. If no transform has occurred, returns
            an empty dictionary.

        Examples
        --------
        >>> asr = ASR(cutoff=20.0).fit_transform(raw)
        >>> diag = asr.get_diagnostics()
        >>> print(diag.keys())
        """
        self._check_is_fitted()
        if not hasattr(self, "diagnostics_"):
            return {}
        return dict(self.diagnostics_)

    def get_calibration_mask(self) -> np.ndarray:
        """Return the boolean mask of data used for calibration.

        The mask is **window-based** for the standard / Riemannian / adaptive
        backends (one bool per calibration window; see
        :attr:`calibration_mask_kind_` ``== "window"``) and **sample-based**
        for :class:`JugglerASR` (one bool per time sample;
        :attr:`calibration_mask_kind_` ``== "sample"``).

        Returns
        -------
        mask : ndarray of bool
            The calibration clean-window or reference-sample mask.

        See Also
        --------
        get_rejection_mask : retained-sample mask after optional window rejection.

        Examples
        --------
        Retrieve and inspect the shape of the calibration mask:

        >>> asr = ASR().fit(raw)
        >>> mask = asr.get_calibration_mask()
        >>> print(mask.sum())  # Number of clean windows/samples kept
        """
        self._check_is_fitted()
        return np.asarray(self.clean_window_mask_, dtype=bool).copy()

    def get_rejection_mask(self) -> np.ndarray:
        """Return the retained-sample mask from final clean_windows-style rejection.

        Returns
        -------
        mask : ndarray of bool, shape (n_times,)
            ``True`` where samples were kept. Requires ``window_criterion`` to
            have been enabled and ``transform`` to have been run.

        Examples
        --------
        Drop badly repaired samples from the output using the mask:

        >>> asr = ASR(window_criterion=0.25).fit(raw)
        >>> clean_data = asr.transform(raw)
        >>> mask = asr.get_rejection_mask()
        >>> fully_clean_data = clean_data[:, mask]
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
        """Convert ASR decisions into MNE annotations.

        One unified entry point for the three annotation kinds. ``"repair"`` and
        ``"rejection"`` are available on every backend that has run
        ``transform``; ``"calibration"`` is available only for sample-based
        reference selection (:class:`JugglerASR`).

        Parameters
        ----------
        kind : {'repair', 'rejection', 'calibration'}, default='repair'
            Which decision to annotate:

            - ``'repair'`` — windows where at least ``min_components`` principal
              components were reconstructed (default).
            - ``'rejection'`` — samples removed by the final
              ``window_criterion`` clean-windows pass.
            - ``'calibration'`` — samples selected as the calibration reference
              (JugglerASR only).
        min_components : int, default=1
            Minimum reconstructed component count for ``kind='repair'``.
        description : str | None, default=None
            Annotation label. Defaults per kind: ``ASR_REPAIR`` / ``ASR_REJECT``
            / ``ASR_REFERENCE``.

        Returns
        -------
        annotations : mne.Annotations
            Annotation spans for the requested decision.

        Examples
        --------
        Extract repair annotations and attach them to the Raw object for visual
        inspection of exactly where ASR modified the data:

        >>> asr = ASR().fit(raw)
        >>> clean_raw = asr.transform(raw)
        >>> repair_annots = asr.to_annotations(kind="repair", min_components=1)
        >>> clean_raw.set_annotations(clean_raw.annotations + repair_annots)
        """
        self._check_is_fitted()
        if mne is None:
            raise RuntimeError("MNE is required to create annotations")
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
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Apply ASR reconstruction to epoched data.

        Processes each epoch independently, accumulating diagnostics across all
        epochs, and aggregating window metrics.

        Parameters
        ----------
        data : ndarray, shape (n_epochs, n_channels, n_times)
            The epoched data array to reconstruct.
        sfreq : float
            The sampling frequency of the data in Hz.

        Returns
        -------
        cleaned : ndarray, shape (n_epochs, n_channels, n_times)
            The reconstructed epoched data.
        diagnostics : dict
            Aggregated reconstruction statistics across all epochs.
        """
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
            selected_clean, diag = self._process(selected, sfreq)
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
        """Store diagnostics from the most recent transform operation.

        Updates the corresponding instance attributes with the diagnostic
        values from the last reconstruction pass. If window rejection was
        not performed, any existing rejection attributes are deleted.

        Parameters
        ----------
        diagnostics : dict
            The diagnostic dictionary returned by the reconstruction function.
        """
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
        """Warn if the input data violates ASR preprocessing assumptions.

        Checks the MNE info dictionary for insufficient high-pass filtering
        (< 0.25 Hz) and active/unapplied projectors, both of which can
        negatively impact ASR covariance estimation and reconstruction.

        Parameters
        ----------
        inst : mne.io.BaseRaw | mne.Epochs | mne.Evoked | None
            The MNE object containing the data and info dictionary.
        mne_type : str
            The type of the input data (e.g., 'raw', 'epochs', 'array').
        """
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
