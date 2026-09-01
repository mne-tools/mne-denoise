"""ZapLine line-noise removal."""

from __future__ import annotations

import warnings

import numpy as np

from .._data import (
    continuous_to_epochs,
    epochs_to_continuous,
    extract_data_from_mne,
    reconstruct_mne_object,
)
from .._logging import logger, verbose
from .._spatial import apply_spatial_transform

# Inherit from DSS
from ..dss._whitening import (
    map_spatial_matrices_to_sensor_space,
)
from ..dss.denoisers.spectral import LineNoiseBias
from ..dss.denoisers.temporal import SmoothingBias
from ..dss.linear import DSS, _as_smoother
from ..dss.segmentation import CovarianceSegmenter
from ..progress import _emit_progress, _ProgressCallback, _validate_callback
from .adaptive import (
    apply_hybrid_cleanup,
    check_artifact_presence,
    check_spectral_qa,
    detect_harmonics,
    find_fine_peak,
    find_noise_freqs,
)


class ZapLine(DSS):
    r"""DSS-based line-noise removal estimator.

    ZapLine fits spatial filters for a line frequency and removes selected
    components. Adaptive mode performs the ZapLine-plus frequency, segmentation, and
    per-segment processing path.

    Parameters
    ----------
    sfreq : float
        Sampling frequency in Hz.
    line_freq : float or None, default=60.0
        Fundamental line frequency in Hz. Required in standard mode; None enables
        detection in adaptive mode.
    n_select : int or {"auto"}, default="auto"
        Number of DSS components to remove, or automatic selection.
    n_harmonics : int or None, default=None
        Number of harmonics; None uses harmonics below Nyquist.
    nfft : int, default=1024
        FFT length for the line-noise bias.
    nkeep : int or None, default=None
        Number of dimensions retained before DSS.
    rank : int or None, default=None
        Whitening rank.
    reg : float, default=1e-9
        DSS covariance regularization.
    threshold : float, default=3.0
        Outlier threshold for automatic component selection.
    knee_rel_floor : float, default=0.01
        Relative score floor for knee selection.
    knee_min_ratio : float, default=3.0
        Minimum score ratio for knee selection.
    adaptive : bool, default=False
        Use adaptive ZapLine-plus processing.
    adaptive_params : dict or None, default=None
        Parameters for adaptive frequency detection and segment processing.
    segmenter : object or None, default=None
        Segmenter used in adaptive mode.
    crossfade : float, default=0.0
        Boundary crossfade duration in seconds in adaptive mode.
    whiten : bool, default=False
        Pre-whiten supported MNE data channels before processing.
    noise_cov : mne.Covariance or None, default=None
        Noise covariance used when whiten=True.
    verbose : bool, str, int, or None, default=None
        Logging level.

    Attributes
    ----------
    filters_, patterns_, eigenvalues_
        Fitted DSS filters, patterns, and eigenvalues.
    n_removed_ : int
        Number of removed components or adaptive component passes.
    n_harmonics_ : int or None
        Number of harmonics used by the bias.
    adaptive_results_ : dict or None
        Diagnostics from adaptive processing.

    See Also
    --------
    mne_denoise.spectrum_interpolation.SpectrumInterpolation
        Spectral-amplitude interpolation around line frequencies.
    mne_denoise.dss.DSS
        General DSS estimator underlying standard ZapLine decomposition.

    Notes
    -----
    Standard fit followed by transform uses a fitted operator. Adaptive mode requires
    fit_transform because fitting and cleaning are performed per segment.
    The adaptive workflow follows the ZapLine and ZapLine-plus methods
    :footcite:p:`decheveigne2020_zapline,klug_kloosterman2022_zapline_plus`.

    References
    ----------
    .. footbibliography::

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.zapline import ZapLine
    >>> rng = np.random.default_rng(0)
    >>> data = rng.standard_normal((8, 2000))
    >>> model = ZapLine(sfreq=1000.0, line_freq=50.0, n_select="auto")
    >>> clean = model.fit_transform(data)
    """

    def __init__(
        self,
        sfreq: float,
        line_freq: float | None = 60.0,
        n_select: int | str = "auto",
        n_harmonics: int | None = None,
        nfft: int = 1024,
        nkeep: int | None = None,
        rank: int | None = None,
        reg: float = 1e-9,
        threshold: float = 3.0,
        knee_rel_floor: float = 0.01,
        knee_min_ratio: float = 3.0,
        adaptive: bool = False,
        adaptive_params: dict | None = None,
        segmenter=None,
        crossfade: float = 0.0,
        whiten: bool = False,
        noise_cov=None,
        verbose: bool | str | int | None = None,
    ):
        self.sfreq = float(sfreq)
        self.line_freq = float(line_freq) if line_freq is not None else None
        self.n_harmonics = n_harmonics
        self.nfft = nfft
        self.nkeep = nkeep
        self.threshold = threshold
        self.knee_rel_floor = knee_rel_floor
        self.knee_min_ratio = knee_min_ratio
        self.adaptive_params = adaptive_params if adaptive_params is not None else {}
        self.verbose = verbose

        # Initialize DSS Bias immediately if line_freq is known and valid
        if self.line_freq is not None and self.line_freq > 0:
            self.bias = LineNoiseBias(
                freq=self.line_freq,
                sfreq=self.sfreq,
                method="fft",
                n_harmonics=self.n_harmonics,
                nfft=self.nfft,
                overlap=0.5,
            )
            self.n_harmonics_ = self.bias.n_harmonics
        else:
            self.bias = None
            self.n_harmonics_ = None

        # Initialize DSS parent with our bias
        super().__init__(
            bias=self.bias,
            n_components=None,
            rank=rank,
            reg=reg,
            normalize_input=False,
            adaptive=adaptive,
            segmenter=segmenter,
            crossfade=crossfade,
            max_prop_remove=0.2,
            min_select=1,
            component_action="subtract",
            n_select=n_select,
            selection_threshold=threshold,
            knee_rel_floor=knee_rel_floor,
            knee_min_ratio=knee_min_ratio,
            whiten=whiten,
            noise_cov=noise_cov,
            verbose=verbose,
        )

        self.n_removed_ = None
        self._target_freq_ = None
        self.adaptive_results_ = None
        self._artifact_mixing_ = None
        self._mne_ch_names_ = None
        self._whitening_info_ = None

    @verbose
    def fit(
        self,
        X,
        y=None,
        *,
        verbose: bool | str | int | None = None,
    ):
        """Fit standard-mode ZapLine filters.

        Parameters
        ----------
        X : Raw, Epochs, Evoked, or ndarray
            Data used to fit the filters.
        y : None, default=None
            Ignored for scikit-learn compatibility.
        verbose : bool, str, int, or None, default=None
            Logging level.

        Returns
        -------
        ZapLine
            The fitted estimator.

        Raises
        ------
        RuntimeError
            If adaptive=True; use fit_transform instead.
        """
        if self.adaptive:
            raise RuntimeError(
                "Adaptive mode requires simultaneous fit and transform (local chunks). "
                "Use fit_transform() instead."
            )

        data, extracted_sfreq, _, orig_inst, _, ch_names = extract_data_from_mne(
            X,
            auto_pick="data" if self.whiten else True,
            concatenate_epochs=True,
        )
        self._mne_ch_names_ = ch_names
        self._whitening_info_ = orig_inst.info if orig_inst is not None else None

        # Validate sfreq consistency
        if extracted_sfreq is not None and not np.isclose(extracted_sfreq, self.sfreq):
            warnings.warn(
                f"Input data sfreq ({extracted_sfreq}) differs from init sfreq ({self.sfreq}). "
                "Using init sfreq. Please verify your data or init parameters.",
                stacklevel=2,
            )

        # Confirm line_freq is set
        if self.line_freq is None:
            raise ValueError("line_freq required for standard fit().")

        # Run core fitting logic
        self._fit_dss(data)

        logger.info(
            "ZapLine: mode=standard, frequency=%.3g Hz, harmonics=%d, "
            "removed=%d component(s) of %d.",
            self.line_freq,
            self.n_harmonics_ or 0,
            self.n_removed_ or 0,
            len(self.eigenvalues_) if self.eigenvalues_ is not None else 0,
        )

        return self

    @verbose
    def transform(
        self,
        X,
        *,
        verbose: bool | str | int | None = None,
    ):
        """Apply fitted standard-mode ZapLine filters.

        Parameters
        ----------
        X : Raw, Epochs, Evoked, or ndarray
            Data with the fitted channel layout.
        verbose : bool, str, int, or None, default=None
            Logging level.

        Returns
        -------
        same type as X
            Cleaned data.

        Raises
        ------
        RuntimeError
            If adaptive=True or the estimator is not fitted.
        """
        if self.adaptive:
            raise RuntimeError(
                "Adaptive mode requires simultaneous fit and transform (local chunks). "
                "Use fit_transform() instead."
            )

        # Check if fitted
        if self.filters_ is None:
            raise RuntimeError("Not fitted")

        data, extracted_sfreq, mne_type, orig_inst, picks, _ = extract_data_from_mne(
            X,
            ch_names=getattr(self, "_mne_ch_names_", None),
            auto_pick=not self.whiten,
        )

        # Validate sfreq consistency
        if extracted_sfreq is not None and not np.isclose(extracted_sfreq, self.sfreq):
            warnings.warn(
                f"Input data sfreq ({extracted_sfreq}) differs from init sfreq ({self.sfreq}). "
                "Using init sfreq.",
                stacklevel=2,
            )

        # Standard Transform
        cleaned = continuous_to_epochs(
            self._apply_standard_cleaning(epochs_to_continuous(data)), data.shape
        )

        return reconstruct_mne_object(cleaned, orig_inst, mne_type, picks=picks)

    @verbose
    def fit_transform(
        self,
        X,
        y=None,
        *,
        callback=None,
        verbose: bool | str | int | None = None,
        **fit_params,
    ):
        """Fit and transform with standard or adaptive ZapLine.

        Parameters
        ----------
        X : Raw, Epochs, Evoked, or ndarray
            Data to clean.
        y : None, default=None
            Ignored for scikit-learn compatibility.
        callback : callable or None, default=None
            Synchronous progress callback in adaptive mode.
        verbose : bool, str, int, or None, default=None
            Logging level.
        **fit_params : dict
            Additional parameters passed to the standard DSS fit_transform path.

        Returns
        -------
        same type as X
            Cleaned data with the input layout.
        """
        callback = _validate_callback(callback)
        if not self.adaptive:
            return super().fit_transform(X, y=y, callback=callback, **fit_params)

        data, extracted_sfreq, mne_type, orig_inst, picks, ch_names = (
            extract_data_from_mne(
                X,
                auto_pick="data" if self.whiten else True,
            )
        )
        self._mne_ch_names_ = ch_names
        self._whitening_info_ = orig_inst.info if orig_inst is not None else None

        if extracted_sfreq is not None and not np.isclose(extracted_sfreq, self.sfreq):
            warnings.warn(
                f"Input data sfreq ({extracted_sfreq}) differs from init sfreq ({self.sfreq}). "
                "Using init sfreq.",
                stacklevel=2,
            )

        # Adaptive logic (ZapLine-plus)
        if data.ndim == 3:
            n_ep, n_ch, n_t = data.shape
            data_cont = epochs_to_continuous(data)
        else:
            n_ch, n_t = data.shape
            data_cont = data

        if self.whiten:
            data_work = self._prewhiten_sensor_data(
                data_cont,
                info=self._whitening_info_,
                ch_names=ch_names,
            )
        else:
            data_work = data_cont

        res = self._run_adaptive(data_work, callback=callback)
        self.n_removed_ = res["n_removed"]
        if self.whiten:
            removed = apply_spatial_transform(self._dewhitener_, res["removed"])
            cleaned = data_cont - removed
            if self.patterns_ is not None:
                self.patterns_ = apply_spatial_transform(
                    self._dewhitener_, self.patterns_
                )
            res["cleaned"] = cleaned
            res["removed"] = removed
        else:
            cleaned = res["cleaned"]
        self.adaptive_results_ = res

        if data.ndim == 3:
            cleaned = continuous_to_epochs(cleaned, (n_ep, n_ch, n_t))

        line_freqs = res.get("line_freqs", ())
        logger.info(
            "ZapLine: mode=adaptive, frequencies=%s Hz, %d segment pass(es), "
            "removed=%d component pass(es).",
            tuple(float(freq) for freq in line_freqs),
            len(res.get("chunk_info", ())),
            res.get("n_removed", 0),
        )

        return reconstruct_mne_object(cleaned, orig_inst, mne_type, picks=picks)

    def _fit_dss(self, data: np.ndarray):
        """Fit DSS filters to residual data and choose removals."""
        if self.whiten:
            data_work = self._prewhiten_sensor_data(
                data,
                info=self._whitening_info_,
                ch_names=self._mne_ch_names_,
            )
        else:
            data_work = data

        # 1. Smooth data
        data_smooth, data_residual = self._get_smooth_residual(data_work, warn=True)

        # 2. Setup (Rank)
        dss_rank = self.nkeep if self.nkeep is not None else self.rank
        self.rank = dss_rank

        # 3. Call DSS fit on Residual
        if self.bias is None:
            self.filters_ = np.zeros((0, data.shape[0]))
            self.patterns_ = np.zeros((data.shape[0], 0))
            self._artifact_mixing_ = np.zeros((data.shape[0], 0))
            self.eigenvalues_ = np.array([])
            self.n_removed_ = 0
            return

        super()._fit_numpy(data_residual)
        self.mixing_ = self.patterns_

        # Keep full DSS solution before truncating to removed components.
        full_filters = self.filters_.copy()
        full_mixing = self.mixing_.copy()
        qa_filters = full_filters
        qa_mixing = full_mixing
        if self.whiten:
            full_filters, full_mixing = map_spatial_matrices_to_sensor_space(
                full_filters,
                full_mixing,
                whitener=self._whitener_,
                dewhitener=self._dewhitener_,
            )
        self.mixing_ = full_mixing

        # 4. Resolve the component count. DSS.auto_select handles both the 'auto' and
        # the explicit-int cases, so the primary policy lives in exactly one place.
        self.n_removed_ = self.auto_select()
        if self.n_select == "auto":
            if self.n_removed_ == 0 and check_artifact_presence(
                data_work, self.sfreq, self.line_freq
            ):
                logger.debug(
                    "ZapLine auto-selection found no DSS component boundary, "
                    "but a line-noise peak is present at %.3g Hz; evaluating "
                    "component counts spectrally.",
                    self.line_freq,
                )
                self.n_removed_ = self._find_min_components_for_line_suppression(
                    data_smooth=data_smooth,
                    data_residual=data_residual,
                    full_filters=qa_filters,
                    full_mixing=qa_mixing,
                )
                logger.debug(
                    "ZapLine spectral fallback selected %d components.",
                    self.n_removed_,
                )
            logger.debug(
                "ZapLine auto-selected %d/%d components "
                "(eigenvalues: max=%.3g, min=%.3g)",
                self.n_removed_,
                len(self.eigenvalues_),
                float(self.eigenvalues_[0]),
                float(self.eigenvalues_[-1]),
            )

        # 5. Truncate to line-dominated DSS components.
        # Use DSS mixing from the full decomposition to reconstruct artifacts.
        if self.n_removed_ > 0:
            self.filters_ = full_filters[: self.n_removed_]
            self._artifact_mixing_ = full_mixing[:, : self.n_removed_]
            self.patterns_ = self._artifact_mixing_
        else:
            self.filters_ = np.zeros((0, data.shape[0]))
            self.patterns_ = np.zeros((data.shape[0], 0))
            self._artifact_mixing_ = np.zeros((data.shape[0], 0))

    def _find_min_components_for_line_suppression(
        self,
        *,
        data_smooth: np.ndarray,
        data_residual: np.ndarray,
        full_filters: np.ndarray,
        full_mixing: np.ndarray,
    ) -> int:
        """Find the smallest DSS subspace that passes spectral QA."""
        n_candidates = full_filters.shape[0]
        last_status = None
        best_before_overcleaning = 0

        for n_selected in range(1, n_candidates + 1):
            filters = full_filters[:n_selected]
            mixing = full_mixing[:, :n_selected]
            sources = filters @ data_residual
            artifact = mixing @ sources
            candidate_clean = data_smooth + (data_residual - artifact)
            status = check_spectral_qa(
                candidate_clean,
                self.sfreq,
                self.line_freq,
            )
            last_status = status

            if status == "ok":
                return n_selected
            if status == "weak":
                best_before_overcleaning = n_selected

        if best_before_overcleaning == 0:
            best_before_overcleaning = n_candidates
        warnings.warn(
            "Line noise remains after evaluating the available DSS components "
            "without a satisfactory spectral-QA result. Consider setting "
            "n_select manually or using adaptive ZapLine.",
            stacklevel=2,
        )
        logger.debug(
            "ZapLine spectral fallback ended with status %r after %d candidates.",
            last_status,
            n_candidates,
        )
        return best_before_overcleaning

    def _apply_standard_cleaning(self, data: np.ndarray) -> np.ndarray:
        """Apply cleaning with fitted DSS filters."""
        if self.n_removed_ <= 0:
            return data.copy()

        # 1. Smooth
        data_smooth, data_residual = self._get_smooth_residual(data, warn=False)

        # 2. Extract artifact sources using fitted filters (manual to avoid recentering)
        # DSS filters are spatial (n_comp, n_ch).
        # data_residual is (n_ch, n_times).
        sources = self.filters_ @ data_residual

        # 3. Project back to artifact using full DSS mixing for selected components.
        artifact = self._artifact_mixing_ @ sources

        # 4. Subtract artifact from residual, add back smooth
        cleaned = data_smooth + (data_residual - artifact)

        return cleaned

    def _get_smooth_residual(self, data: np.ndarray, warn: bool = False):
        """Split data into smooth and residual components."""
        # Use self.sfreq directly
        # If line_freq=0 (unlikely here if fit passed), period undefined.
        # Check integrity
        if self.line_freq is None or self.line_freq == 0:
            # Should not happen in standard mode, effectively no cleaning
            return data, np.zeros_like(data)

        # Calculate exact period (may be non-integer)
        exact_period = self.sfreq / self.line_freq
        int_period = int(round(exact_period))

        # Check if period is close to an integer (within 5%)
        period_error = abs(exact_period - int_period) / exact_period

        if period_error > 1e-4:
            # Significant mismatch - use fractional smoothing
            if warn and period_error > 0.05:
                warnings.warn(
                    f"sfreq/line_freq = {exact_period:.2f} is not close to an integer. "
                    f"Using fractional-period smoothing for accuracy.",
                    UserWarning,
                    stacklevel=2,
                )
            data_smooth = self._fractional_smooth(data, exact_period)
        else:
            # Period is close to integer - use standard smoothing
            if warn and abs(exact_period - int_period) > 0.1:
                warnings.warn(
                    f"sfreq/line_freq = {exact_period:.2f} is not exactly an integer. "
                    f"Smoothing will use period={int_period} samples.",
                    UserWarning,
                    stacklevel=2,
                )
            smoother = SmoothingBias(window=int_period, iterations=1)
            data_smooth = smoother.apply(data)

        data_residual = data - data_smooth
        return data_smooth, data_residual

    def _fractional_smooth(self, data: np.ndarray, period: float) -> np.ndarray:
        """Apply fractional-period boxcar smoothing."""
        from scipy.signal import lfilter

        n_times = data.shape[-1]
        period = float(period)

        if period <= 1:
            # Degenerate case (should not occur for valid line frequencies).
            return data.copy()

        integ = int(np.floor(period))
        frac = period - integ

        if integ >= n_times:
            mean = np.mean(data, axis=-1, keepdims=True)
            return np.repeat(mean, n_times, axis=-1)

        # remove onset step, filter, then restore DC.
        mean_head = np.mean(data[..., : integ + 1], axis=-1, keepdims=True)
        centered = data - mean_head

        if np.isclose(frac, 0.0):
            # Fast path for integer period using cumulative sums.
            smoothed = np.cumsum(centered, axis=-1)
            smoothed[..., integ:] = smoothed[..., integ:] - smoothed[..., :-integ]
            smoothed = smoothed / float(integ)
        else:
            kernel = np.concatenate([np.ones(integ), [frac]]) / period
            smoothed = lfilter(kernel, [1.0], centered, axis=-1)

        smoothed += mean_head
        return smoothed

    # =========================================================================
    # Adaptive Mode (ZapLine-plus) Methods
    # =========================================================================

    def _run_adaptive(
        self,
        data: np.ndarray,
        *,
        callback: _ProgressCallback | None = None,
    ) -> dict:
        """Run the adaptive ZapLine processing path."""
        n_channels, n_times = data.shape
        params = self.adaptive_params.copy()

        # Extract params with defaults
        fmin = params.get("fmin", 17.0)
        fmax = params.get("fmax", 99.0)
        process_harmonics = params.get("process_harmonics", False)
        max_harmonics = params.get("max_harmonics", None)
        min_chunk_len = params.get("min_chunk_len", 30.0)

        # 1. Automatic frequency detection
        line_freqs = self.line_freq
        if line_freqs is None:
            logger.debug("ZapLine adaptive frequency detection started.")
            line_freqs = find_noise_freqs(data, self.sfreq, fmin=fmin, fmax=fmax)
            logger.debug("ZapLine adaptive frequencies detected: %s.", line_freqs)
        elif isinstance(line_freqs, int | float):
            line_freqs = [float(line_freqs)]

        # Quick exit if nothing to clean
        if not line_freqs:
            return {
                "cleaned": data.copy(),
                "removed": np.zeros_like(data),
                "n_removed": 0,
                "line_freq": 0.0,
                "chunk_info": [],
            }

        current_data = data.copy()
        all_chunk_metadata = []

        # Collect all target frequencies
        all_freqs_to_process = []
        for lfreq in line_freqs:
            all_freqs_to_process.append(lfreq)
            if process_harmonics:
                harmonics = detect_harmonics(
                    current_data, self.sfreq, lfreq, max_harmonics
                )
                all_freqs_to_process.extend(harmonics)

        self._smoother = _as_smoother(self.smooth)

        try:
            for freq_idx, target_freq in enumerate(all_freqs_to_process):
                self._target_freq_ = target_freq
                segmenter = CovarianceSegmenter(
                    sfreq=self.sfreq,
                    min_chunk_len=min_chunk_len,
                    bandpass=(target_freq - 3, target_freq + 3),
                )
                current_data = self._run_segmented(
                    current_data,
                    self.sfreq,
                    segmenter=segmenter,
                    callback=None,
                )
                logger.debug(
                    "ZapLine adaptive frequency pass %.3g Hz completed over %d "
                    "segment(s).",
                    target_freq,
                    len(self.segment_results_ or ()),
                )

                for seg in self.segment_results_ or []:
                    all_chunk_metadata.append(
                        {
                            "frequency": target_freq,
                            "fine_freq": seg.get("fine_freq"),
                            "start": seg["start"],
                            "end": seg["end"],
                            "n_removed": seg["n_selected"],
                            "artifact_present": seg.get("artifact_present"),
                        }
                    )
                    # Representative attributes for plotting
                    if seg.get("eigenvalues") is not None:
                        self.eigenvalues_ = seg["eigenvalues"]
                    if seg.get("patterns") is not None:
                        self.patterns_ = seg["patterns"]
                _emit_progress(
                    callback,
                    method="zapline",
                    stage="frequency",
                    current=freq_idx + 1,
                    total=len(all_freqs_to_process),
                    component=None,
                    metric=float(target_freq),
                )
        finally:
            self._target_freq_ = None

        return {
            "cleaned": current_data,
            "removed": data - current_data,
            "n_removed": sum(c.get("n_removed", 0) for c in all_chunk_metadata),
            "line_freq": line_freqs[0] if line_freqs else 0,
            "line_freqs": tuple(float(freq) for freq in all_freqs_to_process),
            "chunk_info": all_chunk_metadata,
        }

    def _process_segment(self, chunk: np.ndarray) -> dict:
        """Clean one segment with the ZapLine spectral-QA retry loop."""
        params = self.adaptive_params
        n_remove_params = params.get("n_remove_params", {})
        qa_params = params.get("qa_params", {})
        hybrid_fallback = params.get("hybrid_fallback", False)

        sigma_init = n_remove_params.get("sigma", self.selection_threshold)
        min_remove = n_remove_params.get("min_remove", self.min_select)
        max_prop_remove = n_remove_params.get("max_prop", self.max_prop_remove)

        target_freq = self._target_freq_ or self.line_freq
        if target_freq is None:
            raise RuntimeError(
                "ZapLine needs a target frequency to clean a segment. Pass "
                "line_freq=..., or use adaptive=True to detect it."
            )

        max_sigma = qa_params.get("max_sigma", 4.0)
        min_sigma = qa_params.get("min_sigma", 2.5)
        # New QA parameters
        max_prop_above = qa_params.get("max_prop_above_upper", 0.005)
        max_prop_below = qa_params.get("max_prop_below_lower", 0.005)
        freq_detect_mult = qa_params.get("freq_detect_mult_fine", 2.0)

        n_channels = chunk.shape[0]
        fine_freq = find_fine_peak(chunk, self.sfreq, target_freq)
        present = check_artifact_presence(chunk, self.sfreq, fine_freq)
        logger.debug(
            "ZapLine segment: target=%.3g Hz, fine=%.3g Hz, artifact_present=%s.",
            target_freq,
            fine_freq,
            present,
        )

        current_sigma = sigma_init
        current_min_remove = min_remove if present else 0

        best_chunk_clean = None
        is_too_strong = False
        status = "ok"

        max_retries = 5
        res_n_removed = 0
        res_cleaned = chunk.copy()

        for _retry in range(max_retries):
            # Fresh ZapLine for this chunk. Deriving it from get_params()
            # carries every setting (rank, reg, whiten, nfft, ...) rather than
            # silently dropping whatever this call site forgot to list.
            est = type(self)(
                **{
                    **self.get_params(),
                    "line_freq": fine_freq,
                    "n_select": "auto",
                    "threshold": current_sigma,
                    "adaptive": False,
                    "crossfade": 0.0,
                    "verbose": "WARNING",
                }
            )

            # Adaptive ZapLine owns the segment-level aggregate report; hide
            # each retry's nested DSS fit from the user-facing INFO stream.
            est.fit(chunk)
            res_cleaned = est.transform(chunk)
            res_n_removed = est.n_removed_

            # Apply constraints
            max_rem_cap = int(n_channels * max_prop_remove)
            n_rem = min(res_n_removed, max_rem_cap)
            n_rem = max(n_rem, current_min_remove)

            # Refit if constraints changed n_removed
            if n_rem != res_n_removed:
                est = ZapLine(
                    sfreq=self.sfreq,
                    line_freq=fine_freq,
                    n_select=int(n_rem),
                    knee_rel_floor=self.knee_rel_floor,
                    knee_min_ratio=self.knee_min_ratio,
                    adaptive=False,
                    # The outer adaptive operation owns the aggregate report.
                    verbose="WARNING",
                )
                est.fit(chunk)
                res_cleaned = est.transform(chunk)
                res_n_removed = est.n_removed_

            status = check_spectral_qa(
                res_cleaned,
                self.sfreq,
                fine_freq,
                max_prop_above=max_prop_above,
                max_prop_below=max_prop_below,
                freq_detect_mult=freq_detect_mult,
            )
            logger.debug(
                "ZapLine segment retry %d/%d: fine=%.3g Hz, sigma=%.3g, "
                "selected=%d, status=%s.",
                _retry + 1,
                max_retries,
                fine_freq,
                current_sigma,
                res_n_removed,
                status,
            )

            if status == "ok":
                best_chunk_clean = res_cleaned
                break
            elif status == "weak":
                if is_too_strong:
                    best_chunk_clean = res_cleaned
                    break
                else:
                    current_sigma = max(current_sigma - 0.25, min_sigma)
                    current_min_remove = current_min_remove + 1
            elif status == "strong":
                is_too_strong = True
                current_sigma = min(current_sigma + 0.25, max_sigma)
                current_min_remove = max(current_min_remove - 1, 0)

        if best_chunk_clean is None:
            best_chunk_clean = res_cleaned

        if hybrid_fallback and status == "weak":
            best_chunk_clean = apply_hybrid_cleanup(
                best_chunk_clean, self.sfreq, fine_freq
            )

        return {
            "cleaned": best_chunk_clean,
            "n_selected": res_n_removed,
            "eigenvalues": getattr(est, "eigenvalues_", None),
            "patterns": getattr(est, "patterns_", None),
            "filters": getattr(est, "filters_", None),
            # ZapLine-specific diagnostics, carried into segment_results_
            "fine_freq": fine_freq,
            "artifact_present": present,
        }
