"""ZapLine: Line noise removal using Denoising Source Separation.

This module implements the ZapLine algorithm for removing power line noise (50/60 Hz)
and its harmonics from M/EEG recordings using Denoising Source Separation (DSS).

The algorithm works by:
1. Decomposing data into a smooth component and a residual (line-frequency related)
2. Applying DSS to the residual to find components that maximize line noise power
3. Projecting out the noise components while preserving neural signals

This module contains:
1. `ZapLine`: Scikit-learn/MNE compatible Transformer (inherits from DSS)

The adaptive mode (ZapLine-plus) extends this with:
- Automatic noise frequency detection
- Data segmentation based on covariance stationarity
- Per-segment adaptive parameter tuning

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)

References
----------
.. [1] de Cheveigné, A. (2020). ZapLine: A simple and effective method to remove
       power line artifacts. NeuroImage, 207, 116356.
       https://doi.org/10.1016/j.neuroimage.2019.116356

.. [2] Klug, M., & Kloosterman, N. A. (2022). Zapline-plus: A Zapline extension for
       automatic and adaptive removal of frequency-specific noise artifacts in M/EEG.
       Human Brain Mapping, 43(9), 2743-2758.
       https://doi.org/10.1002/hbm.25832

.. [3] de Cheveigné, A., & Simon, J. Z. (2008). Denoising based on spatial filtering.
       Journal of Neuroscience Methods, 171(2), 331-339.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np

from .._logging import set_log_level_from_verbose
from .._spatial import (
    apply_spatial_transform,
    continuous_to_epochs,
    epochs_to_continuous,
)

# Inherit from DSS
from ..dss.denoisers.spectral import LineNoiseBias
from ..dss.denoisers.temporal import SmoothingBias
from ..dss.linear import DSS, _as_smoother
from ..dss.utils.segmentation import CovarianceSegmenter
from ..dss.utils.whitening import (
    map_spatial_matrices_to_sensor_space,
)
from ..utils import (
    extract_data_from_mne,
    reconstruct_mne_object,
)
from .adaptive import (
    apply_hybrid_cleanup,
    check_artifact_presence,
    check_spectral_qa,
    detect_harmonics,
    find_fine_peak,
    find_noise_freqs,
)

logger = logging.getLogger(__name__)


class ZapLine(DSS):
    r"""ZapLine Transformer for line noise removal.

    Implements the ZapLine algorithm [1]_ for removing power line noise (50/60 Hz)
    and its harmonics from M/EEG recordings. Inherits from :class:`DSS` and is
    compatible with scikit-learn and MNE-Python objects.

    The algorithm uses Denoising Source Separation (DSS) to find spatial filters
    that maximize power at the line noise frequency, then projects out these
    noise components while preserving neural signals.

    The cleaning process follows these steps:

    1. **Smoothing**: Apply a moving average filter with period matching the line
       frequency to separate slowly-varying ("smooth") and fast ("residual") components
    2. **DSS**: Apply DSS to the residual using :class:`LineNoiseBias` to find
       components that maximize line noise power
    3. **Artifact removal**: Project out the top noise components from the residual
    4. **Reconstruction**: Add back the smooth component to obtain cleaned data

    Parameters
    ----------
    sfreq : float
        Sampling frequency in Hz. Required.
    line_freq : float | None, default=60.0
        Line noise frequency in Hz (typically 50 or 60 Hz).
        If ``None`` and ``adaptive=True``, the frequency is auto-detected.
        If ``None`` and ``adaptive=False``, raises an error.
    n_select : int | 'auto', default='auto'
        Number of noise components to remove. Inherited from :class:`DSS`;
        ``'auto'`` first uses :meth:`DSS.auto_select` and, when that returns
        zero despite a spectral artifact at ``line_freq``, evaluates
        successive leading DSS components with ZapLine's spectral quality
        assessment. An ``int`` removes exactly that many. The resolved count
        is stored in :attr:`n_removed_`.
    n_harmonics : int | None, default=None
        Number of harmonics to include in the bias function.
        If ``None``, auto-determined based on Nyquist frequency.
    nfft : int, default=1024
        FFT length for the spectral bias function.
    nkeep : int | None, default=None
        Number of PCA components to retain before DSS.
        If ``None``, uses ``rank`` or auto-determined from data.
    rank : int | None, default=None
        Rank of the data for whitening. If ``None``, auto-determined.
    reg : float, default=1e-9
        Regularization parameter for DSS covariance inversion.
    threshold : float, default=3.0
        Sigma threshold for iterative outlier removal when ``n_select='auto'``.
    knee_rel_floor : float, default=0.01
        Relative-to-max anchor for the knee-detection fallback used when
        ``n_select='auto'``. Eigenvalues below this fraction of the largest
        are excluded from knee selection. See
        :func:`mne_denoise.dss.utils.detect_eigenvalue_knee`.
    knee_min_ratio : float, default=3.0
        Minimum linear-scale drop between consecutive eigenvalues required to
        qualify as a knee. Defaults to a factor-of-3 drop. Lower values make
        knee detection more permissive; higher values make it stricter.
    adaptive : bool, default=False
        Inherited from :class:`DSS`. If ``True``, use adaptive ZapLine-plus
        mode [2]_ with:
        - Automatic frequency detection
        - Data segmentation based on covariance stationarity
        - Per-segment parameter adaptation
    adaptive_params : dict | None, default=None
        Parameters for adaptive mode. See Notes for available options.
    crossfade : float, default=0.0
        Duration in seconds of the raised-cosine cross-fade applied at segment
        boundaries in adaptive mode. Each segment is cleaned with its own
        filters, so hard concatenation (the default, matching ZapLine-plus)
        leaves a step discontinuity at every boundary — broadband in frequency,
        and therefore smeared across the very spectrum being cleaned. Set to
        ``0.5``–``2.0`` s to blend the boundaries instead. Ignored when
        ``adaptive=False``.
    whiten : bool, default=False
        If ``True``, jointly process all supported data channel types after
        MNE-style pre-whitening. The default keeps the existing homogeneous
        channel-type selection.
    noise_cov : mne.Covariance | None, default=None
        Noise covariance used when ``whiten=True``. It requires an MNE input.
        If ``None``, MNE inputs are standardized by channel type and NumPy
        arrays are standardized per channel.
    verbose : bool | str | int | None, default=None
        Control logging verbosity.

    Attributes
    ----------
    filters_ : ndarray, shape (n_removed, n_channels)
        Spatial filters (unmixing matrix) for the removed noise components.
    patterns_ : ndarray, shape (n_channels, n_removed)
        Spatial patterns (mixing matrix) for the removed noise components.
    eigenvalues_ : ndarray, shape (n_components,)
        DSS eigenvalues (ratio of line-noise power to total power).
    n_removed_ : int
        Number of components actually removed.
    n_harmonics_ : int | None
        Number of harmonics used by the bias function.
    adaptive_results_ : dict | None
        Results from adaptive mode, including chunk information.

    See Also
    --------
    DSS : Parent class implementing Denoising Source Separation.
    LineNoiseBias : Bias function for line noise.
    mne_denoise.zapline.adaptive : Adaptive ZapLine-plus utilities.

    Notes
    -----
    **Adaptive Mode Parameters** (``adaptive_params`` dict):

    - ``fmin`` : float (default 17.0) - Minimum frequency for noise detection
    - ``fmax`` : float (default 99.0) - Maximum frequency for noise detection
    - ``process_harmonics`` : bool (default False) - Process detected harmonics
    - ``max_harmonics`` : int - Maximum number of harmonics to process
    - ``hybrid_fallback`` : bool (default False) - Use notch filter fallback
    - ``min_chunk_len`` : float (default 30.0) - Minimum segment length in seconds
    - ``n_remove_params`` : dict
      Adaptive per-segment component-selection settings:
      ``sigma`` is the selection threshold (defaults to ``threshold``),
      ``min_remove`` is the minimum number of components removed when an
      artifact is present (defaults to 1), and ``max_prop`` is the maximum
      proportion of channels removed (defaults to 0.2)
    - ``qa_params`` : dict - Parameters for QA (max_sigma, min_sigma)

    Examples
    --------
    Basic usage with MNE Raw object:

    >>> from mne_denoise.zapline import ZapLine
    >>> # Remove 50 Hz line noise
    >>> zapline = ZapLine(sfreq=1000, line_freq=50.0)
    >>> raw_clean = zapline.fit_transform(raw)

    Using automatic component selection:

    >>> zapline = ZapLine(sfreq=1000, line_freq=60.0, n_select="auto")
    >>> zapline.fit(data)
    >>> print(f"Removed {zapline.n_removed_} components")
    >>> cleaned = zapline.transform(data)

    Adaptive mode (ZapLine-plus):

    >>> zapline = ZapLine(
    ...     sfreq=1000,
    ...     line_freq=None,
    ...     adaptive=True,
    ...     adaptive_params={"n_remove_params": {"min_remove": 1, "max_prop": 0.2}},
    ... )
    >>> cleaned = zapline.fit_transform(epochs)  # Auto-detects noise frequencies

    References
    ----------
    .. [1] de Cheveigné, A. (2020). ZapLine: A simple and effective method to remove
           power line artifacts. NeuroImage, 207, 116356.
    .. [2] Klug, M., & Kloosterman, N. A. (2022). Zapline-plus: A Zapline extension
           for automatic and adaptive removal of frequency-specific noise artifacts
           in M/EEG. Human Brain Mapping, 43(9), 2743-2758.
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
        set_log_level_from_verbose(self.verbose)

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
            verbose=self.verbose,
        )

        self.n_removed_ = None
        self._target_freq_ = None
        self.adaptive_results_ = None
        self._artifact_mixing_ = None
        self._mne_ch_names_ = None
        self._whitening_info_ = None

    def fit(self, X, y=None):
        """Fit ZapLine spatial filters to data.

        Computes DSS filters that maximize power at the line noise frequency.
        Only available in standard mode (``adaptive=False``).

        Parameters
        ----------
        X : Raw | Epochs | Evoked | ndarray
            The data to fit. Can be:

            - MNE Raw, Epochs, or Evoked object
            - NumPy array of shape ``(n_channels, n_times)`` for continuous data
            - NumPy array of shape ``(n_epochs, n_channels, n_times)`` for epoched data

        y : None
            Ignored. Present for scikit-learn API compatibility.

        Returns
        -------
        self : ZapLine
            The fitted estimator.

        Raises
        ------
        RuntimeError
            If ``adaptive=True``. Use :meth:`fit_transform` instead.
        ValueError
            If ``line_freq`` is ``None``.

        See Also
        --------
        transform : Apply the fitted filters to remove noise.
        fit_transform : Fit and transform in one step.
        """
        set_log_level_from_verbose(self.verbose)
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

        return self

    def transform(self, X):
        """Apply ZapLine cleaning to remove line noise.

        Uses the fitted spatial filters to project out noise components.
        Only available in standard mode (``adaptive=False``).

        Parameters
        ----------
        X : Raw | Epochs | Evoked | ndarray
            The data to clean. Must have the same number of channels as the
            data used for fitting.

        Returns
        -------
        X_clean : Raw | Epochs | Evoked | ndarray
            Cleaned data with line noise removed. Returns the same type as input.

        Raises
        ------
        RuntimeError
            If ``adaptive=True``. Use :meth:`fit_transform` instead.
        RuntimeError
            If the estimator has not been fitted.

        See Also
        --------
        fit : Fit the spatial filters.
        fit_transform : Fit and transform in one step.
        """
        set_log_level_from_verbose(self.verbose)
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

    def fit_transform(self, X, y=None, **fit_params):
        """Fit and transform data in one step.

        This method works for both standard and adaptive modes:

        - **Standard mode** (``adaptive=False``): Fits DSS filters and applies
          noise removal in one step.
        - **Adaptive mode** (``adaptive=True``): Runs the full ZapLine-plus
          algorithm with automatic frequency detection, data segmentation,
          and per-segment parameter adaptation.

        Parameters
        ----------
        X : Raw | Epochs | Evoked | ndarray
            The data to process.
        y : None
            Ignored. Present for scikit-learn API compatibility.
        **fit_params : dict
            Additional parameters passed to the parent :meth:`DSS.fit_transform`
            in standard mode.

        Returns
        -------
        X_clean : Raw | Epochs | Evoked | ndarray
            Cleaned data with line noise removed. Returns the same type as input.

        Notes
        -----
        In adaptive mode, results are stored in :attr:`adaptive_results_` with:

        - ``cleaned``: The cleaned data array
        - ``removed``: The removed artifact array
        - ``n_removed``: Total components removed across all chunks
        - ``line_freq``: Detected line frequency
        - ``chunk_info``: List of per-chunk processing information
        """
        set_log_level_from_verbose(self.verbose)
        if not self.adaptive:
            return super().fit_transform(X, y=y, **fit_params)

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

        res = self._run_adaptive(data_work)
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

        return reconstruct_mne_object(cleaned, orig_inst, mne_type, picks=picks)

    def _fit_dss(self, data: np.ndarray):
        """Fit DSS spatial filters to residual data.

        Core internal fitting logic that:
        1. Separates data into smooth and residual components
        2. Fits DSS on the residual using ``LineNoiseBias``
        3. Determines number of components to remove
        4. Truncates filters to noise-relevant components

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_times)
            Continuous data to fit.
        """
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
                logger.info(
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
                logger.info(
                    "ZapLine spectral fallback selected %d components.",
                    self.n_removed_,
                )
            logger.info(
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
        """Find the smallest DSS subspace that passes spectral QA.

        This is a ZapLine-specific fallback for the case where the generic DSS
        eigenvalue selector finds no component boundary despite a detected line
        artifact. The DSS decomposition is already fitted; only the number of
        leading components used for reconstruction changes between candidates.

        Parameters
        ----------
        data_smooth : ndarray, shape (n_channels, n_times)
            Smooth branch from the fitted data.
        data_residual : ndarray, shape (n_channels, n_times)
            Residual branch from the fitted data.
        full_filters : ndarray, shape (n_components, n_channels)
            Complete DSS filter matrix.
        full_mixing : ndarray, shape (n_channels, n_components)
            Complete DSS mixing matrix.

        Returns
        -------
        n_selected : int
            Smallest candidate for which :func:`check_spectral_qa` returns
            ``"ok"``. If no candidate passes, the largest candidate before an
            over-cleaning result is returned and a warning is emitted.
        """
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
        logger.info(
            "ZapLine spectral fallback ended with status %r after %d candidates.",
            last_status,
            n_candidates,
        )
        return best_before_overcleaning

    def _apply_standard_cleaning(self, data: np.ndarray) -> np.ndarray:
        """Apply noise cleaning using fitted DSS filters.

        Removes line noise by projecting out the artifact subspace from
        the residual (high-frequency) component of the data.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_times)
            Continuous data to clean.

        Returns
        -------
        cleaned : ndarray, shape (n_channels, n_times)
            Cleaned data with line noise removed.
        """
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
        """Decompose data into smooth and residual components.

        Uses a moving average filter with period matching the line frequency
        to separate slowly-varying and fast (line-frequency related) components.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_times)
            Input data to decompose.
        warn : bool, default=False
            If ``True``, warn when ``sfreq/line_freq`` is not close to an integer.

        Returns
        -------
        data_smooth : ndarray, shape (n_channels, n_times)
            Smoothed (low-frequency) component.
        data_residual : ndarray, shape (n_channels, n_times)
            Residual (high-frequency) component containing line noise.
        """
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
        """Apply fractional-period smoothing with a boxcar kernel.

        This mirrors NoiseTools ``nt_smooth`` behavior used by ``nt_zapline``:
        for non-integer period ``T``, the kernel is ``[1 ... 1 frac] / T``
        where ``frac = T - floor(T)``. This preserves the period-locked
        decomposition used by ZapLine and avoids switching to a generic
        highpass/lowpass split.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_times)
            Input data.
        period : float
            Exact (possibly non-integer) smoothing period in samples.

        Returns
        -------
        smoothed : ndarray, shape (n_channels, n_times)
            Smoothed (baseline) component.
        """
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

    def _run_adaptive(self, data: np.ndarray) -> dict:
        """Run ZapLine-plus adaptive algorithm.

        Orchestrates:
        1. Noise frequency detection (if line_freq is None)
        2. Data segmentation based on covariance stationarity
        3. Per-chunk processing with QA loop

        Parameters
        ----------
        data : ndarray (n_channels, n_times)
            Continuous data.

        Returns
        -------
        results : dict
            Contains 'cleaned', 'removed', 'n_removed', 'chunk_info', etc.
        """
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
            logger.info("Detecting line noise frequencies...")
            line_freqs = find_noise_freqs(data, self.sfreq, fmin=fmin, fmax=fmax)
            logger.info(f"Detected: {line_freqs}")
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
            for target_freq in all_freqs_to_process:
                self._target_freq_ = target_freq
                segmenter = CovarianceSegmenter(
                    sfreq=self.sfreq,
                    min_chunk_len=min_chunk_len,
                    bandpass=(target_freq - 3, target_freq + 3),
                )
                current_data = self._run_segmented(
                    current_data, self.sfreq, segmenter=segmenter
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
        finally:
            self._target_freq_ = None

        return {
            "cleaned": current_data,
            "removed": data - current_data,
            "n_removed": sum(c.get("n_removed", 0) for c in all_chunk_metadata),
            "line_freq": line_freqs[0] if line_freqs else 0,
            "chunk_info": all_chunk_metadata,
        }

    def _process_segment(self, chunk: np.ndarray) -> dict:
        """Clean one segment with ZapLine's spectral-QA retry loop.

        Overrides :meth:`DSS._process_segment`. Everything around this — the
        segmentation, the cross-fade, the per-segment bookkeeping — is handled
        by :meth:`DSS._run_segmented`; only the ZapLine-specific behaviour
        lives here:

        1. refine the coarse target frequency to a fine peak,
        2. check whether the artifact is actually present in this segment,
        3. loop, adjusting sigma and the removal floor until
           :func:`check_spectral_qa` reports neither over- nor under-cleaning,
        4. optionally fall back to a notch filter if cleaning stays weak.

        Parameters
        ----------
        chunk : ndarray, shape (n_channels, n_times)
            Data segment supplied by the segmented engine.

        Returns
        -------
        result : dict
            The keys :meth:`DSS._process_segment` promises, plus ZapLine's
            ``'fine_freq'`` and ``'artifact_present'`` diagnostics, which the
            engine stores verbatim in :attr:`segment_results_`.
        """
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
                }
            )

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
