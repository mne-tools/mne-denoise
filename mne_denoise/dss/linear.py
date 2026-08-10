"""Core linear DSS algorithm and Estimator.

This module contains:
1. `compute_dss`: The core mathematical implementation of Linear DSS.
2. `DSS`: The Scikit-learn estimator compatible with MNE-Python objects or NumPy arrays.

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)

References
----------
.. [1] Särelä & Valpola (2005). Denoising Source Separation. J. Mach. Learn. Res., 6, 233-272.
.. [2] de Cheveigné & Simon (2008). Denoising based on spatial filtering. J. Neurosci. Methods.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone

# Optional MNE support
try:
    import mne
    from mne.epochs import BaseEpochs
    from mne.evoked import Evoked
    from mne.io import BaseRaw
except ImportError:
    mne = None

from .._covariance import compute_covariance
from .._logging import set_log_level_from_verbose
from .._spatial import apply_spatial_transform
from ..blending import overlap_add_combine
from ..utils import extract_data_from_mne, reconstruct_mne_object
from .denoisers import LinearDenoiser
from .denoisers.temporal import SmoothingBias
from .utils.segmentation import CovarianceSegmenter, FixedWindowSegmenter
from .utils.selection import auto_select_components_robust
from .utils.whitening import (
    apply_covariance_transform,
    compute_data_covariance_whitener,
    compute_mne_sensor_whitener,
    map_spatial_matrices_to_sensor_space,
)

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 1. Core Algorithm
# -----------------------------------------------------------------------------


def compute_dss(
    covariance_baseline: np.ndarray,
    covariance_biased: np.ndarray,
    *,
    n_components: int | None = None,
    rank: int | None = None,
    reg: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Compute DSS spatial filters from baseline and biased covariances.

    This implements the core Linear DSS algorithm as described in Särelä & Valpola (2005) [1]_.

    The algorithm finds a linear transform (spatial filters) that maximizes the
    biased variance (signal) relative to total/baseline variance (noise).

    The process corresponds to Equation 7 in de Cheveigné & Simon (2008) [2]_:

    .. math:: \\tilde{S}(t) = P Q R_2 N_2 R_1 N_1 S(t)

    where:

    *   **N1** (Initial Normalization): Handled externally (e.g. ``DSS(normalize_input=True)``).
        Ensures equal weight for each sensor.
    *   **R1** (First PCA): Rotation derived from baseline covariance (Sphering/Whitening PCA).
        Discards components with negligible power.
    *   **N2** (Whitening): Normalization to obtain orthonormal "spatially whitened" vectors.
    *   **R2** (Second PCA): Rotation derived from biased covariance in the whitened space.
    *   **Q** (Selector): Selection of the top ``n_components`` with highest bias score.
    *   **P** (Projection): Projection back to sensor space (Spatial Patterns).

    Parameters
    ----------
    covariance_baseline : ndarray
        Baseline covariance.
    covariance_biased : ndarray
        Biased covariance.
    n_components : int, optional
        Number of DSS components to return (The **Q** selector step). If None, return all.
    rank : int, optional
        Rank for whitening stage. If None, auto-determined from data.
    reg : float
        Regularization threshold. Default 1e-9.

    Returns
    -------
    dss_filters : ndarray, shape (n_components, n_channels)
        DSS spatial filters (unmixing matrix transposed).
        Corresponds to the combined transform :math:`Q R_2 N_2 R_1`.
        Apply as: ``sources = dss_filters @ data``.
    dss_patterns : ndarray, shape (n_channels, n_components)
        DSS spatial patterns (mixing matrix).
        Corresponds to the projection matrix **P**.
    eigenvalues : ndarray, shape (n_components,)
        DSS eigenvalues (ratio of biased power to baseline power).

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise import compute_covariance
    >>> from mne_denoise.dss import compute_dss
    >>> # Generate synthetic data (n_channels, n_times)
    >>> data = np.random.randn(10, 1000)
    >>> # Compute covariances
    >>> cov_baseline = compute_covariance(data)
    >>> # Biased covariance: trial-averaged standard example or filtering
    >>> cov_biased = compute_covariance(data)  # Just a placeholder
    >>> # Compute DSS
    >>> filters, patterns, evs = compute_dss(cov_baseline, cov_biased, n_components=5)

    See Also
    --------
    DSS : Estimator class for linear DSS.

    References
    ----------
    .. [1] Särelä, J., & Valpola, H. (2005). Denoising source separation.
           Journal of Machine Learning Research, 6, 233-272.
    .. [2] de Cheveigné, A., & Simon, J. Z. (2008). Denoising based on spatial filtering.
           Journal of Neuroscience Methods, 171(2), 331-339.
    """
    # Check shapes
    if covariance_baseline.shape != covariance_biased.shape:
        raise ValueError(
            f"Covariance shapes mismatch: {covariance_baseline.shape} vs {covariance_biased.shape}"
        )

    n_channels = covariance_baseline.shape[0]
    if covariance_baseline.shape != (n_channels, n_channels):
        raise ValueError(f"Covariance must be square, got {covariance_baseline.shape}")

    # STEP 1 + 2: derive and apply the shared baseline-covariance whitener.
    whitener, _, eigenvalues_white = compute_data_covariance_whitener(
        covariance_baseline,
        rank=rank,
        reg=reg,
    )
    n_keep = eigenvalues_white.size
    max_ev = eigenvalues_white[0]

    if n_keep < n_channels // 4:
        logger.warning(
            "DSS: only %d/%d components kept after rank reduction "
            "(reg=%g, max_eigval=%.3g, smallest_kept_eigval=%.3g). "
            "This is common for MEG data with a large dynamic range "
            "(e.g., raw CTF magnetometers in Tesla). Consider passing "
            "normalize_input=True to DSS, lowering reg, or fitting "
            "homogeneous channel types separately instead of mixing channels "
            "with different physical units.",
            int(n_keep),
            int(n_channels),
            float(reg),
            float(max_ev),
            float(eigenvalues_white[n_keep - 1]),
        )

    covariance_whitened = apply_covariance_transform(whitener, covariance_biased)

    # =========================================================================
    # STEP 3: PCA on whitened covariance_biased -> defines R2
    # =========================================================================
    eigenvalues_biased, eigenvectors_biased = np.linalg.eigh(covariance_whitened)

    # Sort descending
    idx2 = np.argsort(eigenvalues_biased)[::-1]
    eigenvalues_biased = eigenvalues_biased[idx2]
    eigenvectors_biased = eigenvectors_biased[:, idx2]

    # =========================================================================
    # STEP 4: Build DSS matrix (filters = R2 * N2 * R1)
    # =========================================================================
    unmixing_matrix = whitener.T @ eigenvectors_biased

    # =========================================================================
    # STEP 5: Normalize so components have unit variance
    # =========================================================================
    norm_factor = np.diag(unmixing_matrix.T @ covariance_baseline @ unmixing_matrix)
    norm_factor = np.where(norm_factor > 1e-15, norm_factor, 1.0)
    unmixing_matrix = unmixing_matrix @ np.diag(1.0 / np.sqrt(norm_factor))

    # =========================================================================
    # STEP 6: Truncate to n_components
    # =========================================================================
    if n_components is None:
        n_components = unmixing_matrix.shape[1]
    else:
        n_components = min(n_components, unmixing_matrix.shape[1])

    unmixing_matrix = unmixing_matrix[:, :n_components]
    eigenvalues = eigenvalues_biased[:n_components]

    # =========================================================================
    # Convert to our convention: filters are (n_components, n_channels)
    # Corresponds to Q selector on the rows of the combined matrix.
    # =========================================================================
    dss_filters = unmixing_matrix.T

    # DSS patterns (mixing matrix)
    # Note: Patterns are in physical units. Use get_normalized_patterns() for visualization.
    dss_patterns = covariance_baseline @ unmixing_matrix

    return dss_filters, dss_patterns, eigenvalues


# -----------------------------------------------------------------------------
# 2. Scikit-Learn Estimator
# -----------------------------------------------------------------------------


def _as_smoother(smooth: LinearDenoiser | int | None) -> LinearDenoiser | None:
    """Coerce a ``smooth`` parameter value into a denoiser.

    Parameters
    ----------
    smooth : LinearDenoiser | int | None
        ``None`` for no smoothing, an ``int`` window length in samples, or any
        denoiser exposing ``apply()``.

    Returns
    -------
    smoother : LinearDenoiser | None
        ``None`` when ``smooth`` is unset, otherwise a denoiser whose
        ``apply()`` yields the smooth branch of the decomposition.

    Raises
    ------
    TypeError
        If ``smooth`` is neither ``None``, an ``int``, nor ``apply()``-able.
    """
    if smooth is None:
        return None
    if isinstance(smooth, int | np.integer):
        return SmoothingBias(window=int(smooth), iterations=1)
    if hasattr(smooth, "apply"):
        # Covers SmoothingBias and any other LinearDenoiser
        return smooth
    raise TypeError(f"smooth must be SmoothingBias, int, or None, got {type(smooth)}")


class DSS(BaseEstimator, TransformerMixin):
    """Denoising Source Separation (DSS) Transformer.

    Implements DSS as a scikit-learn compatible transformer that fits natively
    on MNE-Python objects (Raw, Epochs, Evoked) or numpy arrays.

    Parameters
    ----------
    n_components : int, optional
        Number of DSS components to keep. If None, keep all.
    bias : LinearDenoiser
        Bias function to define the signal of interest. Must be an instance of
        `mne_denoise.dss.LinearDenoiser` (e.g. `BandpassBias`, `TrialAverageBias`)
        or a callable that takes data and returns biased data.
    n_select : int | 'auto' | None, default=None
        Number of significant components to auto-select after fitting.
        If ``'auto'``, :meth:`auto_select` determines the count via
        :func:`~mne_denoise.dss.utils.selection.auto_select_components_robust`
        and stores it in :attr:`n_selected_`.
        If ``int``, uses that exact number.
        If ``None`` (default), no automatic selection is performed — except
        when ``adaptive=True``, where it defaults to ``'auto'`` because
        per-segment adaptation is the whole point of that mode.
    selection_threshold : float, default=3.0
        Sigma threshold for the outlier arm of automatic selection:
        components with ``eigenvalue > mean + sigma * std`` are significant.
        Same meaning as ``ZapLine(threshold=...)``.
    knee_rel_floor : float, default=0.01
        Relative floor for the knee arm of automatic selection. Eigenvalues
        below this fraction of the largest are not considered valid knee
        anchors. Same meaning as ``ZapLine(knee_rel_floor=...)``.
    knee_min_ratio : float, default=3.0
        Minimum drop ratio required to qualify as a knee, so that smoothly
        decaying (artifact-free) spectra select nothing. Same meaning as
        ``ZapLine(knee_min_ratio=...)``.
    rank : int or dict, optional
        Rank of the data for whitening. If None, rank is estimated automatically.
    reg : float
        Regularization for covariance estimation. Default 1e-9.
    normalize_input : bool
        If True, normalize input data channel-wise (L2 norm) before fitting/transforming.
        Useful when mixing sensors with different scales (e.g. MAG and GRAD). Default True.
        Ignored when ``whiten=True`` (the whitener handles the scaling).
    cov_method : str
        Method for covariance estimation.
        For MNE objects, passed as `method` to `mne.compute_covariance`.
        For NumPy arrays, selects the internal array covariance estimator.
        Default 'empirical'.
    cov_kws : dict, optional
        Additional keywords options for covariance estimation.
        For MNE objects, passed to `mne.compute_covariance` (e.g. `{'tstep': 0.1, 'rank': 'info'}`).
        For NumPy arrays, passed to the internal array covariance estimator
        (e.g. ``{'shrinkage': 0.1}``).
    smooth : SmoothingBias | int | None, default=None
        Optional smoothing decomposition before DSS, inspired by ZapLine.
        When set, data is decomposed into ``smooth + residual`` and DSS
        is fitted/applied on the **residual** only.  This dramatically
        increases eigenvalue contrast for narrowband artifacts because
        DSS no longer competes against broadband EEG variance.

        - If ``SmoothingBias`` instance: used directly.
        - If ``int``: interpreted as the smoothing window in samples
          (e.g., ``int(sfreq / line_freq)`` for line noise).
        - If ``None`` (default): no smoothing, DSS is applied to the
          full data (original behavior).
    adaptive : bool, default=False
        If ``True``, data is split into segments and DSS is fitted
        independently per segment.  This handles **non-stationary**
        artifacts whose spatial or spectral profile changes over
        time.  The per-segment pathway runs in :meth:`fit_transform`;
        :meth:`fit` still produces a single global fit.

        This is the same switch :class:`~mne_denoise.zapline.ZapLine`
        exposes as ``adaptive``, which inherits this parameter directly.
    segmenter : CovarianceSegmenter | FixedWindowSegmenter | None, default=None
        Segmentation strategy.  If ``None`` and ``adaptive=True``,
        a :class:`CovarianceSegmenter` is created automatically
        (requires ``sfreq`` to be determinable from the input or
        from the bias function).
    crossfade : float, default=0.0
        Duration (in seconds) of the cross-fade at segment boundaries
        when ``adaptive=True``.  Adjacent segments are extended by
        this amount on each side, cleaned independently, then blended
        using a raised-cosine (Hann) overlap-add window.  This
        eliminates discontinuities at segment boundaries.
        If ``0.0`` (default), segments are hard-concatenated, matching
        ZapLine-plus, which concatenates cleaned chunks directly
        (Klug & Kloosterman, 2022); the cross-fade is an ``mne-denoise``
        addition for smoother boundaries.  Typical values: ``0.5`` – ``2.0`` s.
    max_prop_remove : float | None, default=None
        Maximum proportion of channels that can be removed per segment.
        E.g. ``0.2`` caps ``n_selected`` at ``int(n_channels × 0.2)``.
        Safety valve to prevent over-cleaning; mirrors ZapLine-plus, which
        caps the automatic component count at one-fifth of the channels
        (Klug & Kloosterman, 2022, §2.4).
    min_select : int, default=0
        Minimum components to select when ``n_select='auto'`` and
        the artifact is present.  Guarantees a floor on cleaning
        strength.  Only effective when ``adaptive=True``.  Mirrors
        ZapLine-plus's fixed-removal floor (``fixedNremove``; Klug &
        Kloosterman, 2022).
    return_type : {'sources', 'epochs', 'raw'}
        Type of object to return from `transform`. 'sources' returns a numpy array
        of DSS components. 'epochs'/'raw' returns the denoised input object.
    whiten : bool, default=False
        If True, decompose all data channel types jointly (e.g. mag + grad + eeg)
        instead of isolating a single homogeneous type. The data is whitened
        before the DSS bias/covariance step and un-whitened on reconstruction, so
        channels with different physical units no longer contaminate one another.
    noise_cov : mne.Covariance | None, default=None
        Noise covariance used to build the whitener when ``whiten=True`` (MNE
        inputs only). If None, MNE inputs are scaled by channel type, matching
        MNE's ICA pre-whitening fallback; NumPy arrays are scaled per channel.
        Ignored when ``whiten=False``.
    verbose : bool | str | int | None, default=None
        Control logging verbosity.

    Attributes
    ----------
    filters_ : array, shape (n_components, n_channels)
        The spatial filters (un-mixing matrix).
    patterns_ : array, shape (n_channels, n_components)
        The spatial patterns (mixing matrix).
    eigenvalues_ : array, shape (n_components,)
        The power of each component in the biased data (bias score).
    n_selected_ : int | None
        Number of significant components detected by automatic selection.
        Only set when ``n_select`` is not ``None``. Use this to determine
        how many components to remove/keep in downstream processing.
    segment_results_ : list of dict | None
        Per-segment metadata when ``adaptive=True``.  Each dict
        contains ``'start'``, ``'end'``, ``'n_selected'``,
        ``'eigenvalues'``, and ``'patterns'``.

    Examples
    --------
    >>> from mne_denoise.dss import DSS, BandpassBias
    >>> from mne_denoise.dss.denoisers import TrialAverageBias
    >>> # Create a bias (e.g. emphasize 10Hz oscillations)
    >>> bias = BandpassBias(sfreq=250, freq=10, bandwidth=2)
    >>> # Initialize DSS
    >>> dss = DSS(bias=bias, n_components=3)
    >>> # Fit on data (MNE Raw/Epochs or NumPy)
    >>> dss.fit(raw_data)
    >>> # Extract sources
    >>> sources = dss.transform(raw_data)
    >>> # Or return denoised data
    >>> dss.return_type = "raw"
    >>> denoised_raw = dss.transform(raw_data)

    See Also
    --------
    compute_dss : Functional interface for computing DSS solutions.
    """

    def __init__(
        self,
        bias: LinearDenoiser | Callable,
        n_components: int | None = None,
        n_select: int | str | None = None,
        selection_threshold: float = 3.0,
        knee_rel_floor: float = 0.01,
        knee_min_ratio: float = 3.0,
        rank: int | dict | None = None,
        reg: float = 1e-9,
        normalize_input: bool = True,
        cov_method: str = "empirical",
        cov_kws: dict | None = None,
        smooth: LinearDenoiser | int | None = None,
        adaptive: bool = False,
        segmenter: CovarianceSegmenter | FixedWindowSegmenter | None = None,
        crossfade: float = 0.0,
        max_prop_remove: float | None = None,
        min_select: int = 0,
        return_type: str = "sources",
        whiten: bool = False,
        noise_cov=None,
        verbose: bool | str | int | None = None,
    ) -> None:
        self.n_components = n_components
        self.bias = bias
        self.n_select = n_select
        self.selection_threshold = selection_threshold
        self.knee_rel_floor = knee_rel_floor
        self.knee_min_ratio = knee_min_ratio
        self.rank = rank
        self.reg = reg
        self.normalize_input = normalize_input
        self.cov_method = cov_method
        self.cov_kws = cov_kws
        self.smooth = smooth
        self.adaptive = adaptive
        self.segmenter = segmenter
        self.crossfade = crossfade
        self.max_prop_remove = max_prop_remove
        self.min_select = min_select
        self.return_type = return_type
        self.whiten = whiten
        self.noise_cov = noise_cov
        self.verbose = verbose
        set_log_level_from_verbose(self.verbose)

        # Fitted attributes
        self.filters_: np.ndarray | None = None
        self.patterns_: np.ndarray | None = None
        self.mixing_: np.ndarray | None = None
        self.eigenvalues_: np.ndarray | None = None
        self.explained_variance_: np.ndarray | None = None
        self.channel_norms_: np.ndarray | None = None
        self.n_selected_: np.ndarray | None = None
        self.segment_results_: list | None = None
        self._whitener_: np.ndarray | None = None
        self._dewhitener_: np.ndarray | None = None
        self._smoother = None  # Resolved SmoothingBias instance
        self._mne_info = None

    def fit(
        self,
        X: BaseRaw | BaseEpochs | Evoked | np.ndarray,
        y=None,
        weights: np.ndarray | None = None,
    ) -> DSS:
        """Compute DSS spatial filters.

        Parameters
        ----------
        X : Raw | Epochs | Evoked | array
            The data to fit.
            - If array, shape must be:
              - `(n_channels, n_times)` for continuous data.
              - `(n_channels, n_times, n_epochs)` for epoch data (evoked DSS).
              - `(n_datasets, n_channels, n_times)` for group data (Joint DSS).
            Note: For group DSS, you must reshape your list of datasets into a 3D array before fitting.
        y : None
            Ignored.
        weights : array, shape (n_times,), optional
             Sample weights for covariance computation. Only used if input is numpy array
             or if internal logic supports weighted covariance for MNE objects.

        Returns
        -------
        self : DSS
            The fitted transformer.
        """
        set_log_level_from_verbose(self.verbose)
        if self.adaptive:
            logger.info(
                "DSS(adaptive=True).fit() computes a single global fit. "
                "Call fit_transform() for the per-segment adaptive pathway."
            )

        if self.whiten:
            # Joint multi-sensor decomposition: the whitener replaces the
            # channel-wise normalization and the homogeneous-type isolation.
            self._fit_whitened(X, weights=weights)
            self.mixing_ = self.patterns_
            return self

        if self.normalize_input:
            X_norm = self._normalize(X, fit=True)
        else:
            X_norm = X

        # Resolve smoothing (if configured)
        self._smoother = _as_smoother(self.smooth)

        # If smoothing is enabled, decompose and fit on the residual only
        if self._smoother is not None:
            data, _, _, _, _, _ = extract_data_from_mne(
                X_norm, channel_first_epochs=True
            )
            data_residual = data - self._smoother.apply(data)
            # Fit DSS on residual (always numpy path)
            self._fit_numpy(data_residual, weights=weights)
        elif mne is not None and isinstance(X_norm, BaseRaw | BaseEpochs | Evoked):
            self._fit_mne(X_norm, weights=weights)
        elif isinstance(X_norm, np.ndarray):
            self._fit_numpy(X_norm, weights=weights)
        else:
            raise TypeError(f"Unsupported input type: {type(X_norm)}")

        # Compute mixing matrix
        # self.patterns_ from compute_dss already satisfy X = P @ S
        self.mixing_ = self.patterns_

        # Automatic component selection
        if self._effective_n_select() is not None and self.eigenvalues_ is not None:
            self.n_selected_ = self.auto_select()

        return self

    def _effective_n_select(self) -> int | str | None:
        """Resolve ``n_select``, defaulting to ``'auto'`` in adaptive mode.

        Adaptive mode exists to adapt the number of removed components to
        each segment, so ``n_select=None`` there would silently clean nothing
        and return the input unchanged. ZapLine's adaptive path makes the same
        choice by hardcoding ``n_select='auto'``.

        Returns
        -------
        n_select : int | 'auto' | None
        """
        if self.n_select is None and self.adaptive:
            return "auto"
        if not (
            self.n_select is None
            or self.n_select == "auto"
            or isinstance(self.n_select, int | np.integer)
        ):
            raise ValueError(
                f"n_select must be an int, 'auto', or None, got {self.n_select!r}. "
                "Selection behaviour is tuned via selection_threshold, "
                "knee_rel_floor, and knee_min_ratio."
            )
        return self.n_select

    def auto_select(self, threshold: float | None = None) -> int:
        """Automatically determine how many DSS components are significant.

        Delegates to :func:`~mne_denoise.dss.utils.selection.auto_select_components_robust`,
        which layers two complementary detectors and takes the larger count:

        - :func:`~mne_denoise.dss.utils.selection.iterative_outlier_removal`
          catches the case where a few components stand out as statistical
          outliers (typical EEG, and any spectrum with high contrast such as
          DSS after ``smooth``).
        - :func:`~mne_denoise.dss.utils.selection.detect_eigenvalue_knee`
          catches the case where many co-equal strong components sit above a
          noise floor, where the outlier test returns 0 (typical
          high-channel-count MEG with coherent line noise; see Issue #34).

        On a smoothly-decaying spectrum both return 0, so clean data is left
        untouched. This is the same selector :class:`~mne_denoise.zapline.ZapLine`
        uses for ``n_select='auto'``.

        Called automatically during :meth:`fit` when ``n_select`` is set; can
        also be called manually after fitting to explore a different threshold.

        Parameters
        ----------
        threshold : float | None
            Override the sigma threshold for the outlier detector. If ``None``,
            uses ``self.selection_threshold``.

        Returns
        -------
        n_selected : int
            Number of significant components detected. When ``n_select`` is an
            ``int``, that value is returned instead (clipped to the number of
            available components).

        Raises
        ------
        RuntimeError
            If the estimator has not been fitted yet.

        Examples
        --------
        >>> dss = DSS(bias=my_bias, n_components=30, n_select="auto")
        >>> dss.fit(raw)
        >>> print(f"{dss.n_selected_} significant components")
        >>> dss.auto_select(threshold=2.5)  # explore a looser threshold
        """
        if self.eigenvalues_ is None:
            raise RuntimeError("DSS not fitted. Call fit() first.")

        n_select = self._effective_n_select()
        if isinstance(n_select, int):
            return min(n_select, len(self.eigenvalues_))

        threshold = threshold if threshold is not None else self.selection_threshold

        return int(
            auto_select_components_robust(
                self.eigenvalues_,
                sigma=threshold,
                knee_rel_floor=self.knee_rel_floor,
                knee_min_ratio=self.knee_min_ratio,
            )
        )

    def _normalize(
        self, X: BaseRaw | BaseEpochs | Evoked | np.ndarray, fit: bool = False
    ) -> BaseRaw | BaseEpochs | Evoked | np.ndarray:
        """Normalize data channel-wise.

        This mimics MNE's Scaling capabilities, ensuring channels with different
        units (e.g. MAG vs GRAD) contribute equally.
        """
        # Helper to get numpy data
        is_mne = False
        mne_type = None
        if mne is not None and isinstance(X, BaseRaw | BaseEpochs | Evoked):
            data = X.get_data()
            is_mne = True
            if isinstance(X, BaseEpochs):
                mne_type = "epochs"
                # MNE Epochs: (n_epochs, n_channels, n_times) -> (n_channels, n_times, n_epochs)
                data = np.transpose(data, (1, 2, 0))
            elif isinstance(X, Evoked):
                mne_type = "evoked"
            else:
                mne_type = "raw"
        else:
            data = X

        # Now data is always (n_channels, ...) for both 2D and 3D
        orig_shape = data.shape
        if data.ndim == 3:
            n_ch, n_times, n_epochs = data.shape
            data_2d = data.reshape(n_ch, -1)
        else:
            n_ch, n_times = data.shape
            data_2d = data

        if fit:
            # unique norms per channel
            self.channel_norms_ = np.linalg.norm(data_2d, axis=1)
            # Avoid division by zero
            self.channel_norms_ = np.where(
                self.channel_norms_ > 0, self.channel_norms_, 1.0
            )

        # Apply normalization
        data_norm = data_2d / self.channel_norms_[:, np.newaxis]

        # Reshape back
        if len(orig_shape) == 3:
            data_norm = data_norm.reshape(orig_shape)

        if is_mne:
            if mne_type == "raw":
                out = mne.io.RawArray(data_norm, X.info.copy(), verbose=False)
                # Preserve annotations
                if hasattr(X, "annotations") and X.annotations is not None:
                    out.set_annotations(X.annotations)
                return out
            elif mne_type == "epochs":
                # Transpose back to MNE format: (n_ch, n_times, n_epochs) -> (n_epochs, n_ch, n_times)
                data_norm = np.transpose(data_norm, (2, 0, 1))
                out = mne.EpochsArray(
                    data_norm,
                    X.info.copy(),
                    events=getattr(X, "events", None),
                    tmin=getattr(X, "tmin", 0),
                    event_id=getattr(X, "event_id", None),
                    verbose=False,
                )
                # Preserve metadata
                if hasattr(X, "metadata") and X.metadata is not None:
                    out.metadata = X.metadata.copy()
                return out
            else:  # Evoked
                out = mne.EvokedArray(
                    data_norm,
                    X.info.copy(),
                    tmin=getattr(X, "tmin", 0),
                    comment=getattr(X, "comment", ""),
                    nave=getattr(X, "nave", 1),
                    verbose=False,
                )
                return out
        else:
            return data_norm

    def _apply_bias(self, data: np.ndarray) -> np.ndarray:
        """Apply bias function to data."""
        if hasattr(self.bias, "apply"):
            return self.bias.apply(data)
        else:
            return self.bias(data)

    def _fit_mne(
        self,
        inst: BaseRaw | BaseEpochs | Evoked,
        weights: np.ndarray | None = None,
    ) -> None:
        """Fit using MNE objects."""
        self.info_ = inst.info

        if weights is not None:
            # If weights provided, extract data and use numpy path
            data = inst.get_data()
            self._fit_numpy(data, weights=weights)
            return

        method = self.cov_method
        kws = self.cov_kws.copy() if self.cov_kws else {}
        # Set defaults if not in kws
        kws.setdefault("rank", self.rank)
        kws.setdefault("verbose", False)

        data, _, _, _, picks, ch_names = extract_data_from_mne(
            inst, channel_first_epochs=True
        )
        self._mne_ch_names_ = ch_names

        # MNE covariance computation requires the inst object to match the array
        if picks is not None:
            inst = inst.copy().pick(picks)

        biased_data = self._apply_bias(data)

        if isinstance(inst, BaseEpochs):
            biased_data = np.transpose(biased_data, (2, 0, 1))

        if isinstance(inst, BaseRaw):
            kws.setdefault("tstep", 2.0)
            baseline_cov = mne.compute_raw_covariance(inst, method=method, **kws)
            biased_inst = mne.io.RawArray(biased_data, inst.info, verbose=False)
            biased_cov = mne.compute_raw_covariance(biased_inst, method=method, **kws)

        elif isinstance(inst, BaseEpochs):
            baseline_cov = mne.compute_covariance(inst, method=method, **kws)
            biased_inst = mne.EpochsArray(biased_data, inst.info, verbose=False)
            biased_cov = mne.compute_covariance(biased_inst, method=method, **kws)

        else:  # Evoked - use numpy path since MNE doesn't support Evoked covariance
            self._fit_numpy(data, weights=weights)
            return

        # Extract data from MNE covariances
        self.filters_, self.patterns_, self.eigenvalues_ = compute_dss(
            covariance_baseline=baseline_cov.data,
            covariance_biased=biased_cov.data,
            n_components=self.n_components,
            reg=self.reg,
        )

        # Calculate explained variance from filters and baseline covariance
        # Diag(filters @ baseline_cov.data @ filters.T)
        sources_cov = self.filters_ @ baseline_cov.data @ self.filters_.T
        self.explained_variance_ = np.diag(sources_cov)

    def _fit_numpy(self, X: np.ndarray, weights: np.ndarray | None = None) -> None:
        """Fit using numpy arrays."""
        biased_X = self._apply_bias(X)

        method = self.cov_method
        kws = self.cov_kws.copy() if self.cov_kws else {}

        baseline_cov = compute_covariance(X, method=method, weights=weights, **kws)
        biased_cov = compute_covariance(biased_X, method=method, weights=weights, **kws)

        # Use rank if provided (compute from covariance if not)
        rank = None
        if self.rank is not None and isinstance(self.rank, int):
            rank = self.rank
            # If rank is a dict (MNE style), ignore for numpy

        self.filters_, self.patterns_, self.eigenvalues_ = compute_dss(
            covariance_baseline=baseline_cov,
            covariance_biased=biased_cov,
            n_components=self.n_components,
            rank=rank,
            reg=self.reg,
        )

        # Calculate explained variance
        sources_cov = self.filters_ @ baseline_cov @ self.filters_.T
        self.explained_variance_ = np.diag(sources_cov)

    def _fit_whitened(
        self,
        X: BaseRaw | BaseEpochs | Evoked | np.ndarray,
        weights: np.ndarray | None = None,
    ) -> None:
        """Fit DSS on all data channels jointly after whitening.

        The whitener ``W`` is baked into ``filters_`` and its inverse into
        ``patterns_``/``mixing_`` so that ``transform`` and ``inverse_transform``
        operate in sensor units without any further change.
        """
        method = self.cov_method
        kws = self.cov_kws.copy() if self.cov_kws else {}
        # The NumPy covariance helper does not accept MNE-only options.
        for key in ("rank", "verbose", "tstep"):
            kws.pop(key, None)

        data, _, _, orig_inst, _, ch_names = extract_data_from_mne(
            X,
            auto_pick="data",
            channel_first_epochs=True,
        )
        self._mne_ch_names_ = ch_names
        self.info_ = orig_inst.info if orig_inst is not None else None
        self._mne_info = self.info_

        data_w = self._prewhiten_sensor_data(
            data,
            info=self.info_,
            ch_names=ch_names,
        )
        biased_w = self._apply_bias(data_w)

        baseline_cov = compute_covariance(data_w, method=method, weights=weights, **kws)
        biased_cov = compute_covariance(biased_w, method=method, weights=weights, **kws)

        rank = self.rank if isinstance(self.rank, int) else None
        filters_w, patterns_w, self.eigenvalues_ = compute_dss(
            baseline_cov,
            biased_cov,
            n_components=self.n_components,
            rank=rank,
            reg=self.reg,
        )

        # Store the fitted spatial matrices in the original sensor coordinates.
        self.filters_, self.patterns_ = map_spatial_matrices_to_sensor_space(
            filters_w,
            patterns_w,
            whitener=self._whitener_,
            dewhitener=self._dewhitener_,
        )
        self.explained_variance_ = np.diag(filters_w @ baseline_cov @ filters_w.T)

    def _prewhiten_sensor_data(
        self,
        data: np.ndarray,
        *,
        info=None,
        ch_names: list[str] | None = None,
    ) -> np.ndarray:
        """Fit the configured sensor whitener and apply it to data."""
        whitener, dewhitener = compute_mne_sensor_whitener(
            data,
            info=info,
            ch_names=ch_names,
            noise_cov=self.noise_cov,
            rank=self.rank,
        )
        self._whitener_ = whitener
        self._dewhitener_ = dewhitener
        return apply_spatial_transform(whitener, data)

    def transform(
        self, X: BaseRaw | BaseEpochs | Evoked | np.ndarray
    ) -> np.ndarray | BaseRaw | BaseEpochs | Evoked:
        """Apply DSS spatial filters.

        Parameters
        ----------
        X : Raw | Epochs | Evoked | array
            Data to transform.
            - If array, must match the shape convention used in fit (see fit docstring).

        Returns
        -------
        out : array | Raw | Epochs | Evoked
            If return_type='sources', returns the source time series.
            If return_type='raw'/'epochs'/'evoked', returns the reconstructed data (denoised)
            projected back to sensor space (keeping n_components).
        """
        set_log_level_from_verbose(self.verbose)
        if self.filters_ is None:
            raise RuntimeError("DSS not fitted. Call fit() first.")

        if self.normalize_input and not self.whiten:
            # Apply normalization using fitted norms
            X_in = self._normalize(X, fit=False)
        else:
            X_in = X

        # Helper to extract data
        # DSS internal convention for Epochs: (n_channels, n_times, n_epochs)
        data, _, mne_type, orig_inst, picks, _ = extract_data_from_mne(
            X_in,
            ch_names=getattr(self, "_mne_ch_names_", None),
            channel_first_epochs=True,
        )

        # If smoothing is enabled, project the residual (not full data)
        if self._smoother is not None:
            data_smooth = self._smoother.apply(data)
            data_for_dss = data - data_smooth
        else:
            data_smooth = None
            data_for_dss = data

        orig_shape = data.shape
        if data_for_dss.ndim == 3:
            n_ch, n_times, n_epochs = data_for_dss.shape
            data_2d = data_for_dss.reshape(n_ch, -1)
        else:
            n_ch, n_times = data_for_dss.shape
            data_2d = data_for_dss

        # Center using mean on data_2d
        # DSS implies zero-mean assumption for correct projection
        mean_ = data_2d.mean(axis=1, keepdims=True)
        data_centered = data_2d - mean_

        sources = self.filters_ @ data_centered

        if self.return_type == "sources":
            if len(orig_shape) == 3:
                sources = sources.reshape(
                    self.n_components or sources.shape[0], n_times, n_epochs
                )
                if mne_type == "epochs":
                    # Return as (n_epochs, n_components, n_times)
                    return sources.transpose(2, 0, 1)
            return sources

        # Use only kept components
        n_keep = self.n_components if self.n_components else self.filters_.shape[0]
        # mixing shape: (n_channels, n_components)
        rec = self.mixing_[:, :n_keep] @ sources[:n_keep]
        rec += mean_

        # Add back smooth component if it was separated
        if data_smooth is not None:
            smooth_2d = (
                data_smooth.reshape(data_smooth.shape[0], -1)
                if data_smooth.ndim == 3
                else data_smooth
            )
            rec = rec + smooth_2d

        # Reshape to original
        if len(orig_shape) == 3:
            rec = rec.reshape(orig_shape)  # (n_ch, n_times, n_epochs)

        # De-normalization
        if self.normalize_input and not self.whiten:
            if len(orig_shape) == 3:  # (n_ch, n_times, n_epochs)
                rec = rec * self.channel_norms_[:, np.newaxis, np.newaxis]
            else:  # (n_ch, n_times)
                rec = rec * self.channel_norms_[:, np.newaxis]

        # Prepare for reconstruction (transpose back if needed)
        if mne_type == "epochs":
            rec = np.transpose(rec, (2, 0, 1))

        return reconstruct_mne_object(
            rec, orig_inst, mne_type, picks=picks, verbose=False
        )

    def inverse_transform(
        self, sources: np.ndarray, component_indices: np.ndarray | None = None
    ) -> np.ndarray:
        """Transform sources back to sensor space.

        Parameters
        ----------
        sources : array, shape (n_components, n_times)
            The latent sources.
        component_indices : array-like of bool or int, optional
            Indices of components to keep. If None, keep all.

        Returns
        -------
        reconstructed : array, shape (n_channels, n_times)
            The reconstructed sensor space data.
        """
        if self.filters_ is None:
            raise RuntimeError("DSS not fitted. Call fit() first.")
        is_epochs_mne = False

        if sources.ndim == 3:
            # Determine orientation: sources from transform() are
            # (n_comps, n_times, n_epochs) for numpy or (n_epochs, n_comps, n_times) for MNE epochs
            # Use shape[0] vs mixing_.shape[1] to detect MNE epoch format
            n_comp_fit = self.mixing_.shape[1]
            if sources.shape[0] != n_comp_fit and sources.shape[1] == n_comp_fit:
                # MNE epochs format: (n_epochs, n_comps, n_times) -> (n_comps, n_times, n_epochs)
                sources_internal = np.transpose(sources, (1, 2, 0))
                is_epochs_mne = True
            else:
                sources_internal = sources
        else:
            sources_internal = sources

        n_comp_sources = sources_internal.shape[0]
        patterns = self.mixing_[:, :n_comp_sources]

        if component_indices is not None:
            # Make a copy to avoid modifying input
            sources_used = sources_internal.copy()
            mask = np.array(component_indices)

            # Handle boolean mask
            if mask.dtype == bool:
                if len(mask) != n_comp_sources:
                    raise ValueError(
                        f"Mask length {len(mask)} != n_sources {n_comp_sources}"
                    )
                sources_used[~mask] = 0
            else:
                # Handle integer indices
                # Create a boolean mask from indices
                full_mask = np.zeros(n_comp_sources, dtype=bool)
                full_mask[mask] = True
                sources_used[~full_mask] = 0

            rec_internal = np.tensordot(patterns, sources_used, axes=(1, 0))
        else:
            rec_internal = np.tensordot(patterns, sources_internal, axes=(1, 0))

        if is_epochs_mne:
            # rec_internal: (n_ch, n_times, n_epochs) -> (n_epochs, n_ch, n_times)
            rec = np.transpose(rec_internal, (2, 0, 1))
        else:
            rec = rec_internal

        if self.normalize_input and not self.whiten:
            # rec is (n_epochs, n_ch, n_times) OR (n_ch, n_times, n_epochs) OR (n_ch, n_times)
            if is_epochs_mne:
                rec = rec * self.channel_norms_[np.newaxis, :, np.newaxis]
            elif rec.ndim == 3:  # (n_ch, n_times, n_epochs)
                rec = rec * self.channel_norms_[:, np.newaxis, np.newaxis]
            else:  # (n_ch, n_times)
                rec = rec * self.channel_norms_[:, np.newaxis]

        return rec

    def get_normalized_patterns(self) -> np.ndarray:
        """Get L2-normalized spatial patterns for visualization.

        Returns
        -------
        patterns_norm : ndarray, shape (n_channels, n_components)
            L2-normalized spatial patterns.
        """
        if self.patterns_ is None:
            raise RuntimeError("DSS not fitted. Call fit() first.")

        norms = np.linalg.norm(self.patterns_, axis=0)
        # Use relative threshold for physical units
        max_norm = np.max(norms)
        threshold = 1e-15 * max_norm if max_norm > 0 else 1e-30
        norms = np.where(norms > threshold, norms, 1.0)
        return self.patterns_ / norms

    # -----------------------------------------------------------------
    # Segmented mode
    # -----------------------------------------------------------------

    def fit_transform(self, X, y=None, **fit_params):
        """Fit and transform data in one step.

        In **adaptive mode** (``adaptive=True``), the data is split into
        segments and each segment gets its own independent DSS fit +
        cleaning pass.  This is the only entry-point for adaptive
        processing because ``fit()`` alone is not meaningful when
        filters differ per segment.

        In standard mode, this is equivalent to
        ``self.fit(X).transform(X)``.

        Parameters
        ----------
        X : Raw | Epochs | Evoked | ndarray
            The data to process.
        y : None
            Ignored.
        **fit_params
            Additional keyword arguments forwarded to :meth:`fit`.

        Returns
        -------
        X_out : ndarray | Raw | Epochs | Evoked
            In adaptive mode, returns cleaned data (same type as input).
            In standard mode with ``return_type='sources'``, returns DSS
            source time-series.  With any other ``return_type``, returns
            cleaned (denoised) data produced by subtracting the artifact
            captured by the first ``n_selected_`` components.
        """
        if not self.adaptive:
            self.fit(X, **fit_params)

            if self.return_type == "sources":
                return self.transform(X)

            # ── Denoise via artifact subtraction ──
            data, _, mne_type, orig_inst, _, _ = extract_data_from_mne(X)

            n_remove = self.n_selected_ if self.n_selected_ is not None else 0
            if n_remove > 0:
                # Temporarily switch to get source time-series
                saved_rt = self.return_type
                self.return_type = "sources"
                try:
                    sources = self.transform(X)
                finally:
                    self.return_type = saved_rt

                artifact = self.inverse_transform(
                    sources, component_indices=np.arange(n_remove)
                )
                cleaned = data - artifact
            else:
                cleaned = data

            return reconstruct_mne_object(cleaned, orig_inst, mne_type, verbose=False)

        # --- adaptive (per-segment) mode ---
        data, extracted_sfreq, mne_type, orig_inst, _, _ = extract_data_from_mne(X)

        # Determine sfreq
        sfreq = extracted_sfreq
        if sfreq is None and hasattr(self.bias, "sfreq"):
            sfreq = self.bias.sfreq
        if sfreq is None:
            raise ValueError(
                "Cannot determine sfreq for adaptive mode. "
                "Pass an MNE object or use a bias with a .sfreq attribute."
            )

        # Handle epochs: concatenate into continuous
        is_epochs = False
        if data.ndim == 3:
            is_epochs = True
            n_ep, n_ch, n_t = data.shape
            data_cont = np.transpose(data, (1, 0, 2)).reshape(n_ch, -1)
        else:
            data_cont = data

        # Resolve smoother once
        self._smoother = _as_smoother(self.smooth)

        # A global fit over the whole recording populates the estimator-level
        # attributes (filters_, patterns_, eigenvalues_) with something that
        # describes all of the data. Per-segment results are kept separately in
        # segment_results_.
        global_est = self._make_segment_estimator()
        global_est.fit(data_cont)
        self.filters_ = global_est.filters_
        self.patterns_ = global_est.patterns_
        self.mixing_ = global_est.patterns_
        self.eigenvalues_ = global_est.eigenvalues_
        self.explained_variance_ = global_est.explained_variance_
        self.channel_norms_ = global_est.channel_norms_

        # Run segmented processing
        cleaned = self._run_segmented(data_cont, sfreq)

        # Reshape back if epochs
        if is_epochs:
            cleaned = cleaned.reshape(n_ch, n_ep, n_t).transpose(1, 0, 2)

        return reconstruct_mne_object(cleaned, orig_inst, mne_type, verbose=False)

    def _resolve_segmenter(self, sfreq: float):
        """Resolve the segmenter parameter.

        If ``self.segmenter`` is ``None``, creates a default
        :class:`CovarianceSegmenter` with optional bandpass from the
        bias function.

        Parameters
        ----------
        sfreq : float
            Sampling frequency in Hz.

        Returns
        -------
        segmenter : CovarianceSegmenter | FixedWindowSegmenter
        """
        if self.segmenter is not None:
            return self.segmenter

        # Build a default CovarianceSegmenter
        bandpass = None
        # If the bias has a target frequency, focus segmentation around it
        if hasattr(self.bias, "freq") and self.bias.freq is not None:
            f = float(self.bias.freq)
            bandpass = (max(1.0, f - 3), min(sfreq / 2 - 1, f + 3))

        return CovarianceSegmenter(
            sfreq=sfreq,
            min_chunk_len=30.0,
            bandpass=bandpass,
        )

    def _run_segmented(
        self,
        data: np.ndarray,
        sfreq: float,
        segmenter: CovarianceSegmenter | FixedWindowSegmenter | None = None,
    ) -> np.ndarray:
        """Run segmented fit-transform on continuous data.

        This is the shared engine for every adaptive denoiser in the package.
        It owns segmentation, the per-segment loop, the cap/floor policy, the
        cross-fade, and the bookkeeping in :attr:`segment_results_`. Subclasses
        customise *what happens inside a segment* by overriding
        :meth:`_process_segment` — see :class:`~mne_denoise.zapline.ZapLine`,
        which adds spectral QA there without reimplementing any of this.

        When :attr:`crossfade` is positive and there are multiple segments,
        adjacent segments are extended by ``crossfade`` seconds on each side
        and combined by raised-cosine overlap-add, eliminating the boundary
        discontinuities that hard concatenation produces.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_times)
            Continuous data.
        sfreq : float
            Sampling frequency.
        segmenter : CovarianceSegmenter | FixedWindowSegmenter | None
            Explicit segmenter, overriding :attr:`segmenter` for this call.
            ZapLine uses this to re-segment around each target frequency.

        Returns
        -------
        cleaned : ndarray, shape (n_channels, n_times)
            Cleaned data (segments blended via cross-fade or concatenated).
        """
        if segmenter is None:
            segmenter = self._resolve_segmenter(sfreq)
        segments = segmenter.segment(data)

        if not segments:
            raise ValueError(
                "Segmenter returned no segments. Check segmenter settings "
                "and data length."
            )

        logger.info(
            f"Segmented DSS: {len(segments)} segment(s) "
            f"over {data.shape[1] / sfreq:.1f}s"
        )

        # ------ cross-fade setup ------
        n_overlap = int(self.crossfade * sfreq) if self.crossfade > 0 else 0
        _n_ch, n_times = data.shape
        use_crossfade = n_overlap > 0 and len(segments) > 1

        if use_crossfade:
            min_seg_len = min(end - start for start, end in segments)
            if n_overlap > min_seg_len // 2:
                n_overlap = max(1, min_seg_len // 2)
                logger.warning(
                    f"Crossfade overlap clamped to {n_overlap} samples "
                    f"({n_overlap / sfreq:.2f}s) — half the smallest "
                    f"segment."
                )

        # ------ per-segment processing ------
        self.segment_results_ = []
        cleaned_chunks: list[dict] = []
        per_segment_n_removed: list[int] = []

        for seg_idx, (start, end) in enumerate(segments):
            # Optionally extend boundaries for cross-fade context
            if use_crossfade:
                is_first = seg_idx == 0
                is_last = seg_idx == len(segments) - 1
                ext_start = start if is_first else max(0, start - n_overlap)
                ext_end = end if is_last else min(n_times, end + n_overlap)
            else:
                ext_start, ext_end = start, end

            chunk = data[:, ext_start:ext_end]
            result = self._process_segment(chunk)

            cleaned_chunks.append(
                {
                    "data": result["cleaned"],
                    "ext_start": ext_start,
                    "ext_end": ext_end,
                    "start": start,
                    "end": end,
                }
            )
            per_segment_n_removed.append(result["n_selected"])

            # Store per-segment metadata. Any extra keys a subclass adds to
            # the result (e.g. ZapLine's fine_freq / artifact_present) are
            # carried through untouched.
            meta = {k: v for k, v in result.items() if k != "cleaned"}
            self.segment_results_.append({"start": start, "end": end, **meta})

        # Per-segment filters live in ``segment_results_``. The estimator-level
        # ``filters_``/``patterns_``/``eigenvalues_`` come from a single global
        # fit performed by the caller, so they always describe the whole
        # recording rather than whichever segment happened to run last.
        self.n_selected_ = max(per_segment_n_removed) if per_segment_n_removed else 0

        # ------ combine segments ------
        if use_crossfade:
            return overlap_add_combine(data.shape, cleaned_chunks)
        return np.concatenate([c["data"] for c in cleaned_chunks], axis=1)

    def _make_segment_estimator(self) -> DSS:
        """Build the per-segment estimator used by adaptive mode.

        Uses :func:`sklearn.base.clone` so every constructor parameter is
        carried over automatically — including ones added later — and then
        overrides only what must differ for a single segment. Hand-copying
        the parameters here is how ``whiten`` and ``noise_cov`` previously
        went missing without any error.

        Returns
        -------
        estimator : DSS
            An unfitted clone configured for one segment.
        """
        est = clone(self)
        est.set_params(
            adaptive=False,  # do NOT recurse
            # Resolve 'auto' here: the clone is no longer adaptive, so it
            # would otherwise fall back to n_select=None and select nothing.
            n_select=self._effective_n_select(),
            segmenter=None,
            crossfade=0.0,
            return_type="sources",
            # Per-segment caps are applied by the caller, not the clone.
            max_prop_remove=None,
            min_select=0,
            # A dict rank is an MNE-object concept; segments are plain arrays.
            rank=self.rank if isinstance(self.rank, int | type(None)) else None,
        )
        return est

    def _process_segment(self, chunk: np.ndarray) -> dict:
        """Fit, select, and clean one segment (subclass extension point).

        :meth:`_run_segmented` calls this once per segment and owns everything
        around it — segmentation, cross-fade, and bookkeeping. Override this
        (and only this) to change what happens *within* a segment;
        :class:`~mne_denoise.zapline.ZapLine` does exactly that to add its
        spectral-QA retry loop.

        Overrides must return at least the keys documented below. Any
        additional keys are stored verbatim in :attr:`segment_results_`, which
        is how ZapLine surfaces its per-chunk ``fine_freq`` and
        ``artifact_present`` diagnostics.

        Parameters
        ----------
        chunk : ndarray, shape (n_channels, n_times)
            Data segment (already extended for cross-fade, if enabled).

        Returns
        -------
        result : dict
            ``'cleaned'`` (ndarray, same shape as ``chunk``), ``'n_selected'``
            (int), ``'eigenvalues'``, ``'patterns'``, and ``'filters'``.
        """
        n_channels = chunk.shape[0]
        seg_dss = self._make_segment_estimator()
        seg_dss.fit(chunk)
        n_sel = seg_dss.n_selected_ if seg_dss.n_selected_ is not None else 0

        # Apply caps
        if self.max_prop_remove is not None:
            n_sel = min(n_sel, int(n_channels * self.max_prop_remove))
        n_sel = max(n_sel, self.min_select)

        # Clean the segment
        cleaned = self._clean_segment(chunk, seg_dss, n_sel)

        return {
            "cleaned": cleaned,
            "n_selected": n_sel,
            "eigenvalues": seg_dss.eigenvalues_,
            "patterns": seg_dss.patterns_,
            "filters": seg_dss.filters_,
        }

    def _clean_segment(
        self, data: np.ndarray, fitted_dss: DSS, n_remove: int
    ) -> np.ndarray:
        """Clean a segment by projecting out *n_remove* DSS components.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_times)
            Segment data.
        fitted_dss : DSS
            A fitted DSS instance (with ``filters_``, ``mixing_``, etc.).
        n_remove : int
            Number of components to remove.

        Returns
        -------
        cleaned : ndarray, shape (n_channels, n_times)
        """
        if n_remove <= 0 or fitted_dss.filters_ is None:
            return data.copy()

        # Smoothing decomposition (if configured)
        if fitted_dss._smoother is not None:
            data_smooth = fitted_dss._smoother.apply(data)
            data_residual = data - data_smooth
        else:
            data_smooth = np.zeros_like(data)
            data_residual = data

        # Center residual before projection (DSS assumes zero-mean)
        mean_ = data_residual.mean(axis=1, keepdims=True)
        residual_centered = data_residual - mean_

        # Project residual through the top n_remove DSS filters
        sources = fitted_dss.filters_[:n_remove] @ residual_centered
        artifact = fitted_dss.mixing_[:, :n_remove] @ sources

        return data_smooth + (data_residual - artifact)
