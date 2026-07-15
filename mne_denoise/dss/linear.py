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
from sklearn.base import BaseEstimator, TransformerMixin

# Optional MNE support
try:
    import mne
    from mne.epochs import BaseEpochs
    from mne.evoked import Evoked
    from mne.io import BaseRaw
except ImportError:
    mne = None

from .._logging import set_log_level_from_verbose
from ..utils import extract_data_from_mne, reconstruct_mne_object
from .denoisers import LinearDenoiser
from .utils import compute_covariance, compute_evoked_covariance
from .utils.segmentation import CovarianceSegmenter, FixedWindowSegmenter

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
        Note: These are returned in original sensor units (not normalized),
        satisfying the identity :math:`X_{rec} = Patterns \times Sources`.
    eigenvalues : ndarray, shape (n_components,)
        DSS eigenvalues (ratio of biased power to baseline power).

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.dss import compute_dss, compute_covariance
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

    # =========================================================================
    # STEP 1: PCA from covariance_baseline -> defines R1
    # =========================================================================
    covariance_baseline_sym = (covariance_baseline + covariance_baseline.T) / 2
    eigenvalues_white, eigenvectors_white = np.linalg.eigh(covariance_baseline_sym)

    # Sort descending
    idx = np.argsort(eigenvalues_white)[::-1]
    eigenvalues_white = eigenvalues_white[idx]
    eigenvectors_white = eigenvectors_white[:, idx]

    eigenvalues_white = np.abs(eigenvalues_white)

    # Apply threshold
    max_ev = np.max(eigenvalues_white)
    if not np.isfinite(max_ev) or max_ev <= 0:
        raise ValueError("Covariance matrix has no significant variance")

    keep_mask = eigenvalues_white / max_ev > reg

    if rank is not None:
        keep_mask[rank:] = False

    n_keep = np.sum(keep_mask)
    if n_keep == 0:
        raise ValueError("No components above regularization threshold")

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

    eigenvalues_white = eigenvalues_white[keep_mask]
    eigenvectors_white = eigenvectors_white[:, keep_mask]

    # =========================================================================
    # STEP 2: Whitening -> defines N2
    # =========================================================================
    W_white = np.diag(np.sqrt(1.0 / eigenvalues_white))
    covariance_whitened = (
        W_white.T
        @ eigenvectors_white.T
        @ covariance_biased
        @ eigenvectors_white
        @ W_white
    )
    covariance_whitened = (covariance_whitened + covariance_whitened.T) / 2

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
    unmixing_matrix = eigenvectors_white @ W_white @ eigenvectors_biased

    # =========================================================================
    # STEP 5: Normalize so components have unit variance on baseline
    # =========================================================================
    norm_factor = np.diag(unmixing_matrix.T @ covariance_baseline @ unmixing_matrix)
    # Use a relative threshold for robustness across physical units (MEG/EEG)
    max_norm = np.max(norm_factor)
    threshold = 1e-18 * max_norm if max_norm > 0 else 1e-30
    norm_factor = np.where(norm_factor > threshold, norm_factor, 1.0)
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
    n_select : int | 'auto' | None
        Component count used for artifact subtraction by ``fit_transform``.
        ``'auto'`` applies ``selection_method`` to fitted eigenvalues.
    selection_method : {'combined', 'outlier', 'ratio', 'max_gap'}
        Automatic component-selection rule.
    selection_threshold : float
        Threshold for the selected rule.
    rank : int or dict, optional
        Rank of the data for whitening. If None, rank is estimated automatically.
    reg : float
        Regularization for covariance estimation. Default 1e-9.
    normalize_input : bool
        If True, normalize input data channel-wise (L2 norm) before fitting/transforming.
        Useful when mixing sensors with different scales (e.g. MAG and GRAD). Default True.
        Ignored when ``whiten=True``.
    whiten : bool
        If True, jointly decompose all M/EEG data-channel types after spatial
        whitening. This is the explicit multi-modal path; the default continues
        to isolate one homogeneous sensor type. Default False.
    noise_cov : mne.Covariance, optional
        Named noise covariance used when ``whiten=True``. If omitted, a
        diagonal per-channel standard-deviation whitener is fitted from X.
    smooth : SmoothingBias | int | None
        Optional smoothing decomposition; DSS is fitted to the residual.
    segmented : bool
        Fit and clean independently within stationary segments. This mode is
        available only through ``fit_transform``.
    segmenter : CovarianceSegmenter | FixedWindowSegmenter | None
        Explicit segmentation strategy. A covariance segmenter is created when
        omitted in segmented mode.
    crossfade : float
        Seconds of overlap-add crossfade around segment boundaries.
    max_prop_remove : float | None
        Per-segment cap on the proportion of removed components.
    min_select : int
        Per-segment floor on the removed component count.
    cov_method : str
        Method for covariance estimation.
        For MNE objects, passed as `method` to `mne.compute_covariance`.
        For NumPy arrays, passed as `method` to `mne_denoise.utils.compute_covariance`.
        Default 'empirical'.
    cov_kws : dict, optional
        Additional keywords options for covariance estimation.
        For MNE objects, passed to `mne.compute_covariance` (e.g. `{'tstep': 0.1, 'rank': 'info'}`).
        For NumPy arrays, passed to `mne_denoise.utils.compute_covariance` (e.g. `{'shrinkage': 0.1}`).
    return_type : {'sources', 'epochs', 'raw'}
        Type of object to return from `transform`. 'sources' returns a numpy array
        of DSS components. 'epochs'/'raw' returns the denoised input object.

    Attributes
    ----------
    filters_ : array, shape (n_components, n_channels)
        The spatial filters (un-mixing matrix).
    patterns_ : array, shape (n_channels, n_components)
        The spatial patterns (mixing matrix).
    eigenvalues_ : array, shape (n_components,)
        The power of each component in the biased data (bias score).

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
        selection_method: str = "combined",
        selection_threshold: float = 3.0,
        rank: int | dict | None = None,
        reg: float = 1e-9,
        normalize_input: bool = True,
        cov_method: str = "empirical",
        cov_kws: dict | None = None,
        return_type: str = "sources",
        whiten: bool = False,
        noise_cov=None,
        smooth: LinearDenoiser | int | None = None,
        segmented: bool = False,
        segmenter: CovarianceSegmenter | FixedWindowSegmenter | None = None,
        crossfade: float = 0.0,
        max_prop_remove: float | None = None,
        min_select: int = 0,
        verbose: bool | str | int | None = None,
    ) -> None:
        self.n_components = n_components
        self.bias = bias
        self.n_select = n_select
        self.selection_method = selection_method
        self.selection_threshold = selection_threshold
        self.rank = rank
        self.reg = reg
        self.normalize_input = normalize_input
        self.cov_method = cov_method
        self.cov_kws = cov_kws
        self.return_type = return_type
        self.whiten = whiten
        self.noise_cov = noise_cov
        self.smooth = smooth
        self.segmented = segmented
        self.segmenter = segmenter
        self.crossfade = crossfade
        self.max_prop_remove = max_prop_remove
        self.min_select = min_select
        self.verbose = verbose
        set_log_level_from_verbose(self.verbose)

        # Fitted attributes
        self.filters_: np.ndarray | None = None
        self.patterns_: np.ndarray | None = None
        self.mixing_: np.ndarray | None = None
        self.eigenvalues_: np.ndarray | None = None
        self.explained_variance_: np.ndarray | None = None
        self.channel_norms_: np.ndarray | None = None
        self._whitener_: np.ndarray | None = None
        self._unwhitener_: np.ndarray | None = None
        self.n_selected_: int | None = None
        self.segment_results_: list[dict] | None = None
        self._smoother = None
        self._mne_info = None

    def _resolve_smoother(self) -> None:
        from .denoisers.temporal import SmoothingBias

        if self.smooth is None:
            self._smoother = None
        elif isinstance(self.smooth, int):
            self._smoother = SmoothingBias(window=self.smooth, iterations=1)
        elif isinstance(self.smooth, SmoothingBias) or hasattr(self.smooth, "apply"):
            self._smoother = self.smooth
        else:
            raise TypeError(
                "smooth must be SmoothingBias, an integer window, or None; "
                f"got {type(self.smooth)}"
            )

    def _decompose_smooth(self, data: np.ndarray) -> tuple[np.ndarray | None, np.ndarray]:
        if self._smoother is None:
            return None, data
        smooth = self._smoother.apply(data)
        return smooth, data - smooth

    def auto_select(self, threshold=None, method=None) -> int:
        """Select a fitted DSS component count using the configured rule."""
        if self.eigenvalues_ is None:
            raise RuntimeError("DSS not fitted. Call fit() first.")
        from .utils.selection import (
            eigenvalue_ratio_selection,
            iterative_outlier_removal,
            max_gap_selection,
        )

        if isinstance(self.n_select, int):
            return min(max(self.n_select, 0), len(self.eigenvalues_))
        threshold = self.selection_threshold if threshold is None else threshold
        method = self.selection_method if method is None else method
        if method == "outlier":
            return iterative_outlier_removal(self.eigenvalues_, threshold)
        if method == "ratio":
            return eigenvalue_ratio_selection(self.eigenvalues_, threshold)
        if method == "max_gap":
            return max_gap_selection(self.eigenvalues_, min(float(threshold), 1.2))
        if method == "combined":
            count = iterative_outlier_removal(self.eigenvalues_, threshold)
            if count:
                return count
            count = eigenvalue_ratio_selection(
                self.eigenvalues_, min(float(threshold), 2.0)
            )
            return count or max_gap_selection(self.eigenvalues_, 1.2)
        raise ValueError(
            "selection_method must be 'combined', 'outlier', 'ratio', or 'max_gap'"
        )

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
        if self.segmented:
            raise RuntimeError(
                "Segmented mode requires simultaneous fit and transform; use fit_transform()."
            )
        self._resolve_smoother()
        if self.whiten and self._smoother is not None:
            raise ValueError("whiten=True and smooth are separate operating modes")
        if self.whiten:
            self._fit_whitened(X, weights=weights)
        else:
            if self.normalize_input:
                X_norm = self._normalize(X, fit=True)
            else:
                X_norm = X
            if self._smoother is not None:
                data, _, mne_type, _, _, _ = extract_data_from_mne(X_norm)
                if mne_type == "epochs":
                    data = np.transpose(data, (1, 2, 0))
                _, residual = self._decompose_smooth(data)
                self._fit_numpy(residual, weights=weights)
            elif mne is not None and isinstance(X_norm, BaseRaw | BaseEpochs | Evoked):
                self._fit_mne(X_norm, weights=weights)
            elif isinstance(X_norm, np.ndarray):
                self._fit_numpy(X_norm, weights=weights)
            else:
                raise TypeError(f"Unsupported input type: {type(X_norm)}")

        # Compute mixing matrix
        # self.patterns_ from compute_dss already satisfy X = P @ S
        self.mixing_ = self.patterns_
        if self.n_select is not None:
            self.n_selected_ = self.auto_select()

        return self

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
            # unique std per channel
            self.channel_norms_ = np.std(data_2d, axis=1)
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

        data, _, mne_type, _, picks, ch_names = extract_data_from_mne(inst)
        self._mne_ch_names_ = ch_names

        # MNE covariance computation requires the inst object to match the array
        if picks is not None:
            inst = inst.copy().pick(picks)

        if mne_type == "epochs":
            # DSS transpose preference
            data = np.transpose(data, (1, 2, 0))

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

        else:  # Evoked
            baseline_cov = compute_evoked_covariance(inst, method=method, **kws)
            biased_inst = mne.EvokedArray(
                biased_data, inst.info, tmin=float(inst.times[0]), verbose=False
            )
            biased_cov = compute_evoked_covariance(biased_inst, method=method, **kws)

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

    @staticmethod
    def _whiten_apply(whitener: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Apply a spatial whitener along the channel axis."""
        if data.ndim == 2:
            return whitener @ data
        n_channels = data.shape[0]
        flat = data.reshape(n_channels, -1)
        return (whitener @ flat).reshape(data.shape)

    def _noise_cov_matrix(self, ch_names: list[str]) -> np.ndarray:
        names = list(self.noise_cov.ch_names)
        missing = [name for name in ch_names if name not in names]
        if missing:
            raise ValueError(f"noise_cov is missing required channels: {missing[:5]}")
        indices = [names.index(name) for name in ch_names]
        return np.asarray(self.noise_cov.data)[np.ix_(indices, indices)]

    def _compute_whitener(
        self, data: np.ndarray, ch_names: list[str] | None
    ) -> tuple[np.ndarray, np.ndarray]:
        n_channels = data.shape[0]
        if self.noise_cov is not None:
            if ch_names is None:
                raise ValueError("noise_cov requires an MNE input with named channels.")
            covariance = self._noise_cov_matrix(ch_names)
            covariance = (covariance + covariance.T) / 2.0
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            eigenvalues = np.abs(eigenvalues)
            maximum = float(eigenvalues.max(initial=0.0))
            floor = self.reg * maximum if maximum > 0 else self.reg
            eigenvalues = np.maximum(eigenvalues, floor)
            whitener = (eigenvectors * (1.0 / np.sqrt(eigenvalues))) @ eigenvectors.T
            unwhitener = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
        else:
            flat = data.reshape(n_channels, -1)
            standard_deviation = flat.std(axis=1)
            standard_deviation = np.where(standard_deviation > 0, standard_deviation, 1.0)
            whitener = np.diag(1.0 / standard_deviation)
            unwhitener = np.diag(standard_deviation)
        return whitener, unwhitener

    def _fit_whitened(
        self,
        X: BaseRaw | BaseEpochs | Evoked | np.ndarray,
        weights: np.ndarray | None = None,
    ) -> None:
        """Fit a joint multi-sensor DSS decomposition after whitening."""
        method = self.cov_method
        kws = self.cov_kws.copy() if self.cov_kws else {}
        for key in ("rank", "verbose", "tstep"):
            kws.pop(key, None)

        if mne is not None and isinstance(X, BaseRaw | BaseEpochs | Evoked):
            self.info_ = X.info
            self._mne_info = X.info
            picks = mne.pick_types(
                X.info,
                meg=True,
                eeg=True,
                seeg=True,
                ecog=True,
                dbs=True,
                fnirs=True,
                exclude=[],
            )
            if len(picks) == 0:
                raise ValueError("No data channels found to fit DSS with whiten=True.")
            ch_names = [X.ch_names[pick] for pick in picks]
            self._mne_ch_names_ = ch_names
            data, _, mne_type, _, _, _ = extract_data_from_mne(X, ch_names=ch_names)
            if mne_type == "epochs":
                data = np.transpose(data, (1, 2, 0))
        elif isinstance(X, np.ndarray):
            data = np.asarray(X)
            ch_names = None
            self._mne_ch_names_ = None
        else:
            raise TypeError(f"Unsupported input type: {type(X)}")

        whitener, unwhitener = self._compute_whitener(data, ch_names)
        self._whitener_ = whitener
        self._unwhitener_ = unwhitener
        whitened = self._whiten_apply(whitener, data)
        biased = self._apply_bias(whitened)
        baseline_covariance = compute_covariance(
            whitened, method=method, weights=weights, **kws
        )
        biased_covariance = compute_covariance(
            biased, method=method, weights=weights, **kws
        )
        rank = self.rank if isinstance(self.rank, int) else None
        filters, patterns, self.eigenvalues_ = compute_dss(
            baseline_covariance,
            biased_covariance,
            n_components=self.n_components,
            rank=rank,
            reg=self.reg,
        )
        # Bake whitening into the fitted state so transform/inverse_transform
        # continue to operate in the original sensor units.
        self.filters_ = filters @ whitener
        self.patterns_ = unwhitener @ patterns
        self.explained_variance_ = np.diag(
            filters @ baseline_covariance @ filters.T
        )

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
        data, _, mne_type, orig_inst, picks, _ = extract_data_from_mne(
            X_in, ch_names=getattr(self, "_mne_ch_names_", None)
        )

        # DSS internal convention for Epochs: (n_channels, n_times, n_epochs)
        if mne_type == "epochs":
            data = np.transpose(data, (1, 2, 0))

        if self._smoother is not None:
            data_smooth, data_for_dss = self._decompose_smooth(data)
        else:
            data_smooth, data_for_dss = None, data

        orig_shape = data_for_dss.shape
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
        if data_smooth is not None:
            rec += data_smooth.reshape(data_smooth.shape[0], -1)

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

    def fit_transform(self, X, y=None, **fit_params):
        """Fit and transform, including adaptive segmented artifact subtraction."""
        if not self.segmented:
            self.fit(X, **fit_params)
            if self.return_type == "sources":
                return self.transform(X)
            data, _, mne_type, original, picks, _ = extract_data_from_mne(
                X, ch_names=getattr(self, "_mne_ch_names_", None)
            )
            n_remove = int(self.n_selected_ or 0)
            if n_remove <= 0:
                cleaned = data.copy()
            else:
                saved = self.return_type
                self.return_type = "sources"
                try:
                    sources = self.transform(X)
                finally:
                    self.return_type = saved
                n_remove = min(n_remove, sources.shape[1] if mne_type == "epochs" else sources.shape[0])
                artifact = self.inverse_transform(
                    sources, component_indices=np.arange(n_remove)
                )
                cleaned = data - artifact
            return reconstruct_mne_object(
                cleaned, original, mne_type, picks=picks, verbose=False
            )

        if self.whiten:
            raise ValueError("segmented=True and whiten=True are not yet composable")
        data, extracted_sfreq, mne_type, original, picks, _ = extract_data_from_mne(X)
        sfreq = extracted_sfreq or getattr(self.bias, "sfreq", None)
        if sfreq is None:
            raise ValueError(
                "Cannot determine sfreq for segmented mode; use an MNE object or a bias with sfreq."
            )
        if mne_type == "epochs":
            n_epochs, n_channels, n_times = data.shape
            continuous = data.transpose(1, 0, 2).reshape(n_channels, -1)
        elif data.ndim == 2:
            continuous = data
            n_epochs = n_channels = n_times = None
        else:
            raise ValueError("segmented DSS accepts continuous 2D arrays or MNE Epochs")
        self._resolve_smoother()
        cleaned = self._run_segmented(continuous, float(sfreq))
        if mne_type == "epochs":
            cleaned = cleaned.reshape(n_channels, n_epochs, n_times).transpose(1, 0, 2)
        return reconstruct_mne_object(
            cleaned, original, mne_type, picks=picks, verbose=False
        )

    def _resolve_segmenter(self, sfreq: float):
        if self.segmenter is not None:
            return self.segmenter
        bandpass = None
        if hasattr(self.bias, "freq_band"):
            bandpass = tuple(float(value) for value in self.bias.freq_band)
        elif getattr(self.bias, "freq", None) is not None:
            frequency = float(self.bias.freq)
            bandpass = (max(1.0, frequency - 3.0), min(sfreq / 2 - 1, frequency + 3.0))
        return CovarianceSegmenter(
            sfreq=sfreq, min_chunk_len=30.0, bandpass=bandpass
        )

    def _run_segmented(self, data: np.ndarray, sfreq: float) -> np.ndarray:
        segments = self._resolve_segmenter(sfreq).segment(data)
        if not segments:
            raise ValueError("segmenter returned no segments")
        overlap = max(0, int(round(float(self.crossfade) * sfreq)))
        use_crossfade = overlap > 0 and len(segments) > 1
        if use_crossfade:
            overlap = min(overlap, max(1, min(end - start for start, end in segments) // 2))
        chunks = []
        self.segment_results_ = []
        selected = []
        for index, (start, end) in enumerate(segments):
            extended_start = start if index == 0 or not use_crossfade else max(0, start - overlap)
            extended_end = (
                end
                if index == len(segments) - 1 or not use_crossfade
                else min(data.shape[1], end + overlap)
            )
            result = self._process_segment(data[:, extended_start:extended_end])
            chunks.append(
                {
                    "cleaned": result["cleaned"],
                    "ext_start": extended_start,
                    "ext_end": extended_end,
                    "start": start,
                    "end": end,
                }
            )
            selected.append(result["n_selected"])
            self.segment_results_.append(
                {
                    "start": start,
                    "end": end,
                    "extended_start": extended_start,
                    "extended_end": extended_end,
                    "n_selected": result["n_selected"],
                    "eigenvalues": result["eigenvalues"],
                    "patterns": result["patterns"],
                }
            )
            self.filters_ = result["filters"]
            self.patterns_ = result["patterns"]
            self.eigenvalues_ = result["eigenvalues"]
            self.mixing_ = result["mixing"]
        self.n_selected_ = max(selected, default=0)
        if use_crossfade:
            return self._crossfade_combine(data.shape, chunks)
        return np.concatenate([chunk["cleaned"] for chunk in chunks], axis=1)

    @staticmethod
    def _crossfade_combine(shape: tuple[int, int], chunks: list[dict]) -> np.ndarray:
        output = np.zeros(shape)
        weights = np.zeros(shape[1])
        for chunk in chunks:
            start, end = chunk["ext_start"], chunk["ext_end"]
            window = np.ones(end - start)
            leading = chunk["start"] - start
            trailing = end - chunk["end"]
            if leading:
                phase = np.arange(leading, dtype=float)
                window[:leading] = 0.5 * (1.0 - np.cos(np.pi * phase / leading))
            if trailing:
                phase = np.arange(trailing, dtype=float)
                window[-trailing:] = 0.5 * (1.0 + np.cos(np.pi * phase / trailing))
            output[:, start:end] += chunk["cleaned"] * window
            weights[start:end] += window
        return output / np.maximum(weights, 1e-10)[None]

    def _process_segment(self, chunk: np.ndarray) -> dict:
        model = DSS(
            bias=self.bias,
            n_components=self.n_components,
            n_select=self.n_select,
            selection_method=self.selection_method,
            selection_threshold=self.selection_threshold,
            rank=self.rank if self.rank is None or isinstance(self.rank, int) else None,
            reg=self.reg,
            normalize_input=self.normalize_input,
            cov_method=self.cov_method,
            cov_kws=self.cov_kws,
            return_type="sources",
            smooth=self.smooth,
            segmented=False,
            verbose=self.verbose,
        ).fit(chunk)
        count = int(model.n_selected_ or 0)
        if self.max_prop_remove is not None:
            count = min(count, int(chunk.shape[0] * float(self.max_prop_remove)))
        count = min(chunk.shape[0], max(count, int(self.min_select)))
        return {
            "cleaned": self._clean_segment(chunk, model, count),
            "n_selected": count,
            "eigenvalues": model.eigenvalues_,
            "patterns": model.patterns_,
            "filters": model.filters_,
            "mixing": model.mixing_,
        }

    @staticmethod
    def _clean_segment(data: np.ndarray, model: DSS, n_remove: int) -> np.ndarray:
        if n_remove <= 0:
            return data.copy()
        if model.normalize_input:
            working = data / model.channel_norms_[:, None]
        else:
            working = data
        smooth, residual = model._decompose_smooth(working)
        if smooth is None:
            smooth = np.zeros_like(residual)
        mean = residual.mean(axis=1, keepdims=True)
        sources = model.filters_[:n_remove] @ (residual - mean)
        artifact = model.mixing_[:, :n_remove] @ sources
        cleaned = smooth + residual - artifact
        if model.normalize_input:
            cleaned = cleaned * model.channel_norms_[:, None]
        return cleaned

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
