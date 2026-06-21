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
from .utils import compute_covariance

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
    rank : int or dict, optional
        Rank of the data for whitening. If None, rank is estimated automatically.
    reg : float
        Regularization for covariance estimation. Default 1e-9.
    normalize_input : bool
        If True, normalize input data channel-wise (L2 norm) before fitting/transforming.
        Useful when mixing sensors with different scales (e.g. MAG and GRAD). Default True.
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
        rank: int | dict | None = None,
        reg: float = 1e-9,
        normalize_input: bool = True,
        cov_method: str = "empirical",
        cov_kws: dict | None = None,
        return_type: str = "sources",
        verbose: bool | str | int | None = None,
    ) -> None:
        self.n_components = n_components
        self.bias = bias
        self.rank = rank
        self.reg = reg
        self.normalize_input = normalize_input
        self.cov_method = cov_method
        self.cov_kws = cov_kws
        self.return_type = return_type
        self.verbose = verbose
        set_log_level_from_verbose(self.verbose)

        # Fitted attributes
        self.filters_: np.ndarray | None = None
        self.patterns_: np.ndarray | None = None
        self.mixing_: np.ndarray | None = None
        self.eigenvalues_: np.ndarray | None = None
        self.explained_variance_: np.ndarray | None = None
        self.channel_norms_: np.ndarray | None = None
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
        if self.normalize_input:
            X_norm = self._normalize(X, fit=True)
        else:
            X_norm = X

        if mne is not None and isinstance(X_norm, BaseRaw | BaseEpochs | Evoked):
            self._fit_mne(X_norm, weights=weights)
        elif isinstance(X_norm, np.ndarray):
            self._fit_numpy(X_norm, weights=weights)
        else:
            raise TypeError(f"Unsupported input type: {type(X_norm)}")

        # Compute mixing matrix
        # self.patterns_ from compute_dss already satisfy X = P @ S
        self.mixing_ = self.patterns_

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

        if self.normalize_input:
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

        orig_shape = data.shape
        if data.ndim == 3:
            n_ch, n_times, n_epochs = data.shape
            data_2d = data.reshape(n_ch, -1)
        else:
            n_ch, n_times = data.shape
            data_2d = data

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

        # Reshape to original
        if len(orig_shape) == 3:
            rec = rec.reshape(orig_shape)  # (n_ch, n_times, n_epochs)

        # De-normalization
        if self.normalize_input:
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

        if self.normalize_input:
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
