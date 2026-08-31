"""Linear DSS algorithms."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone

if TYPE_CHECKING:
    from mne.epochs import BaseEpochs
    from mne.evoked import Evoked
    from mne.io import BaseRaw

from .. import _mne
from .._blending import overlap_add_combine
from .._covariance import compute_covariance, compute_mean
from .._data import (
    _mne_instance_types,
    continuous_to_epochs,
    epochs_to_continuous,
    extract_data_from_mne,
    reconstruct_mne_object,
)
from .._logging import logger, verbose
from .._spatial import apply_spatial_transform
from ..progress import _emit_progress, _ProgressCallback, _validate_callback
from ._whitening import (
    apply_covariance_transform,
    compute_data_covariance_whitener,
    compute_mne_sensor_whitener,
    map_spatial_matrices_to_sensor_space,
)
from .denoisers import LinearDenoiser
from .denoisers.averaging import AverageBias
from .denoisers.periodic import CombFilterBias, PeakFilterBias
from .denoisers.spectral import BandpassBias, LineNoiseBias
from .denoisers.temporal import LagAverageBias, SmoothingBias
from .segmentation import CovarianceSegmenter, FixedWindowSegmenter
from .selection import auto_select_components_robust

_COMPONENT_ACTIONS = frozenset({"extract", "retain", "subtract"})


@verbose
def compute_dss(
    covariance_baseline: np.ndarray,
    covariance_biased: np.ndarray,
    *,
    n_components: int | None = None,
    rank: int | None = None,
    reg: float = 1e-9,
    verbose: bool | str | int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Compute DSS spatial filters from baseline and biased covariances.

    Parameters
    ----------
    covariance_baseline : ndarray, shape (n_channels, n_channels)
        Baseline covariance defining the total-power metric.
    covariance_biased : ndarray, shape (n_channels, n_channels)
        Biased covariance defining the signal-of-interest metric.
    n_components : int or None, default=None
        Number of components to return. ``None`` returns the available rank.
    rank : int or None, default=None
        Whitening rank. ``None`` estimates the rank from the baseline covariance.
    reg : float, default=1e-9
        Relative eigenvalue threshold used during whitening.
    verbose : bool, str, int, or None, default=None
        MNE-style logging level.

    Returns
    -------
    filters : ndarray, shape (n_components, n_channels)
        DSS spatial filters.
    patterns : ndarray, shape (n_channels, n_components)
        DSS spatial patterns.
    eigenvalues : ndarray, shape (n_components,)
        Biased-to-baseline variance ratios.

    See Also
    --------
    DSS
        Estimator that learns and applies the decomposition to recordings.

    Notes
    -----
    The baseline covariance is whitened, the biased covariance is diagonalized in
    that space, and the resulting filters are normalized in the baseline metric.

    References
    ----------
    This implementation follows the linear DSS formulation
    :footcite:p:`sarela2005_dss`.

    .. footbibliography::

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.dss import compute_dss
    >>> rng = np.random.default_rng(0)
    >>> data = rng.standard_normal((8, 2000))
    >>> biased_data = data + 0.1 * rng.standard_normal(data.shape)
    >>> baseline = np.cov(data)
    >>> biased = np.cov(biased_data)
    >>> filters, patterns, scores = compute_dss(baseline, biased, n_components=3)
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

    logger.debug(
        "DSS numerical core: input channels=%d, whitening rank=%d, "
        "returned components=%d.",
        n_channels,
        n_keep,
        n_components,
    )

    return dss_filters, dss_patterns, eigenvalues


def _as_smoother(smooth: LinearDenoiser | int | None) -> LinearDenoiser | None:
    """Return a smoothing denoiser for the ``smooth`` parameter."""
    if smooth is None:
        return None
    if isinstance(smooth, int | np.integer):
        return SmoothingBias(window=int(smooth), iterations=1)
    if hasattr(smooth, "apply"):
        # Covers SmoothingBias and any other LinearDenoiser
        return smooth
    raise TypeError(f"smooth must be SmoothingBias, int, or None, got {type(smooth)}")


def _bias_name(bias: object) -> str:
    """Return a concise scientific description of a DSS bias."""
    if bias is None:
        return "None"
    if isinstance(bias, CombFilterBias):
        return (
            f"CombFilterBias(f0={float(bias.fundamental_freq):.3g} Hz, "
            f"harmonics={int(bias.n_harmonics)})"
        )
    if isinstance(bias, BandpassBias):
        low, high = bias.freq_band
        return f"BandpassBias({float(low):.3g}-{float(high):.3g} Hz)"
    if isinstance(bias, SmoothingBias):
        return f"SmoothingBias(window={int(bias.window)})"
    if isinstance(bias, AverageBias):
        return f"AverageBias(axis={bias.axis})"
    if isinstance(bias, LineNoiseBias):
        harmonics = (
            f", harmonics={int(bias.n_harmonics)}"
            if bias.n_harmonics is not None
            else ""
        )
        return (
            f"LineNoiseBias(freq={float(bias.freq):.3g} Hz, "
            f"method={bias.method}{harmonics})"
        )
    if isinstance(bias, PeakFilterBias):
        return f"PeakFilterBias(freq={float(bias.freq):.3g} Hz)"
    if isinstance(bias, LagAverageBias):
        return f"LagAverageBias(lags={bias.lags})"
    if isinstance(bias, LinearDenoiser):
        return type(bias).__name__
    return getattr(bias, "__name__", type(bias).__name__)


class DSS(BaseEstimator, TransformerMixin):
    """Denoising Source Separation transformer.

    The estimator fits DSS filters from a baseline covariance and a biased
    covariance produced by ``bias``. It accepts channel-first NumPy arrays and
    MNE ``Raw``, ``Epochs``, and ``Evoked`` objects.

    Parameters
    ----------
    bias : LinearDenoiser or callable
        Bias transformation applied before the biased covariance is estimated.
    n_components : int or None, default=None
        Number of fitted components; ``None`` uses the available whitening rank.
    n_select : int, {"auto"}, or None, default=None
        Number of leading components used by ``retain`` or ``subtract``. ``"auto"``
        uses the package component-selection heuristics.
    selection_threshold : float, default=3.0
        Sigma threshold for automatic outlier selection.
    knee_rel_floor : float, default=0.01
        Relative score floor for automatic knee selection.
    knee_min_ratio : float, default=3.0
        Minimum score ratio for automatic knee selection.
    rank : int, dict, or None, default=None
        Whitening rank.
    reg : float, default=1e-9
        Relative covariance-whitening regularization.
    normalize_input : bool, default=True
        Normalize each fitted channel by its L2 norm before the DSS covariance
        calculation and undo that scaling on sensor-space output.
    cov_method : str, default="empirical"
        Covariance method passed to the MNE or NumPy covariance path.
    cov_kws : dict or None, default=None
        Additional covariance-estimator keywords.
    smooth : LinearDenoiser, int, or None, default=None
        Optional smooth branch to subtract before DSS. An integer is a smoothing
        window in samples.
    adaptive : bool, default=False
        Fit independent segment operators in ``fit_transform``. This mode supports
        only ``component_action="subtract"``.
    segmenter : CovarianceSegmenter, FixedWindowSegmenter, or None, default=None
        Segmenter for adaptive processing. ``None`` uses a covariance segmenter.
    crossfade : float, default=0.0
        Boundary cross-fade duration in seconds for adaptive processing.
    max_prop_remove : float or None, default=None
        Maximum fraction of channels selected per adaptive segment.
    min_select : int, default=0
        Minimum automatic selection count in adaptive processing.
    component_action : {"extract", "retain", "subtract"}, default="extract"
        Operation performed by :meth:`transform`.
    whiten : bool, default=False
        Jointly whiten and decompose all selected MNE channel types.
    noise_cov : mne.Covariance or None, default=None
        Noise covariance for joint MNE whitening; ignored when ``whiten=False``.
    verbose : bool, str, int, or None, default=None
        Logging level.
    center : bool, default=True
        Fit one global channel mean and reuse it during transforms. ``False`` uses
        uncentered second moments.

    Attributes
    ----------
    filters_ : ndarray, shape (n_components, n_channels)
        Fitted spatial filters.
    patterns_ : ndarray, shape (n_channels, n_components)
        Fitted spatial patterns.
    eigenvalues_ : ndarray, shape (n_components,)
        Fitted DSS scores.
    mean_ : ndarray, shape (n_channels, 1)
        Fitted channel mean, or zeros when ``center=False``.
    n_selected_ : int or None
        Selected component count when automatic or explicit selection is active.
    segment_results_ : list of dict or None
        Per-segment results from adaptive ``fit_transform``.

    See Also
    --------
    compute_dss
        Low-level covariance-based DSS decomposition.
    IterativeDSS
        Nonlinear iterative DSS.
    TimeShiftDSS
        Lag-augmented DSS for repeated trials.
    mne_denoise.zapline.ZapLine
        DSS-based line-noise removal.

    Notes
    -----
    NumPy input uses ``(n_channels, n_times)`` or
    ``(n_channels, n_times, n_epochs)``. MNE ``Epochs`` uses its native
    ``(n_epochs, n_channels, n_times)`` layout. ``extract`` returns source data;
    ``retain`` and ``subtract`` return the input layout or a copied MNE container.

    References
    ----------
    :footcite:p:`sarela2005_dss`

    .. footbibliography::

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.dss import BandpassBias, DSS
    >>> rng = np.random.default_rng(0)
    >>> data = rng.standard_normal((8, 2000))
    >>> bias = BandpassBias(freq_band=(8.0, 12.0), sfreq=250.0)
    >>> dss = DSS(bias=bias, n_components=3, component_action="extract")
    >>> sources = dss.fit_transform(data)
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
        component_action: str = "extract",
        whiten: bool = False,
        noise_cov=None,
        verbose: bool | str | int | None = None,
        center: bool = True,
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
        self.center = center
        self.cov_method = cov_method
        self.cov_kws = cov_kws
        self.smooth = smooth
        self.adaptive = adaptive
        self.segmenter = segmenter
        self.crossfade = crossfade
        self.max_prop_remove = max_prop_remove
        self.min_select = min_select
        self.component_action = component_action
        self.whiten = whiten
        self.noise_cov = noise_cov
        self.verbose = verbose

        # Fitted attributes
        self.filters_: np.ndarray | None = None
        self.patterns_: np.ndarray | None = None
        self.mixing_: np.ndarray | None = None
        self.eigenvalues_: np.ndarray | None = None
        self.explained_variance_: np.ndarray | None = None
        self.channel_norms_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.n_selected_: np.ndarray | None = None
        self.segment_results_: list | None = None
        self._whitener_: np.ndarray | None = None
        self._dewhitener_: np.ndarray | None = None
        self._smoother = None  # Resolved SmoothingBias instance
        self._mne_info = None
        self._mne_ch_names_: list[str] | None = None

    @verbose
    def fit(
        self,
        X: BaseRaw | BaseEpochs | Evoked | np.ndarray,
        y=None,
        weights: np.ndarray | None = None,
        *,
        verbose: bool | str | int | None = None,
    ) -> DSS:
        """Fit the DSS filters and fitted metadata.

        Parameters
        ----------
        X : mne.io.BaseRaw, mne.BaseEpochs, mne.Evoked, or ndarray
            Training data. NumPy input is channel-first and may be 2D or 3D.
        y : None, default=None
            Ignored for scikit-learn compatibility.
        weights : ndarray or None, default=None
            Non-negative observation weights for NumPy input.
        verbose : bool, str, int, or None, default=None
            Logging level for this call.

        Returns
        -------
        DSS
            The fitted estimator.
        """
        self._mne_ch_names_ = None
        self._validate_component_action()
        self._validate_decomposition_parameters()
        if not isinstance(self.center, bool):
            raise TypeError("center must be a bool")
        if self.adaptive:
            logger.debug(
                "DSS(adaptive=True).fit() computes a single global fit. "
                "Call fit_transform() for the per-segment adaptive pathway."
            )

        if self.whiten:
            # Joint multi-sensor decomposition: the whitener replaces the
            # channel-wise normalization and the homogeneous-type isolation.
            self._fit_whitened(X, weights=weights)
            self.mixing_ = self.patterns_
            logger.info(
                "DSS: bias=%s, channels=%d, rank=%d, components=%d, action=%s.",
                _bias_name(self.bias),
                self.filters_.shape[1],
                self.filters_.shape[0],
                self.filters_.shape[0],
                self.component_action,
            )
            return self

        if self.normalize_input:
            X_norm = self._normalize(X, fit=True)
        else:
            X_norm = X

        # Resolve smoothing (if configured)
        self._smoother = _as_smoother(self.smooth)

        # If smoothing is enabled, decompose and fit on the residual only
        if self._smoother is not None:
            data, _, _, orig_inst, picks, ch_names = extract_data_from_mne(
                X_norm,
                ch_names=self._mne_ch_names_,
                exclude_bads=self._mne_ch_names_ is None,
                channel_first_epochs=True,
            )
            self._mne_ch_names_ = ch_names
            if orig_inst is not None:
                fitted_inst = orig_inst.copy()
                if picks is not None:
                    fitted_inst.pick(picks)
                self.info_ = fitted_inst.info
                self._mne_info = self.info_
            data_residual = data - self._smoother.apply(data)
            # Fit DSS on residual (always numpy path)
            self._fit_numpy(data_residual, weights=weights)
        elif isinstance(X_norm, _mne_instance_types()):
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

        logger.info(
            "DSS: bias=%s, channels=%d, rank=%d, components=%d, action=%s%s.",
            _bias_name(self.bias),
            self.filters_.shape[1],
            self.filters_.shape[0],
            self.filters_.shape[0],
            self.component_action,
            f" (selected {self.n_selected_})" if self.n_selected_ is not None else "",
        )

        return self

    def _effective_n_select(self) -> int | str | None:
        """Resolve the component count, including adaptive-mode defaults."""
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

    def _validate_component_action(self) -> None:
        """Validate the explicit component-operation contract."""
        if self.component_action not in _COMPONENT_ACTIONS:
            allowed = ", ".join(sorted(_COMPONENT_ACTIONS))
            raise ValueError(
                "component_action must be one of "
                f"{{{allowed}}}, got {self.component_action!r}."
            )

    def _validate_decomposition_parameters(self) -> None:
        """Reject nonsensical component, rank, and selection counts early."""
        if self.n_components is not None and (
            isinstance(self.n_components, bool)
            or not isinstance(self.n_components, int | np.integer)
            or int(self.n_components) <= 0
        ):
            raise ValueError("n_components must be a positive integer or None")
        if isinstance(self.rank, int | np.integer) and (
            isinstance(self.rank, bool) or int(self.rank) <= 0
        ):
            raise ValueError("rank must be a positive integer when specified as an int")
        if isinstance(self.n_select, int | np.integer) and (
            isinstance(self.n_select, bool) or int(self.n_select) < 0
        ):
            raise ValueError("n_select must be a non-negative integer, 'auto', or None")

    def auto_select(self, threshold: float | None = None) -> int:
        """Return the number of leading DSS components selected by the package heuristics.

        Parameters
        ----------
        threshold : float or None, default=None
            Override the outlier sigma threshold.

        Returns
        -------
        int
            Selected component count.
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
        fitted_ch_names = None if fit else self._mne_ch_names_
        data, _, mne_type, orig_inst, picks, ch_names = extract_data_from_mne(
            X,
            ch_names=fitted_ch_names,
            exclude_bads=fit,
            channel_first_epochs=True,
        )
        is_mne = mne_type != "array"
        if fit and is_mne:
            self._mne_ch_names_ = ch_names

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

        if not is_mne:
            return data_norm

        if mne_type == "epochs":
            data_norm = np.transpose(data_norm, (2, 0, 1))
        return reconstruct_mne_object(
            data_norm,
            orig_inst,
            mne_type,
            picks=picks,
        )

    def _apply_bias(self, data: np.ndarray) -> np.ndarray:
        """Apply bias function to data."""
        if hasattr(self.bias, "apply"):
            return self.bias.apply(data)
        else:
            return self.bias(data)

    def _fit_mean(self, data: np.ndarray, weights: np.ndarray | None = None) -> None:
        """Store the channel origin used by every subsequent transform."""
        data = np.asarray(data, dtype=np.float64)
        self.mean_ = (
            compute_mean(data, weights=weights)
            if self.center
            else np.zeros((data.shape[0], 1), dtype=np.float64)
        )

    def _fit_mne(
        self,
        inst: BaseRaw | BaseEpochs | Evoked,
        weights: np.ndarray | None = None,
    ) -> None:
        """Fit using MNE objects."""
        _mne.require_mne("DSS MNE covariance estimation")
        method = self.cov_method
        kws = self.cov_kws.copy() if self.cov_kws else {}
        # Set defaults if not in kws
        kws.setdefault("rank", self.rank)
        kws.setdefault("verbose", False)

        data, _, mne_type, _, picks, ch_names = extract_data_from_mne(
            inst,
            ch_names=self._mne_ch_names_,
            exclude_bads=self._mne_ch_names_ is None,
            channel_first_epochs=True,
        )
        self._mne_ch_names_ = ch_names

        # MNE covariance computation and the fitted spatial matrices must use
        # exactly the same good-channel contract.
        if picks is not None:
            inst = inst.copy().pick(picks)
        self.info_ = inst.info
        self._mne_info = self.info_

        if weights is not None or not self.center or mne_type == "evoked":
            # Weighted or explicitly uncentered MNE input uses the canonical
            # channel-first NumPy path.
            self._fit_numpy(data, weights=weights)
            return

        self._fit_mean(data)
        biased_data = self._apply_bias(data)

        if mne_type == "epochs":
            biased_data = np.transpose(biased_data, (2, 0, 1))

        biased_inst = reconstruct_mne_object(
            biased_data,
            inst,
            mne_type,
        )

        if mne_type == "raw":
            kws.setdefault("tstep", 2.0)
            baseline_cov = _mne.mne.compute_raw_covariance(inst, method=method, **kws)
            biased_cov = _mne.mne.compute_raw_covariance(
                biased_inst, method=method, **kws
            )

        elif mne_type == "epochs":
            baseline_cov = _mne.mne.compute_covariance(inst, method=method, **kws)
            biased_cov = _mne.mne.compute_covariance(biased_inst, method=method, **kws)

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
        self._fit_mean(X, weights)
        biased_X = self._apply_bias(X)

        method = self.cov_method
        kws = self.cov_kws.copy() if self.cov_kws else {}
        kws["assume_centered"] = not self.center

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
        kws["assume_centered"] = not self.center

        data, _, _, orig_inst, picks, ch_names = extract_data_from_mne(
            X,
            auto_pick="data",
            exclude_bads=True,
            channel_first_epochs=True,
        )
        self._mne_ch_names_ = ch_names
        if orig_inst is not None:
            fitted_inst = orig_inst.copy()
            if picks is not None:
                fitted_inst.pick(picks)
            self.info_ = fitted_inst.info
        else:
            self.info_ = None
        self._mne_info = self.info_

        self._fit_mean(data, weights)
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

    @verbose
    def transform(
        self,
        X: BaseRaw | BaseEpochs | Evoked | np.ndarray,
        *,
        verbose: bool | str | int | None = None,
    ) -> np.ndarray | BaseRaw | BaseEpochs | Evoked:
        """Apply the configured DSS component operation.

        Parameters
        ----------
        X : mne.io.BaseRaw, mne.BaseEpochs, mne.Evoked, or ndarray
            Data compatible with the fitted channel layout.
        verbose : bool, str, int, or None, default=None
            Logging level for this call.

        Returns
        -------
        ndarray or MNE object
            ``extract`` returns source data. ``retain`` and ``subtract`` return a copy
            with the input array layout or MNE container type.
        """
        self._validate_component_action()
        return self._transform_with_action(X, self.component_action)

    def _operation_component_count(self, action: str, n_available: int) -> int:
        """Return the leading component count used in sensor space."""
        if action == "retain" and self.n_selected_ is None:
            return n_available
        if self.n_selected_ is None:
            return 0
        return min(max(int(self.n_selected_), 0), n_available)

    def _transform_with_action(
        self,
        X: BaseRaw | BaseEpochs | Evoked | np.ndarray,
        action: str,
    ) -> np.ndarray | BaseRaw | BaseEpochs | Evoked:
        """Apply a validated operation without mutating estimator parameters."""
        if action not in _COMPONENT_ACTIONS:
            raise ValueError(f"Unknown component action {action!r}.")
        if self.filters_ is None:
            raise RuntimeError("DSS not fitted. Call fit() first.")

        if self.normalize_input and not self.whiten:
            # Apply normalization using fitted norms
            X_in = self._normalize(X, fit=False)
        else:
            X_in = X

        # Helper to extract data
        # DSS internal convention for Epochs: (n_channels, n_times, n_epochs)
        data, _, mne_type, _, picks, _ = extract_data_from_mne(
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
            full_data_2d = data.reshape(n_ch, -1)
        else:
            n_ch, n_times = data_for_dss.shape
            data_2d = data_for_dss
            full_data_2d = data

        # Apply the centering measure fitted with the DSS filters. Recomputing
        # this mean from each transform batch would make a learned transform
        # depend on unrelated observations supplied alongside it.
        if self.mean_ is None:
            raise RuntimeError("DSS fitted centering state is unavailable")
        data_centered = data_2d - self.mean_

        sources = self.filters_ @ data_centered

        if action == "extract":
            if len(orig_shape) == 3:
                sources = sources.reshape(sources.shape[0], n_times, n_epochs)
                if mne_type == "epochs":
                    # Return as (n_epochs, n_components, n_times)
                    return sources.transpose(2, 0, 1)
            return sources

        n_action = self._operation_component_count(action, sources.shape[0])
        if action == "subtract" and n_action == 0:
            if hasattr(X, "copy"):
                return X.copy()
            return np.array(X, copy=True)

        selected = self.mixing_[:, :n_action] @ sources[:n_action]
        if action == "subtract":
            rec = full_data_2d - selected
        else:
            rec = selected + self.mean_
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
            rec,
            X if mne_type != "array" else None,
            mne_type,
            picks=picks,
        )

    @verbose
    def inverse_transform(
        self,
        sources: np.ndarray,
        component_indices: np.ndarray | None = None,
        *,
        verbose: bool | str | int | None = None,
    ) -> np.ndarray:
        """Reconstruct sensor-space data from DSS sources.

        Parameters
        ----------
        sources : ndarray, shape (n_components, n_times) or 3D
            Sources returned by ``component_action="extract"``. NumPy epochs use
            channel-first layout; MNE Epochs source arrays use epoch-first layout.
        component_indices : array-like of int or bool, default=None
            Components to include. ``None`` includes all supplied sources.
        verbose : bool, str, int, or None, default=None
            Logging level for this call.

        Returns
        -------
        ndarray
            Reconstructed sensor-space data. The fitted global mean is not added.
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
        """Fit DSS and apply the configured component operation.

        In standard mode this composes :meth:`fit` and :meth:`transform`. In adaptive
        mode it performs transductive per-segment fitting and subtraction; adaptive
        mode requires ``component_action="subtract"``.

        Parameters
        ----------
        X : ndarray or MNE object
            Data used for fitting and transformation.
        y : None, default=None
            Ignored for scikit-learn compatibility.
        callback : callable or None, default=None
            Synchronous progress callback for adaptive processing.
        verbose : bool, str, int, or None, default=None
            Logging level for this call.
        **fit_params
            Additional keyword arguments passed to :meth:`fit` in standard mode.

        Returns
        -------
        ndarray or MNE object
            Transformed data with the configured component-action semantics.
        """
        callback = _validate_callback(callback)
        self._validate_component_action()
        if not self.adaptive:
            self.fit(X, **fit_params)
            return self.transform(X)

        if self.component_action != "subtract":
            raise ValueError(
                "adaptive fit_transform supports only "
                "component_action='subtract' because each segment has a "
                "different fitted component basis. Use fit().transform() for "
                "global extraction or retention."
            )

        # --- adaptive (per-segment) mode ---
        data, extracted_sfreq, mne_type, orig_inst, picks, ch_names = (
            extract_data_from_mne(X, exclude_bads=True)
        )
        self._mne_ch_names_ = ch_names

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
            data_cont = epochs_to_continuous(data)
        else:
            data_cont = data

        # Resolve smoother once
        self._smoother = _as_smoother(self.smooth)

        # A global fit over the whole recording populates the estimator-level
        # attributes (filters_, patterns_, eigenvalues_) with something that
        # describes all of the data. Per-segment results are kept separately in
        # segment_results_.
        global_est = self._make_segment_estimator()
        # Adaptive DSS owns the aggregate INFO record; hide the global helper
        # fit so one operation does not report its internal DSS twice.
        global_est.fit(data_cont, verbose="WARNING")
        self.filters_ = global_est.filters_
        self.patterns_ = global_est.patterns_
        self.mixing_ = global_est.patterns_
        self.eigenvalues_ = global_est.eigenvalues_
        self.explained_variance_ = global_est.explained_variance_
        self.channel_norms_ = global_est.channel_norms_

        # Run segmented processing
        cleaned = self._run_segmented(data_cont, sfreq, callback=callback)

        # Reshape back if epochs
        if is_epochs:
            cleaned = continuous_to_epochs(cleaned, (n_ep, n_ch, n_t))

        result = reconstruct_mne_object(
            cleaned,
            orig_inst,
            mne_type,
            picks=picks,
        )
        logger.info(
            "DSS: adaptive fit, %d segment(s), max %d component(s) selected, "
            "action=%s.",
            len(self.segment_results_ or ()),
            self.n_selected_ or 0,
            self.component_action,
        )
        return result

    def _resolve_segmenter(self, sfreq: float):
        """Resolve the configured DSS segmenter."""
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
        *,
        callback: _ProgressCallback | None = None,
    ) -> np.ndarray:
        """Run segmented fit-transform on continuous data."""
        if segmenter is None:
            segmenter = self._resolve_segmenter(sfreq)
        segments = segmenter.segment(data)

        if not segments:
            raise ValueError(
                "Segmenter returned no segments. Check segmenter settings "
                "and data length."
            )

        logger.debug(
            "Segmented DSS engine: %d segment(s) over %.1f s.",
            len(segments),
            data.shape[1] / sfreq,
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
            logger.debug(
                "Segmented DSS segment %d/%d: samples=%d:%d, selected=%d.",
                seg_idx + 1,
                len(segments),
                start,
                end,
                result.get("n_selected", 0),
            )
            _emit_progress(
                callback,
                method="dss",
                stage="segment",
                current=seg_idx + 1,
                total=len(segments),
                component=None,
                metric=float(result["n_selected"]),
            )

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
        """Build an unfitted DSS clone configured for one segment."""
        est = clone(self)
        est.set_params(
            adaptive=False,  # do NOT recurse
            # Resolve 'auto' here: the clone is no longer adaptive, so it
            # would otherwise fall back to n_select=None and select nothing.
            n_select=self._effective_n_select(),
            segmenter=None,
            crossfade=0.0,
            # Per-segment caps are applied by the caller, not the clone.
            max_prop_remove=None,
            min_select=0,
            # A dict rank is an MNE-object concept; segments are plain arrays.
            rank=self.rank if isinstance(self.rank, int | type(None)) else None,
            # The adaptive parent owns the aggregate report; segment clones
            # are numerical helpers and must not emit duplicate DSS INFO.
            verbose="WARNING",
        )
        return est

    def _process_segment(self, chunk: np.ndarray) -> dict:
        """Fit, select, and clean one DSS segment."""
        n_channels = chunk.shape[0]
        seg_dss = self._make_segment_estimator()
        # Adaptive DSS owns the aggregate INFO record; segment DSS fits are
        # implementation details rather than separate user-facing results.
        seg_dss.fit(chunk, verbose="WARNING")
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
        """Clean a segment by projecting out selected DSS components."""
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
