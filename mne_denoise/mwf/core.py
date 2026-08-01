"""GEVD multi-channel Wiener filtering for semi-supervised EEG denoising.

This module implements the zero-delay form of the multi-channel Wiener filter
(MWF) described by Somers, Francart, and Bertrand (2018). The filter learns an
artifact covariance from explicitly marked artifact samples and a clean
covariance from marked clean samples or a separate clean reference. A
generalized eigendecomposition (GEVD) then constructs a low-rank artifact model.

The high-frequency detector provided here is only an optional mask-authoring
heuristic; it is not part of the core MWF algorithm and must be selected
explicitly. The authors' MATLAB toolbox also supports delay embedding, which is
outside the scope of this zero-delay implementation.

References
----------
Somers, B., Francart, T., & Bertrand, A. (2018). A generic EEG artifact removal
algorithm based on the multi-channel Wiener filter. Journal of Neural
Engineering, 15(3), 036007. https://doi.org/10.1088/1741-2552/aaac92
"""

from __future__ import annotations

import logging
from numbers import Integral, Real
from typing import Any, Literal

import numpy as np
from scipy.linalg import LinAlgError, eigh
from scipy.signal import butter, sosfiltfilt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ..utils import extract_data_from_mne

logger = logging.getLogger(__name__)

RankSpec = Literal["positive", "full"] | int
MaskStrategy = Literal["hf_power"] | None


def _as_2d_finite(
    X: np.ndarray,
    *,
    name: str,
    min_channels: int = 1,
    min_samples: int = 2,
) -> np.ndarray:
    """Validate and return channel-by-sample floating-point data."""
    try:
        data = np.asarray(X, dtype=float)
    except (TypeError, ValueError) as err:
        raise TypeError(f"{name} must be a numeric array.") from err
    if data.ndim != 2:
        raise ValueError(
            f"{name} must have shape (n_channels, n_samples), got {data.shape}."
        )
    if data.shape[0] < min_channels:
        raise ValueError(f"{name} must contain at least {min_channels} channels.")
    if data.shape[1] < min_samples:
        raise ValueError(f"{name} must contain at least {min_samples} samples.")
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{name} must contain only finite values.")
    return data


def _validate_real(
    value: float,
    *,
    name: str,
    minimum: float | None = None,
    strict: bool = False,
) -> float:
    """Validate a finite real scalar."""
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real number.")
    value = float(value)
    if minimum is not None:
        valid = value > minimum if strict else value >= minimum
        if not valid:
            relation = ">" if strict else ">="
            raise ValueError(f"{name} must be {relation} {minimum}.")
    return value


def _normalize_mask(
    artifact_mask: np.ndarray,
    *,
    n_samples: int,
    treat_nan: Literal["ignore", "artifact", "clean"],
) -> np.ndarray:
    """Return a flat float mask containing only zero, one, or NaN."""
    if treat_nan not in {"ignore", "artifact", "clean"}:
        raise ValueError("treat_nan must be 'ignore', 'artifact', or 'clean'.")
    mask = np.asarray(artifact_mask)
    if mask.ndim not in (1, 2) or mask.size != n_samples:
        raise ValueError(
            "artifact_mask must be one-dimensional, or epoch-by-time, with "
            f"exactly {n_samples} entries; got shape {mask.shape}."
        )
    if mask.dtype == np.bool_:
        normalized = mask.astype(float, copy=False).reshape(-1)
    elif np.issubdtype(mask.dtype, np.number):
        normalized = mask.astype(float, copy=False).reshape(-1)
        finite = normalized[np.isfinite(normalized)]
        if not np.all(np.isin(finite, (0.0, 1.0))):
            raise ValueError("artifact_mask values must be 0, 1, or NaN.")
        if np.any(np.isinf(normalized)):
            raise ValueError("artifact_mask values must be 0, 1, or NaN.")
    else:
        raise TypeError("artifact_mask must contain boolean or numeric values.")

    if treat_nan == "artifact":
        normalized = np.nan_to_num(normalized, nan=1.0)
    elif treat_nan == "clean":
        normalized = np.nan_to_num(normalized, nan=0.0)
    return normalized


def hf_power_mask(
    X: np.ndarray,
    sfreq: float,
    *,
    hf_hz: float = 20.0,
    quantile: float = 0.6,
    smooth_s: float = 0.1,
) -> np.ndarray:
    """Create an artifact mask from smoothed high-frequency power.

    This detector is a convenience heuristic, not part of the reference MWF
    algorithm. Its operating point must be validated for the acquisition regime.

    Parameters
    ----------
    X : ndarray, shape (n_channels, n_samples)
        Multichannel signal in any consistent physical unit.
    sfreq : float
        Sampling frequency in Hz.
    hf_hz : float
        High-pass cutoff in Hz. It must lie strictly below Nyquist.
    quantile : float
        Fractional power quantile strictly between zero and one. Samples above
        this threshold are marked as artifact.
    smooth_s : float
        Positive moving-average duration in seconds.

    Returns
    -------
    artifact_mask : ndarray of bool, shape (n_samples,)
        True for samples selected as artifact-present.

    Notes
    -----
    The output depends on ``sfreq``, ``hf_hz``, ``quantile``, and ``smooth_s``;
    these values are part of the MWF operating point, not universal defaults.
    """
    data = _as_2d_finite(X, name="X")
    sfreq = _validate_real(sfreq, name="sfreq", minimum=0.0, strict=True)
    hf_hz = _validate_real(hf_hz, name="hf_hz", minimum=0.0, strict=True)
    quantile = _validate_real(quantile, name="quantile", minimum=0.0, strict=True)
    smooth_s = _validate_real(smooth_s, name="smooth_s", minimum=0.0, strict=True)
    if hf_hz >= sfreq / 2.0:
        raise ValueError("hf_hz must be strictly below the Nyquist frequency.")
    if quantile >= 1.0:
        raise ValueError("quantile must be strictly between 0 and 1.")

    sos = butter(4, hf_hz, btype="highpass", fs=sfreq, output="sos")
    try:
        high_frequency = sosfiltfilt(sos, data, axis=-1)
    except ValueError as err:
        raise ValueError(
            "X is too short for zero-phase high-frequency mask estimation; "
            "provide an explicit artifact_mask."
        ) from err
    envelope = np.mean(high_frequency**2, axis=0)
    window = max(1, min(int(round(smooth_s * sfreq)), envelope.size))
    envelope = np.convolve(envelope, np.ones(window) / window, mode="same")
    return envelope > np.quantile(envelope, quantile)


def _covariance(data: np.ndarray) -> np.ndarray:
    """Return a symmetric sample covariance matrix."""
    covariance = np.atleast_2d(np.cov(data, rowvar=True, bias=False))
    return (covariance + covariance.T) / 2.0


def _select_rank(rank: RankSpec, artifact_eigenvalues: np.ndarray) -> np.ndarray:
    """Select generalized eigen-directions for the artifact model."""
    n_channels = artifact_eigenvalues.size
    if isinstance(rank, str):
        if rank == "full":
            return np.arange(n_channels)
        if rank != "positive":
            raise ValueError("rank must be 'positive', 'full', or a positive integer.")
        tolerance = (
            np.finfo(float).eps
            * n_channels
            * max(1.0, float(np.max(np.abs(artifact_eigenvalues))))
        )
        return np.flatnonzero(artifact_eigenvalues > tolerance)
    if isinstance(rank, bool) or not isinstance(rank, Integral):
        raise TypeError("rank must be 'positive', 'full', or a positive integer.")
    if not 1 <= int(rank) <= n_channels:
        raise ValueError(f"Integer rank must be between 1 and {n_channels}.")
    return np.arange(int(rank))


def _compute_operator(
    artifact_data: np.ndarray,
    clean_data: np.ndarray,
    *,
    rank: RankSpec,
    artifact_weight: float,
    reg: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Construct the zero-delay GEVD MWF cleaning operator."""
    artifact_weight = _validate_real(
        artifact_weight, name="artifact_weight", minimum=0.0, strict=True
    )
    reg = _validate_real(reg, name="reg", minimum=0.0)
    if artifact_data.shape[0] != clean_data.shape[0]:
        raise ValueError("Artifact and clean data must have the same channel count.")
    if artifact_data.shape[1] < 2 or clean_data.shape[1] < 2:
        raise ValueError(
            "MWF needs at least two artifact and two clean training samples."
        )

    artifact_covariance = _covariance(artifact_data)
    clean_covariance = _covariance(clean_data)
    n_channels = artifact_covariance.shape[0]
    artifact_rank = int(np.linalg.matrix_rank(artifact_covariance))
    clean_rank = int(np.linalg.matrix_rank(clean_covariance))
    covariance_scale = max(
        float(np.trace(artifact_covariance) / n_channels),
        float(np.trace(clean_covariance) / n_channels),
    )
    if not np.isfinite(covariance_scale) or covariance_scale <= np.finfo(float).tiny:
        raise ValueError("MWF training covariances have zero numerical energy.")
    loading = reg * covariance_scale
    identity = np.eye(n_channels)
    artifact_covariance_loaded = artifact_covariance + loading * identity
    clean_covariance_loaded = clean_covariance + loading * identity

    try:
        generalized_eigenvalues, eigenvectors = eigh(
            artifact_covariance_loaded,
            clean_covariance_loaded,
            check_finite=False,
        )
    except LinAlgError as err:
        raise ValueError(
            "The clean covariance is not positive definite. Use a positive reg "
            "or provide more independent clean samples."
        ) from err
    order = np.argsort(generalized_eigenvalues)[::-1]
    generalized_eigenvalues = generalized_eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    artifact_eigenvalues = generalized_eigenvalues - 1.0
    selected = _select_rank(rank, artifact_eigenvalues)

    denominators = generalized_eigenvalues + artifact_weight - 1.0
    tolerance = np.finfo(float).eps * max(1.0, float(np.max(np.abs(denominators))))
    if selected.size and np.any(np.abs(denominators[selected]) <= tolerance):
        raise ValueError(
            "artifact_weight creates a singular MWF denominator for a selected "
            "component."
        )
    component_weights = np.zeros(n_channels)
    component_weights[selected] = (
        artifact_eigenvalues[selected] / denominators[selected]
    )
    weighted_vectors = eigenvectors * component_weights[None, :]
    try:
        reference_artifact_filter = np.linalg.solve(
            eigenvectors.T, weighted_vectors.T
        ).T
    except np.linalg.LinAlgError as err:
        raise ValueError("The generalized eigenvector matrix is singular.") from err

    # The reference MATLAB implementation applies W.T as the artifact operator.
    artifact_operator = reference_artifact_filter.T
    spatial_filter = identity - artifact_operator
    if not np.all(np.isfinite(spatial_filter)):
        raise ValueError("MWF produced a non-finite spatial filter.")

    diagnostics = {
        "rank_requested": rank,
        "rank_used": int(selected.size),
        "selected_components": selected.copy(),
        "generalized_eigenvalues": generalized_eigenvalues.copy(),
        "artifact_eigenvalues": artifact_eigenvalues.copy(),
        "artifact_covariance_rank": artifact_rank,
        "clean_covariance_rank": clean_rank,
        "regularization_loading": float(loading),
        "used_identity": bool(selected.size == 0),
    }
    return spatial_filter, diagnostics


def _resolve_mask(
    data: np.ndarray,
    artifact_mask: np.ndarray | None,
    *,
    clean_reference: np.ndarray | None,
    mask_strategy: MaskStrategy,
    sfreq: float | None,
    hf_hz: float,
    quantile: float,
    smooth_s: float,
    treat_nan: Literal["ignore", "artifact", "clean"],
) -> np.ndarray:
    """Resolve an explicit or explicitly requested heuristic mask."""
    if artifact_mask is not None and mask_strategy is not None:
        raise ValueError("Pass artifact_mask or mask_strategy, not both.")
    if artifact_mask is not None:
        return _normalize_mask(
            artifact_mask, n_samples=data.shape[1], treat_nan=treat_nan
        )
    if mask_strategy is None and clean_reference is not None:
        return np.ones(data.shape[1], dtype=float)
    if mask_strategy is None:
        raise ValueError(
            "An explicit artifact_mask is required. To opt into the unvalidated "
            "high-frequency heuristic, set mask_strategy='hf_power'."
        )
    if mask_strategy != "hf_power":
        raise ValueError("mask_strategy must be None or 'hf_power'.")
    if sfreq is None:
        raise ValueError("sfreq is required when mask_strategy='hf_power'.")
    return hf_power_mask(
        data,
        sfreq,
        hf_hz=hf_hz,
        quantile=quantile,
        smooth_s=smooth_s,
    ).astype(float)


def _fit_from_training_data(
    data: np.ndarray,
    artifact_mask: np.ndarray | None,
    *,
    clean_reference: np.ndarray | None,
    mask_strategy: MaskStrategy,
    sfreq: float | None,
    hf_hz: float,
    quantile: float,
    smooth_s: float,
    rank: RankSpec,
    artifact_weight: float,
    reg: float,
    treat_nan: Literal["ignore", "artifact", "clean"],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Resolve training segments and fit a spatial operator."""
    data = _as_2d_finite(data, name="X", min_channels=2)
    if clean_reference is not None:
        clean_reference = _as_2d_finite(
            clean_reference, name="clean_reference", min_channels=2
        )
        if clean_reference.shape[0] != data.shape[0]:
            raise ValueError(
                "clean_reference must have the same number of channels as X."
            )
    normalized_mask = _resolve_mask(
        data,
        artifact_mask,
        clean_reference=clean_reference,
        mask_strategy=mask_strategy,
        sfreq=sfreq,
        hf_hz=hf_hz,
        quantile=quantile,
        smooth_s=smooth_s,
        treat_nan=treat_nan,
    )
    artifact_samples = normalized_mask == 1.0
    clean_samples = normalized_mask == 0.0
    ignored_samples = np.isnan(normalized_mask)
    if artifact_samples.sum() < 2:
        raise ValueError("artifact_mask must select at least two artifact samples.")
    if clean_reference is None:
        if clean_samples.sum() < 2:
            raise ValueError("artifact_mask must select at least two clean samples.")
        clean_data = data[:, clean_samples]
    else:
        clean_data = clean_reference
    spatial_filter, diagnostics = _compute_operator(
        data[:, artifact_samples],
        clean_data,
        rank=rank,
        artifact_weight=artifact_weight,
        reg=reg,
    )
    valid_samples = (~ignored_samples).sum()
    diagnostics.update(
        {
            "artifact_samples": int(artifact_samples.sum()),
            "clean_samples": int(clean_data.shape[1]),
            "ignored_samples": int(ignored_samples.sum()),
            "artifact_fraction": float(artifact_samples.sum() / max(1, valid_samples)),
            "used_clean_reference": clean_reference is not None,
            "mask_strategy": mask_strategy,
        }
    )
    return spatial_filter, normalized_mask, diagnostics


def _apply_spatial_filter(data: np.ndarray, spatial_filter: np.ndarray) -> np.ndarray:
    """Apply a spatial filter after removing and then restoring channel means."""
    if data.ndim == 2:
        channel_means = data.mean(axis=1, keepdims=True)
        return spatial_filter @ (data - channel_means) + channel_means
    if data.ndim == 3:
        n_epochs, n_channels, n_times = data.shape
        continuous = np.transpose(data, (1, 0, 2)).reshape(n_channels, -1)
        channel_means = continuous.mean(axis=1, keepdims=True)
        cleaned = spatial_filter @ (continuous - channel_means) + channel_means
        return cleaned.reshape(n_channels, n_epochs, n_times).transpose(1, 0, 2)
    raise ValueError(f"Data must be 2D or 3D, got shape {data.shape}.")


def compute_mwf(
    X: np.ndarray,
    artifact_mask: np.ndarray | None = None,
    *,
    clean_reference: np.ndarray | None = None,
    mask_strategy: MaskStrategy = None,
    sfreq: float | None = None,
    hf_hz: float = 20.0,
    quantile: float = 0.6,
    smooth_s: float = 0.1,
    rank: RankSpec = "positive",
    artifact_weight: float = 1.0,
    reg: float = 1e-6,
    treat_nan: Literal["ignore", "artifact", "clean"] = "ignore",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit and apply a zero-delay GEVD multi-channel Wiener filter.

    Parameters
    ----------
    X : ndarray, shape (n_channels, n_samples)
        Training data and the data to clean.
    artifact_mask : ndarray | None
        Values 1 (artifact), 0 (clean), or NaN (handled by ``treat_nan``).
        Required unless ``clean_reference`` contains the clean covariance data or
        ``mask_strategy='hf_power'`` is selected explicitly.
    clean_reference : ndarray | None
        Optional clean data with the same channel order and physical units as
        ``X``. When supplied, zero-valued mask samples in ``X`` are excluded and
        this reference supplies the clean covariance.
    mask_strategy : None | 'hf_power'
        Optional explicit opt-in to :func:`hf_power_mask`.
    sfreq, hf_hz, quantile, smooth_s
        High-frequency mask parameters, used only by ``mask_strategy='hf_power'``.
    rank : {'positive', 'full'} | int
        GEVD artifact rank. ``'positive'`` matches the reference toolbox's
        ``poseig`` default; an integer retains that many leading directions.
    artifact_weight : float
        Positive reference-toolbox noise weighting (``mu``); 1 is unweighted.
    reg : float
        Non-negative relative diagonal loading applied to both covariances.
    treat_nan : {'ignore', 'artifact', 'clean'}
        Interpretation of NaNs in ``artifact_mask``.

    Returns
    -------
    cleaned : ndarray, shape (n_channels, n_samples)
        Filtered data in the same physical units as ``X``.
    diagnostics : dict
        Mask, sample counts, covariance ranks, GEVD values, and rank selection.
    """
    data = _as_2d_finite(X, name="X", min_channels=2)
    spatial_filter, normalized_mask, diagnostics = _fit_from_training_data(
        data,
        artifact_mask,
        clean_reference=clean_reference,
        mask_strategy=mask_strategy,
        sfreq=sfreq,
        hf_hz=hf_hz,
        quantile=quantile,
        smooth_s=smooth_s,
        rank=rank,
        artifact_weight=artifact_weight,
        reg=reg,
        treat_nan=treat_nan,
    )
    cleaned = _apply_spatial_filter(data, spatial_filter)
    diagnostics = {
        **diagnostics,
        "artifact_mask": normalized_mask.copy(),
        "spatial_filter": spatial_filter.copy(),
    }
    return cleaned, diagnostics


def mwf_filter(
    X: np.ndarray,
    artifact_mask: np.ndarray,
    *,
    clean_reference: np.ndarray | None = None,
    rank: RankSpec = "positive",
    artifact_weight: float = 1.0,
    reg: float = 1e-6,
    treat_nan: Literal["ignore", "artifact", "clean"] = "ignore",
) -> np.ndarray:
    """Fit and apply MWF from an explicit artifact mask."""
    cleaned, _ = compute_mwf(
        X,
        artifact_mask,
        clean_reference=clean_reference,
        rank=rank,
        artifact_weight=artifact_weight,
        reg=reg,
        treat_nan=treat_nan,
    )
    return cleaned


def _reconstruct_like(
    cleaned: np.ndarray,
    orig_inst: Any,
    mne_type: str,
    picks: np.ndarray | None,
) -> Any:
    """Insert cleaned channels into an exact copy of an MNE object."""
    if orig_inst is None or mne_type == "array":
        return cleaned
    output = orig_inst.copy()
    selected = slice(None) if picks is None else picks
    if mne_type == "raw":
        output.load_data()
        output._data[selected, :] = cleaned
    elif mne_type == "epochs":
        output.load_data()
        output._data[:, selected, :] = cleaned
    elif mne_type == "evoked":
        output.data[selected, :] = cleaned
    else:  # pragma: no cover - guarded by extract_data_from_mne
        raise TypeError(f"Unsupported MNE data type {mne_type!r}.")
    return output


class MultichannelWienerFilter(BaseEstimator, TransformerMixin):
    """Zero-delay GEVD multi-channel Wiener filter.

    The estimator is semi-supervised: ``fit`` requires an explicit artifact mask,
    a clean reference, or an explicit opt-in to the high-frequency mask heuristic.
    ``transform`` applies the frozen operator without fitting on evaluation data.

    Parameters
    ----------
    rank : {'positive', 'full'} | int
        GEVD artifact rank. ``'positive'`` retains positive artifact
        eigenvalues, matching the reference toolbox default.
    artifact_weight : float
        Positive artifact/noise weighting parameter (reference ``mu``).
    reg : float
        Non-negative relative covariance diagonal loading.
    treat_nan : {'ignore', 'artifact', 'clean'}
        Interpretation of NaNs in an explicit artifact mask.
    mask_strategy : None | 'hf_power'
        Mask-authoring strategy. The default requires explicit evidence; set
        ``'hf_power'`` to opt into the heuristic detector.
    sfreq : float | None
        Sampling frequency in Hz for ``mask_strategy='hf_power'``. MNE metadata
        is used when available and must agree with this value when both exist.
    hf_hz, quantile, smooth_s : float
        Operating point of the high-frequency mask heuristic.
    verbose : bool | str | int | None
        Enable a concise fit summary when truthy.

    Attributes
    ----------
    spatial_filter_ : ndarray, shape (n_channels, n_channels)
        Frozen clean-signal spatial operator.
    artifact_mask_ : ndarray, shape (n_training_samples,)
        Normalized 0/1/NaN training mask.
    generalized_eigenvalues_ : ndarray, shape (n_channels,)
        Sorted GEVD eigenvalues.
    artifact_eigenvalues_ : ndarray, shape (n_channels,)
        GEVD values relative to the clean baseline (``lambda - 1``).
    selected_components_ : ndarray
        Artifact directions retained by ``rank``.
    fit_diagnostics_ : dict
        Sample counts, covariance ranks, and operating-point diagnostics.

    Notes
    -----
    Inputs and a separate ``clean_reference`` must use the same channel scaling
    and physical units. The relative regularization makes the operator invariant
    to a common global rescaling, but not to channel-specific unit mismatches.
    Delay embedding from the reference MATLAB toolbox is not implemented here.
    """

    def __init__(
        self,
        *,
        rank: RankSpec = "positive",
        artifact_weight: float = 1.0,
        reg: float = 1e-6,
        treat_nan: Literal["ignore", "artifact", "clean"] = "ignore",
        mask_strategy: MaskStrategy = None,
        sfreq: float | None = None,
        hf_hz: float = 20.0,
        quantile: float = 0.6,
        smooth_s: float = 0.1,
        verbose: bool | str | int | None = None,
    ) -> None:
        self.rank = rank
        self.artifact_weight = artifact_weight
        self.reg = reg
        self.treat_nan = treat_nan
        self.mask_strategy = mask_strategy
        self.sfreq = sfreq
        self.hf_hz = hf_hz
        self.quantile = quantile
        self.smooth_s = smooth_s
        self.verbose = verbose

    def _resolve_sfreq(self, sfreq_data: float | None) -> float | None:
        """Resolve and cross-check configured and metadata sampling rates."""
        configured = self.sfreq
        if configured is not None:
            configured = _validate_real(
                configured, name="sfreq", minimum=0.0, strict=True
            )
        if sfreq_data is not None:
            sfreq_data = _validate_real(
                sfreq_data, name="data sfreq", minimum=0.0, strict=True
            )
        if (
            configured is not None
            and sfreq_data is not None
            and not np.isclose(configured, sfreq_data, rtol=1e-9, atol=0.0)
        ):
            raise ValueError(
                f"Configured sfreq ({configured}) does not match MNE metadata "
                f"({sfreq_data})."
            )
        return sfreq_data if sfreq_data is not None else configured

    @staticmethod
    def _extract_fit_data(
        X: Any, *, ch_names: list[str] | None = None
    ) -> tuple[np.ndarray, float | None, list[str] | None]:
        """Extract and concatenate training data."""
        data, sfreq, _, _, _, extracted_names = extract_data_from_mne(
            X,
            ch_names=ch_names,
            auto_pick=True,
            concatenate_epochs=True,
        )
        data = _as_2d_finite(data, name="X", min_channels=2)
        return data, sfreq, extracted_names

    def fit(
        self,
        X: Any,
        y=None,
        *,
        artifact_mask: np.ndarray | None = None,
        clean_reference: Any | None = None,
    ) -> MultichannelWienerFilter:
        """Estimate the frozen MWF operator.

        Parameters
        ----------
        X : Raw | Epochs | Evoked | ndarray
            Training data. Epochs are concatenated in epoch-major time order.
        y : None
            Ignored.
        artifact_mask : ndarray | None
            Explicit 0/1/NaN mask. Epoch-shaped masks are flattened in the same
            order as the training epochs.
        clean_reference : Raw | Epochs | Evoked | ndarray | None
            Optional independent clean data with matching channels and units. If
            no mask is supplied, all samples in ``X`` train the artifact
            covariance and all reference samples train the clean covariance.

        Returns
        -------
        self : MultichannelWienerFilter
            Fitted estimator.
        """
        del y
        data, sfreq_data, ch_names = self._extract_fit_data(X)
        sfreq = self._resolve_sfreq(sfreq_data)
        reference_data = None
        if clean_reference is not None:
            reference_data, reference_sfreq, _ = self._extract_fit_data(
                clean_reference, ch_names=ch_names
            )
            if reference_data.shape[0] != data.shape[0]:
                raise ValueError(
                    "clean_reference must have the same number of channels as X."
                )
            if (
                sfreq_data is not None
                and reference_sfreq is not None
                and not np.isclose(sfreq_data, reference_sfreq, rtol=1e-9, atol=0.0)
            ):
                raise ValueError(
                    "clean_reference sampling frequency must match the training data."
                )

        spatial_filter, normalized_mask, diagnostics = _fit_from_training_data(
            data,
            artifact_mask,
            clean_reference=reference_data,
            mask_strategy=self.mask_strategy,
            sfreq=sfreq,
            hf_hz=self.hf_hz,
            quantile=self.quantile,
            smooth_s=self.smooth_s,
            rank=self.rank,
            artifact_weight=self.artifact_weight,
            reg=self.reg,
            treat_nan=self.treat_nan,
        )
        self.spatial_filter_ = spatial_filter
        self.artifact_mask_ = normalized_mask
        self.artifact_fraction_ = diagnostics["artifact_fraction"]
        self.generalized_eigenvalues_ = diagnostics["generalized_eigenvalues"]
        self.artifact_eigenvalues_ = diagnostics["artifact_eigenvalues"]
        self.selected_components_ = diagnostics["selected_components"]
        self.fit_diagnostics_ = diagnostics
        self._n_channels_ = data.shape[0]
        self._fit_ch_names_ = ch_names
        if self.verbose:
            logger.info(
                "MWF fit: %d artifact samples, %d clean samples, rank %d.",
                diagnostics["artifact_samples"],
                diagnostics["clean_samples"],
                diagnostics["rank_used"],
            )
        return self

    def transform(self, X: Any) -> Any:
        """Apply the frozen spatial operator without refitting."""
        check_is_fitted(
            self,
            attributes=["spatial_filter_", "_n_channels_", "_fit_ch_names_"],
        )
        data, _, mne_type, orig_inst, picks, _ = extract_data_from_mne(
            X,
            ch_names=self._fit_ch_names_,
            auto_pick=True,
        )
        data = np.asarray(data, dtype=float)
        if not np.all(np.isfinite(data)):
            raise ValueError("X must contain only finite values.")
        n_channels = data.shape[1] if data.ndim == 3 else data.shape[0]
        if n_channels != self._n_channels_:
            raise ValueError(
                "X has a different number of channels than the fitted data "
                f"({n_channels} vs {self._n_channels_})."
            )
        cleaned = _apply_spatial_filter(data, self.spatial_filter_)
        return _reconstruct_like(cleaned, orig_inst, mne_type, picks)

    def fit_transform(
        self,
        X: Any,
        y=None,
        *,
        artifact_mask: np.ndarray | None = None,
        clean_reference: Any | None = None,
        **fit_params,
    ) -> Any:
        """Fit the operator and apply it to ``X``."""
        if fit_params:
            unknown = ", ".join(sorted(fit_params))
            raise TypeError(f"Unexpected fit parameters: {unknown}.")
        return self.fit(
            X,
            y,
            artifact_mask=artifact_mask,
            clean_reference=clean_reference,
        ).transform(X)


# Short established acronym retained as a documented compatibility alias.
MWF = MultichannelWienerFilter


__all__ = [
    "MWF",
    "MultichannelWienerFilter",
    "compute_mwf",
    "hf_power_mask",
    "mwf_filter",
]
