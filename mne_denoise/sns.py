"""Sensor Noise Suppression (SNS)."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ._covariance import compute_covariance
from ._data import (
    continuous_to_epochs,
    epochs_to_continuous,
    extract_data_from_mne,
    reconstruct_mne_object,
)
from ._logging import logger, verbose
from ._spatial import apply_spatial_transform
from ._validation import (
    check_channel_first_data,
    check_channel_layout,
    check_chunk_size,
)
from .progress import _emit_progress, _ProgressCallback, _validate_callback

_DEFAULT_RCOND = 1e-12

__all__ = ["SNS", "compute_sns", "compute_sns_weights"]


def _automatic_sample_mask(
    data: np.ndarray,
    manual_weight: np.ndarray,
    threshold: float | None,
) -> np.ndarray:
    """Reject samples with a large robust deviation in any channel."""
    if threshold is None:
        return np.ones(data.shape[1], dtype=np.float64)
    included = manual_weight > 0
    reference = data[:, included]
    center = np.median(reference, axis=1, keepdims=True)
    mad = np.median(np.abs(reference - center), axis=1, keepdims=True)
    scale = 1.4826 * mad
    fallback = np.std(reference, axis=1, keepdims=True)
    scale = np.where(scale > 0, scale, fallback)
    scale = np.where(scale > 0, scale, 1.0)
    max_abs_z = np.max(np.abs((data - center) / scale), axis=0)
    return (max_abs_z <= threshold).astype(np.float64)


def _compute_sns_weights(
    cov: np.ndarray,
    n_neighbors: int = 0,
    skip: int = 0,
    *,
    rcond: float = _DEFAULT_RCOND,
    callback: _ProgressCallback | None = None,
    progress_offset: int = 0,
    progress_total: int | None = None,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Compute the SNS spatial operator from a channel covariance matrix."""
    cov = np.asarray(cov, dtype=np.float64)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(
            "cov must be a square (n_channels, n_channels) matrix, "
            f"got shape {cov.shape}"
        )
    if cov.shape[0] < 2:
        raise ValueError("SNS requires at least two channels")
    if not np.isfinite(cov).all():
        raise ValueError("cov must contain only finite values")
    scale = max(float(np.max(np.abs(cov))), np.finfo(np.float64).tiny)
    tolerance = 100 * np.finfo(np.float64).eps * scale * cov.shape[0]
    if float(np.max(np.abs(cov - cov.T))) > tolerance:
        raise ValueError("cov must be symmetric")
    cov = (cov + cov.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(cov)
    eigen_scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
    eigen_tolerance = 100 * np.finfo(float).eps * eigen_scale * cov.shape[0]
    if float(eigenvalues.min()) < -eigen_tolerance:
        raise ValueError("cov must be positive semidefinite")

    for value, name in ((n_neighbors, "n_neighbors"), (skip, "skip")):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be a non-negative integer")
        if value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if isinstance(rcond, bool) or not isinstance(rcond, Real):
        raise TypeError("rcond must be a finite number")
    rcond = float(rcond)
    if not np.isfinite(rcond) or not 0.0 < rcond < 1.0:
        raise ValueError("rcond must be finite and strictly between 0 and 1")

    n_channels = cov.shape[0]
    if progress_total is None:
        progress_total = n_channels
    if skip > n_channels - 2:
        raise ValueError("skip must leave at least one candidate neighbor")
    max_neighbors = n_channels - int(skip) - 1
    k_neighbors = (
        max_neighbors if n_neighbors == 0 else min(int(n_neighbors), max_neighbors)
    )
    standard_deviation = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    denominator = np.outer(standard_deviation, standard_deviation)
    denominator[denominator == 0.0] = 1.0
    correlation = cov / denominator

    weights = np.zeros_like(cov)
    neighbor_ranks = np.zeros(n_channels, dtype=int)
    for channel in range(n_channels):
        order = np.argsort(correlation[:, channel] ** 2, kind="stable")[::-1]
        order = order[order != channel]
        neighbors = order[int(skip) : int(skip) + k_neighbors]
        neighbor_cov = cov[np.ix_(neighbors, neighbors)]
        singular_values = np.linalg.eigvalsh(neighbor_cov)
        cutoff = rcond * max(float(singular_values.max()), 0.0)
        neighbor_ranks[channel] = np.count_nonzero(singular_values > cutoff)
        weights[channel, neighbors] = (
            np.linalg.pinv(neighbor_cov, rcond=rcond, hermitian=True)
            @ cov[neighbors, channel]
        )
        _emit_progress(
            callback,
            method="sns",
            stage="channel",
            current=progress_offset + channel + 1,
            total=progress_total,
            component=None,
            metric=float(neighbor_ranks[channel]),
        )
    return weights, k_neighbors, neighbor_ranks


def compute_sns_weights(
    cov: np.ndarray,
    n_neighbors: int = 0,
    skip: int = 0,
    *,
    rcond: float = _DEFAULT_RCOND,
    callback=None,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Compute SNS weights from a channel covariance matrix.

    Parameters
    ----------
    cov : ndarray, shape (n_channels, n_channels)
        Finite symmetric positive-semidefinite covariance matrix.
    n_neighbors : int, default=0
        Number of neighbors per channel; zero uses all available neighbors.
    skip : int, default=0
        Number of most-correlated neighbors to omit.
    rcond : float, default=1e-12
        Relative pseudoinverse cutoff.
    callback : callable or None, default=None
        Synchronous callback after each channel solve.

    Returns
    -------
    weights : ndarray, shape (n_channels, n_channels)
        Operator for centered channel-first data.
    n_neighbors_used : int
        Effective neighbor count.
    neighbor_ranks : ndarray, shape (n_channels,)
        Numerical rank of each selected neighbor covariance.

    See Also
    --------
    SNS
        Estimator that fits and reuses the SNS operator.
    compute_sns
        One-shot SNS operation for channel-first data.

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.sns import compute_sns_weights
    >>> rng = np.random.default_rng(0)
    >>> data = rng.standard_normal((8, 1000))
    >>> weights, n_neighbors, ranks = compute_sns_weights(np.cov(data), n_neighbors=4)
    """
    callback = _validate_callback(callback)
    cov = np.asarray(cov, dtype=np.float64)
    n_channels = cov.shape[0] if cov.ndim else 0
    return _compute_sns_weights(
        cov,
        n_neighbors=n_neighbors,
        skip=skip,
        rcond=rcond,
        callback=callback,
        progress_offset=0,
        progress_total=n_channels,
    )


@verbose
def compute_sns(
    X: np.ndarray,
    n_neighbors: int = 0,
    skip: int = 0,
    *,
    rcond: float = _DEFAULT_RCOND,
    preserve_mean: bool = False,
    n_iter: int = 1,
    outlier_threshold: float | None = None,
    chunk_size: int | None = None,
    sample_weight: np.ndarray | None = None,
    callback=None,
    verbose: bool | str | int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Learn and apply Sensor Noise Suppression to channel-first data.

    Parameters
    ----------
    X : ndarray, shape (n_channels, n_times) or (n_epochs, n_channels, n_times)
        Continuous or epoched data.
    n_neighbors : int, default=0
        Number of neighbors per channel; zero uses all available neighbors.
    skip : int, default=0
        Number of most-correlated neighbors to omit.
    rcond : float, default=1e-12
        Relative pseudoinverse cutoff.
    preserve_mean : bool, default=False
        Add the fitted channel means after regeneration.
    n_iter : int, default=1
        Number of SNS projections to compose.
    outlier_threshold : float or None, default=None
        Robust channel-wise z-score threshold for samples used during fitting.
    chunk_size : int or None, default=None
        Samples per covariance/application chunk.
    sample_weight : ndarray or None, shape (n_times,) or (n_epochs, n_times)
        Non-negative fitting weights.
    callback : callable or None, default=None
        Synchronous callback after each channel solve.
    verbose : bool, str, int, or None, default=None
        Logging level.

    Returns
    -------
    X_clean : ndarray
        Data with the same shape as X.
    info : dict
        Fitted operators and diagnostics.

    Notes
    -----
    SNS reconstructs each channel from spatially redundant signals in other
    channels using the channel covariance. It targets noise specific to individual
    sensors rather than a source or artifact shared across the array. With centered
    data, the learned operator is applied in channel space
    :footcite:p:`decheveigne_simon2008_sensor`.

    References
    ----------
    .. footbibliography::
    """
    callback = _validate_callback(callback)
    X = check_channel_first_data(X, name="SNS")
    if not isinstance(preserve_mean, bool):
        raise TypeError("preserve_mean must be a bool")
    if isinstance(n_iter, bool) or not isinstance(n_iter, Integral):
        raise TypeError("n_iter must be a positive integer")
    if n_iter < 1:
        raise ValueError("n_iter must be a positive integer")
    chunk_size = check_chunk_size(chunk_size)
    if outlier_threshold is not None:
        if isinstance(outlier_threshold, bool) or not isinstance(
            outlier_threshold, Real
        ):
            raise TypeError("outlier_threshold must be a positive number or None")
        outlier_threshold = float(outlier_threshold)
        if not np.isfinite(outlier_threshold) or outlier_threshold <= 0:
            raise ValueError("outlier_threshold must be finite and positive")

    continuous = epochs_to_continuous(X)
    expected_weight_shape = (X.shape[0], X.shape[2]) if X.ndim == 3 else (X.shape[1],)
    if sample_weight is None:
        manual_weight = np.ones(continuous.shape[1], dtype=np.float64)
    else:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
        if sample_weight.shape != expected_weight_shape:
            raise ValueError(
                "sample_weight must have shape "
                f"{expected_weight_shape}, got {sample_weight.shape}"
            )
        manual_weight = sample_weight.reshape(-1)
        if not np.isfinite(manual_weight).all() or np.any(manual_weight < 0):
            raise ValueError("sample_weight must be finite and non-negative")
        if np.count_nonzero(manual_weight > 0) < 2:
            raise ValueError(
                "sample_weight must weight at least two samples positively"
            )

    automatic_weight = _automatic_sample_mask(
        continuous, manual_weight, outlier_threshold
    )
    combined_weight = manual_weight * automatic_weight
    if np.count_nonzero(combined_weight > 0) < 2:
        raise ValueError(
            "fewer than two positively weighted samples remain after rejection"
        )
    training_mean = (continuous @ combined_weight / combined_weight.sum())[
        :, np.newaxis
    ]
    centered = continuous - training_mean
    current = centered
    composite = np.eye(continuous.shape[0], dtype=np.float64)
    matrices = []
    ranks = []
    effective_neighbors = 0
    n_iter = int(n_iter)
    n_channels = continuous.shape[0]
    for iteration in range(n_iter):
        cov = compute_covariance(
            current,
            weights=combined_weight,
            assume_centered=True,
            chunk_size=chunk_size,
        )
        matrix, effective_neighbors, iteration_ranks = _compute_sns_weights(
            cov,
            n_neighbors=n_neighbors,
            skip=skip,
            rcond=rcond,
            callback=callback,
            progress_offset=iteration * n_channels,
            progress_total=n_iter * n_channels,
        )
        matrices.append(matrix)
        ranks.append(iteration_ranks)
        composite = matrix @ composite
        logger.debug(
            "SNS iteration %d/%d: effective neighbours=%d, median local rank=%.1f.",
            iteration + 1,
            n_iter,
            effective_neighbors,
            float(np.median(iteration_ranks)),
        )
        if iteration + 1 < n_iter:
            current = apply_spatial_transform(matrix, current, chunk_size=chunk_size)

    cleaned = apply_spatial_transform(composite, centered, chunk_size=chunk_size)
    if preserve_mean:
        cleaned += training_mean
    cleaned = continuous_to_epochs(cleaned, X.shape)
    info = {
        "weights": composite,
        "denoising_matrix": composite,
        "denoising_matrices": tuple(matrices),
        "training_mean": training_mean,
        "n_neighbors": effective_neighbors,
        "requested_n_neighbors": int(n_neighbors),
        "skip": int(skip),
        "rcond": float(rcond),
        "preserve_mean": preserve_mean,
        "n_iter": n_iter,
        "outlier_threshold": outlier_threshold,
        "chunk_size": chunk_size,
        "neighbor_ranks": ranks[-1],
        "neighbor_ranks_per_iteration": tuple(ranks),
        "input_rank": int(np.linalg.matrix_rank(centered)),
        "effective_weight_sum": float(combined_weight.sum()),
        "rejected_sample_count": int(np.count_nonzero(automatic_weight == 0)),
    }
    logger.info(
        "SNS: learned %d iteration(s) on %d channels (%d neighbours each; "
        "%d samples rejected).",
        info["n_iter"],
        X.shape[-2],
        info["n_neighbors"],
        info["rejected_sample_count"],
    )
    return cleaned, info


class SNS(BaseEstimator, TransformerMixin):
    """Sensor Noise Suppression estimator.

    The estimator learns a channel mean and spatial operator from training data and
    reuses both during transform.

    Parameters
    ----------
    n_neighbors : int, default=0
        Number of neighbors per channel; zero uses all available neighbors.
    skip : int, default=0
        Number of most-correlated neighbors to omit.
    rcond : float, default=1e-12
        Relative pseudoinverse cutoff.
    preserve_mean : bool, default=False
        Add the fitted channel mean after regeneration.
    verbose : bool, str, int, or None, default=None
        Logging level.
    n_iter : int, default=1
        Number of SNS projections to compose.
    outlier_threshold : float or None, default=None
        Robust channel-wise z-score threshold for fitting.
    chunk_size : int or None, default=None
        Samples per covariance/application chunk.

    Attributes
    ----------
    training_mean_ : ndarray
        Weighted channel mean.
    denoising_matrix_ : ndarray
        Composite spatial operator.
    denoising_matrices_ : tuple of ndarray
        One operator per iteration.
    neighbor_ranks_per_iteration_ : tuple of ndarray
        Local covariance ranks by iteration.

    See Also
    --------
    compute_sns
        One-shot SNS operation for channel-first arrays.
    compute_sns_weights
        Construct the local sensor reconstruction operator.
    mne_denoise.sound.SOUND
        Forward-model-based sensor-noise suppression.

    Notes
    -----
    Channel-first NumPy arrays and MNE Raw, Epochs, and Evoked inputs are supported;
    transform returns the corresponding type without mutating the input. SNS
    reconstructs each channel from spatially redundant signals in other channels;
    it is intended for noise specific to individual sensors, not for a source or
    artifact shared across the array :footcite:p:`decheveigne_simon2008_sensor`.

    References
    ----------
    .. footbibliography::

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.sns import SNS
    >>> rng = np.random.default_rng(0)
    >>> data = rng.standard_normal((8, 1000))
    >>> model = SNS(n_neighbors=4)
    >>> clean = model.fit_transform(data)
    """

    def __init__(
        self,
        n_neighbors: int = 0,
        skip: int = 0,
        rcond: float = _DEFAULT_RCOND,
        preserve_mean: bool = False,
        verbose: bool | str | int | None = None,
        n_iter: int = 1,
        outlier_threshold: float | None = None,
        chunk_size: int | None = None,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.skip = skip
        self.rcond = rcond
        self.preserve_mean = preserve_mean
        self.verbose = verbose
        self.n_iter = n_iter
        self.outlier_threshold = outlier_threshold
        self.chunk_size = chunk_size

    @verbose
    def fit(
        self,
        X: Any,
        y=None,
        sample_weight: np.ndarray | None = None,
        *,
        callback=None,
        verbose: bool | str | int | None = None,
    ) -> SNS:
        """Fit the SNS mean and spatial operator.

        Parameters
        ----------
        X : array-like or MNE Raw, Epochs, or Evoked
            Data used to learn the operator.
        y : None, default=None
            Ignored for scikit-learn compatibility.
        sample_weight : ndarray or None, default=None
            Non-negative fitting weights.
        callback : callable or None, default=None
            Synchronous channel-solve callback.
        verbose : bool, str, int, or None, default=None
            Logging level.

        Returns
        -------
        SNS
            The fitted estimator.
        """
        callback = _validate_callback(callback)
        data, _sfreq, _mne_type, _orig, _picks, names = extract_data_from_mne(
            X, auto_pick=True
        )
        _cleaned, info = compute_sns(
            np.asarray(data, dtype=np.float64),
            n_neighbors=self.n_neighbors,
            skip=self.skip,
            rcond=self.rcond,
            preserve_mean=self.preserve_mean,
            n_iter=self.n_iter,
            outlier_threshold=self.outlier_threshold,
            chunk_size=self.chunk_size,
            sample_weight=sample_weight,
            callback=callback,
        )
        self.training_mean_ = info["training_mean"]
        self.denoising_matrix_ = info["denoising_matrix"]
        self.denoising_matrices_ = info["denoising_matrices"]
        self.n_neighbors_ = info["n_neighbors"]
        self.neighbor_ranks_per_iteration_ = info["neighbor_ranks_per_iteration"]
        self.neighbor_ranks_ = self.neighbor_ranks_per_iteration_[-1]
        self.input_rank_ = info["input_rank"]
        self.n_channels_in_ = self.denoising_matrix_.shape[0]
        self.n_iter_ = info["n_iter"]
        self.chunk_size_ = info["chunk_size"]
        self.effective_weight_sum_ = info["effective_weight_sum"]
        self.rejected_sample_count_ = info["rejected_sample_count"]
        self.feature_names_in_ = None if names is None else tuple(names)
        return self

    @verbose
    def transform(
        self,
        X: Any,
        y=None,
        *,
        verbose: bool | str | int | None = None,
    ) -> Any:
        """Apply the fitted SNS operator.

        Parameters
        ----------
        X : array-like or MNE Raw, Epochs, or Evoked
            Data with the fitted channel layout.
        y : None, default=None
            Ignored for scikit-learn compatibility.
        verbose : bool, str, int, or None, default=None
            Logging level.

        Returns
        -------
        same type as X
            A cleaned copy.
        """
        check_is_fitted(self, ("denoising_matrix_", "training_mean_"))
        data, _sfreq, mne_type, orig_inst, picks, names = extract_data_from_mne(
            X, auto_pick=True
        )
        data = check_channel_first_data(data, name="SNS")
        if not isinstance(self.preserve_mean, bool):
            raise TypeError("preserve_mean must be a bool")
        check_channel_layout(
            "SNS",
            n_channels=data.shape[-2],
            fitted_n_channels=self.n_channels_in_,
            ch_names=None if names is None else tuple(names),
            fitted_ch_names=self.feature_names_in_,
        )
        continuous = epochs_to_continuous(data)
        cleaned = apply_spatial_transform(
            self.denoising_matrix_,
            continuous - self.training_mean_,
            chunk_size=self.chunk_size_,
        )
        if self.preserve_mean:
            cleaned += self.training_mean_
        cleaned = continuous_to_epochs(cleaned, data.shape)
        return reconstruct_mne_object(cleaned, orig_inst, mne_type, picks=picks)

    @verbose
    def fit_transform(
        self,
        X: Any,
        y=None,
        *,
        sample_weight: np.ndarray | None = None,
        callback=None,
        verbose: bool | str | int | None = None,
        **fit_params,
    ) -> Any:
        """Fit SNS and apply the fitted operator.

        Parameters
        ----------
        X : array-like or MNE Raw, Epochs, or Evoked
            Data to fit and transform.
        y : None, default=None
            Ignored for scikit-learn compatibility.
        sample_weight : ndarray or None, default=None
            Non-negative fitting weights.
        callback : callable or None, default=None
            Synchronous channel-solve callback.
        verbose : bool, str, int, or None, default=None
            Logging level.
        **fit_params : dict
            Unexpected fit parameters raise TypeError.

        Returns
        -------
        same type as X
            Sensor-noise-suppressed data.
        """
        if fit_params:
            unexpected = ", ".join(sorted(fit_params))
            raise TypeError(f"Unexpected fit parameters: {unexpected}")
        return self.fit(X, y, sample_weight=sample_weight, callback=callback).transform(
            X
        )
