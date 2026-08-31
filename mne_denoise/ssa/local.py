"""Local Singular Spectrum Analysis."""

from __future__ import annotations

from numbers import Integral
from typing import Any

import numpy as np
from sklearn.cluster import KMeans

from .._logging import logger, verbose
from .._validation import (
    check_channel_first_data,
    check_positive_integer,
    check_positive_real,
)
from ..progress import _emit_progress, _ProgressCallback, _validate_callback
from ._common import (
    _BaseSSATransformer,
    _diagonal_average,
    _resolve_window_length,
    _trajectory_matrix,
)


def _mdl_order(eigenvalues: np.ndarray, n_observations: int) -> tuple[int, np.ndarray]:
    """Select a local PCA dimension with the implemented MDL scores."""
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    n_dimensions = eigenvalues.size
    if n_dimensions < 2:
        return 1, np.empty(0)
    if not np.any(eigenvalues > 0):
        return 1, np.full(n_dimensions - 1, np.inf)
    tiny = np.finfo(float).tiny
    scores = np.empty(n_dimensions - 1, dtype=np.float64)
    for index, order in enumerate(range(1, n_dimensions)):
        discarded = np.maximum(eigenvalues[order:], tiny)
        log_ratio = np.mean(np.log(discarded)) - np.log(np.mean(discarded))
        negative_log_likelihood = -n_observations * (n_dimensions - order) * log_ratio
        degrees = order * n_dimensions - 0.5 * order * (order - 1) + 1.0
        scores[index] = negative_log_likelihood + 0.5 * degrees * np.log(n_observations)
    return int(np.argmin(scores) + 1), scores


def _fit_local_clusters(
    trajectory: np.ndarray,
    n_clusters: int | str,
    *,
    max_clusters: int,
    random_state: int | None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Cluster delay vectors and fit one local PCA model per cluster."""
    window_length, n_vectors = trajectory.shape
    if n_clusters == "auto" and np.all(trajectory == trajectory[:, :1]):
        n_clusters = 1
    if n_clusters == "auto":
        first = min(max_clusters, n_vectors // window_length)
        candidates = range(first, 0, -1)
    else:
        candidates = (n_clusters,)
    last_reason = ""
    for candidate in candidates:
        if candidate == 1:
            labels = np.zeros(n_vectors, dtype=int)
        else:
            labels = KMeans(
                n_clusters=candidate,
                n_init=10,
                random_state=random_state,
            ).fit_predict(trajectory.T)
        sizes = np.bincount(labels, minlength=candidate)
        if np.any(sizes < window_length):
            last_reason = "each cluster must contain at least window_length vectors"
            continue
        models = []
        reliable = True
        for cluster in range(candidate):
            indices = np.flatnonzero(labels == cluster)
            values = trajectory[:, indices]
            mean = values.mean(axis=1, keepdims=True)
            centered = values - mean
            covariance = centered @ centered.T / indices.size
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            order = np.argsort(eigenvalues)[::-1]
            eigenvalues = np.maximum(eigenvalues[order], 0.0)
            eigenvectors = eigenvectors[:, order]
            dimension, scores = _mdl_order(eigenvalues, indices.size)
            if dimension > window_length // 2:
                reliable = False
            models.append(
                {
                    "indices": indices,
                    "mean": mean,
                    "eigenvalues": eigenvalues,
                    "eigenvectors": eigenvectors,
                    "dimension": dimension,
                    "mdl_scores": scores,
                }
            )
        if reliable or n_clusters != "auto":
            return labels, models
        last_reason = "MDL selected more than half the embedding dimensions"
    raise ValueError("No reliable local SSA clustering was found; " + last_reason)


def _check_local_parameters(
    n_clusters: int | str,
    max_clusters: int,
    random_state: int | None,
) -> tuple[int | str, int, int | None]:
    """Validate local-SSA clustering parameters."""
    if n_clusters != "auto":
        n_clusters = check_positive_integer(n_clusters, name="n_clusters")
    max_clusters = check_positive_integer(max_clusters, name="max_clusters")
    if random_state is not None:
        if isinstance(random_state, bool) or not isinstance(random_state, Integral):
            raise TypeError("random_state must be an integer or None")
        random_state = int(random_state)
    return n_clusters, max_clusters, random_state


def local_ssa_clean_channel(
    x: np.ndarray,
    window_length: int | None = None,
    *,
    window_seconds: float | None = None,
    sfreq: float | None = None,
    n_clusters: int | str = "auto",
    max_clusters: int = 10,
    max_window: int = 100,
    random_state: int | None = 0,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Clean one channel with clustered local SSA reconstruction.

    Parameters
    ----------
    x : array-like, shape (n_times,)
        Finite scalar time series.
    window_length : int | None, default=None
        Delay-vector dimension in samples; None selects it automatically.
    window_seconds : float | None, default=None
        Delay-vector duration in seconds, requiring sfreq and mutually exclusive
        with window_length.
    sfreq : float | None, default=None
        Sampling frequency in Hz.
    n_clusters : int or "auto", default="auto"
        Number of delay-vector clusters, or automatic reliable selection.
    max_clusters : int, default=10
        Upper bound for automatic cluster selection.
    max_window : int, default=100
        Maximum automatic delay-vector dimension.
    random_state : int | None, default=0
        Seed passed to k-means.
    return_info : bool, default=False
        If True, also return clustering and reconstruction diagnostics.

    Returns
    -------
    x_clean : ndarray, shape (n_times,)
        Residual after subtracting the local-subspace reconstruction.
    info : dict
        Diagnostics returned only when return_info=True.

    Notes
    -----
    Delay vectors are clustered, projected onto the cluster-specific MDL-selected
    subspaces, and reconstructed by anti-diagonal averaging. Genuine structure
    matching the selected subspaces can also be removed.
    :footcite:p:`teixeira2006_local_ssa`.

    References
    ----------
    .. footbibliography::
    """
    n_clusters, max_clusters, random_state = _check_local_parameters(
        n_clusters, max_clusters, random_state
    )
    if not isinstance(return_info, bool):
        raise TypeError("return_info must be a bool")
    if sfreq is not None:
        sfreq = check_positive_real(sfreq, name="sfreq")
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("x must be one-dimensional")
    if not np.isfinite(x).all():
        raise ValueError("x must contain only finite values")
    resolved = _resolve_window_length(
        x.size,
        window_length,
        window_seconds=window_seconds,
        sfreq=sfreq,
        max_window=max_window,
    )
    trajectory = _trajectory_matrix(x, resolved)
    labels, models = _fit_local_clusters(
        trajectory,
        n_clusters=n_clusters,
        max_clusters=max_clusters,
        random_state=random_state,
    )
    artifact_trajectory = np.empty_like(trajectory)
    for model in models:
        indices = model["indices"]
        values = trajectory[:, indices]
        centered = values - model["mean"]
        basis = model["eigenvectors"][:, : model["dimension"]]
        artifact_trajectory[:, indices] = model["mean"] + basis @ (basis.T @ centered)
    artifact = _diagonal_average(artifact_trajectory)
    cleaned = x - artifact
    info = {
        "artifact": artifact,
        "window_length": resolved,
        "trajectory_shape": trajectory.shape,
        "n_clusters": len(models),
        "labels": labels,
        "cluster_sizes": np.array([model["indices"].size for model in models]),
        "subspace_dimensions": np.array(
            [model["dimension"] for model in models], dtype=int
        ),
        "eigenvalues": [model["eigenvalues"] for model in models],
        "mdl_scores": [model["mdl_scores"] for model in models],
    }
    if return_info:
        return cleaned, info
    return cleaned


@verbose
def compute_local_ssa(
    X: np.ndarray,
    window_length: int | None = None,
    *,
    window_seconds: float | None = None,
    sfreq: float | None = None,
    n_clusters: int | str = "auto",
    max_clusters: int = 10,
    max_window: int = 100,
    random_state: int | None = 0,
    callback=None,
    verbose: bool | str | int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply local SSA independently to each channel.

    Parameters
    ----------
    X : array-like, shape (n_channels, n_times)
        Finite channel-first data. Channels are not mixed.
    window_length : int | None, default=None
        Delay-vector dimension in samples; None selects it automatically.
    window_seconds : float | None, default=None
        Delay-vector duration in seconds, requiring sfreq and mutually exclusive
        with window_length.
    sfreq : float | None, default=None
        Sampling frequency in Hz.
    n_clusters : int or "auto", default="auto"
        Number of delay-vector clusters, or automatic reliable selection.
    max_clusters : int, default=10
        Upper bound for automatic cluster selection.
    max_window : int, default=100
        Maximum automatic delay-vector dimension.
    random_state : int | None, default=0
        Seed passed to k-means.
    callback : callable | None, default=None
        Synchronous callback after each channel; return values are ignored and
        callback exceptions propagate.
    verbose : bool, str, int, or None
        Logging level.

    Returns
    -------
    X_clean : ndarray, shape (n_channels, n_times)
        Independently cleaned channels.
    info : dict
        Per-channel cluster, subspace, and reconstruction diagnostics.

    Notes
    -----
    This is repeated univariate local SSA, not multivariate SSA.
    """
    callback = _validate_callback(callback)
    X = check_channel_first_data(
        X, name="local SSA", allow_epochs=False, min_channels=1, min_times=3
    )
    cleaned = np.empty_like(X)
    records = []
    for channel_idx, channel in enumerate(X):
        result, info = local_ssa_clean_channel(
            channel,
            window_length,
            window_seconds=window_seconds,
            sfreq=sfreq,
            n_clusters=n_clusters,
            max_clusters=max_clusters,
            max_window=max_window,
            random_state=random_state,
            return_info=True,
        )
        cleaned[channel_idx] = result
        records.append(info)
        _emit_progress(
            callback,
            method="local_ssa",
            stage="channel",
            current=channel_idx + 1,
            total=X.shape[0],
            component=None,
            metric=float(info["n_clusters"]),
        )
    info = {
        "method": "local-mdl",
        "window_length": records[0]["window_length"],
        "n_clusters": np.array([record["n_clusters"] for record in records]),
        "cluster_sizes": [record["cluster_sizes"] for record in records],
        "subspace_dimensions": [record["subspace_dimensions"] for record in records],
        "eigenvalues": [record["eigenvalues"] for record in records],
        "mdl_scores": [record["mdl_scores"] for record in records],
        "artifacts": np.stack([record["artifact"] for record in records]),
    }
    logger.info(
        "Local SSA: window=%d samples, channels=%d, mean clusters=%.1f, "
        "mean subspace dimension=%.1f.",
        info["window_length"],
        X.shape[0],
        float(np.mean(info["n_clusters"])),
        float(
            np.mean([np.mean(dimensions) for dimensions in info["subspace_dimensions"]])
        ),
    )
    return cleaned, info


class LocalSingularSpectrumAnalysis(_BaseSSATransformer):
    """Channel-wise local-SSA transformer for high-amplitude artifact reconstruction.

    Parameters
    ----------
    window_length : int | None, default=None
        Delay-vector dimension in samples; None selects it automatically.
    window_seconds : float | None, default=None
        Delay-vector duration in seconds, requiring sfreq and mutually exclusive
        with window_length.
    sfreq : float | None, default=None
        Sampling frequency in Hz.
    n_clusters : int or "auto", default="auto"
        Number of delay-vector clusters, or automatic reliable selection.
    max_clusters : int, default=10
        Upper bound for automatic cluster selection.
    max_window : int, default=100
        Maximum automatic delay-vector dimension.
    random_state : int | None, default=0
        Seed passed to k-means.
    verbose : bool, str, int, or None, default=None
        Logging level.

    Attributes
    ----------
    sfreq_ : float | None
        Validated sampling frequency.
    n_channels_in_ : int
        Number of fitted data channels.
    ch_names_in_ : tuple of str or None
        Fitted MNE channel names and order, or None for arrays.
    diagnostics_ : dict | list of dict
        Diagnostics from the most recent transform.
    n_clusters_ : ndarray
        Effective cluster counts.
    subspace_dimensions_ : list
        Selected local subspace dimensions.

    See Also
    --------
    SingularSpectrumAnalysis
        Frequency-guided Basic SSA.
    compute_local_ssa
        One-shot Local SSA interface.

    Notes
    -----
    The estimator is transductive: each transform clusters and reconstructs the
    records supplied to it. Local SSA can remove genuine structure that matches
    the learned local subspaces.
    :footcite:p:`teixeira2006_local_ssa`.

    References
    ----------
    .. footbibliography::

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.ssa import LocalSingularSpectrumAnalysis
    >>> rng = np.random.default_rng(0)
    >>> data = rng.standard_normal((8, 200))
    >>> model = LocalSingularSpectrumAnalysis(window_length=40, n_clusters="auto")
    >>> clean = model.fit_transform(data)
    """

    _progress_method = "local_ssa"

    def __init__(
        self,
        window_length: int | None = None,
        *,
        window_seconds: float | None = None,
        sfreq: float | None = None,
        n_clusters: int | str = "auto",
        max_clusters: int = 10,
        max_window: int = 100,
        random_state: int | None = 0,
        verbose: bool | str | int | None = None,
    ) -> None:
        self.window_length = window_length
        self.window_seconds = window_seconds
        self.sfreq = sfreq
        self.n_clusters = n_clusters
        self.max_clusters = max_clusters
        self.max_window = max_window
        self.random_state = random_state
        self.verbose = verbose

    def _validate_fit_parameters(self, data: np.ndarray, sfreq: float | None) -> None:
        _check_local_parameters(self.n_clusters, self.max_clusters, self.random_state)
        _resolve_window_length(
            data.shape[-1],
            self.window_length,
            window_seconds=self.window_seconds,
            sfreq=sfreq,
            max_window=self.max_window,
        )

    def _compute_record(
        self,
        data: np.ndarray,
        sfreq: float | None,
        *,
        callback: _ProgressCallback | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        # The estimator owns one aggregate SSA report; suppress the core's
        # standalone summary for each record while retaining its computation.
        return compute_local_ssa(
            data,
            self.window_length,
            window_seconds=self.window_seconds,
            sfreq=sfreq,
            n_clusters=self.n_clusters,
            max_clusters=self.max_clusters,
            max_window=self.max_window,
            random_state=self.random_state,
            callback=callback,
            verbose="WARNING",
        )

    def _set_diagnostic_attributes(
        self, records: list[dict[str, Any]], *, epoched: bool
    ) -> None:
        if epoched:
            self.n_clusters_ = np.stack([record["n_clusters"] for record in records])
            self.subspace_dimensions_ = [
                record["subspace_dimensions"] for record in records
            ]
        else:
            self.n_clusters_ = records[0]["n_clusters"]
            self.subspace_dimensions_ = records[0]["subspace_dimensions"]
        dimensions = [
            np.mean(values)
            for record_dimensions in self.subspace_dimensions_
            for values in record_dimensions
        ]
        logger.info(
            "Local SSA: window=%s samples, channels=%d, mean clusters=%.1f, "
            "mean subspace dimension=%.1f.",
            records[0].get("window_length", self.window_length or "auto"),
            self.n_channels_in_,
            float(np.mean(self.n_clusters_)),
            float(np.mean(dimensions)),
        )
