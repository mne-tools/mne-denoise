"""Local Singular Spectrum Analysis for high-amplitude artifact removal.

Local SSA embeds a scalar series as overlapping delay vectors, partitions those
vectors into locally similar states, and estimates a separate principal
subspace in every cluster. Minimum-description-length model selection controls
the local subspace dimensions. Reversing the clustering and applying
anti-diagonal averaging yields a coherent, high-energy reconstruction that is
treated as artifact and subtracted from the observation [1]_.

This signal interpretation is method-specific: genuine high-amplitude neural
activity can also occupy the leading local subspaces and be attenuated.

References
----------
.. [1] Teixeira, A. R., Tome, A. M., Lang, E. W., Gruber, P., & Martins da
       Silva, A. (2006). Automatic removal of high-amplitude artefacts from
       single-channel electroencephalograms. Computer Methods and Programs in
       Biomedicine, 83, 125-138. https://doi.org/10.1016/j.cmpb.2006.06.003
"""

from __future__ import annotations

import logging
from numbers import Integral
from typing import Any

import numpy as np
from sklearn.cluster import KMeans

from .._validation import check_channel_first_data, check_positive_integer, check_sfreq
from ._common import (
    _BaseSSATransformer,
    _diagonal_average,
    _resolve_window_length,
    _trajectory_matrix,
)

logger = logging.getLogger(__name__)


def _mdl_order(eigenvalues: np.ndarray, n_observations: int) -> tuple[int, np.ndarray]:
    """Select a local PCA dimension using Teixeira et al. Eqs. (4)-(6).

    The likelihood term compares the geometric and arithmetic means of the
    discarded covariance eigenvalues. The complexity penalty increases with
    the candidate subspace dimension.
    """
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
    """Cluster delay vectors and fit the local PCA/MDL models.

    Automatic selection searches from the largest admissible cluster count
    downward. A clustering is admissible when every cluster contains at least
    one observation per embedding dimension; its MDL result is considered
    reliable when no selected local dimension exceeds half the embedding
    dimension. Explicit cluster counts retain the MDL result after enforcing
    only the observation-count condition.
    """
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
    """Validate local SSA clustering parameters."""
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
    """Remove a locally reconstructed high-energy artifact from one channel.

    Delay vectors are clustered, projected onto cluster-specific subspaces
    selected by MDL, returned to their temporal positions, and averaged along
    trajectory-matrix anti-diagonals.

    Parameters
    ----------
    x : array-like, shape (n_times,)
        Finite scalar time series.
    window_length : int | None, default=None
        Delay-vector dimension in samples. It must satisfy the canonical SSA
        orientation and is mutually exclusive with ``window_seconds``.
    window_seconds : float | None, default=None
        Delay-vector duration in seconds. It requires ``sfreq`` and is mutually
        exclusive with ``window_length``.
    sfreq : float | None, default=None
        Sampling frequency in Hz. Required for ``window_seconds`` and used by
        automatic window selection when available.
    n_clusters : int | "auto", default="auto"
        Number of delay-vector clusters. ``"auto"`` searches downward from the
        largest admissible value until the reliability conditions are met.
    max_clusters : int, default=10
        Upper bound for automatic cluster-count selection. The source does not
        prescribe this computational bound.
    max_window : int, default=100
        Maximum delay-vector dimension used by automatic window selection.
    random_state : int | None, default=0
        Random seed passed to k-means. None permits nondeterministic
        initialization.
    return_info : bool, default=False
        If True, also return clustering, eigenspectrum, MDL, and reconstructed
        artifact diagnostics.

    Returns
    -------
    x_clean : ndarray, shape (n_times,)
        Residual after subtracting the local-subspace reconstruction.
    info : dict
        Returned only when ``return_info=True``. Contains the artifact,
        trajectory shape, cluster labels and sizes, covariance eigenvalues, MDL
        scores, and selected subspace dimensions.

    Raises
    ------
    TypeError
        If a scalar parameter has an invalid type.
    ValueError
        If ``x``, the embedding, or the requested clustering is invalid, or no
        reliable automatic clustering can be found.

    See Also
    --------
    compute_local_ssa : Apply local SSA independently across channels.
    LocalSingularSpectrumAnalysis : MNE/scikit-learn estimator interface.

    Notes
    -----
    The method assumes that coherent, high-energy structure belongs to the
    artifact and that desired EEG is represented more strongly in the residual
    subspace. This assumption can fail for genuine rhythmic neural activity.
    K-means initialization, the maximum cluster count, and zero-eigenvalue
    regularization are explicit numerical choices because the source does not
    uniquely specify them [1]_.

    References
    ----------
    .. [1] Teixeira, A. R., Tome, A. M., Lang, E. W., Gruber, P., & Martins da
           Silva, A. (2006). Automatic removal of high-amplitude artefacts from
           single-channel electroencephalograms. Computer Methods and Programs
           in Biomedicine, 83, 125-138.
           https://doi.org/10.1016/j.cmpb.2006.06.003

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.ssa import local_ssa_clean_channel
    >>> time = np.arange(300) / 100.0
    >>> observed = np.sin(2 * np.pi * 0.5 * time)
    >>> cleaned = local_ssa_clean_channel(
    ...     observed, window_length=20, n_clusters=2, random_state=0
    ... )
    >>> cleaned.shape
    (300,)
    """
    n_clusters, max_clusters, random_state = _check_local_parameters(
        n_clusters, max_clusters, random_state
    )
    if not isinstance(return_info, bool):
        raise TypeError("return_info must be a bool")
    if sfreq is not None:
        sfreq = check_sfreq(sfreq)
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
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply local SSA independently to every input channel.

    This is the channel-first functional interface to
    :func:`local_ssa_clean_channel`. It returns both cleaned data and the local
    model diagnostics required to assess the reconstruction.

    Parameters
    ----------
    X : array-like, shape (n_channels, n_times)
        Finite channel-first data. Channels are never mixed.
    window_length : int | None, default=None
        Delay-vector dimension in samples. If None, it is selected
        automatically.
    window_seconds : float | None, default=None
        Delay-vector duration in seconds, mutually exclusive with
        ``window_length``.
    sfreq : float | None, default=None
        Sampling frequency in Hz. Required for ``window_seconds`` and used by
        automatic window selection when available.
    n_clusters : int | "auto", default="auto"
        Number of delay-vector clusters, or automatic reliable selection.
    max_clusters : int, default=10
        Upper bound for automatic cluster-count selection.
    max_window : int, default=100
        Maximum delay-vector dimension used by automatic window selection.
    random_state : int | None, default=0
        Random seed passed to k-means.

    Returns
    -------
    X_clean : ndarray, shape (n_channels, n_times)
        Residual data after independently subtracting each channel's local
        reconstruction.
    info : dict
        Per-channel cluster counts, cluster sizes, selected dimensions,
        covariance eigenvalues, MDL scores, and reconstructed artifacts.

    Raises
    ------
    TypeError
        If a scalar parameter has an invalid type.
    ValueError
        If ``X``, the embedding, or the requested clustering is invalid.

    See Also
    --------
    local_ssa_clean_channel : Canonical single-channel implementation.
    LocalSingularSpectrumAnalysis : MNE/scikit-learn estimator interface.

    Notes
    -----
    This function calls :func:`local_ssa_clean_channel` independently for every
    channel. It is repeated univariate local SSA, not multivariate SSA. No
    spatial covariance or cross-channel trajectory matrix is estimated.

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.ssa import compute_local_ssa
    >>> time = np.arange(300) / 100.0
    >>> observed = np.vstack(
    ...     [np.sin(2 * np.pi * 0.5 * time), np.sin(2 * np.pi * 1.0 * time)]
    ... )
    >>> cleaned, info = compute_local_ssa(
    ...     observed, window_length=20, n_clusters=2, random_state=0
    ... )
    >>> cleaned.shape
    (2, 300)
    >>> info["n_clusters"].shape
    (2,)
    """
    X = check_channel_first_data(
        X, name="local SSA", allow_epochs=False, min_channels=1, min_times=3
    )
    cleaned = np.empty_like(X)
    records = []
    for channel in X:
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
        cleaned[len(records)] = result
        records.append(info)
    return cleaned, {
        "method": "local-mdl",
        "window_length": records[0]["window_length"],
        "n_clusters": np.array([record["n_clusters"] for record in records]),
        "cluster_sizes": [record["cluster_sizes"] for record in records],
        "subspace_dimensions": [record["subspace_dimensions"] for record in records],
        "eigenvalues": [record["eigenvalues"] for record in records],
        "mdl_scores": [record["mdl_scores"] for record in records],
        "artifacts": np.stack([record["artifact"] for record in records]),
    }


class LocalSingularSpectrumAnalysis(_BaseSSATransformer):
    """Local-SSA high-amplitude artifact transformer.

    The estimator applies the clustered local-subspace reconstruction of
    Teixeira et al. independently to each selected channel and exposes the
    fitted clustering diagnostics after transformation.

    Parameters
    ----------
    window_length : int | None, default=None
        Delay-vector dimension in samples. It is mutually exclusive with
        ``window_seconds``. None selects an automatic value.
    window_seconds : float | None, default=None
        Delay-vector duration in seconds. It requires a sampling frequency and
        is mutually exclusive with ``window_length``.
    sfreq : float | None, default=None
        Sampling frequency in Hz. MNE input supplies it from metadata and must
        agree with an explicit value.
    n_clusters : int | "auto", default="auto"
        Number of delay-vector clusters, or automatic reliable selection.
    max_clusters : int, default=10
        Upper bound for automatic cluster-count selection.
    max_window : int, default=100
        Maximum delay-vector dimension used by automatic window selection.
    random_state : int | None, default=0
        Random seed passed to k-means.
    verbose : bool | str | int | None, default=None
        MNE-style logging level.

    Attributes
    ----------
    sfreq_ : float | None
        Validated sampling frequency, or None when sample-based parameters and
        NumPy input do not require one.
    n_channels_in_ : int
        Number of data channels seen during fitting.
    ch_names_in_ : tuple of str | None
        Fitted MNE channel names and order, or None for NumPy input.
    diagnostics_ : dict | list of dict
        Diagnostics from the most recent transformation. Epoched input stores
        one dictionary per epoch.
    n_clusters_ : ndarray
        Effective cluster count per channel, or per epoch and channel.
    subspace_dimensions_ : list
        Selected local subspace dimensions for every channel.

    See Also
    --------
    compute_local_ssa : Functional interface for channel-first arrays.
    local_ssa_clean_channel : Canonical single-channel implementation.
    mne_denoise.ssa.SingularSpectrumAnalysis : Basic SSA with frequency grouping.

    Notes
    -----
    The estimator is transductive. ``fit`` validates the operating point and
    records the channel layout; every ``transform`` clusters and decomposes the
    records supplied to that call. Record and epoch boundaries can therefore
    change the delay vectors, clusters, covariance spectra, and reconstruction.

    Local SSA assumes that coherent, high-energy structure is artifact. Genuine
    neural activity that satisfies the same local-subspace model can be removed
    [1]_.

    References
    ----------
    .. [1] Teixeira, A. R., Tome, A. M., Lang, E. W., Gruber, P., & Martins da
           Silva, A. (2006). Automatic removal of high-amplitude artefacts from
           single-channel electroencephalograms. Computer Methods and Programs
           in Biomedicine, 83, 125-138.
           https://doi.org/10.1016/j.cmpb.2006.06.003

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.ssa import LocalSingularSpectrumAnalysis
    >>> sfreq = 100.0
    >>> time = np.arange(500) / sfreq
    >>> data = np.vstack(
    ...     [np.sin(2 * np.pi * 0.5 * time), np.sin(2 * np.pi * 10.0 * time)]
    ... )
    >>> model = LocalSingularSpectrumAnalysis(
    ...     window_length=20, n_clusters=2, random_state=0
    ... )
    >>> cleaned = model.fit_transform(data)
    >>> cleaned.shape
    (2, 500)
    """

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
        self, data: np.ndarray, sfreq: float | None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        return compute_local_ssa(
            data,
            self.window_length,
            window_seconds=self.window_seconds,
            sfreq=sfreq,
            n_clusters=self.n_clusters,
            max_clusters=self.max_clusters,
            max_window=self.max_window,
            random_state=self.random_state,
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
        logger.info(
            "Local SSA: used a mean of %.1f clusters/channel.",
            float(np.mean(self.n_clusters_)),
        )
