"""Sensor Noise Suppression (SNS) for channel-specific noise removal.

Sensor Noise Suppression (de Cheveigne & Simon 2008) [1]_ suppresses noise that
is *specific to individual sensors* by regenerating each channel from a
projection onto the subspace spanned by its most-correlated neighbour channels.
The method assumes that signals of interest are spatially correlated across
sensors and that sensor-specific noise is not. Whether those assumptions hold
must be validated for each acquisition regime.

For each channel ``k``, SNS selects the ``n_neighbors`` channels most correlated
with ``k`` (optionally skipping the ``skip`` closest, which may share the same
local noise), and replaces ``x_k`` with its least-squares projection onto those
neighbours. Applied to centered data as a single spatial operator ``W``
(``X_clean = W @ X_centered``), with ``W[k, k] = 0`` so a channel is never
regenerated from itself.

This is the reference-free, purely spatial counterpart to the
line/component-based denoisers in the package. The canonical reference
implementations are de Cheveigne's MATLAB NoiseTools
(``nt_sns``) and the maintained MEEGkit Python port.

This module contains:

- ``compute_sns_weights``: build the SNS spatial operator from a covariance.
- ``compute_sns``: one-shot SNS cleaning of an array (learn + apply).
- ``SNS``: the scikit-learn estimator (leakage-safe ``fit``/``transform``),
  compatible with MNE-Python objects or NumPy arrays.

References
----------
.. [1] de Cheveigne, A., & Simon, J. Z. (2008). Sensor noise suppression.
       Journal of Neuroscience Methods, 168(1), 195-202.
       https://doi.org/10.1016/j.jneumeth.2007.09.012
"""

from __future__ import annotations

import logging
import math
from numbers import Integral, Real
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ..utils import extract_data_from_mne

logger = logging.getLogger(__name__)

_DEFAULT_RCOND = 1e-12


def _validate_count(value: int, *, name: str) -> int:
    """Validate a non-negative integer operating-point parameter."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _validate_rcond(rcond: float) -> float:
    """Validate the relative pseudoinverse cutoff."""
    if isinstance(rcond, bool) or not isinstance(rcond, Real):
        raise TypeError("rcond must be a finite number")
    rcond = float(rcond)
    if not math.isfinite(rcond) or not 0.0 < rcond < 1.0:
        raise ValueError("rcond must be finite and strictly between 0 and 1")
    return rcond


def _validate_covariance(cov: np.ndarray) -> np.ndarray:
    """Return a finite, symmetric, positive-semidefinite covariance."""
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
    return cov


def _validate_data(X: np.ndarray, *, allow_epochs: bool) -> np.ndarray:
    """Validate channel-first continuous or epoched data."""
    X = np.asarray(X, dtype=np.float64)
    expected = (2, 3) if allow_epochs else (2,)
    if X.ndim not in expected:
        shape_text = "2-D or 3-D" if allow_epochs else "2-D"
        raise ValueError(f"Expected a {shape_text} channel-first array, got {X.shape}")
    if X.shape[-2] < 2:
        raise ValueError("SNS requires at least two channels")
    if X.shape[-1] < 2:
        raise ValueError("SNS requires at least two time samples")
    if X.ndim == 3 and X.shape[0] < 1:
        raise ValueError("SNS requires at least one epoch")
    if not np.isfinite(X).all():
        raise ValueError("X must contain only finite values")
    return X


def _correlation(cov: np.ndarray) -> np.ndarray:
    """Convert a covariance matrix to a correlation matrix (safe on zero diag)."""
    d = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    denom = np.outer(d, d)
    denom[denom == 0.0] = 1.0
    return cov / denom


def _compute_sns_weights(
    cov: np.ndarray,
    n_neighbors: int,
    skip: int,
    rcond: float,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Build an SNS operator and retain the effective local ranks."""
    cov = _validate_covariance(cov)
    n_neighbors = _validate_count(n_neighbors, name="n_neighbors")
    skip = _validate_count(skip, name="skip")
    rcond = _validate_rcond(rcond)
    n_channels = cov.shape[0]
    if skip > n_channels - 2:
        raise ValueError("skip must leave at least one candidate neighbor")
    max_neighbors = n_channels - skip - 1
    k_neighbors = max_neighbors if n_neighbors == 0 else min(n_neighbors, max_neighbors)
    if k_neighbors < 1:  # pragma: no cover - guarded by validation above
        raise ValueError("SNS requires at least one neighbor per channel")

    corr = _correlation(cov)
    weights = np.zeros((n_channels, n_channels), dtype=np.float64)
    neighbor_ranks = np.zeros(n_channels, dtype=int)
    for channel in range(n_channels):
        order = np.argsort(corr[:, channel] ** 2, kind="stable")[::-1]
        order = order[order != channel]
        neighbors = order[skip : skip + k_neighbors]
        neighbor_cov = cov[np.ix_(neighbors, neighbors)]
        cross_cov = cov[neighbors, channel]
        singular_values = np.linalg.eigvalsh(neighbor_cov)
        cutoff = rcond * max(float(singular_values.max()), 0.0)
        neighbor_ranks[channel] = int(np.count_nonzero(singular_values > cutoff))
        coefficients = (
            np.linalg.pinv(neighbor_cov, rcond=rcond, hermitian=True) @ cross_cov
        )
        weights[channel, neighbors] = coefficients
    return weights, k_neighbors, neighbor_ranks


def compute_sns_weights(
    cov: np.ndarray,
    n_neighbors: int = 0,
    skip: int = 0,
    *,
    rcond: float = _DEFAULT_RCOND,
) -> tuple[np.ndarray, int]:
    """Build the SNS spatial operator ``W`` from a channel covariance matrix.

    Each channel is regenerated from a least-squares projection onto its
    ``n_neighbors`` most-correlated neighbours. Returns ``(W, n_neighbors_used)``
    where ``X_clean = W @ X_centered`` and ``W[k, k] == 0``.

    Parameters
    ----------
    cov : ndarray, shape (n_channels, n_channels)
        Channel covariance (e.g. ``X @ X.T / n_times`` on demeaned data).
    n_neighbors : int
        Number of neighbour channels used to regenerate each channel. ``0``
        means "all others" (``n_channels - skip - 1``).
    skip : int
        Number of closest (most-correlated) neighbours to skip, e.g. to avoid
        regenerating a channel from immediate neighbours that share local noise.
    rcond : float
        Relative cutoff used for the Hermitian pseudoinverse of each neighbor
        covariance. It must be strictly between zero and one.

    Returns
    -------
    W : ndarray, shape (n_channels, n_channels)
        The SNS denoising operator (apply to centered data).
    n_neighbors_used : int
        The effective number of neighbours after capping.
    """
    weights, k_neighbors, _ = _compute_sns_weights(cov, n_neighbors, skip, rcond)
    return weights, k_neighbors


def compute_sns(
    X: np.ndarray,
    n_neighbors: int = 0,
    skip: int = 0,
    *,
    rcond: float = _DEFAULT_RCOND,
    preserve_mean: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Sensor-noise-suppress a data array (learn + apply in one call).

    Convenience one-shot function. For a leakage-safe train/evaluation split use
    the :class:`SNS` estimator instead.

    Parameters
    ----------
    X : ndarray, shape (n_channels, n_samples)
        Multichannel signal.
    n_neighbors : int
        Number of neighbour channels per regeneration (``0`` = all others).
    skip : int
        Number of closest neighbours to skip.
    rcond : float
        Relative pseudoinverse cutoff used for rank-deficient neighbor sets.
    preserve_mean : bool
        If ``False`` (default), reproduce the reference SNS convention and
        return centered regenerations. If ``True``, restore each input-channel
        mean after applying the operator to its centered fluctuations.

    Returns
    -------
    X_clean : ndarray, shape (n_channels, n_samples)
        Sensor-noise-suppressed signal.
    info : dict
        Diagnostics include the operator, effective neighbor count, local
        neighbor ranks, input rank, and the complete numerical operating point.
    """
    X = _validate_data(X, allow_epochs=False)
    if not isinstance(preserve_mean, bool):
        raise TypeError("preserve_mean must be a bool")
    input_mean = X.mean(axis=1, keepdims=True)
    Xd = X - input_mean
    cov = (Xd @ Xd.T) / max(Xd.shape[1], 1)
    weights, effective_neighbors, neighbor_ranks = _compute_sns_weights(
        cov, n_neighbors, skip, rcond
    )
    X_clean = weights @ Xd
    if preserve_mean:
        X_clean += input_mean
    return X_clean, {
        "weights": weights,
        "n_neighbors": effective_neighbors,
        "requested_n_neighbors": int(n_neighbors),
        "skip": int(skip),
        "rcond": float(rcond),
        "preserve_mean": preserve_mean,
        "neighbor_ranks": neighbor_ranks,
        "input_rank": int(np.linalg.matrix_rank(Xd)),
    }


class SNS(BaseEstimator, TransformerMixin):
    """Sensor Noise Suppression (de Cheveigne & Simon 2008).

    ``fit`` learns the spatial operator ``W`` on the training data (each channel
    regressed onto its most-correlated neighbours); ``transform`` applies that
    fixed operator to new data (leakage-safe). Accepts MNE ``Raw``/``Epochs``
    objects or NumPy ``(n_channels, n_samples)`` and
    ``(n_epochs, n_channels, n_samples)`` arrays.

    Parameters
    ----------
    n_neighbors : int
        Number of neighbour channels used to regenerate each channel. ``0``
        (default) uses all other channels (``n_channels - skip - 1``). On dense
        arrays a smaller value (e.g. 10-40) is faster and more robust.
    skip : int
        Number of closest (most-correlated) neighbours to skip when regenerating
        a channel, to avoid channels that share the same local noise.
    rcond : float
        Relative cutoff used for rank-deficient neighbor covariance matrices.
    preserve_mean : bool
        If ``False`` (default), match the reference SNS convention by returning
        centered regenerations. If ``True``, restore the input-channel means.
    verbose : bool | str | int | None
        Control logging verbosity (MNE-style).

    Attributes
    ----------
    denoising_matrix_ : ndarray, shape (n_channels, n_channels)
        The learned SNS operator, applied to centered data.
    n_neighbors_ : int
        Effective number of neighbours used per channel.
    neighbor_ranks_ : ndarray, shape (n_channels,)
        Numerical rank of each channel's selected neighbor covariance.
    input_rank_ : int
        Rank of the centered training data.
    feature_names_in_ : tuple of str | None
        Names and order of fitted MNE channels, when available.

    Notes
    -----
    SNS suppresses noise that is specific to individual sensors; it does **not**
    target physiological artifacts (ocular, muscle, cardiac) that are themselves
    spatially correlated. Use it as a first-stage sensor-cleanup, complementary
    to the artifact-specific denoisers.

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.sns import SNS
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((32, 5000))
    >>> cleaned = SNS(n_neighbors=8).fit_transform(X)
    >>> cleaned.shape
    (32, 5000)
    """

    def __init__(
        self,
        n_neighbors: int = 0,
        skip: int = 0,
        rcond: float = _DEFAULT_RCOND,
        preserve_mean: bool = False,
        verbose: bool | str | int | None = None,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.skip = skip
        self.rcond = rcond
        self.preserve_mean = preserve_mean
        self.verbose = verbose

    def _learn(self, data2d: np.ndarray) -> None:
        data2d = _validate_data(data2d, allow_epochs=False)
        if not isinstance(self.preserve_mean, bool):
            raise TypeError("preserve_mean must be a bool")
        Xd = data2d - data2d.mean(axis=1, keepdims=True)
        cov = (Xd @ Xd.T) / max(Xd.shape[1], 1)
        (
            self.denoising_matrix_,
            self.n_neighbors_,
            self.neighbor_ranks_,
        ) = _compute_sns_weights(cov, self.n_neighbors, self.skip, self.rcond)
        self.input_rank_ = int(np.linalg.matrix_rank(Xd))
        self.n_channels_in_ = data2d.shape[0]

    @staticmethod
    def _as_continuous(data: np.ndarray) -> np.ndarray:
        """Concatenate epochs without allowing channel-axis ambiguity."""
        data = _validate_data(data, allow_epochs=True)
        if data.ndim == 2:
            return data
        n_epochs, n_channels, n_times = data.shape
        return np.transpose(data, (1, 0, 2)).reshape(n_channels, n_epochs * n_times)

    def _apply_operator(self, data: np.ndarray) -> np.ndarray:
        """Apply the fixed operator using the reference centering convention."""
        data = _validate_data(data, allow_epochs=True)
        if not isinstance(self.preserve_mean, bool):
            raise TypeError("preserve_mean must be a bool")
        if data.shape[-2] != self.n_channels_in_:
            raise ValueError(
                "X has a different channel count from fit: "
                f"expected {self.n_channels_in_}, got {data.shape[-2]}"
            )
        if data.ndim == 2:
            input_mean = data.mean(axis=1, keepdims=True)
            cleaned = self.denoising_matrix_ @ (data - input_mean)
        else:
            input_mean = data.mean(axis=(0, 2), keepdims=True)
            cleaned = np.einsum(
                "ij,ejt->eit", self.denoising_matrix_, data - input_mean
            )
        if self.preserve_mean:
            cleaned += input_mean
        self.last_input_means_ = np.asarray(input_mean).copy()
        return cleaned

    @staticmethod
    def _restore_container(
        cleaned: np.ndarray,
        *,
        original: Any,
        mne_type: str,
        picks: np.ndarray | None,
    ) -> Any:
        """Restore cleaned channels in a copy of the original MNE container."""
        if mne_type == "array" or original is None:
            return cleaned
        output = original.copy()
        if mne_type in ("raw", "epochs"):
            output.load_data()
            if picks is None:
                output._data[...] = cleaned
            elif mne_type == "epochs":
                output._data[:, picks, :] = cleaned
            else:
                output._data[picks, :] = cleaned
        elif mne_type == "evoked":
            if picks is None:
                output.data[...] = cleaned
            else:
                output.data[picks, :] = cleaned
        else:  # pragma: no cover - guarded by extract_data_from_mne
            raise TypeError(f"unsupported container type {mne_type!r}")
        return output

    def _check_channel_names(self, names: list[str] | None) -> None:
        """Prevent applying a spatial operator to reordered named channels."""
        fitted = self.feature_names_in_
        current = None if names is None else tuple(names)
        if fitted is not None and current != fitted:
            raise ValueError(
                "MNE channel names/order differ from fit; apply SNS to the exact "
                "fitted channel layout"
            )

    def fit(self, X: Any, y=None) -> SNS:
        """Learn the SNS spatial operator.

        Parameters
        ----------
        X : Raw | Epochs | ndarray
            Training data. Epochs are concatenated along time.
        y : None
            Ignored.

        Returns
        -------
        self : SNS
        """
        data, _sfreq, _mne_type, _orig, _picks, names = extract_data_from_mne(
            X, auto_pick=True
        )
        data2d = self._as_continuous(np.asarray(data, dtype=np.float64))
        self._learn(data2d)
        self.feature_names_in_ = None if names is None else tuple(names)
        if self.verbose:
            logger.info(
                "SNS: learned operator on %d channels (%d neighbours each).",
                self.denoising_matrix_.shape[0],
                self.n_neighbors_,
            )
        return self

    def transform(self, X: Any, y=None) -> Any:
        """Apply the learned SNS operator.

        Parameters
        ----------
        X : Raw | Epochs | ndarray
            Data to clean (same channel layout as the fitted data).
        y : None
            Ignored.

        Returns
        -------
        X_clean : Raw | Epochs | ndarray
            Cleaned data in the same format as the input.
        """
        check_is_fitted(self, "denoising_matrix_")
        data, _sfreq, mne_type, orig_inst, picks, names = extract_data_from_mne(
            X, auto_pick=True
        )
        self._check_channel_names(names)
        cleaned = self._apply_operator(np.asarray(data, dtype=np.float64))
        return self._restore_container(
            cleaned, original=orig_inst, mne_type=mne_type, picks=picks
        )

    def fit_transform(self, X: Any, y=None, **fit_params) -> Any:
        """Fit on ``X`` and apply to ``X`` in one step.

        Parameters
        ----------
        X : Raw | Epochs | ndarray
            Input data.
        y : None
            Ignored.
        **fit_params
            Ignored.

        Returns
        -------
        X_clean : Raw | Epochs | ndarray
            Cleaned data.
        """
        return self.fit(X, y).transform(X)
