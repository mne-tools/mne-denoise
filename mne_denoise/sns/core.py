"""Sensor Noise Suppression (SNS) for channel-specific noise removal.

Sensor Noise Suppression (de Cheveigne & Simon 2008) [1]_ suppresses noise that
is *specific to individual sensors* by regenerating each channel from a
projection onto the subspace spanned by its most-correlated neighbour channels.
The rationale: genuine brain signal is spatially correlated across sensors and
is therefore well predicted by the other channels, whereas sensor-specific noise
(amplifier noise, a flaky electrode, SQUID noise) is uncorrelated across sensors
and cannot be predicted — so it is removed by the projection.

For each channel ``k``, SNS selects the ``n_neighbors`` channels most correlated
with ``k`` (optionally skipping the ``skip`` closest, which may share the same
local noise), and replaces ``x_k`` with its least-squares projection onto those
neighbours. Applied as a single spatial operator ``W`` (``X_clean = W @ X``),
with ``W[k, k] = 0`` so a channel is never regenerated from itself.

This is the reference-free, purely spatial counterpart to the
line/component-based denoisers in the package. It has no maintained standalone
Python package; the canonical implementations are de Cheveigne's MATLAB
NoiseTools (``nt_sns``) and its meegkit port.

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
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ..utils import extract_data_from_mne, reconstruct_mne_object

logger = logging.getLogger(__name__)


def _correlation(cov: np.ndarray) -> np.ndarray:
    """Convert a covariance matrix to a correlation matrix (safe on zero diag)."""
    d = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    denom = np.outer(d, d)
    denom[denom == 0.0] = 1.0
    return cov / denom


def compute_sns_weights(
    cov: np.ndarray, n_neighbors: int = 0, skip: int = 0
) -> tuple[np.ndarray, int]:
    """Build the SNS spatial operator ``W`` from a channel covariance matrix.

    Each channel is regenerated from a least-squares projection onto its
    ``n_neighbors`` most-correlated neighbours. Returns ``(W, n_neighbors_used)``
    where ``X_clean = W @ X`` and ``W[k, k] == 0``.

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

    Returns
    -------
    W : ndarray, shape (n_channels, n_channels)
        The SNS denoising operator (apply as ``X_clean = W @ X``).
    n_neighbors_used : int
        The effective number of neighbours after capping.
    """
    cov = np.asarray(cov, dtype=np.float64)
    n = cov.shape[0]
    if cov.shape != (n, n):
        raise ValueError(f"cov must be square (n_channels, n_channels), got {cov.shape}.")
    max_neighbors = max(0, n - skip - 1)
    k_neighbors = max_neighbors if n_neighbors in (0, None) else int(n_neighbors)
    k_neighbors = min(k_neighbors, max_neighbors)

    corr = _correlation(cov)
    W = np.zeros((n, n), dtype=np.float64)
    for k in range(n):
        if k_neighbors <= 0:
            W[k, k] = 1.0  # nothing to regenerate from -> pass channel through
            continue
        order = np.argsort(corr[:, k] ** 2)[::-1]  # most correlated first
        order = order[order != k]  # drop self explicitly (robust to correlation ties)
        idx = order[skip : skip + k_neighbors]  # optionally skip nearest, take neighbours
        if idx.size == 0:
            W[k, k] = 1.0
            continue
        c_nn = cov[np.ix_(idx, idx)]
        c_nk = cov[idx, k]
        # least-squares projection weights (pinv -> robust to rank deficiency);
        # basis-invariant to the PCA-whitening used by NoiseTools/meegkit.
        b = np.linalg.pinv(c_nn) @ c_nk
        W[k, idx] = b
    return W, k_neighbors


def compute_sns(
    X: np.ndarray, n_neighbors: int = 0, skip: int = 0
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

    Returns
    -------
    X_clean : ndarray, shape (n_channels, n_samples)
        Sensor-noise-suppressed signal.
    info : dict
        Diagnostics: ``weights`` (the operator ``W``) and ``n_neighbors`` used.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(
            f"Expected a 2-D (n_channels, n_samples) array, got shape {X.shape}."
        )
    Xd = X - X.mean(axis=1, keepdims=True)
    cov = (Xd @ Xd.T) / max(Xd.shape[1], 1)
    W, k = compute_sns_weights(cov, n_neighbors, skip)
    X_clean = W @ X
    return X_clean, {"weights": W, "n_neighbors": k}


class SNS(BaseEstimator, TransformerMixin):
    """Sensor Noise Suppression (de Cheveigne & Simon 2008).

    ``fit`` learns the spatial operator ``W`` on the training data (each channel
    regressed onto its most-correlated neighbours); ``transform`` applies that
    fixed operator to new data (leakage-safe). Accepts MNE ``Raw``/``Epochs``
    objects or NumPy ``(n_channels, n_samples)`` arrays.

    Parameters
    ----------
    n_neighbors : int
        Number of neighbour channels used to regenerate each channel. ``0``
        (default) uses all other channels (``n_channels - skip - 1``). On dense
        arrays a smaller value (e.g. 10-40) is faster and more robust.
    skip : int
        Number of closest (most-correlated) neighbours to skip when regenerating
        a channel, to avoid channels that share the same local noise.
    verbose : bool | str | int | None
        Control logging verbosity (MNE-style).

    Attributes
    ----------
    denoising_matrix_ : ndarray, shape (n_channels, n_channels)
        The learned SNS operator (``X_clean = denoising_matrix_ @ X``).
    n_neighbors_ : int
        Effective number of neighbours used per channel.

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
        verbose: bool | str | int | None = None,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.skip = skip
        self.verbose = verbose

    def _learn(self, data2d: np.ndarray) -> None:
        Xd = data2d - data2d.mean(axis=1, keepdims=True)
        cov = (Xd @ Xd.T) / max(Xd.shape[1], 1)
        self.denoising_matrix_, self.n_neighbors_ = compute_sns_weights(
            cov, self.n_neighbors, self.skip
        )

    def fit(self, X: Any, y=None) -> "SNS":
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
        data, _sfreq, mne_type, _orig, _picks, _names = extract_data_from_mne(
            X, auto_pick=True
        )
        if mne_type == "epochs":
            n_ep, n_ch, n_t = data.shape
            data2d = np.transpose(data, (1, 0, 2)).reshape(n_ch, n_ep * n_t)
        else:
            data2d = np.asarray(data, dtype=np.float64)
        self._learn(data2d)
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
        data, _sfreq, mne_type, orig_inst, picks, _names = extract_data_from_mne(
            X, auto_pick=True
        )
        if mne_type == "epochs":
            cleaned = np.empty_like(data, dtype=np.float64)
            for e in range(data.shape[0]):
                cleaned[e] = self.denoising_matrix_ @ np.asarray(data[e], dtype=np.float64)
        else:
            cleaned = self.denoising_matrix_ @ np.asarray(data, dtype=np.float64)
        return reconstruct_mne_object(
            cleaned, orig_inst, mne_type, picks=picks, verbose=False
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
