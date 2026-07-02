"""Reference-free BSS-CCA (auto-CCA) for muscle/EMG artifact removal.

Canonical-Correlation-Analysis blind source separation (BSS-CCA), introduced by
De Clercq et al. (2006) [1]_ for muscle-artifact removal, separates a
multichannel recording into components ordered by *autocorrelation* by running
CCA between the signal and a one-sample-lagged copy of itself. Slow, highly
autocorrelated components (neural rhythms) receive high canonical correlations;
broadband, weakly autocorrelated components (EMG/muscle) receive low ones.
Dropping the low-autocorrelation components and reconstructing removes muscle
activity while preserving neural structure.

This is the reference-*free* counterpart to the reference-aware
:class:`~mne_denoise.icanclean.ICanClean`: both are CCA-based spatial cleaners,
but iCanClean cancels artifacts shared with dedicated reference channels, while
auto-CCA needs no reference and separates on temporal autocorrelation alone. It
is a widely used muscle comparator in the mobile- and wearable-EEG literature
and has no maintained standalone Python implementation.

The CCA solver itself is reused from
:func:`mne_denoise.icanclean._cca.canonical_correlation`.

This module contains:

- ``compute_autocca``: the one-shot array-based algorithm (learn + apply).
- ``AutoCCA``: the scikit-learn estimator (leakage-safe ``fit``/``transform``),
  compatible with MNE-Python objects or NumPy arrays.

References
----------
.. [1] De Clercq, W., Vergult, A., Vanrumste, B., Van Paesschen, W., &
       Van Huffel, S. (2006). Canonical correlation analysis applied to remove
       muscle artifacts from the electroencephalogram. IEEE Transactions on
       Biomedical Engineering, 53(12), 2583-2587.
       https://doi.org/10.1109/TBME.2006.879459
.. [2] Safieddine, D., et al. (2012). Removal of muscle artifact from EEG data:
       comparison between stochastic (ICA and CCA) and deterministic (EMD and
       wavelet-based) approaches. EURASIP Journal on Advances in Signal
       Processing, 2012, 127. https://doi.org/10.1186/1687-6180-2012-127
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ..icanclean._cca import canonical_correlation
from ..utils import extract_data_from_mne, reconstruct_mne_object

logger = logging.getLogger(__name__)


def _learn_autocca_filter(
    X: np.ndarray, rho_threshold: float = 0.9, n_keep: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Learn the BSS-CCA spatial operator on ``X`` (n_channels, n_samples).

    Runs CCA between the signal and a one-sample-lagged copy, ranks the
    canonical components by autocorrelation, and builds a channel-space cleaning
    matrix that keeps only the high-autocorrelation (neural) components.

    Returns
    -------
    cleaning_matrix : ndarray, shape (n_channels, n_channels)
        Left-multiplying operator ``C`` such that
        ``X_clean = C @ (X - mean) + mean``.
    filters : ndarray, shape (n_components, n_channels)
        Canonical spatial filters (rows project channels to components).
    patterns : ndarray, shape (n_channels, n_components)
        Canonical spatial patterns (columns are component topographies).
    correlations : ndarray, shape (n_components,)
        Canonical correlations against the lagged signal (autocorrelations),
        sorted descending.
    kept_mask : ndarray of bool, shape (n_components,)
        ``True`` for components kept as neural, ``False`` for dropped artifact.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(
            f"Expected a 2-D (n_channels, n_samples) array, got shape {X.shape}."
        )
    n_ch = X.shape[0]
    Xs = X.T  # (n_samples, n_ch)
    Ys = np.roll(Xs, 1, axis=0)
    Ys[0] = Xs[0]  # one-sample lag without wrap-around leakage
    A, _B, R, _U, _V = canonical_correlation(Xs, Ys)  # A (n_ch, d), R (d,) descending
    d = A.shape[1]
    if d == 0:
        eye = np.eye(n_ch)
        empty_f = np.zeros((0, n_ch))
        empty_p = np.zeros((n_ch, 0))
        return eye, empty_f, empty_p, R, np.zeros(0, dtype=bool)

    if n_keep is not None:  # keep exactly the top-k by autocorrelation
        keep = np.zeros(d, dtype=bool)
        keep[: max(1, int(n_keep))] = True
    else:  # keep high-autocorrelation (neural); drop low (EMG/muscle)
        keep = R >= float(rho_threshold)
        if not keep.any():
            keep[0] = True  # never drop everything

    Ainv = np.linalg.pinv(A)  # (d, n_ch)
    cleaning = (A @ np.diag(keep.astype(np.float64)) @ Ainv).T  # (n_ch, n_ch)
    filters = A.T  # (d, n_ch)
    patterns = Ainv.T  # (n_ch, d)
    return cleaning, filters, patterns, R, keep


def compute_autocca(
    X: np.ndarray, rho_threshold: float = 0.9, n_keep: int | None = None
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reference-free BSS-CCA cleaning of a single data array (learn + apply).

    Convenience one-shot function that learns the CCA operator on ``X`` and
    immediately applies it to ``X``. For a leakage-safe train/evaluation split,
    use the :class:`AutoCCA` estimator (``fit`` on train, ``transform`` on
    evaluation data) instead.

    Parameters
    ----------
    X : ndarray, shape (n_channels, n_samples)
        Multichannel signal (single homogeneous channel type).
    rho_threshold : float
        Keep canonical components whose autocorrelation is at least this value;
        drop the rest as muscle. Ignored when ``n_keep`` is given.
    n_keep : int | None
        If set, keep exactly the top-``n_keep`` components by autocorrelation.

    Returns
    -------
    X_clean : ndarray, shape (n_channels, n_samples)
        Cleaned signal.
    info : dict
        Diagnostics: ``cleaning_matrix``, ``filters``, ``patterns``,
        ``correlations``, ``kept_mask``, ``n_kept``, ``n_removed``.
    """
    X = np.asarray(X, dtype=np.float64)
    cleaning, filters, patterns, R, keep = _learn_autocca_filter(
        X, rho_threshold, n_keep
    )
    m = X.mean(axis=1, keepdims=True)
    X_clean = cleaning @ (X - m) + m
    info = {
        "cleaning_matrix": cleaning,
        "filters": filters,
        "patterns": patterns,
        "correlations": R,
        "kept_mask": keep,
        "n_kept": int(keep.sum()),
        "n_removed": int(keep.size - keep.sum()),
    }
    return X_clean, info


class AutoCCA(BaseEstimator, TransformerMixin):
    """Reference-free BSS-CCA muscle/EMG remover (De Clercq et al. 2006).

    ``fit`` learns the spatial cleaning operator on the training data by CCA
    against a one-sample lag; ``transform`` applies that fixed operator to new
    data, so the train/evaluation leakage barrier is respected (unlike the
    one-shot :func:`compute_autocca`). Accepts MNE ``Raw``/``Epochs`` objects or
    NumPy arrays of shape ``(n_channels, n_samples)``.

    Parameters
    ----------
    rho_threshold : float
        Keep canonical components whose autocorrelation (canonical correlation
        against the lagged signal) is at least this value; drop the rest as
        muscle. Ignored when ``n_keep`` is given. Defaults to ``0.9``.
    n_keep : int | None
        If set, keep exactly the top-``n_keep`` components by autocorrelation.
    verbose : bool | str | int | None
        Control logging verbosity (MNE-style).

    Attributes
    ----------
    cleaning_matrix_ : ndarray, shape (n_channels, n_channels)
        The learned left-multiplying cleaning operator.
    filters_ : ndarray, shape (n_components, n_channels)
        Canonical spatial filters (rows project channels to components).
    patterns_ : ndarray, shape (n_channels, n_components)
        Canonical spatial patterns (columns are component topographies).
    correlations_ : ndarray, shape (n_components,)
        Canonical autocorrelations, sorted descending.
    kept_mask_ : ndarray of bool, shape (n_components,)
        ``True`` for kept (neural) components.
    n_kept_ : int
        Number of components kept.
    n_removed_ : int
        Number of components removed as muscle.

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.cca import AutoCCA
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((16, 2000))
    >>> cleaned = AutoCCA(rho_threshold=0.9).fit_transform(X)
    >>> cleaned.shape
    (16, 2000)
    """

    def __init__(
        self,
        rho_threshold: float = 0.9,
        n_keep: int | None = None,
        verbose: bool | str | int | None = None,
    ) -> None:
        self.rho_threshold = rho_threshold
        self.n_keep = n_keep
        self.verbose = verbose

    def _learn(self, data2d: np.ndarray) -> None:
        (
            self.cleaning_matrix_,
            self.filters_,
            self.patterns_,
            self.correlations_,
            self.kept_mask_,
        ) = _learn_autocca_filter(data2d, float(self.rho_threshold), self.n_keep)
        self.n_kept_ = int(self.kept_mask_.sum())
        self.n_removed_ = int(self.kept_mask_.size - self.n_kept_)

    def fit(self, X: Any, y=None) -> "AutoCCA":
        """Learn the BSS-CCA cleaning operator.

        Parameters
        ----------
        X : Raw | Epochs | ndarray
            Training data. Epochs are concatenated along time to learn a single
            spatial operator.
        y : None
            Ignored.

        Returns
        -------
        self : AutoCCA
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
                "AutoCCA: kept %d / removed %d components (rho_threshold=%s).",
                self.n_kept_,
                self.n_removed_,
                self.rho_threshold,
            )
        return self

    def _apply(self, x2d: np.ndarray) -> np.ndarray:
        x2d = np.asarray(x2d, dtype=np.float64)
        m = x2d.mean(axis=1, keepdims=True)
        return self.cleaning_matrix_ @ (x2d - m) + m

    def transform(self, X: Any, y=None) -> Any:
        """Apply the learned cleaning operator.

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
        check_is_fitted(self, "cleaning_matrix_")
        data, _sfreq, mne_type, orig_inst, picks, _names = extract_data_from_mne(
            X, auto_pick=True
        )
        if mne_type == "epochs":
            cleaned = np.empty_like(data, dtype=np.float64)
            for e in range(data.shape[0]):
                cleaned[e] = self._apply(data[e])
        else:
            cleaned = self._apply(data)
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
