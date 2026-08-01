"""Reference-free lagged CCA for broadband artifact attenuation.

The implementation follows the BSS-CCA construction of De Clercq et al.
(2006): CCA is solved between a multichannel signal and a delayed copy of the
same signal.  The lag is always explicit.  It can be declared in samples or in
physical time; a one-sample lag is never inserted silently.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ..icanclean._cca import canonical_correlation
from ..utils import extract_data_from_mne

logger = logging.getLogger(__name__)


def _validate_selection(rho_threshold: float, n_keep: int | None) -> None:
    """Validate component-selection parameters without data-dependent bounds."""
    if type(rho_threshold) not in (int, float) or not math.isfinite(
        float(rho_threshold)
    ):
        raise TypeError("rho_threshold must be a finite number")
    if not 0.0 <= float(rho_threshold) <= 1.0:
        raise ValueError("rho_threshold must be between 0 and 1")
    if n_keep is not None and (type(n_keep) is not int or n_keep < 1):
        raise ValueError("n_keep must be a positive integer or None")


def _resolve_lag_samples(
    *,
    lag_samples: int | None,
    lag_seconds: float | None,
    sfreq: float | None,
    n_times: int,
) -> int:
    """Resolve one explicit lag declaration to a positive sample count."""
    if (lag_samples is None) == (lag_seconds is None):
        raise ValueError("set exactly one of lag_samples or lag_seconds")
    if lag_samples is not None:
        if type(lag_samples) is not int or lag_samples < 1:
            raise ValueError("lag_samples must be a positive integer")
        resolved = lag_samples
    else:
        if type(lag_seconds) not in (int, float) or not math.isfinite(
            float(lag_seconds)
        ):
            raise TypeError("lag_seconds must be a finite number")
        if float(lag_seconds) <= 0:
            raise ValueError("lag_seconds must be positive")
        if sfreq is None:
            raise ValueError("sfreq is required when lag_seconds is used")
        if type(sfreq) not in (int, float) or not math.isfinite(float(sfreq)):
            raise TypeError("sfreq must be a finite number")
        if float(sfreq) <= 0:
            raise ValueError("sfreq must be positive")
        resolved = int(np.floor(float(lag_seconds) * float(sfreq) + 0.5))
        if resolved < 1:
            raise ValueError("lag_seconds resolves to less than one sample")
    if resolved >= n_times - 1:
        raise ValueError(
            f"lag ({resolved} samples) leaves fewer than two paired samples "
            f"for n_times={n_times}"
        )
    return resolved


def _lagged_pairs(X: np.ndarray, lag_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Return current/past CCA matrices without wrap or epoch-boundary pairs."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 2:
        if X.shape[0] < 1:
            raise ValueError("X must contain at least one channel")
        current = X[:, lag_samples:].T
        past = X[:, :-lag_samples].T
    elif X.ndim == 3:
        # Public 3-D convention: (n_epochs, n_channels, n_times).
        if X.shape[0] < 1 or X.shape[1] < 1:
            raise ValueError("X must contain at least one epoch and channel")
        current = np.transpose(X[:, :, lag_samples:], (0, 2, 1)).reshape(-1, X.shape[1])
        past = np.transpose(X[:, :, :-lag_samples], (0, 2, 1)).reshape(-1, X.shape[1])
    else:
        raise ValueError(
            "X must be 2-D (channels, samples) or 3-D (epochs, channels, samples)"
        )
    if current.shape[0] < 2:
        raise ValueError("lagged CCA requires at least two paired samples")
    if not np.isfinite(current).all() or not np.isfinite(past).all():
        raise ValueError("X must contain only finite values")
    return current, past


def _learn_lagged_cca_filter(
    X: np.ndarray,
    *,
    lag_samples: int,
    rho_threshold: float = 0.9,
    n_keep: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Learn the lagged-CCA spatial operator from continuous or epoched data."""
    _validate_selection(rho_threshold, n_keep)
    current, past = _lagged_pairs(X, lag_samples)
    filters_x, _filters_y, correlations, _scores_x, _scores_y = canonical_correlation(
        current, past
    )
    n_components = filters_x.shape[1]
    if n_components == 0:
        raise ValueError("lagged CCA found zero-rank current or lagged data")
    if n_keep is not None and n_keep > n_components:
        raise ValueError(f"n_keep={n_keep} exceeds the fitted CCA rank {n_components}")

    if n_keep is None:
        keep = correlations >= float(rho_threshold)
        if not keep.any():
            keep[0] = True  # Explicitly prevent numerical annihilation.
    else:
        keep = np.zeros(n_components, dtype=bool)
        keep[:n_keep] = True

    mixing = np.linalg.pinv(filters_x)
    cleaning = (filters_x @ np.diag(keep.astype(float)) @ mixing).T
    filters = filters_x.T
    patterns = mixing.T
    return cleaning, filters, patterns, correlations, keep


def _apply_operator(X: np.ndarray, cleaning_matrix: np.ndarray) -> np.ndarray:
    """Apply a fitted channel-space operator without crossing epoch boundaries."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 2:
        mean = X.mean(axis=1, keepdims=True)
        return cleaning_matrix @ (X - mean) + mean
    if X.ndim == 3:
        means = X.mean(axis=2, keepdims=True)
        return np.einsum("ij,ejt->eit", cleaning_matrix, X - means) + means
    raise ValueError(
        "X must be 2-D (channels, samples) or 3-D (epochs, channels, samples)"
    )


def _restore_container(
    cleaned: np.ndarray,
    *,
    original: object,
    mne_type: str,
    picks: np.ndarray | None,
) -> object:
    """Restore cleaned channels by copying the original MNE container."""
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


def compute_lagged_cca(
    X: np.ndarray,
    *,
    lag_samples: int | None = None,
    lag_seconds: float | None = None,
    sfreq: float | None = None,
    rho_threshold: float = 0.9,
    n_keep: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Learn and apply reference-free lagged CCA to an array.

    Exactly one of ``lag_samples`` or ``lag_seconds`` must be supplied.  For a
    leakage-safe train/evaluation split, use :class:`LaggedCCA`.

    Parameters
    ----------
    X : ndarray, shape (n_channels, n_times) | (n_epochs, n_channels, n_times)
        Continuous or epoched multichannel data.
    lag_samples : int | None
        Positive lag in samples.
    lag_seconds : float | None
        Positive physical lag in seconds. Requires ``sfreq``.
    sfreq : float | None
        Sampling frequency used only to resolve ``lag_seconds``.
    rho_threshold : float
        Keep canonical components with correlation at least this value.
    n_keep : int | None
        If provided, keep exactly the first ``n_keep`` components instead.

    Returns
    -------
    X_clean : ndarray
        Cleaned data with the same shape as ``X``.
    diagnostics : dict
        Fitted operator, component matrices, correlations, selection, and lag.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim not in (2, 3):
        raise ValueError("X must be a 2-D or 3-D array")
    n_times = X.shape[-1]
    resolved_lag = _resolve_lag_samples(
        lag_samples=lag_samples,
        lag_seconds=lag_seconds,
        sfreq=sfreq,
        n_times=n_times,
    )
    cleaning, filters, patterns, correlations, keep = _learn_lagged_cca_filter(
        X,
        lag_samples=resolved_lag,
        rho_threshold=rho_threshold,
        n_keep=n_keep,
    )
    cleaned = _apply_operator(X, cleaning)
    diagnostics = {
        "cleaning_matrix": cleaning,
        "filters": filters,
        "patterns": patterns,
        "correlations": correlations,
        "kept_mask": keep,
        "n_kept": int(keep.sum()),
        "n_removed": int(keep.size - keep.sum()),
        "lag_samples": resolved_lag,
        "lag_seconds": (None if sfreq is None else float(resolved_lag) / float(sfreq)),
    }
    return cleaned, diagnostics


class LaggedCCA(BaseEstimator, TransformerMixin):
    """Reference-free lagged-CCA muscle/EMG attenuation estimator.

    Exactly one lag representation is required.  ``fit`` learns a fixed
    channel-space operator, and ``transform`` applies that operator without
    refitting.

    Parameters
    ----------
    lag_samples : int | None
        Positive lag in samples.
    lag_seconds : float | None
        Positive lag in seconds. MNE inputs supply their own sampling frequency;
        NumPy inputs require ``sfreq``.
    sfreq : float | None
        Sampling frequency for NumPy data when ``lag_seconds`` is used. If an
        MNE input is supplied, a provided value must agree with ``info['sfreq']``.
    rho_threshold : float
        Keep canonical components with correlation at least this value.
    n_keep : int | None
        If provided, keep exactly the first ``n_keep`` components.
    verbose : bool | str | int | None
        Control logging verbosity.
    """

    def __init__(
        self,
        *,
        lag_samples: int | None = None,
        lag_seconds: float | None = None,
        sfreq: float | None = None,
        rho_threshold: float = 0.9,
        n_keep: int | None = None,
        verbose: bool | str | int | None = None,
    ) -> None:
        self.lag_samples = lag_samples
        self.lag_seconds = lag_seconds
        self.sfreq = sfreq
        self.rho_threshold = rho_threshold
        self.n_keep = n_keep
        self.verbose = verbose

    def _resolve_sfreq(self, data_sfreq: float | None) -> float | None:
        if (
            self.sfreq is not None
            and data_sfreq is not None
            and not np.isclose(float(self.sfreq), float(data_sfreq))
        ):
            raise ValueError(
                f"sfreq={self.sfreq} disagrees with MNE info sfreq={data_sfreq}"
            )
        return data_sfreq if data_sfreq is not None else self.sfreq

    def fit(self, X: Any, y=None) -> LaggedCCA:
        """Learn the lagged-CCA cleaning operator."""
        del y
        data, data_sfreq, mne_type, _orig, _picks, ch_names = extract_data_from_mne(
            X, auto_pick=True
        )
        if data.ndim == 3 and mne_type != "epochs" and not isinstance(X, np.ndarray):
            raise ValueError("three-dimensional inputs must be Epochs or an ndarray")
        sfreq = self._resolve_sfreq(data_sfreq)
        lag = _resolve_lag_samples(
            lag_samples=self.lag_samples,
            lag_seconds=self.lag_seconds,
            sfreq=sfreq,
            n_times=data.shape[-1],
        )
        (
            self.cleaning_matrix_,
            self.filters_,
            self.patterns_,
            self.correlations_,
            self.kept_mask_,
        ) = _learn_lagged_cca_filter(
            data,
            lag_samples=lag,
            rho_threshold=self.rho_threshold,
            n_keep=self.n_keep,
        )
        self.lag_samples_ = lag
        self.sfreq_ = None if sfreq is None else float(sfreq)
        self.n_features_in_ = int(self.cleaning_matrix_.shape[0])
        self.feature_names_in_ = None if ch_names is None else np.asarray(ch_names)
        self._mne_ch_names_ = ch_names
        self.n_kept_ = int(self.kept_mask_.sum())
        self.n_removed_ = int(self.kept_mask_.size - self.n_kept_)
        if self.verbose:
            logger.info(
                "LaggedCCA: lag=%d samples, kept %d and removed %d components.",
                self.lag_samples_,
                self.n_kept_,
                self.n_removed_,
            )
        return self

    def transform(self, X: Any) -> Any:
        """Apply the fixed cleaning operator to new data."""
        check_is_fitted(self, ("cleaning_matrix_", "lag_samples_"))
        data, data_sfreq, mne_type, orig_inst, picks, _names = extract_data_from_mne(
            X,
            ch_names=self._mne_ch_names_,
            auto_pick=True,
        )
        transform_sfreq = self._resolve_sfreq(data_sfreq)
        if (
            self.sfreq_ is not None
            and transform_sfreq is not None
            and not np.isclose(self.sfreq_, float(transform_sfreq))
        ):
            raise ValueError(
                f"transform sfreq={transform_sfreq} disagrees with fitted "
                f"sfreq={self.sfreq_}"
            )
        n_channels = data.shape[1] if data.ndim == 3 else data.shape[0]
        if n_channels != self.n_features_in_:
            raise ValueError(
                f"input has {n_channels} channels; fitted data had "
                f"{self.n_features_in_}"
            )
        cleaned = _apply_operator(data, self.cleaning_matrix_)
        return _restore_container(
            cleaned,
            original=orig_inst,
            mne_type=mne_type,
            picks=picks,
        )

    def fit_transform(self, X: Any, y=None, **fit_params) -> Any:
        """Fit on ``X`` and apply the fitted operator to ``X``."""
        return self.fit(X, y=y, **fit_params).transform(X)
