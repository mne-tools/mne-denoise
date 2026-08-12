"""Shared mechanics for Basic and local Singular Spectrum Analysis."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .._logging import set_log_level_from_verbose
from .._validation import (
    check_channel_first_data,
    check_channel_layout,
    check_sfreq,
    resolve_sfreq,
)
from ..utils import extract_data_from_mne, reconstruct_mne_object


def _check_positive_integer(value: int, *, name: str) -> int:
    """Validate a positive integer parameter."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer")
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _resolve_window_length(
    n_times: int,
    window_length: int | None,
    *,
    window_seconds: float | None = None,
    sfreq: float | None = None,
    max_window: int = 100,
) -> int:
    """Resolve an SSA embedding dimension in the canonical ``L <= K`` range."""
    if n_times < 3:
        raise ValueError("SSA requires at least 3 time samples")
    max_window = _check_positive_integer(max_window, name="max_window")
    if max_window < 2:
        raise ValueError("max_window must be at least 2")
    if window_length is not None and window_seconds is not None:
        raise ValueError("Specify only one of window_length and window_seconds")
    if window_seconds is not None:
        if isinstance(window_seconds, bool) or not isinstance(window_seconds, Real):
            raise TypeError("window_seconds must be a positive, finite number")
        window_seconds = float(window_seconds)
        if not np.isfinite(window_seconds) or window_seconds <= 0:
            raise ValueError("window_seconds must be a positive, finite number")
        sfreq = check_sfreq(sfreq, context="window_seconds")
        resolved = int(np.floor(window_seconds * sfreq + 0.5))
    elif window_length is not None:
        resolved = _check_positive_integer(window_length, name="window_length")
    elif sfreq is not None:
        sfreq = check_sfreq(sfreq)
        resolved = min(int(np.floor(0.5 * sfreq + 0.5)), max_window, (n_times + 1) // 2)
    else:
        resolved = min(n_times // 2, max_window)
    if resolved < 2:
        raise ValueError("window_length must be at least 2 samples")
    largest = (n_times + 1) // 2
    if resolved > largest:
        raise ValueError(
            "window_length must satisfy 2 <= window_length <= (n_times + 1) // 2"
        )
    return resolved


def _trajectory_matrix(x: np.ndarray, window_length: int) -> np.ndarray:
    """Return the standard ``(L, K)`` Hankel trajectory matrix."""
    return np.lib.stride_tricks.sliding_window_view(x, window_length).T


def _diagonal_average(matrix: np.ndarray) -> np.ndarray:
    """Average anti-diagonals of a 2-D matrix, including edge weights."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    n_rows, n_columns = matrix.shape
    out = np.zeros(n_rows + n_columns - 1, dtype=np.float64)
    weights = np.zeros_like(out)
    for row in range(n_rows):
        out[row : row + n_columns] += matrix[row]
        weights[row : row + n_columns] += 1.0
    return out / weights


class _BaseSSATransformer(BaseEstimator, TransformerMixin):
    """Shared MNE/scikit-learn integration for transductive SSA cleaners."""

    _name = "SSA"
    _requires_sfreq = False

    def _validate_fit_parameters(self, data: np.ndarray, sfreq: float | None) -> None:
        raise NotImplementedError

    def _compute_record(
        self, data: np.ndarray, sfreq: float | None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        raise NotImplementedError

    def fit(self, X: Any, y=None):
        """Validate the operating point and record the fitted channel layout."""
        set_log_level_from_verbose(self.verbose)
        data, data_sfreq, _kind, _orig, _picks, names = extract_data_from_mne(
            X, auto_pick=True
        )
        data = check_channel_first_data(
            data, name=self._name, allow_epochs=True, min_channels=1, min_times=3
        )
        sfreq = resolve_sfreq(
            self.sfreq,
            data_sfreq,
            context=self._name,
            required=self._requires_sfreq,
        )
        self._validate_fit_parameters(data, sfreq)
        self.sfreq_ = sfreq
        self.n_channels_in_ = data.shape[-2]
        self.ch_names_in_ = None if names is None else tuple(names)
        self.is_fitted_ = True
        return self

    def transform(self, X: Any, y=None) -> Any:
        """Apply the transductive SSA decomposition to the supplied records."""
        check_is_fitted(self, "is_fitted_")
        set_log_level_from_verbose(self.verbose)
        data, data_sfreq, kind, original, picks, names = extract_data_from_mne(
            X, auto_pick=True
        )
        data = check_channel_first_data(
            data, name=self._name, allow_epochs=True, min_channels=1, min_times=3
        )
        sfreq = resolve_sfreq(
            self.sfreq,
            data_sfreq,
            context=self._name,
            required=self._requires_sfreq,
        )
        if self.sfreq_ is not None and not np.isclose(sfreq, self.sfreq_):
            raise ValueError(
                f"transform sfreq={sfreq} differs from fitted sfreq={self.sfreq_}"
            )
        check_channel_layout(
            self._name,
            n_channels=data.shape[-2],
            fitted_n_channels=self.n_channels_in_,
            ch_names=names,
            fitted_ch_names=self.ch_names_in_,
        )
        if data.ndim == 2:
            cleaned, info = self._compute_record(data, sfreq)
            records = [info]
        else:
            cleaned = np.empty_like(data)
            records = []
            for epoch, values in enumerate(data):
                cleaned[epoch], info = self._compute_record(values, sfreq)
                records.append(info)
        self.diagnostics_ = records[0] if data.ndim == 2 else records
        self._set_diagnostic_attributes(records, epoched=data.ndim == 3)
        return reconstruct_mne_object(cleaned, original, kind, picks=picks)

    def _set_diagnostic_attributes(
        self, records: list[dict[str, Any]], *, epoched: bool
    ) -> None:
        """Expose algorithm-specific diagnostics after transform."""
        raise NotImplementedError
