"""Shared dataclasses and types for the ASR module.

This module holds the fitted-state container shared by the standard, adaptive,
Juggler, and Riemannian ASR variants.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ASRState:
    """Fitted ASR calibration state.

    Parameters
    ----------
    M : ndarray, shape (n_channels, n_channels)
        Matrix square root of the robust calibration covariance.
    T : ndarray, shape (n_channels, n_channels)
        Direction-dependent threshold matrix.
    thresholds : ndarray, shape (n_channels,)
        Per-calibration-component RMS thresholds.
    calibration_patterns : ndarray, shape (n_channels, n_channels)
        Calibration covariance eigenvectors.
    filter_b : ndarray
        Numerator coefficients for the statistics-only filter.
    filter_a : ndarray
        Denominator coefficients for the statistics-only filter.
    filter_zi : ndarray
        Final causal filter state from calibration, used to initialize the
        statistics path during reconstruction without a filter discontinuity.
    cov : ndarray, shape (n_channels, n_channels)
        Robust calibration covariance.
    rank : int
        Numerical rank after regularization.
    method : {'standard', 'riemannian'}
        Covariance geometry used by the fitted state.
    riemannian_solver : str | None
        Experimental eigenspace strategy used for Riemannian ASR.
    """

    M: np.ndarray
    T: np.ndarray
    thresholds: np.ndarray
    calibration_patterns: np.ndarray
    filter_b: np.ndarray
    filter_a: np.ndarray
    cov: np.ndarray
    rank: int
    filter_zi: np.ndarray | None = None
    method: str = "standard"
    riemannian_solver: str | None = None


def _copy_asr_state(state: ASRState) -> ASRState:
    """Create a deep copy of an ASRState object.

    Parameters
    ----------
    state : ASRState
        The state object to copy.

    Returns
    -------
    ASRState
        A new ASRState instance with copied arrays.
    """
    return ASRState(
        M=state.M.copy(),
        T=state.T.copy(),
        thresholds=state.thresholds.copy(),
        calibration_patterns=state.calibration_patterns.copy(),
        filter_b=state.filter_b.copy(),
        filter_a=state.filter_a.copy(),
        filter_zi=None if state.filter_zi is None else state.filter_zi.copy(),
        cov=state.cov.copy(),
        rank=int(state.rank),
        method=state.method,
        riemannian_solver=state.riemannian_solver,
    )


def _copy_process_state(state: dict[str, Any]) -> dict[str, Any]:
    """Create a deep copy of an ASR process state dictionary.

    Parameters
    ----------
    state : dict
        The process state dictionary containing NumPy arrays or scalar values.

    Returns
    -------
    dict
        A new dictionary with copied arrays and values.
    """
    copied: dict[str, Any] = {}
    for key, value in state.items():
        copied[key] = value.copy() if isinstance(value, np.ndarray) else value
    return copied
