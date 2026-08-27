"""Whitening utilities for data covariance and MNE sensor data.

The data-covariance API computes a reduced-rank transform from an empirical
covariance. The sensor API follows MNE pre-whitening conventions for mixed
channel types.

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any

import numpy as np
from numpy.linalg import LinAlgError

from ... import _mne
from ..._covariance import compute_covariance
from ..._spatial import apply_spatial_transform


def apply_covariance_transform(
    matrix: np.ndarray, covariance: np.ndarray
) -> np.ndarray:
    """Apply a spatial transform to both axes of a covariance matrix."""
    matrix = np.asarray(matrix)
    covariance = np.asarray(covariance)
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2D, got {matrix.ndim}D")
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError(f"covariance must be square, got shape {covariance.shape}")
    if matrix.shape[1] != covariance.shape[0]:
        raise ValueError(
            "matrix and covariance dimensions do not match "
            f"({matrix.shape[1]} != {covariance.shape[0]})"
        )
    transformed = matrix @ covariance @ matrix.T
    return (transformed + transformed.T) / 2.0


def map_spatial_matrices_to_sensor_space(
    filters: np.ndarray,
    patterns: np.ndarray,
    *,
    whitener: np.ndarray,
    dewhitener: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map spatial filters and patterns from whitened to sensor coordinates."""
    filters = np.asarray(filters)
    patterns = np.asarray(patterns)
    whitener = np.asarray(whitener)
    dewhitener = np.asarray(dewhitener)
    if any(matrix.ndim != 2 for matrix in (filters, patterns, whitener, dewhitener)):
        raise ValueError("filters, patterns, whitener, and dewhitener must be 2D")
    if filters.shape[1] != whitener.shape[0]:
        raise ValueError(
            "filter and whitener dimensions do not match "
            f"({filters.shape[1]} != {whitener.shape[0]})"
        )
    if dewhitener.shape[1] != patterns.shape[0]:
        raise ValueError(
            "dewhitener and pattern dimensions do not match "
            f"({dewhitener.shape[1]} != {patterns.shape[0]})"
        )
    if filters.shape[0] != patterns.shape[1]:
        raise ValueError(
            "filters and patterns must describe the same number of components "
            f"({filters.shape[0]} != {patterns.shape[1]})"
        )
    return filters @ whitener, dewhitener @ patterns


# Data-covariance whitening
# -------------------------
def compute_data_covariance_whitener(
    cov: np.ndarray,
    *,
    rank: int | None = None,
    reg: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute whitening and de-whitening matrices from covariance.

    Parameters
    ----------
    cov : ndarray, shape (n_channels, n_channels)
        Covariance matrix of the data.
    rank : int, optional
        Number of principal components to retain. If None, determined
        automatically based on eigenvalue threshold.
    reg : float
        Regularization threshold. Eigenvalues smaller than
        reg * max(eigenvalue) are discarded. Default 1e-9.

    Returns
    -------
    whitener : ndarray, shape (rank, n_channels)
        Matrix to whiten data: ``X_white = whitener @ X``.
    dewhitener : ndarray, shape (n_channels, rank)
        Matrix to de-whiten: ``X = dewhitener @ X_white``.
    eigenvalues : ndarray, shape (rank,)
        Retained eigenvalues (descending order).


    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.dss.utils import compute_data_covariance_whitener
    >>> rng = np.random.default_rng(0)
    >>> data = rng.standard_normal((4, 1000))
    >>> cov = data @ data.T / data.shape[1]
    >>> whitener, dewhitener, eigenvalues = compute_data_covariance_whitener(cov)
    >>> np.allclose(whitener @ cov @ whitener.T, np.eye(4))
    True
    """
    cov = np.asarray(cov, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"cov must be square, got shape {cov.shape}")
    if cov.shape[0] == 0:
        raise ValueError("cov must contain at least one channel")
    if not np.all(np.isfinite(cov)):
        raise ValueError("cov must contain only finite values")
    if rank is not None and (isinstance(rank, bool) or not isinstance(rank, Integral)):
        raise TypeError("rank must be an int or None")
    if rank is not None and rank <= 0:
        raise ValueError("rank must be positive")
    if rank is not None and rank > cov.shape[0]:
        raise ValueError(
            f"rank ({rank}) cannot exceed the channel count ({cov.shape[0]})"
        )
    if isinstance(reg, bool) or not isinstance(reg, Real):
        raise TypeError("reg must be a real number")
    if not np.isfinite(reg) or reg < 0:
        raise ValueError("reg must be a finite non-negative number")

    # Covariance matrices are symmetric by definition. Averaging also removes
    # harmless floating-point asymmetry from empirical estimates.
    cov = (cov + cov.T) / 2

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
    except LinAlgError as e:
        raise ValueError(f"Eigendecomposition failed: {e}") from e

    # Sort in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    max_abs_eigenvalue = float(np.max(np.abs(eigenvalues)))
    max_eigenvalue = float(eigenvalues[0])
    if not np.isfinite(max_abs_eigenvalue) or max_eigenvalue <= 0:
        raise ValueError("Covariance matrix has no significant variance")
    if eigenvalues[-1] < -1e-10 * max_abs_eigenvalue:
        raise ValueError("Covariance matrix must be positive semi-definite")

    # Clip only numerical negative noise, then discard negligible directions.
    eigenvalues = np.clip(eigenvalues, a_min=0, a_max=None)
    threshold = reg * max_eigenvalue
    keep_mask = eigenvalues > threshold

    if rank is not None:
        keep_mask[rank:] = False

    eigenvalues = eigenvalues[keep_mask]
    eigenvectors = eigenvectors[:, keep_mask]
    if eigenvalues.size == 0:
        raise ValueError("No components above regularization threshold")

    scales = 1.0 / np.sqrt(eigenvalues)
    whitener = scales[:, np.newaxis] * eigenvectors.T
    dewhitener = eigenvectors * np.sqrt(eigenvalues)

    return whitener, dewhitener, eigenvalues


def whiten_from_data_covariance(
    data: np.ndarray,
    *,
    rank: int | None = None,
    reg: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Whiten multichannel data using its empirical covariance.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times) or (n_channels, n_times, n_epochs)
        Input data to whiten.
    rank : int, optional
        Number of principal components to retain. If None, auto-determined.
    reg : float
        Regularization threshold for eigenvalue cutoff. Default 1e-9.

    Returns
    -------
    whitened : ndarray, shape (rank, n_times) or (rank, n_times, n_epochs)
        Whitened data with unit covariance.
    whitener : ndarray, shape (rank, n_channels)
        Whitening matrix.
    dewhitener : ndarray, shape (n_channels, rank)
        De-whitening matrix for reconstruction.

    Examples
    --------
    >>> data = np.random.randn(64, 1000)  # 64 channels, 1000 samples
    >>> whitened, W, D = whiten_from_data_covariance(data)
    >>> # Verify whitened covariance is approximately identity
    >>> np.allclose(whitened @ whitened.T / 1000, np.eye(whitened.shape[0]), atol=0.1)
    True
    """
    # Handle 2D and 3D data
    input_shape = data.shape
    if data.ndim == 3:
        # Epochs: (n_channels, n_times, n_epochs) -> (n_channels, n_times*n_epochs)
        n_channels, n_times, n_epochs = data.shape
        data_2d = data.reshape(n_channels, -1)
    elif data.ndim == 2:
        data_2d = data
    else:
        raise ValueError(f"Data must be 2D or 3D, got {data.ndim}D")

    # Remove mean for covariance computation
    data_centered = data_2d - data_2d.mean(axis=1, keepdims=True)

    cov = compute_covariance(data_centered, assume_centered=True)

    # Get whitening matrices
    whitener, dewhitener, eigenvalues = compute_data_covariance_whitener(
        cov, rank=rank, reg=reg
    )

    # Apply whitening through the shared channel-axis operation.
    whitened_2d = apply_spatial_transform(whitener, data_centered)

    # Reshape back if input was 3D
    if len(input_shape) == 3:
        whitened = whitened_2d.reshape(whitener.shape[0], n_times, n_epochs)
    else:
        whitened = whitened_2d

    return whitened, whitener, dewhitener


# MNE-style sensor pre-whitening
# ------------------------------
def compute_mne_sensor_whitener(
    data: np.ndarray,
    *,
    info: Any = None,
    ch_names: list[str] | None = None,
    noise_cov: Any = None,
    rank: int | dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute MNE-style sensor pre-whitening and coloring matrices.

    A supplied MNE noise covariance is handled by MNE's covariance whitening
    implementation, including its projectors, channel scaling, and rank logic.
    Without a noise covariance, MNE inputs are standardized by channel type as
    in :class:`mne.preprocessing.ICA`; NumPy arrays are standardized per channel.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim not in (2, 3):
        raise ValueError(f"data must be 2D or 3D, got {data.ndim}D")
    if not np.all(np.isfinite(data)):
        raise ValueError("data must contain only finite values")

    n_channels = data.shape[0]
    if noise_cov is not None:
        _mne.require_mne("DSS MNE noise-covariance whitening")
        if info is None or ch_names is None:
            raise ValueError("noise_cov requires an MNE input with named channels")
        if not isinstance(noise_cov, _mne.mne.Covariance):
            raise TypeError("noise_cov must be an mne.Covariance")
        missing = [name for name in ch_names if name not in noise_cov.ch_names]
        if missing:
            raise ValueError(
                f"noise_cov is missing required channels: {', '.join(missing)}"
            )
        from mne.cov import compute_whitener

        whitener, names, colorer = compute_whitener(
            noise_cov,
            info,
            picks=ch_names,
            # Integer rank is applied by DSS after pre-whitening. MNE's
            # covariance whitener accepts its channel-type rank dictionary.
            rank=rank if isinstance(rank, dict) else None,
            return_colorer=True,
            verbose=False,
        )
        if list(names) != list(ch_names):
            raise RuntimeError("MNE whitener channel order does not match the input")
        return np.asarray(whitener), np.asarray(colorer)

    flat = data.reshape(n_channels, -1)
    scales = np.ones(n_channels, dtype=float)
    if info is not None and ch_names is not None:
        _mne.require_mne("DSS MNE sensor whitening")
        indices = [info["ch_names"].index(name) for name in ch_names]
        picked_info = _mne.mne.pick_info(info, indices)
        for picks in _mne.mne.channel_indices_by_type(picked_info, exclude=()).values():
            if len(picks):
                scales[picks] = np.std(flat[picks])
    else:
        scales = np.std(flat, axis=1)
    scales = np.where(np.isfinite(scales) & (scales > 0), scales, 1.0)
    return np.diag(1.0 / scales), np.diag(scales)
