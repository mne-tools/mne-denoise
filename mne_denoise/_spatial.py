"""Internal spatial operations shared by denoising algorithms."""

from __future__ import annotations

from numbers import Integral

import numpy as np


def apply_spatial_transform(
    matrix: np.ndarray,
    data: np.ndarray,
    *,
    chunk_size: int | None = None,
) -> np.ndarray:
    """Apply a spatial matrix along the first axis of 2D or 3D data.

    Parameters
    ----------
    matrix : ndarray, shape (n_output_channels, n_input_channels)
        Spatial transformation matrix.
    data : ndarray, shape (n_input_channels, ...)
        Channel-first continuous or multidimensional data.
    chunk_size : int | None, default=None
        Number of flattened samples transformed at a time. None applies the
        matrix in one operation.

    Returns
    -------
    transformed : ndarray
        Transformed data with the trailing dimensions of ``data`` preserved.
    """
    matrix = np.asarray(matrix)
    data = np.asarray(data)
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2D, got {matrix.ndim}D")
    if data.ndim not in (2, 3):
        raise ValueError(f"data must be 2D or 3D, got {data.ndim}D")
    if matrix.shape[1] != data.shape[0]:
        raise ValueError(
            "matrix and data channel dimensions do not match "
            f"({matrix.shape[1]} != {data.shape[0]})"
        )
    if chunk_size is not None:
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, Integral):
            raise TypeError("chunk_size must be a positive integer or None")
        if chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer or None")
        chunk_size = int(chunk_size)

    flat = data.reshape(data.shape[0], -1)
    if chunk_size is None:
        transformed = matrix @ flat
    else:
        transformed = np.empty((matrix.shape[0], flat.shape[1]), dtype=np.float64)
        for start in range(0, flat.shape[1], chunk_size):
            stop = min(start + chunk_size, flat.shape[1])
            transformed[:, start:stop] = matrix @ flat[:, start:stop]
    return transformed.reshape((matrix.shape[0], *data.shape[1:]))
