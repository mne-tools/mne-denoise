"""Internal spatial operations shared by denoising algorithms."""

from __future__ import annotations

import numpy as np

from ._validation import check_chunk_size


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
    chunk_size = check_chunk_size(chunk_size)

    flat = data.reshape(data.shape[0], -1)
    if chunk_size is None:
        transformed = matrix @ flat
    else:
        transformed = np.empty((matrix.shape[0], flat.shape[1]), dtype=np.float64)
        for start in range(0, flat.shape[1], chunk_size):
            stop = min(start + chunk_size, flat.shape[1])
            transformed[:, start:stop] = matrix @ flat[:, start:stop]
    return transformed.reshape((matrix.shape[0], *data.shape[1:]))


def epochs_to_continuous(data: np.ndarray) -> np.ndarray:
    """Concatenate epoched channel-first data along time.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times) | (n_epochs, n_channels, n_times)
        Channel-first data. Two-dimensional input is returned unchanged.

    Returns
    -------
    continuous : ndarray, shape (n_channels, n_times) | (n_channels, n_epochs * n_times)
        Channel-first continuous data.

    See Also
    --------
    continuous_to_epochs : Inverse operation.
    """
    data = np.asarray(data)
    if data.ndim == 2:
        return data
    if data.ndim != 3:
        raise ValueError(f"data must be 2D or 3D, got {data.ndim}D")
    return data.transpose(1, 0, 2).reshape(data.shape[1], -1)


def continuous_to_epochs(continuous: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Split continuous channel-first data back into epochs.

    Parameters
    ----------
    continuous : ndarray, shape (n_channels, n_epochs * n_times)
        Output of :func:`epochs_to_continuous`.
    shape : tuple
        Target shape. A two-element shape returns ``continuous`` unchanged.

    Returns
    -------
    epoched : ndarray
        Data reshaped to ``shape``.

    See Also
    --------
    epochs_to_continuous : Inverse operation.
    """
    if len(shape) == 2:
        return continuous
    if len(shape) != 3:
        raise ValueError(f"shape must have 2 or 3 entries, got {len(shape)}")
    n_epochs, _n_channels, n_times = shape
    return continuous.reshape(continuous.shape[0], n_epochs, n_times).transpose(1, 0, 2)
