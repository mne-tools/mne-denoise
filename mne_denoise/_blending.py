"""Raised-cosine blending utilities."""

from __future__ import annotations

import numpy as np

__all__ = ["raised_cosine_ramp", "overlap_add_combine"]


def raised_cosine_ramp(width: int) -> np.ndarray:
    """Return a rising raised-cosine ramp of the requested width.

    Parameters
    ----------
    width : int
        Number of samples; must be positive.

    Returns
    -------
    ramp : ndarray, shape (width,)
        Increasing blend weights.
    """
    if width <= 0:
        raise ValueError(f"width must be positive, got {width}")
    return (1.0 - np.cos(np.pi * np.arange(1, width + 1) / width)) / 2.0


def overlap_add_combine(
    shape: tuple[int, int],
    chunks: list[dict],
    *,
    eps: float = 1e-10,
) -> np.ndarray:
    """Blend processed chunks by their tapered overlap weights.

    Parameters
    ----------
    shape : tuple of int
        Output shape (n_channels, n_times).
    chunks : list of dict
        Chunk data and extended/owned half-open ranges.
    eps : float, default=1e-10
        Minimum accumulated weight before division.

    Returns
    -------
    output : ndarray, shape (n_channels, n_times)
        Weighted-average reconstruction.

    Notes
    -----
    Each chunk has full weight on its owned range and raised-cosine tapers over
    extensions.
    """
    n_channels, n_times = shape
    output = np.zeros((n_channels, n_times))
    weights = np.zeros(n_times)

    for chunk in chunks:
        data = chunk["data"]
        ext_start, ext_end = chunk["ext_start"], chunk["ext_end"]
        start, end = chunk["start"], chunk["end"]

        window = np.ones(ext_end - ext_start)

        # Fade in across the leading extension.
        lead = start - ext_start
        if lead > 0:
            window[:lead] = raised_cosine_ramp(lead)

        # Fade out across the trailing extension.
        trail = ext_end - end
        if trail > 0:
            window[len(window) - trail :] = 1.0 - raised_cosine_ramp(trail)

        output[:, ext_start:ext_end] += data * window[np.newaxis, :]
        weights[ext_start:ext_end] += window

    return output / np.maximum(weights, eps)[np.newaxis, :]
