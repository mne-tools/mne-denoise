"""Overcorrection metrics for linear denoisers."""

from __future__ import annotations

import numpy as np

__all__ = ["quantify_overcorrection"]


def quantify_overcorrection(
    operator: np.ndarray, leadfield: np.ndarray
) -> dict[str, np.ndarray]:
    """Compute source-topography distortion metrics for a spatial operator.

    Parameters
    ----------
    operator : ndarray, shape (n_channels, n_channels)
        Linear spatial filter.
    leadfield : ndarray, shape (n_channels, n_sources)
        Lead field in the same channel order and reference as operator.

    Returns
    -------
    dict of ndarray
        Per-source arrays for:

        amplitude_change
            (norm(operator @ l) - norm(l)) / norm(l).
        correlation
            (operator @ l) dot l / (norm(operator @ l) * norm(l)).
        relative_error
            norm(operator @ l - l) / norm(l).
        goodness_of_fit
            1 - relative_error**2.

    Notes
    -----
    These are mne-denoise evaluation definitions. Sources with zero input or output
    topography produce NaN where the corresponding ratio is undefined.
    """
    operator = np.asarray(operator, dtype=float)
    leadfield = np.asarray(leadfield, dtype=float)
    if operator.ndim != 2 or operator.shape[0] != operator.shape[1]:
        raise ValueError(
            f"operator must be square (n_channels, n_channels), got {operator.shape}."
        )
    if leadfield.ndim != 2 or leadfield.shape[0] != operator.shape[0]:
        raise ValueError(
            f"leadfield has {leadfield.shape[0]} channels but operator has "
            f"{operator.shape[0]}."
        )

    filtered = operator @ leadfield
    norm_before = np.linalg.norm(leadfield, axis=0)
    norm_after = np.linalg.norm(filtered, axis=0)

    # A source with no topography, or one deleted outright, has no meaningful
    # relative change; report NaN instead of dividing by zero.
    scale = np.where(norm_before > 0, norm_before, np.nan)
    cosine_scale = np.where(
        (norm_before > 0) & (norm_after > 0), norm_before * norm_after, np.nan
    )

    relative_error = np.linalg.norm(filtered - leadfield, axis=0) / scale
    return {
        "amplitude_change": (norm_after - norm_before) / scale,
        "correlation": np.einsum("ij,ij->j", filtered, leadfield) / cosine_scale,
        "relative_error": relative_error,
        "goodness_of_fit": 1.0 - relative_error**2,
    }
