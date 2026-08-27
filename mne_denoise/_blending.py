"""Raised-cosine blending shared by the adaptive denoisers.

Any denoiser that adapts over time eventually faces the same problem: two
different cleaning operators are correct on either side of a boundary, and
switching between them abruptly injects a step discontinuity into every
channel.  A step is broadband in frequency, so it smears energy across the
whole spectrum — including the band the denoiser was trying to clean — and
creates transients that downstream epoching or averaging will treat as real.

The fix in both cases is a raised-cosine (Hann) taper that moves weight
smoothly from one operator to the other.  Two flavours are needed:

- :func:`raised_cosine_ramp` — when both operators can be applied to the same
  samples, blend their *outputs* pointwise over the transition window.  Used
  by ASR, where the reconstruction matrix is re-estimated per update.
- :func:`overlap_add_combine` — when each operator is only valid on its own
  stretch of data, process overlapping extensions and combine by a normalised
  weighted average.  Used by segmented DSS/ZapLine, where each segment gets an
  independent fit.

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

from __future__ import annotations

import numpy as np

__all__ = ["raised_cosine_ramp", "overlap_add_combine"]


def raised_cosine_ramp(width: int) -> np.ndarray:
    """Rising raised-cosine ramp from ~0 to 1 over ``width`` samples.

    The complementary falling ramp is ``1 - raised_cosine_ramp(width)``, so a
    pair of them forms a partition of unity: blending two signals with weights
    ``w`` and ``1 - w`` preserves a constant wherever the two agree.

    Parameters
    ----------
    width : int
        Number of samples in the transition. Must be positive.

    Returns
    -------
    ramp : ndarray, shape (width,)
        Monotonically increasing weights in ``(0, 1]``.

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise._blending import raised_cosine_ramp
    >>> w = raised_cosine_ramp(4)
    >>> np.round(w, 3)
    array([0.146, 0.5  , 0.854, 1.   ])
    >>> # Cross-fade old -> new over the transition
    >>> blended = new * w + old * (1.0 - w)  # doctest: +SKIP
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
    """Combine independently-processed, overlapping chunks by weighted average.

    Each chunk was processed over an *extended* range that reaches into its
    neighbours.  Inside its own ``[start, end)`` the chunk carries full weight;
    across the extensions the weight tapers with a raised cosine.  Summing the
    weighted chunks and dividing by the accumulated weight yields a smooth
    result, and because the division normalises, the tapers do not need to be
    exactly complementary.

    Parameters
    ----------
    shape : tuple of (int, int)
        ``(n_channels, n_times)`` of the output array.
    chunks : list of dict
        One dict per chunk with keys:

        - ``'data'`` : ndarray, shape ``(n_channels, ext_end - ext_start)``
          — the processed chunk covering the extended range.
        - ``'ext_start'``, ``'ext_end'`` : int — the extended range.
        - ``'start'``, ``'end'`` : int — the chunk's own (unextended) range,
          over which it carries full weight.
    eps : float, default=1e-10
        Floor applied to the accumulated weights before dividing, guarding
        against division by zero if the chunks fail to tile the output.

    Returns
    -------
    output : ndarray, shape (n_channels, n_times)
        The blended result.

    See Also
    --------
    raised_cosine_ramp : The taper used for each transition.
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
