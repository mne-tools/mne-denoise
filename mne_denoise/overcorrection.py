"""Quantify how much a linear spatial filter distorts cortical signals.

Every spatial denoiser in this package ultimately produces a linear operator
``W`` that maps measured EEG onto cleaned EEG.  Cleaning is never free: the
same projection that removes an artifact also removes whatever brain signal
happened to share that direction in sensor space.  The usual way to judge the
damage is to compare cleaned data against a ground truth, which real
recordings do not have.

Because ``W`` is explicit, the damage can be computed directly instead.
Pushing the operator through a forward model,

``L_after = W @ L``

turns each column of the lead field — the scalp topography one cortical source
would produce — into the topography that source would produce *after*
filtering.  Comparing the two answers the question without any ground-truth
data: had this cortical region been active, how much of its signal would the
filter have destroyed?

The four metrics here characterise that comparison from complementary angles:
:func:`quantify_overcorrection` returns amplitude change and topography
correlation, which isolate loss of *magnitude* and loss of *shape*
respectively, alongside relative error and goodness of fit, which fold both
together.

The metrics are estimator-agnostic: any channel-space cleaning operator works,
including those fitted by SOUND, SSP-SIR, ZapLine, DSS and iCanClean.

References
----------
Mutanen, T. P., Metsomaa, J., Makkonen, M., Varone, G., Marzetti, L., &
Ilmoniemi, R. J. (2022). Source-based artifact-rejection techniques for
TMS-EEG. Journal of Neuroscience Methods, 382, 109693.

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

from __future__ import annotations

import numpy as np

__all__ = ["quantify_overcorrection"]


def quantify_overcorrection(
    operator: np.ndarray, leadfield: np.ndarray
) -> dict[str, np.ndarray]:
    """Measure the distortion a spatial filter imposes on cortical sources.

    Parameters
    ----------
    operator : array of shape (n_channels, n_channels)
        The linear spatial filter to characterise, e.g. ``SOUND.operator_`` or
        ``SSPSIR.operator_``.
    leadfield : array of shape (n_channels, n_sources)
        Lead field whose columns are the scalp topographies of the cortical
        sources of interest, e.g. ``SOUND.leadfield_``. Must use the same
        channel order and reference as ``operator``.

    Returns
    -------
    metrics : dict of ndarray
        Four per-source arrays, each of shape ``(n_sources,)``:

        ``amplitude_change``
            Relative change in topography magnitude. 0 means the source is
            passed at full strength, -1 that it is deleted entirely. Usually
            negative, since spatial filtering attenuates.
        ``correlation``
            Cosine similarity between the topography before and after, in
            [-1, 1]. Near 1 means the *shape* survived even if the amplitude
            did not.
        ``relative_error``
            Combined magnitude-and-shape error; 0 is ideal.
        ``goodness_of_fit``
            1 is perfect, 0 corresponds to complete deletion of the source.

        Sources whose topography is identically zero yield NaN rather than a
        division by zero, as does ``correlation`` for a fully deleted source.

    Raises
    ------
    ValueError
        If ``operator`` is not square, or its channel count disagrees with
        ``leadfield``.

    See Also
    --------
    mne_denoise.sound.SOUND : Exposes a fitted ``operator_`` and ``leadfield_``.
    mne_denoise.sspsir.SSPSIR : Likewise.

    Notes
    -----
    Writing ``l`` for a source's topography before filtering and ``l'`` for
    ``operator @ l`` after, the four metrics are

    ``amplitude_change = (||l'|| - ||l||) / ||l||``

    ``correlation = (l' . l) / (||l'|| ||l||)``

    ``relative_error = ||l' - l|| / ||l||``

    ``goodness_of_fit = 1 - relative_error ** 2``

    ``amplitude_change`` and ``correlation`` are deliberately complementary: a
    filter that halves every topography without changing its shape scores -0.5
    and 1.0 respectively, which is a very different outcome from one that
    preserves magnitude while scrambling the pattern. ``relative_error`` and
    ``goodness_of_fit`` cannot separate the two, so report them alongside
    rather than instead of the first pair.

    ``goodness_of_fit`` turns negative where a filter amplifies a source away
    from its original topography, i.e. where the result is further from the
    truth than deleting the source outright would have been.

    Values are not comparable across different sensor geometries or different
    EEG references, so use them to compare filters on one dataset rather than
    to compare datasets.

    Examples
    --------
    A filter that projects out one source's topography deletes that source and
    largely spares an orthogonal one:

    >>> import numpy as np
    >>> from mne_denoise.overcorrection import quantify_overcorrection
    >>> leadfield = np.eye(3)[:, :2]
    >>> direction = np.array([1.0, 0.0, 0.0])
    >>> projector = np.eye(3) - np.outer(direction, direction)
    >>> metrics = quantify_overcorrection(projector, leadfield)
    >>> metrics["goodness_of_fit"].round(3)
    array([0., 1.])
    >>> metrics["amplitude_change"].round(3)
    array([-1.,  0.])
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
