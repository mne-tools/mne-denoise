"""Internal utility functions for visualization.

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

import mne
import numpy as np


def _compute_gfp(inst_or_data):
    """Compute GFP (RMS across channels) from 2D/3D signal data.

    Parameters
    ----------
    inst_or_data : MNE object | ndarray
        Signal data as:
        - 2D ``(n_channels, n_times)``, or
        - 3D ``(n_epochs, n_channels, n_times)``.
        MNE inputs are converted via ``get_data()``.

    Returns
    -------
    gfp : ndarray of shape (n_times,)
        Global field power time series.

    Raises
    ------
    ValueError
        If input is not 2D or 3D.
    """
    if hasattr(inst_or_data, "get_data"):
        data = np.asarray(inst_or_data.get_data(), dtype=float)
    else:
        data = np.asarray(inst_or_data, dtype=float)

    if data.ndim == 3:
        data = data.mean(axis=0)
    if data.ndim != 2:
        raise ValueError(
            "GFP input must be 2D (n_channels, n_times) or 3D "
            "(n_epochs, n_channels, n_times)."
        )
    return np.sqrt(np.mean(data**2, axis=0))


def _get_info(estimator, info=None):
    """Resolve MNE info from estimator or argument."""
    if info is not None:
        return info
    if hasattr(estimator, "info_") and estimator.info_ is not None:
        return estimator.info_
    if hasattr(estimator, "_mne_info") and estimator._mne_info is not None:
        return estimator._mne_info
    return None


def _get_patterns(estimator):
    """Extract spatial patterns from estimator."""
    if hasattr(estimator, "patterns_") and estimator.patterns_ is not None:
        return estimator.patterns_
    raise ValueError(
        "Estimator does not have a 'patterns_' attribute or it is None. "
        "Make sure the estimator is fitted."
    )


def _get_filters(estimator):
    """Extract spatial filters from estimator."""
    if hasattr(estimator, "filters_") and estimator.filters_ is not None:
        return estimator.filters_
    raise ValueError("Estimator does not have a 'filters_' attribute or it is None.")


def _get_scores(estimator):
    """Extract scores (eigenvalues or similar) from estimator.

    Checks for both eigenvalues_ (DSS) and scores_ (ZapLine).
    """
    # Check for eigenvalues_ (LinearDSS)
    if hasattr(estimator, "eigenvalues_") and estimator.eigenvalues_ is not None:
        ev = estimator.eigenvalues_
        if isinstance(ev, np.ndarray) and ev.size > 0:
            return ev
    # Check for scores_ (ZapLine)
    if hasattr(estimator, "scores_") and estimator.scores_ is not None:
        sc = estimator.scores_
        if isinstance(sc, np.ndarray) and sc.size > 0:
            return sc
    # For iterative DSS, we might not have a simple "score"
    if hasattr(estimator, "convergence_info_"):
        return None
    return None


def _get_components(estimator, data=None):
    """
    Get component time series.

    If 'sources_' (cached) is present (e.g. IterativeDSS), use it.
    Otherwise, compute from data using 'transform'.
    """
    # 1. Try cached sources if no new data provided
    if data is None:
        if hasattr(estimator, "sources_") and estimator.sources_ is not None:
            return estimator.sources_
        raise ValueError(
            "Data must be provided if sources are not cached in the estimator."
        )

    # 2. Compute sources for the provided data
    # Check if we need to force 'return_type' for LinearDSS
    if hasattr(estimator, "return_type"):
        old_return_type = estimator.return_type
        try:
            estimator.return_type = "sources"
            sources = estimator.transform(data)
        finally:
            estimator.return_type = old_return_type
    else:
        # Assumed to be IterativeDSS or similar which transforms to sources
        sources = estimator.transform(data)

    # 3. Standardize shape to (n_components, n_times, [n_epochs])
    # LinearDSS with MNE Epochs input returns (n_epochs, n_components, n_times)
    # We want dimension 0 to be components for easier plotting.

    if (
        isinstance(data, mne.BaseEpochs)
        and sources.ndim == 3
        and sources.shape[1] == _get_filters(estimator).shape[0]
    ):
        # Transpose to (n_comp, n_times, n_epochs)
        sources = np.transpose(sources, (1, 2, 0))

    return sources


def _handle_picks(info, picks=None):
    """Wrap mne.pick_types/pick_channels using public API."""
    if picks is None:
        return mne.pick_types(
            info, meg=True, eeg=True, seeg=True, ecog=True, fnirs=True, exclude="bads"
        )
    # Use public API for picking
    if isinstance(picks, str):
        if picks == "all":
            return np.arange(len(info["ch_names"]))
        # Use pick_types for string type specifiers
        return mne.pick_types(info, **{picks: True}, exclude="bads")
    elif isinstance(picks, list | np.ndarray):
        # Could be channel names or indices
        if len(picks) > 0 and isinstance(picks[0], str):
            return mne.pick_channels(info["ch_names"], include=picks)
        else:
            return np.asarray(picks)
    return np.asarray(picks)
