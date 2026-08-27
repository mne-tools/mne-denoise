"""MNE Annotations helpers for ASR diagnostics.

This module converts internal boolean masks and window diagnostics into
MNE-Python ``Annotations`` objects for visualization and subsequent processing.
"""

from typing import Any

import numpy as np

from .. import _mne
from ._windowing import _mask_to_sample_spans, _merge_sample_spans


def _repair_annotations(
    diagnostics: dict[str, Any], sfreq: float, min_components: int, description: str
):
    """Create annotations for ASR burst repair periods.

    Parameters
    ----------
    diagnostics : dict
        ASR transform diagnostics dictionary containing 'window_starts',
        'window_stops', and 'n_components_reconstructed'.
    sfreq : float
        The sampling frequency of the data.
    min_components : int
        The minimum number of reconstructed components in a window to be annotated.
    description : str
        The annotation description string.

    Returns
    -------
    mne.Annotations
        An MNE Annotations object containing the repair spans.
    """
    _mne.require_mne("ASR repair annotations")
    starts = diagnostics["window_starts"]
    stops = diagnostics["window_stops"]
    counts = diagnostics["n_components_reconstructed"]
    spans = [
        (int(s), int(e))
        for s, e, c in zip(starts, stops, counts)
        if c >= min_components
    ]
    spans = _merge_sample_spans(spans)
    onsets = [s / sfreq for s, _ in spans]
    durations = [(e - s) / sfreq for s, e in spans]
    return _mne.mne.Annotations(onsets, durations, [description] * len(spans))


def _rejection_annotations(
    rejection_sample_mask: np.ndarray, sfreq: float, description: str
):
    """Create annotations for rejected (unreparable) data segments.

    Parameters
    ----------
    rejection_sample_mask : ndarray of bool, shape (n_times,)
        Boolean mask where False indicates a rejected sample.
    sfreq : float
        The sampling frequency of the data.
    description : str
        The annotation description string.

    Returns
    -------
    mne.Annotations
        An MNE Annotations object containing the rejected spans.
    """
    _mne.require_mne("ASR rejection annotations")
    mask = np.asarray(rejection_sample_mask, dtype=bool)
    if mask.ndim != 1:
        raise RuntimeError(
            "Rejection annotations require continuous transform diagnostics."
        )
    rejected = ~mask
    if not np.any(rejected):
        return _mne.mne.Annotations([], [], [])
    spans = _mask_to_sample_spans(rejected)
    onsets = [s / sfreq for s, _ in spans]
    durations = [(e - s) / sfreq for s, e in spans]
    return _mne.mne.Annotations(onsets, durations, [description] * len(spans))


def _calibration_annotations(
    calibration_mask_kind: str,
    reference_sample_mask: np.ndarray,
    sfreq: float,
    description: str,
):
    """Create annotations for the calibration reference segments.

    Parameters
    ----------
    calibration_mask_kind : str
        The type of calibration mask used. Must be "sample".
    reference_sample_mask : ndarray of bool, shape (n_times,)
        Boolean mask where True indicates a sample was used for calibration.
    sfreq : float
        The sampling frequency of the data.
    description : str
        The annotation description string.

    Returns
    -------
    mne.Annotations
        An MNE Annotations object containing the reference spans.
    """
    _mne.require_mne("ASR calibration annotations")
    if calibration_mask_kind != "sample":
        raise RuntimeError(
            "Calibration annotations are only available for sample-based "
            "reference selection (JugglerASR). This backend uses window-based "
            "calibration; use get_calibration_mask() instead."
        )
    mask = np.asarray(reference_sample_mask, dtype=bool)
    spans = _mask_to_sample_spans(mask)
    onsets = [start / sfreq for start, _ in spans]
    durations = [(stop - start) / sfreq for start, stop in spans]
    return _mne.mne.Annotations(onsets, durations, [description] * len(spans))
