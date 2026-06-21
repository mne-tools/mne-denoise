"""General utilities for mne-denoise."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import mne
    from mne.epochs import BaseEpochs
    from mne.evoked import Evoked
    from mne.io import BaseRaw

    _HAS_MNE = True
except ImportError:
    mne = None
    # Mock classes for type hinting/checking when MNE not available

    class BaseRaw:  # noqa: D101
        """Mock BaseRaw class."""

    class BaseEpochs:  # noqa: D101
        """Mock BaseEpochs class."""

    class Evoked:  # noqa: D101
        """Mock Evoked class."""

    _HAS_MNE = False

import logging
import warnings

logger = logging.getLogger(__name__)


def _get_homogeneous_picks(
    inst: Any, auto_pick: bool | str = "auto"
) -> np.ndarray | None:
    """Choose one homogeneous data channel type from an MNE object."""
    if not (_HAS_MNE and isinstance(inst, BaseRaw | BaseEpochs | Evoked)):
        return None

    ch_types = inst.get_channel_types()
    pick_specs = [
        (
            "mag",
            {
                "meg": "mag",
                "eeg": False,
                "ref_meg": False,
                "stim": False,
                "misc": False,
            },
        ),
        (
            "grad",
            {
                "meg": "grad",
                "eeg": False,
                "ref_meg": False,
                "stim": False,
                "misc": False,
            },
        ),
        (
            "eeg",
            {
                "meg": False,
                "eeg": True,
                "ref_meg": False,
                "stim": False,
                "misc": False,
            },
        ),
    ]

    present_types = []
    best_picks = None
    best_type = None

    for ch_type, pick_kws in pick_specs:
        if ch_type not in ch_types:
            continue
        picks = mne.pick_types(inst.info, exclude=(), **pick_kws)
        if len(picks) > 0:
            present_types.append(ch_type)
            if best_picks is None:
                best_picks = np.asarray(picks, dtype=int)
                best_type = ch_type

    if len(present_types) > 1:
        msg = (
            f"Found multiple data channel types {present_types} in the object. "
            "MNE-Denoise estimators should be fitted on a single homogeneous channel type. "
            "Please use `inst.pick()` or `inst.pick_types()` to select a single data channel type before fitting."
        )
        if auto_pick == "auto" or auto_pick is True:
            msg += f" Automatically picking '{best_type}'."
            warnings.warn(msg, UserWarning, stacklevel=2)
        else:
            raise ValueError(msg)

    if best_picks is not None:
        if len(best_picks) == len(inst.ch_names):
            return None
        logger.info(
            "Auto-picking %d/%d %s channels and preserving other channels.",
            len(best_picks),
            len(inst.ch_names),
            best_type,
        )
        return best_picks

    return None


def extract_data_from_mne(
    X: Any, ch_names: list[str] | None = None, auto_pick: bool = True
) -> tuple[np.ndarray, float | None, str, Any, np.ndarray | None, list[str] | None]:
    """Extract data and metadata from input (MNE object or array).

    Parameters
    ----------
    X : Raw | Epochs | Evoked | array
        Input data.
    ch_names : list of str | None
        Explicit list of channel names to extract. If None and X is an MNE object,
        the function auto-picks a single homogeneous channel type (if auto_pick=True).
    auto_pick : bool
        Whether to automatically pick a homogeneous channel type if ch_names is None.

    Returns
    -------
    data : array
        Extracted data. If MNE Epochs, data is (n_epochs, n_channels, n_times).
    sfreq : float | None
        Sampling frequency.
    mne_type : str
        'raw', 'epochs', 'evoked', or 'array'.
    orig_inst : object
        Original MNE instance (or None).
    picks : array of int | None
        Indices of channels extracted (if any filtering occurred).
    extracted_ch_names : list of str | None
        Names of the extracted channels (if an MNE object was passed).
    """
    sfreq = None
    mne_type = "array"
    orig_inst = None
    picks = None
    extracted_ch_names = None

    if _HAS_MNE and isinstance(X, BaseRaw | BaseEpochs | Evoked):
        orig_inst = X
        sfreq = X.info["sfreq"]

        if ch_names is not None:
            missing = [ch for ch in ch_names if ch not in X.ch_names]
            if missing:
                raise ValueError(
                    f"Input MNE object is missing required channels: {missing[:5]}"
                )
            picks = np.array([X.ch_names.index(ch) for ch in ch_names])
            data = X.get_data(picks=picks)
            extracted_ch_names = [X.ch_names[p] for p in picks]
        else:
            if auto_pick is not False:
                picks = _get_homogeneous_picks(X, auto_pick=auto_pick)
            else:
                picks = None

            if picks is not None:
                data = X.get_data(picks=picks)
                extracted_ch_names = [X.ch_names[p] for p in picks]
            else:
                data = X.get_data()
                extracted_ch_names = X.ch_names.copy()

        if isinstance(X, BaseEpochs):
            mne_type = "epochs"
        elif isinstance(X, Evoked):
            mne_type = "evoked"
        else:
            mne_type = "raw"
    else:
        # Assume array
        data = np.asarray(X)

    return data, sfreq, mne_type, orig_inst, picks, extracted_ch_names


def reconstruct_mne_object(
    data: np.ndarray,
    orig_inst: Any,
    mne_type: str,
    picks: np.ndarray | None = None,
    verbose: bool = False,
) -> Any:
    """Reconstruct MNE object from data and template instance.

    Parameters
    ----------
    data : array
        The cleaned/processed data.
    orig_inst : object
        The original MNE instance (template).
    mne_type : str
        Type string returned by extract_data_from_mne ('raw', 'epochs', 'evoked', 'array').
    picks : array of int | None
        If provided, `data` is re-inserted into a copy of `orig_inst` only at these channel indices.
    verbose : bool
        Verbosity flag for MNE object creation.

    Returns
    -------
    out : Raw | Epochs | Evoked | array
        Reconstructed object or the data array.
    """
    if mne_type == "array" or orig_inst is None:
        return data

    if not _HAS_MNE:
        return data

    if picks is not None:
        data_full = orig_inst.get_data().copy()
        if mne_type == "epochs":
            data_full[:, picks, :] = data
        else:
            data_full[picks, :] = data
        data = data_full

    if mne_type == "raw":
        out = mne.io.RawArray(data, orig_inst.info, verbose=verbose)
        if hasattr(orig_inst, "annotations") and orig_inst.annotations:
            out.set_annotations(orig_inst.annotations)
        return out

    elif mne_type == "epochs":
        events = getattr(orig_inst, "events", None)
        event_id = getattr(orig_inst, "event_id", None)
        tmin = getattr(orig_inst, "tmin", 0)
        metadata = getattr(orig_inst, "metadata", None)

        out = mne.EpochsArray(
            data,
            orig_inst.info,
            events=events,
            event_id=event_id,
            tmin=tmin,
            verbose=verbose,
        )
        if metadata is not None:
            out.metadata = metadata.copy()
        return out

    elif mne_type == "evoked":
        nave = getattr(orig_inst, "nave", 1)
        tmin = getattr(orig_inst, "tmin", 0)
        comment = getattr(orig_inst, "comment", "")

        out = mne.EvokedArray(
            data, orig_inst.info, tmin=tmin, nave=nave, comment=comment, verbose=verbose
        )
        return out

    return data
