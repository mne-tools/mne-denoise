"""Shared data extraction and reconstruction for arrays and MNE objects."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from . import _mne
from ._logging import logger


def _mne_instance_types() -> tuple[type, ...]:
    """Return supported MNE instance types when MNE-Python is available."""
    if _mne.mne is None:
        return ()

    return _mne.mne.io.BaseRaw, _mne.mne.BaseEpochs, _mne.mne.Evoked


def _get_homogeneous_picks(
    inst: Any, auto_pick: bool | str = "auto"
) -> np.ndarray | None:
    """Choose one homogeneous data channel type from an MNE object."""
    mne_types = _mne_instance_types()
    if not isinstance(inst, mne_types):
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
        picks = _mne.mne.pick_types(inst.info, exclude=(), **pick_kws)
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

    if best_picks is None or len(best_picks) == len(inst.ch_names):
        return None

    logger.debug(
        "Auto-picking %d/%d %s channels and preserving other channels.",
        len(best_picks),
        len(inst.ch_names),
        best_type,
    )
    return best_picks


def _resolve_mne_picks(
    inst: Any,
    ch_names: list[str] | None,
    auto_pick: bool | str,
    exclude_bads: bool,
) -> np.ndarray | None:
    """Resolve the channel-selection policy for an MNE instance."""
    if ch_names is not None:
        missing = [ch for ch in ch_names if ch not in inst.ch_names]
        if missing:
            raise ValueError(
                f"Input MNE object is missing required channels: {missing[:5]}"
            )
        return np.array([inst.ch_names.index(ch) for ch in ch_names])

    if auto_pick == "data":
        picks = _mne.mne.pick_types(
            inst.info,
            meg=True,
            ref_meg=False,
            eeg=True,
            seeg=True,
            ecog=True,
            dbs=True,
            fnirs=True,
            csd=True,
            exclude=(),
        )
        if picks.size == 0:
            raise ValueError("No data channels found for joint decomposition")
    elif auto_pick is not False:
        picks = _get_homogeneous_picks(inst, auto_pick=auto_pick)
    else:
        picks = None

    if not exclude_bads:
        return picks

    candidate_picks = (
        np.arange(len(inst.ch_names), dtype=int)
        if picks is None
        else np.asarray(picks, dtype=int)
    )
    bads = set(inst.info["bads"])
    picks = np.asarray(
        [pick for pick in candidate_picks if inst.ch_names[pick] not in bads],
        dtype=int,
    )
    if picks.size == 0:
        raise ValueError("No good data channels remain after excluding bads")
    return picks


def epochs_to_continuous(data: np.ndarray) -> np.ndarray:
    """Convert standard epoch-major data to continuous channel-first data.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times) | (n_epochs, n_channels, n_times)
        Standard MNE-style data. Two-dimensional input has shape
        ``(n_channels, n_times)`` and is returned unchanged. Three-dimensional
        input has shape ``(n_epochs, n_channels, n_times)`` and is concatenated
        epoch-by-epoch along time.

    Returns
    -------
    continuous : ndarray, shape (n_channels, n_times) | (n_channels, n_epochs * n_times)
        Continuous channel-first data.

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
    """Convert continuous data to the original standard epoch-major shape.

    Parameters
    ----------
    continuous : ndarray, shape (n_channels, n_epochs * n_times)
        Continuous channel-first data, typically the output of
        :func:`epochs_to_continuous`.
    shape : tuple
        Original shape. It must represent either ``(n_channels, n_times)`` or
        ``(n_epochs, n_channels, n_times)``. A two-dimensional shape returns
        ``continuous`` unchanged.

    Returns
    -------
    epoched : ndarray, shape (n_epochs, n_channels, n_times)
        Data reshaped to the original standard epoch-major representation.

    See Also
    --------
    epochs_to_continuous : Forward operation.
    """
    if len(shape) == 2:
        return continuous
    if len(shape) != 3:
        raise ValueError(f"shape must have 2 or 3 entries, got {len(shape)}")
    n_epochs, n_channels, n_times = shape
    if continuous.ndim != 2:
        raise ValueError(f"continuous must be 2D, got {continuous.ndim}D")
    if continuous.shape[0] != n_channels:
        raise ValueError(
            f"continuous has {continuous.shape[0]} channels but target shape "
            f"expects {n_channels}"
        )
    expected_samples = n_epochs * n_times
    if continuous.shape[1] != expected_samples:
        raise ValueError(
            f"continuous has {continuous.shape[1]} samples but target shape "
            f"expects {expected_samples}"
        )
    return continuous.reshape(n_channels, n_epochs, n_times).transpose(1, 0, 2)


def extract_data_from_mne(
    X: Any,
    ch_names: list[str] | None = None,
    auto_pick: bool | str = True,
    concatenate_epochs: bool = False,
    channel_first_epochs: bool = False,
    exclude_bads: bool = False,
) -> tuple[np.ndarray, float | None, str, Any, np.ndarray | None, list[str] | None]:
    """
    Extract data and metadata from an MNE object or NumPy-compatible array.

    The function centralizes channel selection and type detection. Epoched
    inputs can optionally be concatenated along time for algorithms that fit a
    single spatial model across epochs.

    Parameters
    ----------
    X : Raw | Epochs | Evoked | array
        Input data.
    ch_names : list of str | None, default=None
        Explicit list of channel names to extract. If None and X is an MNE object,
        the function auto-picks a single homogeneous channel type (if auto_pick=True).
    auto_pick : bool | 'data', default=True
        Channel selection when ``ch_names`` is None. If True, automatically pick
        one homogeneous channel type. If ``'data'``, pick all supported data
        channels jointly. If False, retain every channel.
    concatenate_epochs : bool, default=False
        If True, convert three-dimensional ``(n_epochs, n_channels, n_times)``
        data to ``(n_channels, n_epochs * n_times)``. The returned ``mne_type``
        remains ``'epochs'`` for MNE Epochs input.
    channel_first_epochs : bool, default=False
        If True, return MNE Epochs as
        ``(n_channels, n_times, n_epochs)``. Cannot be combined with
        ``concatenate_epochs=True``.
    exclude_bads : bool, default=False
        If True, omit channels listed in ``X.info['bads']`` from automatic
        channel selection. This has no effect when ``ch_names`` explicitly
        defines the fitted channel contract.

    Returns
    -------
    data : array
        Extracted data. MNE Epochs are returned as
        ``(n_epochs, n_channels, n_times)`` unless
        ``concatenate_epochs=True`` or ``channel_first_epochs=True``.
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

    See Also
    --------
    reconstruct_mne_object : Rebuild an MNE object after processing.

    Examples
    --------
    Concatenate an epoched array along its time axis:

    >>> import numpy as np
    >>> epochs = np.arange(24).reshape(2, 3, 4)
    >>> continuous, *_ = extract_data_from_mne(epochs, concatenate_epochs=True)
    >>> continuous.shape
    (3, 8)
    """
    sfreq = None
    mne_type = "array"
    orig_inst = None
    picks = None
    extracted_ch_names = None

    if concatenate_epochs and channel_first_epochs:
        raise ValueError(
            "concatenate_epochs and channel_first_epochs cannot both be True"
        )

    mne_types = _mne_instance_types()
    if isinstance(X, mne_types):
        if isinstance(X, mne_types[1]):
            mne_type = "epochs"
        elif isinstance(X, mne_types[2]):
            mne_type = "evoked"
        else:
            mne_type = "raw"

        orig_inst = X
        sfreq = X.info["sfreq"]
        picks = _resolve_mne_picks(X, ch_names, auto_pick, exclude_bads)
        if picks is None:
            data = X.get_data()
            extracted_ch_names = X.ch_names.copy()
        else:
            data = X.get_data(picks=picks)
            extracted_ch_names = [X.ch_names[p] for p in picks]
    else:
        if _mne.mne is None and hasattr(X, "get_data") and hasattr(X, "info"):
            _mne.require_mne("MNE data input support")
        # Assume array
        data = np.asarray(X)

    if data.ndim not in (2, 3):
        if isinstance(X, np.ndarray):
            raise ValueError(f"Data must be 2D or 3D, got {data.ndim}D")
        raise TypeError(f"Unsupported input type: {type(X)}")
    if concatenate_epochs and data.ndim == 3:
        data = epochs_to_continuous(data)
    elif channel_first_epochs and mne_type == "epochs":
        data = np.transpose(data, (1, 2, 0))

    return data, sfreq, mne_type, orig_inst, picks, extracted_ch_names


def reconstruct_mne_object(
    data: np.ndarray,
    orig_inst: Any,
    mne_type: str,
    picks: np.ndarray | None = None,
) -> Any:
    """Insert processed data into a copy of an MNE object.

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

    Returns
    -------
    out : Raw | Epochs | Evoked | array
        Reconstructed object or the data array.
    """
    if mne_type == "array" or orig_inst is None:
        return data

    if _mne.mne is None:
        _mne.require_mne("MNE object reconstruction")

    if mne_type == "evoked":
        out = orig_inst.copy()
        target = out.data
    elif mne_type in ("raw", "epochs"):
        out = orig_inst.copy().load_data()
        target = out._data
    else:
        return data

    if picks is None:
        if target.shape != data.shape:
            raise ValueError(
                f"Processed data shape {data.shape} does not match {mne_type} "
                f"shape {target.shape}"
            )
        target[...] = data
    elif mne_type == "epochs":
        target[:, picks, :] = data
    else:
        target[picks, :] = data

    return out
