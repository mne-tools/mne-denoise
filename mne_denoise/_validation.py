"""Internal input validation shared by array-based algorithms.

These helpers centralize the preconditions that every channel-first denoiser
checks at its public boundary, so error messages and accepted types stay
consistent across the package.
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral, Real

import numpy as np


def check_positive_integer(value: int, *, name: str) -> int:
    """Validate a positive integer parameter."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer")
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def check_channel_first_data(
    X: np.ndarray,
    *,
    name: str,
    allow_epochs: bool = True,
    min_channels: int = 2,
    min_times: int = 2,
) -> np.ndarray:
    """Validate and convert channel-first continuous or epoched data.

    Parameters
    ----------
    X : array-like
        Data shaped ``(n_channels, n_times)`` or, when ``allow_epochs`` is
        True, ``(n_epochs, n_channels, n_times)``.
    name : str
        Algorithm name used in error messages, e.g. ``"SNS"``.
    allow_epochs : bool, default=True
        Whether three-dimensional epoched input is accepted.
    min_channels : int, default=2
        Minimum number of channels required.
    min_times : int, default=2
        Minimum number of time samples required.

    Returns
    -------
    X : ndarray
        ``X`` as a float64 array.

    Raises
    ------
    ValueError
        If the shape, size, or finiteness preconditions are not met.
    """
    X = np.asarray(X, dtype=np.float64)
    expected = (2, 3) if allow_epochs else (2,)
    if X.ndim not in expected:
        shape_text = "2-D or 3-D" if allow_epochs else "2-D"
        raise ValueError(f"Expected a {shape_text} channel-first array, got {X.shape}")
    if X.shape[-2] < min_channels:
        count = "one channel" if min_channels == 1 else "two channels"
        raise ValueError(f"{name} requires at least {count}")
    if X.shape[-1] < min_times:
        raise ValueError(f"{name} requires at least two time samples")
    if X.ndim == 3 and X.shape[0] < 1:
        raise ValueError(f"{name} requires at least one epoch")
    if not np.isfinite(X).all():
        raise ValueError("X must contain only finite values")
    return X


def check_sfreq(sfreq: float | None, *, context: str | None = None) -> float:
    """Validate a sampling frequency and return it as a float.

    Parameters
    ----------
    sfreq : float | None
        Candidate sampling frequency.
    context : str | None, default=None
        What requires the value, used to explain a missing one, e.g.
        ``"lag_seconds"`` produces "sfreq is required when lag_seconds is used".

    Returns
    -------
    sfreq : float
        The validated sampling frequency.

    Raises
    ------
    TypeError
        If ``sfreq`` is a bool or not a real number.
    ValueError
        If ``sfreq`` is None, non-finite, or not positive.
    """
    if sfreq is None:
        where = f" when {context} is used" if context else ""
        raise ValueError(f"sfreq is required{where}")
    if isinstance(sfreq, bool) or not isinstance(sfreq, Real):
        raise TypeError("sfreq must be a real number")
    sfreq = float(sfreq)
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("sfreq must be a positive, finite number")
    return sfreq


def resolve_sample_window(
    window: Sequence[int | float],
    *,
    unit: str,
    sfreq: float | None = None,
    name: str = "window",
) -> tuple[int, int]:
    """Validate a half-open time window and resolve it to sample offsets.

    Parameters
    ----------
    window : sequence of int or float
        Ordered ``(start, stop)`` boundaries.
    unit : {"samples", "seconds"}
        Unit of the supplied boundaries. Sample-valued boundaries must be
        integers. Second-valued boundaries use nearest-sample rounding with
        ties to even.
    sfreq : float | None, default=None
        Sampling frequency. Required for second-valued windows and validated
        when supplied for sample-valued windows.
    name : str, default="window"
        Parameter name used in error messages.

    Returns
    -------
    sample_window : tuple of int
        Nonempty half-open ``(start, stop)`` sample offsets.
    """
    if unit not in {"samples", "seconds"}:
        raise ValueError(f"{name}_unit must be 'samples' or 'seconds', got {unit!r}.")

    values = np.asarray(window, dtype=object)
    if values.shape != (2,):
        raise ValueError(f"{name} must contain exactly two numeric boundaries.")
    normalized = []
    for value in values.tolist():
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name} must contain exactly two numeric boundaries.")
        if isinstance(value, Integral):
            normalized.append(int(value))
            continue
        if not np.isfinite(float(value)):
            raise ValueError(f"{name} boundaries must be finite.")
        normalized.append(float(value))
    if normalized[0] >= normalized[1]:
        raise ValueError(f"{name} start must be strictly less than {name} stop.")

    if unit == "samples":
        if any(not isinstance(value, Integral) for value in normalized):
            raise ValueError(
                f"{name} boundaries must be integers when {name}_unit='samples'."
            )
        if sfreq is not None:
            check_sfreq(sfreq)
        resolved = [int(value) for value in normalized]
    else:
        sfreq = check_sfreq(sfreq, context=f"{name}_unit='seconds'")
        scaled = [float(value) * sfreq for value in normalized]
        if not np.all(np.isfinite(scaled)):
            raise ValueError(
                f"{name} boundaries in seconds resolve outside the finite sample range."
            )
        resolved = [round(value) for value in scaled]

    bounds = np.iinfo(np.int64)
    if any(value < int(bounds.min) or value > int(bounds.max) for value in resolved):
        raise ValueError(f"{name} boundaries must fit in signed 64-bit samples.")
    if resolved[0] >= resolved[1]:
        raise ValueError(
            f"{name} resolves to an empty or reversed sample interval; "
            "increase its duration or sfreq."
        )
    return int(resolved[0]), int(resolved[1])


def check_chunk_size(chunk_size: int | None) -> int | None:
    """Validate an optional chunk size for blockwise processing.

    Parameters
    ----------
    chunk_size : int | None
        Number of samples per block, or None to process everything at once.

    Returns
    -------
    chunk_size : int | None
        The validated value.

    Raises
    ------
    TypeError
        If ``chunk_size`` is a bool or not an integer.
    ValueError
        If ``chunk_size`` is not positive.
    """
    if chunk_size is None:
        return None
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, Integral):
        raise TypeError("chunk_size must be a positive integer or None")
    if chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer or None")
    return int(chunk_size)


def resolve_sfreq(
    declared: float | None,
    data_sfreq: float | None,
    *,
    context: str | None = None,
    required: bool = True,
) -> float | None:
    """Reconcile a user-declared sampling frequency with container metadata.

    An MNE container carries its own sampling frequency. When the caller also
    declares one, the two must agree: silently preferring either would discard
    a stated intention.

    Parameters
    ----------
    declared : float | None
        Sampling frequency supplied by the caller, e.g. an estimator parameter.
    data_sfreq : float | None
        Sampling frequency read from an MNE container, or None for arrays.
    context : str | None, default=None
        What requires the value, used when neither source provides one.
    required : bool, default=True
        If False, return None instead of raising when neither is available.

    Returns
    -------
    sfreq : float | None
        The effective sampling frequency.

    Raises
    ------
    ValueError
        If the two sources disagree, or if none is available and ``required``.
    """
    if (
        declared is not None
        and data_sfreq is not None
        and not np.isclose(float(declared), float(data_sfreq))
    ):
        raise ValueError(f"sfreq={declared} disagrees with MNE info sfreq={data_sfreq}")
    value = data_sfreq if data_sfreq is not None else declared
    if value is None and not required:
        return None
    return check_sfreq(value, context=context)


def check_channel_layout(
    name: str,
    *,
    n_channels: int,
    fitted_n_channels: int,
    ch_names: tuple[str, ...] | list[str] | None = None,
    fitted_ch_names: tuple[str, ...] | list[str] | None = None,
) -> None:
    """Verify that transform input matches the layout seen during fit.

    Parameters
    ----------
    name : str
        Algorithm name used in error messages, e.g. ``"SNS"``.
    n_channels, fitted_n_channels : int
        Channel counts of the current input and of the fitted data.
    ch_names, fitted_ch_names : sequence of str | None, default=None
        Channel names of the current input and of the fitted data. The check is
        skipped when either is None, as it is for array input.

    Raises
    ------
    ValueError
        If the names, their order, or the channel counts differ.
    """
    if (
        ch_names is not None
        and fitted_ch_names is not None
        and tuple(ch_names) != tuple(fitted_ch_names)
    ):
        raise ValueError(
            f"MNE channel names/order differ from fit; apply {name} to the "
            "exact fitted channel layout"
        )
    if n_channels != fitted_n_channels:
        raise ValueError(
            f"X has {n_channels} channels; fitted data had {fitted_n_channels}"
        )
