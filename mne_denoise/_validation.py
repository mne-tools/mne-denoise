"""Shared validation helpers."""

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


def check_positive_real(value: float, *, name: str) -> float:
    """Validate a positive finite real parameter and return it as a float."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a positive, finite number")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive, finite number")
    return value


def check_option(value, *, name: str, allowed: Sequence[object]):
    """Validate a categorical parameter against the supplied allowed values."""
    allowed = tuple(allowed)
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed!r}; received value {value!r}")
    return value


def check_channel_first_data(
    X: np.ndarray,
    *,
    name: str,
    allow_epochs: bool = True,
    min_channels: int = 2,
    min_times: int = 2,
) -> np.ndarray:
    """Validate a finite 2-D or optional 3-D channel-first array.

    Parameters
    ----------
    X : array-like
        Shape (n_channels, n_times), or (n_epochs, n_channels, n_times) when
        allow_epochs=True.
    name : str
        Name used in validation errors.
    allow_epochs : bool, default=True
        Whether 3-D input is accepted.
    min_channels, min_times : int, default=2
        Minimum channel and time dimensions.

    Returns
    -------
    X : ndarray
        Float64 view or copy of the input.
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


def check_matching_sfreq(
    input_sfreq: float | None,
    fitted_sfreq: float | None,
    *,
    name: str,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> None:
    """Verify that input sampling frequency matches a fitted value.

    A missing value is accepted because some array-based transforms do not
    carry sampling-frequency metadata. Standalone validity checks belong to
    :func:`check_positive_real`.
    """
    if input_sfreq is None or fitted_sfreq is None:
        return
    if not np.isclose(input_sfreq, fitted_sfreq, rtol=rtol, atol=atol):
        raise ValueError(
            f"{name}: transform sfreq={input_sfreq} does not match fitted "
            f"sfreq={fitted_sfreq}; sampling frequency must match the fitted value"
        )


def resolve_sample_window(
    window: Sequence[int | float],
    *,
    unit: str,
    sfreq: float | None = None,
    name: str = "window",
) -> tuple[int, int]:
    """Validate a half-open window and resolve it to sample offsets.

    Parameters
    ----------
    window : sequence of int or float
        Start and stop boundaries.
    unit : {"samples", "seconds"}
        Units of the boundaries.
    sfreq : float or None, default=None
        Sampling frequency for second-valued windows.
    name : str, default="window"
        Name used in validation errors.

    Returns
    -------
    sample_window : tuple of int
        Non-empty half-open sample interval.
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
            check_positive_real(sfreq, name="sfreq")
        resolved = [int(value) for value in normalized]
    else:
        if sfreq is None:
            raise ValueError(f"sfreq is required when {name}_unit='seconds' is used")
        sfreq = check_positive_real(sfreq, name="sfreq")
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
    """Validate an optional positive block size and return it unchanged as an int."""
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
    """Resolve a declared sampling frequency against MNE metadata.

    A mismatch raises; when required=False, missing values return None.
    """
    if declared is not None:
        declared = check_positive_real(declared, name="sfreq")
    if data_sfreq is not None:
        data_sfreq = check_positive_real(data_sfreq, name="sfreq")
    if (
        declared is not None
        and data_sfreq is not None
        and not np.isclose(declared, data_sfreq)
    ):
        raise ValueError(f"sfreq={declared} disagrees with MNE info sfreq={data_sfreq}")
    value = data_sfreq if data_sfreq is not None else declared
    if value is None and not required:
        return None
    if value is None:
        where = f" when {context} is used" if context else ""
        raise ValueError(f"sfreq is required{where}")
    return value


def check_channel_layout(
    name: str,
    *,
    n_channels: int,
    fitted_n_channels: int,
    ch_names: tuple[str, ...] | list[str] | None = None,
    fitted_ch_names: tuple[str, ...] | list[str] | None = None,
) -> None:
    """Verify channel count and, when available, name/order equality with the fitted layout."""
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
            f"{name}: X has {n_channels} channels; fitted data had {fitted_n_channels}"
        )
