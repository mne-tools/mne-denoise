"""Event-locked bias functions for DSS."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np

from ..._validation import (
    check_positive_integer,
    check_positive_real,
    resolve_sample_window,
)
from .base import LinearDenoiser


def _as_int64_coordinates(
    values: object, name: str, *, ndim: tuple[int, ...]
) -> np.ndarray:
    """Return exact integer coordinates without signed overflow."""
    array = np.asarray(values, dtype=object)
    if array.ndim not in ndim:
        dimensions = " or ".join(f"{value}-D" for value in ndim)
        raise ValueError(f"{name} must be a {dimensions} integer array.")
    if array.ndim == 2 and array.shape[1] != 2:
        raise ValueError(
            f"{name} must have shape (n_events, 2) for per-epoch coordinates."
        )

    bounds = np.iinfo(np.int64)
    converted = []
    for value in array.ravel().tolist():
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise ValueError(f"{name} must contain only integer coordinates.")
        integer = int(value)
        if integer < int(bounds.min) or integer > int(bounds.max):
            raise ValueError(f"{name} values must fit in signed 64-bit samples.")
        converted.append(integer)
    return np.asarray(converted, dtype=np.int64).reshape(array.shape)


def _prepare_events(
    event_samples: object, event_origin: str, first_samp: int | None
) -> tuple[np.ndarray, int | None, np.ndarray]:
    """Validate, deduplicate, and map public event coordinates."""
    if event_origin not in {"data", "raw"}:
        raise ValueError(f"event_origin must be 'data' or 'raw', got {event_origin!r}.")
    if event_origin == "raw" and first_samp is None:
        raise ValueError("first_samp is required when event_origin='raw'.")
    if event_origin == "data" and first_samp is not None:
        raise ValueError(
            "first_samp must be omitted when event_origin='data'; events are "
            "already relative to the supplied data."
        )

    events = np.unique(
        _as_int64_coordinates(event_samples, "event_samples", ndim=(1, 2)), axis=0
    )
    if events.ndim == 2:
        if np.any(events[:, 0] < 0):
            raise ValueError("epoch indices in event_samples must be non-negative.")
        if event_origin == "raw":
            raise ValueError(
                "event_origin='raw' is not defined for per-epoch coordinates; "
                "use data-relative (epoch_index, sample_index) pairs."
            )
        return events, None, events.copy()

    if first_samp is None:
        return events, None, events.copy()
    first_samp = int(_as_int64_coordinates([first_samp], "first_samp", ndim=(1,))[0])
    mapped = _as_int64_coordinates(
        [int(event) - first_samp for event in events],
        "event_samples after applying first_samp",
        ndim=(1,),
    )
    return events, first_samp, mapped


def _as_float_data(data: np.ndarray) -> np.ndarray:
    """Validate numerical input and convert it without mutating the caller."""
    array = np.asarray(data)
    if array.ndim not in (2, 3):
        raise ValueError(f"Data must be 2D or 3D, got {array.ndim}D.")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("Data must contain real numerical values.")
    if np.iscomplexobj(array):
        raise TypeError("Data must be real-valued, not complex.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Data must contain only finite values.")
    dtype = np.float32 if array.dtype == np.float32 else np.float64
    return np.asarray(array, dtype=dtype)


def _event_segments(
    data: np.ndarray,
    events: np.ndarray,
    window: tuple[int, int],
    min_events: int,
) -> list[tuple[slice, ...]]:
    """Validate coordinates and return safe indexing segments."""
    if data.ndim == 2 and events.ndim != 1:
        raise ValueError("2-D data requires one-dimensional event sample coordinates.")
    if data.ndim == 3 and events.ndim != 2:
        raise ValueError(
            "3-D data requires (epoch_index, sample_index) event coordinates; "
            "flat global sample indices are not accepted."
        )
    if len(events) < min_events:
        raise ValueError(
            f"CycleAverageBias requires at least {min_events} unique complete "
            f"events; received {len(events)} after deduplication."
        )

    n_times = data.shape[1]
    n_epochs = None if data.ndim == 2 else data.shape[2]
    pre, post = window
    segments = []
    for coordinate in events.tolist():
        if events.ndim == 1:
            sample = int(coordinate)
            label = f"event sample {sample}"
            segment = (slice(sample + pre, sample + post),)
        else:
            epoch, sample = (int(value) for value in coordinate)
            label = f"event ({epoch}, {sample})"
            assert n_epochs is not None
            if epoch >= n_epochs:
                raise ValueError(
                    f"{label} is out of range for data with {n_epochs} epochs."
                )
            segment = (slice(sample + pre, sample + post), epoch)
        if sample < 0 or sample >= n_times:
            raise ValueError(
                f"{label} is out of range for the sample coordinates [0, {n_times})."
            )
        if sample + pre < 0 or sample + post > n_times:
            raise ValueError(
                f"{label} with half-open window [{sample + pre}, {sample + post}) "
                f"crosses the data boundary [0, {n_times}); adjust events or window."
            )
        segments.append(segment)
    return segments


class CycleAverageBias(LinearDenoiser):
    """Fixed-window event-locked averaging bias.

    Parameters
    ----------
    event_samples : array-like of int, shape (n_events,) or (n_events, 2)
        One-dimensional sample coordinates for 2D input, or
        ``(epoch_index, sample_index)`` pairs for 3D channel-first input.
    window : tuple of int or float, default=(-100, 200)
        Half-open interval ``[event + start, event + stop)``.
    window_unit : {"samples", "seconds"}, default="samples"
        Unit for ``window``; seconds require ``sfreq``.
    sfreq : float or None, default=None
        Sampling frequency in Hz for second-valued windows.
    event_origin : {"data", "raw"}, default="data"
        Origin for one-dimensional coordinates. ``"raw"`` requires ``first_samp``.
    first_samp : int or None, default=None
        Acquisition sample offset used with ``event_origin="raw"``.
    min_events : int, default=2
        Minimum number of unique complete events.

    Notes
    -----
    Windows are half-open. Boundary-crossing or incomplete events raise an error;
    overlapping contributions are averaged and duplicate coordinates are removed.
    This fixed-window bias is not the complete variable-period quasiperiodic
    procedure described in the original DSS work.
    """

    def __init__(
        self,
        event_samples: Sequence[int] | Sequence[tuple[int, int]],
        window: tuple[int | float, int | float] = (-100, 200),
        *,
        window_unit: Literal["samples", "seconds"] = "samples",
        sfreq: float | None = None,
        event_origin: Literal["data", "raw"] = "data",
        first_samp: int | None = None,
        min_events: int = 2,
    ) -> None:
        self.window = resolve_sample_window(
            window, unit=window_unit, sfreq=sfreq, name="window"
        )
        self.window_input = tuple(window)
        self.window_unit = window_unit
        self.sfreq = (
            check_positive_real(sfreq, name="sfreq") if sfreq is not None else None
        )
        self.event_origin = event_origin
        self.event_samples, self.first_samp, self.event_samples_data_ = _prepare_events(
            event_samples, event_origin, first_samp
        )
        self.min_events = check_positive_integer(min_events, name="min_events")
        if self.min_events < 2:
            raise ValueError("min_events must be greater than or equal to 2")
        self._window_length = self.window[1] - self.window[0]

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply fixed-window event-locked averaging.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_times) or (n_channels, n_times, n_epochs)
            Continuous or channel-first epoched data.

        Returns
        -------
        ndarray
            Floating-point event-locked estimate with the input shape.
        """
        data = _as_float_data(data)
        segments = _event_segments(
            data, self.event_samples_data_, self.window, self.min_events
        )
        windows = [data[(slice(None), *segment)] for segment in segments]
        cycle_average = np.mean(np.stack(windows), axis=0, dtype=data.dtype)

        biased = np.zeros_like(data)
        counts = np.zeros(data.shape[1:], dtype=np.int64)
        for segment in segments:
            biased[(slice(None), *segment)] += cycle_average
            counts[segment] += 1
        np.divide(
            biased,
            counts[np.newaxis, ...],
            out=biased,
            where=counts[np.newaxis, ...] > 0,
        )
        return biased
