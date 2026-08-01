"""Artifact-based bias functions for DSS.

Implements cycle averaging for quasi-periodic artifacts like ECG and blinks.
This emphasizes reproducible artifact morphology while canceling neural activity.

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)

References
----------
.. [1] Särelä & Valpola (2005). Denoising Source Separation. J. Mach. Learn. Res., 6, 233-272.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Literal

import numpy as np

from .base import LinearDenoiser


class CycleAverageBias(LinearDenoiser):
    """Bias for removing quasi-periodic artifacts (e.g., ECG, EOG).

    Applies cycle averaging synchronized to artifact events (e.g., R-peaks
    for ECG, blink onsets for EOG). This emphasizes the stereotyped
    artifact waveform while canceling non-phase-locked neural activity.

    Parameters
    ----------
    event_samples : array-like of int
        Sample indices of artifact events (e.g., R-peak locations). Their
        coordinate system is declared by ``event_origin``. Values must fit in
        the signed 64-bit sample range; out-of-range coordinates are rejected
        instead of wrapped.
    window : tuple of int or float
        Half-open interval ``(start, stop)`` around each event. Its unit is
        declared by ``window_unit``. Resolved sample boundaries must fit in the
        signed 64-bit sample range. The default is ``(-100, 200)``.
    window_unit : {"samples", "seconds"} | None
        Unit of ``window``. Canonical calls must specify this explicitly. If
        ``None``, the 0.x compatibility behavior infers seconds when ``sfreq``
        is provided and samples otherwise, and emits ``FutureWarning``.
    sfreq : float, optional
        Sampling frequency in Hz. Required when ``window_unit="seconds"``.
        Second-valued boundaries are converted exactly once at construction by
        rounding to the nearest sample (ties to even).
    event_origin : {"data", "raw"} | None
        ``"data"`` means event zero is the first sample of the array passed to
        :meth:`apply`. ``"raw"`` means MNE acquisition sample numbering (for
        example, column zero from an MNE events array); ``first_samp`` is then
        subtracted exactly once at construction. Canonical calls must specify
        this explicitly. ``None`` retains the 0.x data-relative behavior and
        emits ``FutureWarning``.
    first_samp : int, optional
        First acquisition sample of the corresponding MNE Raw object. Required
        for ``event_origin="raw"`` and forbidden for ``event_origin="data"``.

    Examples
    --------
    >>> # ECG artifact removal
    >>> from mne.preprocessing import find_ecg_events
    >>> from mne_denoise.dss.denoisers import CycleAverageBias
    >>> r_peaks, _, _ = find_ecg_events(raw)  # MNE returns events array
    >>> # Extract sample indices (column 0)
    >>> r_peak_samples = r_peaks[:, 0]
    >>> bias = CycleAverageBias(
    ...     event_samples=r_peak_samples,
    ...     window=(-0.1, 0.2),
    ...     window_unit="seconds",
    ...     sfreq=raw.info["sfreq"],
    ...     event_origin="raw",
    ...     first_samp=raw.first_samp,
    ... )
    >>> biased_data = bias.apply(raw.get_data())

    >>> # EOG (blink) artifact removal
    >>> from mne.preprocessing import find_eog_events
    >>> blinks = find_eog_events(raw)
    >>> blink_samples = blinks[:, 0]
    >>> bias_eog = CycleAverageBias(
    ...     event_samples=blink_samples,
    ...     window=(-200, 200),
    ...     window_unit="samples",
    ...     event_origin="raw",
    ...     first_samp=raw.first_samp,
    ... )
    >>> biased_eog = bias_eog.apply(raw.get_data())

    References
    ----------
    Särelä & Valpola (2005). Section 4.1.4 "DENOISING OF QUASIPERIODIC SIGNALS"
    """

    def __init__(
        self,
        event_samples: Sequence[int],
        window: tuple[int | float, int | float] = (-100, 200),
        *,
        window_unit: Literal["samples", "seconds"] | None = None,
        sfreq: float | None = None,
        event_origin: Literal["data", "raw"] | None = None,
        first_samp: int | None = None,
    ) -> None:
        self.event_samples = self._validate_event_samples(event_samples)
        self.window_input = self._validate_window(window)
        self.window_unit = self._resolve_window_unit(window_unit, sfreq)
        self.sfreq = self._validate_sfreq(sfreq, self.window_unit)
        self.window = self._window_to_samples(
            self.window_input,
            window_unit=self.window_unit,
            sfreq=self.sfreq,
        )
        self.event_origin = self._resolve_event_origin(event_origin)
        self.first_samp = self._validate_first_samp(first_samp, self.event_origin)
        event_offset = 0 if self.first_samp is None else self.first_samp
        self.event_offset_samples_ = int(event_offset)
        data_events = [
            int(event) - self.event_offset_samples_ for event in self.event_samples
        ]
        self.event_samples_data_ = self._as_int64(
            data_events,
            "event_samples after applying first_samp",
        )
        self._window_length = self.window[1] - self.window[0]

    @staticmethod
    def _as_int64(values: Sequence[int | float], name: str) -> np.ndarray:
        """Return exact integral values represented safely as signed int64."""
        array = np.asarray(values)
        if array.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional sequence.")

        minimum = int(np.iinfo(np.int64).min)
        maximum = int(np.iinfo(np.int64).max)
        converted: list[int] = []
        for value in array.tolist():
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{name} must contain only finite integers.")
            if isinstance(value, (int, np.integer)):
                integer = int(value)
            elif isinstance(value, (float, np.floating)):
                scalar = float(value)
                if not np.isfinite(scalar) or not scalar.is_integer():
                    raise ValueError(f"{name} must contain only finite integers.")
                integer = int(scalar)
            else:
                raise ValueError(f"{name} must contain only finite integers.")
            if integer < minimum or integer > maximum:
                raise ValueError(f"{name} values must fit in signed 64-bit samples.")
            converted.append(integer)
        return np.asarray(converted, dtype=np.int64)

    @classmethod
    def _validate_event_samples(cls, event_samples: Sequence[int]) -> np.ndarray:
        """Return event positions without truncation or integer wraparound."""
        events = np.asarray(event_samples)
        if events.ndim != 1:
            raise ValueError("event_samples must be a one-dimensional sequence.")
        try:
            return cls._as_int64(events, "event_samples")
        except ValueError as error:
            if "one-dimensional" in str(error) or "signed 64-bit" in str(error):
                raise
            raise ValueError(
                "event_samples must contain integer sample indices."
            ) from error

    @staticmethod
    def _validate_window(
        window: tuple[int | float, int | float],
    ) -> tuple[int | float, int | float]:
        """Validate a two-boundary finite window."""
        values = np.asarray(window, dtype=object)
        if values.shape != (2,):
            raise ValueError("window must contain exactly two numeric boundaries.")
        normalized: list[int | float] = []
        for value in values.tolist():
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                raise ValueError("window must contain exactly two numeric boundaries.")
            if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                raise ValueError("window boundaries must be finite.")
            normalized.append(
                int(value) if isinstance(value, (int, np.integer)) else float(value)
            )
        start, stop = normalized
        if start >= stop:
            raise ValueError("window start must be strictly less than window stop.")
        return start, stop

    @staticmethod
    def _resolve_window_unit(
        window_unit: Literal["samples", "seconds"] | None,
        sfreq: float | None,
    ) -> Literal["samples", "seconds"]:
        """Resolve the deprecated unit inference used by released 0.x code."""
        if window_unit is None:
            inferred = "seconds" if sfreq is not None else "samples"
            warnings.warn(
                "Implicit CycleAverageBias window units are deprecated and will "
                "be removed in mne-denoise 1.0; pass window_unit='samples' or "
                "window_unit='seconds' explicitly.",
                FutureWarning,
                stacklevel=3,
            )
            return inferred
        if window_unit not in {"samples", "seconds"}:
            raise ValueError(
                f"window_unit must be 'samples' or 'seconds', got {window_unit!r}."
            )
        return window_unit

    @staticmethod
    def _validate_sfreq(
        sfreq: float | None,
        window_unit: Literal["samples", "seconds"],
    ) -> float | None:
        """Validate sampling frequency only when supplied or required."""
        if sfreq is None:
            if window_unit == "seconds":
                raise ValueError("sfreq is required when window_unit='seconds'.")
            return None
        if isinstance(sfreq, bool) or not np.isfinite(sfreq) or float(sfreq) <= 0:
            raise ValueError("sfreq must be a finite positive number.")
        return float(sfreq)

    @staticmethod
    def _window_to_samples(
        window: tuple[int | float, int | float],
        *,
        window_unit: Literal["samples", "seconds"],
        sfreq: float | None,
    ) -> tuple[int, int]:
        """Convert a validated interval to integer sample boundaries once."""
        if window_unit == "samples":
            try:
                samples = CycleAverageBias._as_int64(window, "window boundaries")
            except ValueError as error:
                if "signed 64-bit" in str(error):
                    raise
                raise ValueError(
                    "window boundaries must be integers when window_unit='samples'."
                ) from error
        else:
            # _validate_sfreq guarantees this for the seconds pathway.
            assert sfreq is not None
            rounded: list[int] = []
            for boundary in window:
                scaled = float(boundary) * sfreq
                if not np.isfinite(scaled):
                    raise ValueError(
                        "window boundaries in seconds resolve outside the finite "
                        "sample range."
                    )
                rounded.append(round(scaled))
            samples = CycleAverageBias._as_int64(rounded, "window boundaries")
        if samples[0] >= samples[1]:
            raise ValueError(
                "window resolves to an empty or reversed sample interval; "
                "increase its duration or sfreq."
            )
        return int(samples[0]), int(samples[1])

    @staticmethod
    def _resolve_event_origin(
        event_origin: Literal["data", "raw"] | None,
    ) -> Literal["data", "raw"]:
        """Resolve the deprecated data-relative origin used by 0.x code."""
        if event_origin is None:
            warnings.warn(
                "Implicit CycleAverageBias event origin is deprecated and will "
                "be removed in mne-denoise 1.0; pass event_origin='data' or "
                "event_origin='raw' explicitly.",
                FutureWarning,
                stacklevel=3,
            )
            return "data"
        if event_origin not in {"data", "raw"}:
            raise ValueError(
                f"event_origin must be 'data' or 'raw', got {event_origin!r}."
            )
        return event_origin

    @staticmethod
    def _validate_first_samp(
        first_samp: int | None,
        event_origin: Literal["data", "raw"],
    ) -> int | None:
        """Validate the sole offset used to map events into data coordinates."""
        if event_origin == "raw" and first_samp is None:
            raise ValueError("first_samp is required when event_origin='raw'.")
        if event_origin == "data" and first_samp is not None:
            raise ValueError(
                "first_samp must be omitted when event_origin='data'; the event "
                "samples are already relative to the supplied data."
            )
        if first_samp is None:
            return None
        if not isinstance(first_samp, int | np.integer) or isinstance(first_samp, bool):
            raise TypeError("first_samp must be an integer.")
        value = int(first_samp)
        bounds = np.iinfo(np.int64)
        if value < int(bounds.min) or value > int(bounds.max):
            raise ValueError("first_samp must fit in signed 64-bit samples.")
        return value

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply cycle averaging bias.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_times) or (n_channels, n_times, n_epochs)
            Input data.

        Returns
        -------
        biased : ndarray, same shape as input
            Data where artifact-locked segments are replaced by cycle average.
        """
        data = np.asarray(data)
        original_shape = data.shape

        # Handle 3D epoched data by concatenating
        if data.ndim == 3:
            n_channels, n_times, n_epochs = data.shape
            # Adjust events for concatenated epochs
            data_2d = data.transpose(0, 2, 1).reshape(n_channels, -1)
            total_samples = n_times * n_epochs
        elif data.ndim == 2:
            data_2d = data
            n_channels, total_samples = data.shape
        else:
            raise ValueError(f"Data must be 2D or 3D, got {data.ndim}D")

        # Filter valid events (within data bounds)
        pre, post = self.window
        # Compare before adding the window offsets so extreme, invalid event
        # coordinates cannot wrap around signed int64 and appear valid.
        minimum_event = -pre
        maximum_event = total_samples - post
        valid_events = np.asarray(
            [
                int(event)
                for event in self.event_samples_data_
                if minimum_event <= int(event) <= maximum_event
            ],
            dtype=np.int64,
        )

        if len(valid_events) == 0:
            # No valid events, return zeros (no artifact signal)
            return np.zeros(original_shape)

        # Compute cycle average
        window_len = post - pre
        epochs_matrix = np.zeros((len(valid_events), n_channels, window_len))

        for i, event in enumerate(valid_events):
            start = event + pre
            end = event + post
            epochs_matrix[i] = data_2d[:, start:end]

        # Average across artifact cycles
        cycle_average = np.mean(epochs_matrix, axis=0)  # (n_channels, window_len)

        # Create biased output: each artifact window gets the average
        biased_2d = np.zeros_like(data_2d)

        for event in valid_events:
            start = event + pre
            end = event + post
            biased_2d[:, start:end] = cycle_average

        # Reshape back if needed
        if len(original_shape) == 3:
            biased = biased_2d.reshape(
                original_shape[0], original_shape[2], original_shape[1]
            ).transpose(0, 2, 1)
        else:
            biased = biased_2d

        return biased
