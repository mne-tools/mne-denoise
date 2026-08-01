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
        coordinate system is declared by ``event_origin``.
    window : tuple of int or float
        Half-open interval ``(start, stop)`` around each event. Its unit is
        declared by ``window_unit``. The default is ``(-100, 200)``.
    window_unit : {"samples", "seconds"} | None
        Unit of ``window``. Canonical calls must specify this explicitly. If
        ``None``, the 0.x compatibility behavior infers seconds when ``sfreq``
        is provided and samples otherwise, and emits ``FutureWarning``.
    sfreq : float, optional
        Sampling frequency in Hz. Required when ``window_unit="seconds"``.
        Second-valued boundaries are converted exactly once at construction by
        rounding to the nearest sample with :func:`numpy.rint`.
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
        self.event_samples_data_ = self.event_samples - self.event_offset_samples_
        self._window_length = self.window[1] - self.window[0]

    @staticmethod
    def _validate_event_samples(event_samples: Sequence[int]) -> np.ndarray:
        """Return event positions without silently truncating values."""
        events = np.asarray(event_samples)
        if events.ndim != 1:
            raise ValueError("event_samples must be a one-dimensional sequence.")
        if not np.issubdtype(events.dtype, np.number) or not np.all(
            np.isfinite(events)
        ):
            raise ValueError("event_samples must contain only finite integers.")
        if not np.all(events == np.rint(events)):
            raise ValueError("event_samples must contain integer sample indices.")
        return events.astype(np.int64, copy=True)

    @staticmethod
    def _validate_window(
        window: tuple[int | float, int | float],
    ) -> tuple[float, float]:
        """Validate a two-boundary finite window."""
        values = np.asarray(window)
        if values.shape != (2,) or not np.issubdtype(values.dtype, np.number):
            raise ValueError("window must contain exactly two numeric boundaries.")
        if not np.all(np.isfinite(values)):
            raise ValueError("window boundaries must be finite.")
        start, stop = (float(values[0]), float(values[1]))
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
        window: tuple[float, float],
        *,
        window_unit: Literal["samples", "seconds"],
        sfreq: float | None,
    ) -> tuple[int, int]:
        """Convert a validated interval to integer sample boundaries once."""
        values = np.asarray(window)
        if window_unit == "samples":
            if not np.all(values == np.rint(values)):
                raise ValueError(
                    "window boundaries must be integers when window_unit='samples'."
                )
            samples = np.rint(values).astype(int)
        else:
            # _validate_sfreq guarantees this for the seconds pathway.
            assert sfreq is not None
            samples = np.rint(values * sfreq).astype(int)
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
        return int(first_samp)

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
        valid_mask = (self.event_samples_data_ + pre >= 0) & (
            self.event_samples_data_ + post <= total_samples
        )
        valid_events = self.event_samples_data_[valid_mask]

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
