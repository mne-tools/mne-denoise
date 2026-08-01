"""Experimental cardiac-artifact DSS recipe.

This module deliberately keeps event-coordinate handling separate from true
time-shift DSS.  ``CardiacDSS`` is a thin, inductive estimator around linear
DSS with a cycle-average bias: QRS events are consumed during ``fit`` only,
and the fitted spatial transform is used during ``transform``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from ...utils import extract_data_from_mne
from ..denoisers import CycleAverageBias
from ..linear import DSS

ComponentAction = Literal["extract", "retain", "subtract"]
CoordinateUnit = Literal["samples", "seconds"]
EventOrigin = Literal["data", "raw"]


class CardiacDSSStatus(str, Enum):
    """Terminal status of an experimental :class:`CardiacDSS` fit."""

    APPLIED = "applied"
    ABSTAINED = "abstained"
    NO_OP = "no_op"
    INADMISSIBLE = "inadmissible"


@dataclass(frozen=True, slots=True)
class CardiacDSSDiagnostics:
    """Immutable descriptive diagnostics for a ``CardiacDSS`` fit.

    These fields describe execution; they are not evidence that attenuation or
    neural-signal preservation met a scientific acceptance criterion.
    """

    status: CardiacDSSStatus
    reason: str | None
    input_layout: str
    component_action: str
    component_selection: int
    input_event_count: int
    valid_event_count: int
    excluded_event_count: int
    n_channels: int
    n_times: int
    n_epochs: int | None
    sfreq: float | None
    window_samples: tuple[int, int]
    event_origin: str
    first_samp: int | None
    n_selected: int | None
    eigenvalues: tuple[float, ...]

    def __post_init__(self) -> None:
        """Reject internally inconsistent diagnostic records."""
        if not isinstance(self.status, CardiacDSSStatus):
            raise TypeError("status must be a CardiacDSSStatus.")
        if self.status is CardiacDSSStatus.APPLIED and self.reason is not None:
            raise ValueError("Applied diagnostics cannot include a reason.")
        if self.status is not CardiacDSSStatus.APPLIED and not self.reason:
            raise ValueError("Non-applied diagnostics require a reason.")
        if self.component_action not in {"extract", "retain", "subtract"}:
            raise ValueError("component_action is invalid.")
        if self.component_selection < 0:
            raise ValueError("component_selection must be non-negative.")
        counts = (
            self.input_event_count,
            self.valid_event_count,
            self.excluded_event_count,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("Event counts must be non-negative integers.")
        if self.valid_event_count + self.excluded_event_count != self.input_event_count:
            raise ValueError(
                "Valid and excluded event counts must sum to the input count."
            )
        if self.n_channels <= 0 or self.n_times <= 0:
            raise ValueError("Input channel and time counts must be positive.")
        if self.n_epochs is not None and self.n_epochs <= 0:
            raise ValueError("n_epochs must be positive when present.")
        if self.sfreq is not None and (not np.isfinite(self.sfreq) or self.sfreq <= 0):
            raise ValueError("sfreq must be finite and positive when present.")
        start, stop = self.window_samples
        if type(start) is not int or type(stop) is not int or start >= stop:
            raise ValueError("window_samples must be an increasing integer pair.")
        if self.event_origin not in {"data", "raw"}:
            raise ValueError("event_origin is invalid.")
        if self.event_origin == "raw" and self.first_samp is None:
            raise ValueError("Raw-origin diagnostics require first_samp.")
        if self.event_origin == "data" and self.first_samp is not None:
            raise ValueError("Data-origin diagnostics cannot include first_samp.")
        if self.n_selected is not None and self.n_selected < 0:
            raise ValueError("n_selected must be non-negative when present.")
        if not all(np.isfinite(value) for value in self.eigenvalues):
            raise ValueError("eigenvalues must be finite.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe copy of the diagnostics."""
        return {
            "status": self.status.value,
            "reason": self.reason,
            "input_layout": self.input_layout,
            "component_action": self.component_action,
            "component_selection": self.component_selection,
            "input_event_count": self.input_event_count,
            "valid_event_count": self.valid_event_count,
            "excluded_event_count": self.excluded_event_count,
            "n_channels": self.n_channels,
            "n_times": self.n_times,
            "n_epochs": self.n_epochs,
            "sfreq": self.sfreq,
            "window_samples": list(self.window_samples),
            "event_origin": self.event_origin,
            "first_samp": self.first_samp,
            "n_selected": self.n_selected,
            "eigenvalues": list(self.eigenvalues),
        }


@dataclass(frozen=True, slots=True)
class _FitInput:
    """Validated input layout used to prepare event coordinates."""

    data: np.ndarray
    layout: str
    mne_type: str
    sfreq: float | None
    n_channels: int
    n_times: int
    n_epochs: int | None


@dataclass(frozen=True, slots=True)
class _PreparedEvents:
    """Cycle-average bias plus its descriptive event accounting."""

    bias: CycleAverageBias
    valid_data_samples: tuple[int, ...]
    input_count: int
    valid_count: int
    window_samples: tuple[int, int]


class CardiacDSS(BaseEstimator, TransformerMixin):
    """Experimental QRS-synchronized cardiac-artifact DSS estimator.

    ``CardiacDSS`` fits ordinary linear DSS with a
    :class:`~mne_denoise.dss.CycleAverageBias`. It does not detect QRS events;
    event coordinates and their units must be supplied explicitly. The method
    is an unvalidated research prototype and does not imply a scientifically
    safe operating point.

    Parameters
    ----------
    qrs_events : array-like
        For continuous two-dimensional data, a one-dimensional sequence of QRS
        coordinates. For NumPy 3-D arrays and MNE Epochs, an ``(n_events, 2)``
        array of ``(epoch_index, coordinate_within_epoch)`` pairs. Epoch indices
        are always zero-based; the second column uses ``event_unit``.
    component_action : {'extract', 'retain', 'subtract'}
        Explicit DSS operation. No implicit polarity is inferred.
    component_selection : int
        Explicit number of leading components used by sensor-space actions.
        Automatic selection is intentionally not part of this experimental
        recipe because its thresholds would need their own operating-point
        policy. Zero with ``component_action='subtract'`` is an exact no-op.
    window : tuple of int or float, default=(-0.2, 0.4)
        Half-open cycle window around each QRS event.
    window_unit : {'samples', 'seconds'}, default='seconds'
        Unit of ``window``.
    event_unit : {'samples', 'seconds'}, default='samples'
        Unit of continuous coordinates or the second column of epoched pairs.
        ``event_origin='raw'`` requires acquisition-sample coordinates and
        therefore cannot be combined with ``event_unit='seconds'``.
    sfreq : float | None, default=None
        Sampling frequency for second-valued coordinates. It is inferred from
        MNE inputs. If supplied with an MNE input, it must match the metadata.
    event_origin : {'data', 'raw'}, default='data'
        ``'data'`` makes zero the first supplied sample. ``'raw'`` uses MNE Raw
        acquisition numbering and subtracts ``first_samp`` exactly once. Raw
        origin is not available for epoched inputs.
    first_samp : int | None, default=None
        Required with ``event_origin='raw'`` and forbidden otherwise. For MNE
        Raw input, it must equal ``raw.first_samp``.
    min_valid_events : int, default=1
        User-controlled execution policy. If fewer complete, in-bounds windows
        remain, subtraction abstains with an exact passthrough. Extraction or
        retention is marked inadmissible because no fitted output is defined.
    n_components : int | None, default=None
        Number of DSS components to fit. ``None`` retains the whitening rank.
    whitening_rank : int | dict | None, default=None
        Rank used by the underlying DSS whitening operation.
    reg : float, default=1e-9
        Covariance regularization passed to :class:`~mne_denoise.dss.DSS`.
    normalize_input : bool, default=True
        Whether DSS normalizes channels before fitting.
    whiten : bool, default=False
        Whether mixed MNE data-channel types are jointly pre-whitened.
    noise_cov : mne.Covariance | None, default=None
        Optional MNE noise covariance used when ``whiten=True``.
    verbose : bool | str | int | None, default=None
        Logging verbosity passed to DSS.

    Attributes
    ----------
    diagnostics_ : CardiacDSSDiagnostics | None
        Immutable execution diagnostics after fitting.
    estimator_ : DSS | None
        Fitted DSS estimator for an applied run, otherwise ``None``.
    bias_ : CycleAverageBias | None
        Resolved cycle-average bias. Its event positions are already mapped to
        data coordinates exactly once.
    filters_, patterns_, eigenvalues_ : ndarray | None
        Fitted DSS quantities, or ``None`` for abstained/inadmissible/no-op runs.

    Notes
    -----
    Epoched QRS windows are accepted only when fully contained in their declared
    epoch. They are mapped into the epoch-major concatenation used by
    ``CycleAverageBias`` only after this check, so a window cannot borrow samples
    from an adjacent epoch. QRS events are not used by :meth:`transform`.
    """

    def __init__(
        self,
        qrs_events,
        *,
        component_action: ComponentAction,
        component_selection: int,
        window: tuple[int | float, int | float] = (-0.2, 0.4),
        window_unit: CoordinateUnit = "seconds",
        event_unit: CoordinateUnit = "samples",
        sfreq: float | None = None,
        event_origin: EventOrigin = "data",
        first_samp: int | None = None,
        min_valid_events: int = 1,
        n_components: int | None = None,
        whitening_rank: int | dict | None = None,
        reg: float = 1e-9,
        normalize_input: bool = True,
        whiten: bool = False,
        noise_cov=None,
        verbose: bool | str | int | None = None,
    ) -> None:
        self.qrs_events = qrs_events
        self.component_action = component_action
        self.component_selection = component_selection
        self.window = window
        self.window_unit = window_unit
        self.event_unit = event_unit
        self.sfreq = sfreq
        self.event_origin = event_origin
        self.first_samp = first_samp
        self.min_valid_events = min_valid_events
        self.n_components = n_components
        self.whitening_rank = whitening_rank
        self.reg = reg
        self.normalize_input = normalize_input
        self.whiten = whiten
        self.noise_cov = noise_cov
        self.verbose = verbose

        self.diagnostics_: CardiacDSSDiagnostics | None = None
        self.estimator_: DSS | None = None
        self.bias_: CycleAverageBias | None = None
        self.filters_: np.ndarray | None = None
        self.patterns_: np.ndarray | None = None
        self.mixing_: np.ndarray | None = None
        self.eigenvalues_: np.ndarray | None = None
        self.n_selected_: int | None = None
        self.valid_event_samples_: tuple[int, ...] = ()

    def _validate_parameters(self) -> None:
        """Validate the explicit recipe operating point."""
        if self.component_action not in {"extract", "retain", "subtract"}:
            raise ValueError(
                "component_action must be 'extract', 'retain', or 'subtract'."
            )
        if (
            not isinstance(self.component_selection, int | np.integer)
            or isinstance(self.component_selection, bool)
            or int(self.component_selection) < 0
        ):
            raise ValueError("component_selection must be a non-negative integer.")
        if (
            not isinstance(self.min_valid_events, int | np.integer)
            or isinstance(self.min_valid_events, bool)
            or int(self.min_valid_events) < 1
        ):
            raise ValueError("min_valid_events must be a positive integer.")
        if self.event_unit not in {"samples", "seconds"}:
            raise ValueError("event_unit must be 'samples' or 'seconds'.")
        if self.window_unit not in {"samples", "seconds"}:
            raise ValueError("window_unit must be 'samples' or 'seconds'.")
        if self.event_origin not in {"data", "raw"}:
            raise ValueError("event_origin must be 'data' or 'raw'.")
        if self.event_origin == "raw" and self.event_unit != "samples":
            raise ValueError(
                "event_origin='raw' requires event_unit='samples' because Raw "
                "acquisition coordinates are sample numbers."
            )
        if self.n_components is not None and (
            not isinstance(self.n_components, int | np.integer)
            or isinstance(self.n_components, bool)
            or int(self.n_components) < 1
        ):
            raise ValueError("n_components must be a positive integer or None.")

    def _inspect_input(self, X) -> _FitInput:
        """Extract the fit layout and resolve its sampling frequency."""
        auto_pick: bool | str = "data" if self.whiten else True
        data, inferred_sfreq, mne_type, orig_inst, _, _ = extract_data_from_mne(
            X,
            auto_pick=auto_pick,
            channel_first_epochs=True,
        )
        data = np.asarray(data)
        if not np.issubdtype(data.dtype, np.number):
            raise TypeError("CardiacDSS input data must be numeric.")
        if not np.all(np.isfinite(data)):
            raise ValueError("CardiacDSS input data must contain only finite values.")

        supplied_sfreq = self.sfreq
        if supplied_sfreq is not None and (
            isinstance(supplied_sfreq, bool)
            or not np.isfinite(supplied_sfreq)
            or float(supplied_sfreq) <= 0
        ):
            raise ValueError("sfreq must be a finite positive number.")
        if inferred_sfreq is not None:
            inferred_sfreq = float(inferred_sfreq)
            if supplied_sfreq is not None and not np.isclose(
                float(supplied_sfreq), inferred_sfreq, rtol=1e-12, atol=0.0
            ):
                raise ValueError(
                    "sfreq does not match the sampling frequency in the MNE input."
                )
            sfreq = inferred_sfreq
        else:
            sfreq = None if supplied_sfreq is None else float(supplied_sfreq)
        if (self.event_unit == "seconds" or self.window_unit == "seconds") and (
            sfreq is None
        ):
            raise ValueError(
                "sfreq is required for second-valued event or window coordinates."
            )

        if data.ndim == 2:
            n_channels, n_times = data.shape
            n_epochs = None
        else:
            n_channels, n_times, n_epochs = data.shape
        if n_channels < 1 or n_times < 2 or (n_epochs is not None and n_epochs < 1):
            raise ValueError("CardiacDSS requires channels and at least two samples.")

        if mne_type == "array":
            layout = "array-3d" if data.ndim == 3 else "array-2d"
        else:
            layout = mne_type

        if self.event_origin == "raw":
            if self.first_samp is None:
                raise ValueError("first_samp is required when event_origin='raw'.")
            if not isinstance(self.first_samp, int | np.integer) or isinstance(
                self.first_samp, bool
            ):
                raise TypeError("first_samp must be an integer.")
            if data.ndim == 3 or mne_type in {"epochs", "evoked"}:
                raise ValueError(
                    "event_origin='raw' is supported only for continuous arrays "
                    "and MNE Raw input."
                )
            if mne_type == "raw" and int(self.first_samp) != int(orig_inst.first_samp):
                raise ValueError("first_samp does not match the MNE Raw input.")
        elif self.first_samp is not None:
            raise ValueError("first_samp must be omitted when event_origin='data'.")

        return _FitInput(
            data=data,
            layout=layout,
            mne_type=mne_type,
            sfreq=sfreq,
            n_channels=int(n_channels),
            n_times=int(n_times),
            n_epochs=None if n_epochs is None else int(n_epochs),
        )

    def _coordinates_to_samples(
        self,
        values: np.ndarray,
        *,
        sfreq: float | None,
        name: str,
    ) -> np.ndarray:
        """Validate coordinates and convert them to samples exactly once."""
        if not np.issubdtype(values.dtype, np.number) or not np.all(
            np.isfinite(values)
        ):
            raise ValueError(f"{name} must contain only finite numeric coordinates.")
        if self.event_unit == "samples":
            rounded = np.rint(values)
            if not np.all(values == rounded):
                raise ValueError(
                    f"{name} must contain integers when event_unit='samples'."
                )
        else:
            assert sfreq is not None
            rounded = np.rint(values * sfreq)
        limit = np.iinfo(np.int64).max
        if np.any(np.abs(rounded) > limit):
            raise ValueError(f"{name} cannot be represented as 64-bit sample indices.")
        return rounded.astype(np.int64)

    @staticmethod
    def _reject_duplicate_events(events: np.ndarray) -> None:
        """Reject duplicate coordinates rather than silently reweighting cycles."""
        if events.shape[0] != np.unique(events, axis=0).shape[0]:
            raise ValueError("qrs_events contains duplicate event coordinates.")

    def _prepare_events(self, fit_input: _FitInput) -> _PreparedEvents:
        """Resolve units/origin and exclude incomplete event windows."""
        # A zero-event bias is used only to apply CycleAverageBias's public
        # window conversion contract before event-boundary checks.
        window_contract = CycleAverageBias(
            event_samples=[],
            window=self.window,
            window_unit=self.window_unit,
            sfreq=fit_input.sfreq,
            event_origin="data",
        )
        pre, post = window_contract.window
        events = np.asarray(self.qrs_events)

        if fit_input.n_epochs is None:
            if events.size == 0:
                events = np.empty(0, dtype=float)
            elif events.ndim != 1:
                raise ValueError(
                    "Continuous qrs_events must be a one-dimensional sequence."
                )
            event_samples = self._coordinates_to_samples(
                events,
                sfreq=fit_input.sfreq,
                name="qrs_events",
            )
            self._reject_duplicate_events(event_samples.reshape(-1, 1))
            offset = 0 if self.first_samp is None else int(self.first_samp)
            event_samples_data = event_samples - offset
            valid = (event_samples_data + pre >= 0) & (
                event_samples_data + post <= fit_input.n_times
            )
            valid_input_samples = event_samples[valid]
            valid_data_samples = event_samples_data[valid]
            bias = CycleAverageBias(
                event_samples=valid_input_samples,
                window=self.window,
                window_unit=self.window_unit,
                sfreq=fit_input.sfreq,
                event_origin=self.event_origin,
                first_samp=self.first_samp,
            )
            input_count = int(event_samples.size)
        else:
            if events.size == 0:
                events = np.empty((0, 2), dtype=float)
            if events.ndim != 2 or events.shape[1] != 2:
                raise ValueError(
                    "Epoched qrs_events must have shape (n_events, 2) with "
                    "(epoch_index, within_epoch_coordinate) rows."
                )
            epoch_values = events[:, 0]
            if not np.issubdtype(epoch_values.dtype, np.number) or not np.all(
                np.isfinite(epoch_values)
            ):
                raise ValueError("qrs_events epoch indices must be finite integers.")
            epoch_indices = np.rint(epoch_values)
            if not np.all(epoch_values == epoch_indices):
                raise ValueError("qrs_events epoch indices must be integers.")
            epoch_indices = epoch_indices.astype(np.int64)
            local_samples = self._coordinates_to_samples(
                events[:, 1],
                sfreq=fit_input.sfreq,
                name="qrs_events within-epoch coordinates",
            )
            pairs = np.column_stack([epoch_indices, local_samples])
            self._reject_duplicate_events(pairs)
            valid = (
                (epoch_indices >= 0)
                & (epoch_indices < fit_input.n_epochs)
                & (local_samples + pre >= 0)
                & (local_samples + post <= fit_input.n_times)
            )
            valid_data_samples = (
                epoch_indices[valid] * fit_input.n_times + local_samples[valid]
            )
            bias = CycleAverageBias(
                event_samples=valid_data_samples,
                window=self.window,
                window_unit=self.window_unit,
                sfreq=fit_input.sfreq,
                event_origin="data",
            )
            input_count = int(events.shape[0])

        return _PreparedEvents(
            bias=bias,
            valid_data_samples=tuple(int(value) for value in valid_data_samples),
            input_count=input_count,
            valid_count=int(valid_data_samples.size),
            window_samples=(int(pre), int(post)),
        )

    def _make_diagnostics(
        self,
        fit_input: _FitInput,
        prepared: _PreparedEvents,
        *,
        status: CardiacDSSStatus,
        reason: str | None,
        n_selected: int | None = None,
        eigenvalues: tuple[float, ...] = (),
    ) -> CardiacDSSDiagnostics:
        """Build one immutable terminal diagnostics record."""
        return CardiacDSSDiagnostics(
            status=status,
            reason=reason,
            input_layout=fit_input.layout,
            component_action=str(self.component_action),
            component_selection=int(self.component_selection),
            input_event_count=prepared.input_count,
            valid_event_count=prepared.valid_count,
            excluded_event_count=prepared.input_count - prepared.valid_count,
            n_channels=fit_input.n_channels,
            n_times=fit_input.n_times,
            n_epochs=fit_input.n_epochs,
            sfreq=fit_input.sfreq,
            window_samples=prepared.window_samples,
            event_origin=str(self.event_origin),
            first_samp=None if self.first_samp is None else int(self.first_samp),
            n_selected=n_selected,
            eigenvalues=eigenvalues,
        )

    def _finish_without_fit(
        self,
        fit_input: _FitInput,
        prepared: _PreparedEvents,
        *,
        reason: str,
        no_op: bool = False,
    ) -> CardiacDSS:
        """Record a safe abstention or an output-shape inadmissibility."""
        if no_op:
            status = CardiacDSSStatus.NO_OP
        elif self.component_action == "subtract":
            status = CardiacDSSStatus.ABSTAINED
        else:
            status = CardiacDSSStatus.INADMISSIBLE
        self.diagnostics_ = self._make_diagnostics(
            fit_input,
            prepared,
            status=status,
            reason=reason,
            n_selected=0 if no_op else None,
        )
        return self

    def fit(self, X, y=None) -> CardiacDSS:
        """Fit the experimental cycle-locked spatial decomposition."""
        del y
        self._validate_parameters()
        self.diagnostics_ = None
        self.estimator_ = None
        self.filters_ = None
        self.patterns_ = None
        self.mixing_ = None
        self.eigenvalues_ = None
        self.n_selected_ = None

        fit_input = self._inspect_input(X)
        prepared = self._prepare_events(fit_input)
        self.bias_ = prepared.bias
        self.valid_event_samples_ = prepared.valid_data_samples

        if self.component_action == "subtract" and int(self.component_selection) == 0:
            return self._finish_without_fit(
                fit_input,
                prepared,
                reason="component_selection=0 requests exact subtraction passthrough",
                no_op=True,
            )
        if prepared.valid_count < int(self.min_valid_events):
            return self._finish_without_fit(
                fit_input,
                prepared,
                reason=(
                    f"{prepared.valid_count} complete QRS windows remain, fewer than "
                    f"min_valid_events={int(self.min_valid_events)}"
                ),
            )

        flat = fit_input.data.reshape(fit_input.n_channels, -1)
        centered = flat - flat.mean(axis=1, keepdims=True)
        if not np.any(centered != 0):
            return self._finish_without_fit(
                fit_input,
                prepared,
                reason="fit data have no non-zero centered variance",
            )
        if not np.any(prepared.bias.apply(fit_input.data) != 0):
            return self._finish_without_fit(
                fit_input,
                prepared,
                reason="cycle-averaged bias has exactly zero energy",
            )

        estimator = DSS(
            bias=prepared.bias,
            n_components=self.n_components,
            component_action=self.component_action,
            component_selection=int(self.component_selection),
            whitening_rank=self.whitening_rank,
            reg=self.reg,
            normalize_input=self.normalize_input,
            whiten=self.whiten,
            noise_cov=self.noise_cov,
            verbose=self.verbose,
        ).fit(X)
        self.estimator_ = estimator
        self.filters_ = estimator.filters_
        self.patterns_ = estimator.patterns_
        self.mixing_ = estimator.mixing_
        self.eigenvalues_ = estimator.eigenvalues_
        self.n_selected_ = estimator.n_selected_
        eigenvalues = tuple(float(value) for value in estimator.eigenvalues_)
        self.diagnostics_ = self._make_diagnostics(
            fit_input,
            prepared,
            status=CardiacDSSStatus.APPLIED,
            reason=None,
            n_selected=estimator.n_selected_,
            eigenvalues=eigenvalues,
        )
        return self

    @staticmethod
    def _copy_input(X):
        """Return an exact container-preserving copy for safe passthrough."""
        if hasattr(X, "copy"):
            return X.copy()
        return np.array(X, copy=True)

    def transform(self, X):
        """Apply the fitted spatial transform without consulting QRS events."""
        diagnostics = self.get_diagnostics()
        if diagnostics.status in {
            CardiacDSSStatus.ABSTAINED,
            CardiacDSSStatus.NO_OP,
        }:
            return self._copy_input(X)
        if diagnostics.status is CardiacDSSStatus.INADMISSIBLE:
            raise RuntimeError(
                "CardiacDSS fit is inadmissible for the requested output: "
                f"{diagnostics.reason}"
            )
        assert self.estimator_ is not None
        return self.estimator_.transform(X)

    def fit_transform(self, X, y=None, **fit_params):
        """Fit, then apply the same inductive transform contract."""
        if fit_params:
            names = ", ".join(sorted(fit_params))
            raise TypeError(f"Unexpected fit parameters: {names}")
        return self.fit(X, y=y).transform(X)

    def get_diagnostics(self) -> CardiacDSSDiagnostics:
        """Return immutable execution diagnostics from the latest fit."""
        if self.diagnostics_ is None:
            raise RuntimeError("CardiacDSS is not fitted. Call fit() first.")
        return self.diagnostics_


__all__ = ["CardiacDSS", "CardiacDSSDiagnostics", "CardiacDSSStatus"]
