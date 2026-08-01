"""Experimental time-shift denoising source separation.

This module implements data-side lag augmentation for time-shift DSS (TSDSS).
It is deliberately separate from :class:`~mne_denoise.dss.LagAveragingBias`,
which averages shifted signals on the bias side but still learns an
instantaneous spatial filter.

References
----------
.. [1] de Cheveigne, A. (2010). Time-shift denoising source separation.
       Journal of Neuroscience Methods, 189(1), 113-120.
.. [2] de Cheveigne, A., & Parra, L. C. (2014). Joint decorrelation, a
       versatile tool for multichannel data analysis. NeuroImage, 98, 487-505.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

try:
    import mne
    from mne.epochs import BaseEpochs
    from mne.evoked import Evoked
    from mne.io import BaseRaw
except ImportError:  # pragma: no cover - MNE is a required package dependency
    mne = None

    class BaseRaw:  # noqa: D101
        pass

    class BaseEpochs:  # noqa: D101
        pass

    class Evoked:  # noqa: D101
        pass


from .._logging import set_log_level_from_verbose
from ..utils import extract_data_from_mne
from .denoisers import LinearDenoiser
from .linear import compute_dss
from .utils import compute_covariance
from .utils.selection import auto_select_components_robust

_COMPONENT_ACTIONS = frozenset({"extract", "retain", "subtract"})


@dataclass(frozen=True, slots=True)
class TimeShiftDSSDiagnostics:
    """Immutable descriptive diagnostics for one TSDSS fit.

    These fields describe the fitted geometry and time alignment. They are not
    scientific validation thresholds and do not claim that a fitted component
    is signal, noise, or safe to remove.
    """

    lag_input_unit: str
    lag_samples: tuple[int, ...]
    lag_times_seconds: tuple[float, ...] | None
    sampling_frequency: float | None
    channel_count: int
    epoch_count: int
    lag_count: int
    augmented_feature_count: int
    input_sample_count_per_epoch: int
    valid_sample_count_per_epoch: int
    valid_start: int
    valid_stop: int
    left_edge_samples: int
    right_edge_samples: int
    requested_whitening_rank: int | None
    requested_component_count: int | None
    fitted_component_count: int
    normalization_applied: bool
    edge_policy: str = "preserve_input"
    experimental: bool = True

    def __post_init__(self) -> None:
        if self.lag_input_unit not in {"samples", "seconds"}:
            raise ValueError("lag_input_unit must be 'samples' or 'seconds'.")
        if self.lag_input_unit == "seconds" and self.sampling_frequency is None:
            raise ValueError(
                "Second-based lag diagnostics require a sampling frequency."
            )
        if len(self.lag_samples) < 2 or 0 not in self.lag_samples:
            raise ValueError("lag_samples must include zero and a non-zero lag.")
        if tuple(sorted(set(self.lag_samples))) != self.lag_samples:
            raise ValueError("lag_samples must be sorted and unique.")
        if self.lag_count != len(self.lag_samples):
            raise ValueError("lag_count must equal the number of lag samples.")
        if self.lag_times_seconds is not None and len(self.lag_times_seconds) != len(
            self.lag_samples
        ):
            raise ValueError(
                "lag_times_seconds must align one-to-one with lag_samples."
            )
        if self.lag_times_seconds is not None and not all(
            np.isfinite(value) for value in self.lag_times_seconds
        ):
            raise ValueError("lag_times_seconds must contain only finite values.")
        if self.lag_times_seconds is not None and self.sampling_frequency is not None:
            expected_times = tuple(
                value / self.sampling_frequency for value in self.lag_samples
            )
            if not np.allclose(
                self.lag_times_seconds, expected_times, rtol=0.0, atol=1e-12
            ):
                raise ValueError("lag_times_seconds must agree with resolved samples.")
        for name in (
            "channel_count",
            "epoch_count",
            "lag_count",
            "augmented_feature_count",
            "input_sample_count_per_epoch",
            "valid_sample_count_per_epoch",
            "valid_stop",
            "fitted_component_count",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.valid_start < 0 or self.left_edge_samples < 0:
            raise ValueError("left-edge alignment values must be non-negative.")
        if self.right_edge_samples < 0:
            raise ValueError("right_edge_samples must be non-negative.")
        if self.left_edge_samples != max(self.lag_samples):
            raise ValueError("left_edge_samples must equal the largest lag.")
        if self.right_edge_samples != -min(self.lag_samples):
            raise ValueError("right_edge_samples must equal the negated smallest lag.")
        if self.valid_stop - self.valid_start != self.valid_sample_count_per_epoch:
            raise ValueError("The valid interval and valid sample count disagree.")
        if self.valid_start != self.left_edge_samples:
            raise ValueError("valid_start must equal left_edge_samples.")
        if (
            self.valid_stop + self.right_edge_samples
            != self.input_sample_count_per_epoch
        ):
            raise ValueError("The valid interval and right edge must span the input.")
        if self.augmented_feature_count != self.channel_count * self.lag_count:
            raise ValueError("augmented_feature_count must equal channels times lags.")
        for name in ("requested_whitening_rank", "requested_component_count"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive or None.")
        if self.edge_policy != "preserve_input":
            raise ValueError("The only supported edge policy is 'preserve_input'.")
        if not isinstance(self.normalization_applied, bool) or not isinstance(
            self.experimental, bool
        ):
            raise TypeError("Diagnostic flags must be booleans.")
        if self.sampling_frequency is not None and (
            not np.isfinite(self.sampling_frequency) or self.sampling_frequency <= 0
        ):
            raise ValueError("sampling_frequency must be finite and positive.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe copy of the diagnostics."""
        record = asdict(self)
        record["lag_samples"] = list(self.lag_samples)
        if self.lag_times_seconds is not None:
            record["lag_times_seconds"] = list(self.lag_times_seconds)
        return record


def _validate_optional_positive_int(value: int | None, name: str) -> int | None:
    """Validate a positive integer parameter while rejecting booleans."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise TypeError(f"{name} must be a positive integer or None.")
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive.")
    return int(value)


def _validate_sfreq(sfreq: float | None) -> float | None:
    """Validate an optional sampling frequency."""
    if sfreq is None:
        return None
    if isinstance(sfreq, bool) or not isinstance(sfreq, int | float | np.number):
        raise TypeError("sfreq must be a finite positive number or None.")
    sfreq = float(sfreq)
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("sfreq must be finite and positive.")
    return sfreq


def _one_dimensional_numeric(values: Any, name: str) -> np.ndarray:
    """Convert a lag declaration to a finite one-dimensional numeric array."""
    if isinstance(values, str | bytes) or np.isscalar(values):
        raise TypeError(f"{name} must be a one-dimensional sequence, not a scalar.")
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence.")
    if np.issubdtype(array.dtype, np.bool_) or not np.issubdtype(
        array.dtype, np.number
    ):
        raise TypeError(f"{name} must contain numeric values, not booleans.")
    array = np.asarray(array, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _resolve_lags(
    *,
    lag_samples: Sequence[int] | np.ndarray | None,
    lag_times: Sequence[float] | np.ndarray | None,
    sfreq: float | None,
) -> tuple[tuple[int, ...], tuple[float, ...] | None, float | None, str]:
    """Resolve exactly one explicit lag declaration to sample offsets."""
    if (lag_samples is None) == (lag_times is None):
        raise ValueError("Pass exactly one of lag_samples or lag_times.")
    sfreq = _validate_sfreq(sfreq)

    if lag_samples is not None:
        values = _one_dimensional_numeric(lag_samples, "lag_samples")
        rounded = np.rint(values)
        if not np.array_equal(values, rounded):
            raise ValueError("lag_samples must contain whole sample offsets.")
        resolved = rounded.astype(int)
        unit = "samples"
    else:
        if sfreq is None:
            raise ValueError("sfreq is required when lag_times are specified.")
        values = _one_dimensional_numeric(lag_times, "lag_times")
        sample_positions = values * sfreq
        rounded = np.rint(sample_positions)
        tolerance = 1e-9 * np.maximum(1.0, np.abs(sample_positions))
        if np.any(np.abs(sample_positions - rounded) > tolerance):
            bad = int(np.flatnonzero(np.abs(sample_positions - rounded) > tolerance)[0])
            nearest = rounded[bad] / sfreq
            raise ValueError(
                "Every lag_time must fall exactly on the sampling grid; "
                f"lag_times[{bad}]={values[bad]!r} s is nearest to "
                f"{nearest!r} s at sfreq={sfreq!r} Hz."
            )
        resolved = rounded.astype(int)
        unit = "seconds"

    if len(set(resolved.tolist())) != resolved.size:
        raise ValueError("Lag offsets must remain unique after conversion to samples.")
    if 0 not in resolved:
        raise ValueError(
            "Lag offsets must explicitly include zero so sensor-space output "
            "has an unambiguous reference time."
        )
    if resolved.size < 2 or np.all(resolved == 0):
        raise ValueError(
            "Time-shift DSS requires zero and at least one non-zero lag; "
            "use DSS for an instantaneous filter."
        )

    samples = tuple(sorted(int(value) for value in resolved))
    seconds = (
        tuple(float(value) / sfreq for value in samples) if sfreq is not None else None
    )
    return samples, seconds, sfreq, unit


def _validate_array(data: np.ndarray) -> np.ndarray:
    """Validate channel-first continuous or epoched input data."""
    data = np.asarray(data)
    if data.ndim not in (2, 3):
        raise ValueError(
            "Time-shift DSS data must have shape (channels, times) or "
            f"(channels, times, epochs), got {data.shape}."
        )
    if not np.issubdtype(data.dtype, np.number) or np.iscomplexobj(data):
        raise TypeError("Time-shift DSS data must be real-valued numeric data.")
    if (
        data.shape[0] == 0
        or data.shape[1] == 0
        or (data.ndim == 3 and data.shape[2] == 0)
    ):
        raise ValueError("Time-shift DSS dimensions must all be non-empty.")
    data = np.asarray(data, dtype=float)
    if not np.all(np.isfinite(data)):
        raise ValueError("Time-shift DSS data must contain only finite values.")
    return data


def _valid_interval(n_times: int, lag_samples: tuple[int, ...]) -> tuple[int, int]:
    """Return the reference-time interval shared by every lag block."""
    start = max(lag_samples)
    stop = n_times + min(lag_samples)
    if stop - start < 2:
        raise ValueError(
            "Lag span leaves fewer than two valid samples per epoch: "
            f"n_times={n_times}, lags={lag_samples}."
        )
    return start, stop


def _lag_augment(
    data: np.ndarray, lag_samples: tuple[int, ...]
) -> tuple[np.ndarray, int, int]:
    """Stack lagged channel blocks without wrapping or joining epochs."""
    start, stop = _valid_interval(data.shape[1], lag_samples)
    blocks = [data[:, start - lag : stop - lag, ...] for lag in lag_samples]
    return np.concatenate(blocks, axis=0), start, stop


def _apply_bias(
    bias: LinearDenoiser | Callable[[np.ndarray], np.ndarray], data: np.ndarray
) -> np.ndarray:
    """Apply and validate the target criterion on augmented data."""
    if hasattr(bias, "apply"):
        biased = bias.apply(data)
    elif callable(bias):
        biased = bias(data)
    else:
        raise TypeError("bias must be callable or expose an apply(data) method.")
    biased = np.asarray(biased)
    if biased.shape != data.shape:
        raise ValueError(
            "The bias must preserve augmented data shape; "
            f"expected {data.shape}, got {biased.shape}."
        )
    if not np.issubdtype(biased.dtype, np.number) or np.iscomplexobj(biased):
        raise TypeError("The bias output must be real-valued numeric data.")
    biased = np.asarray(biased, dtype=float)
    if not np.all(np.isfinite(biased)):
        raise ValueError("The bias output must contain only finite values.")
    return biased


def compute_time_shift_dss(
    data: np.ndarray,
    *,
    bias: LinearDenoiser | Callable[[np.ndarray], np.ndarray],
    lag_samples: Sequence[int] | np.ndarray | None = None,
    lag_times: Sequence[float] | np.ndarray | None = None,
    sfreq: float | None = None,
    n_components: int | None = None,
    whitening_rank: int | None = None,
    reg: float = 1e-9,
    normalize_input: bool = True,
    cov_method: str = "empirical",
    cov_kws: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, TimeShiftDSSDiagnostics]:
    r"""Compute a time-shift DSS solution on channel-first NumPy data.

    For a lag :math:`\ell`, the corresponding augmented block is
    :math:`X(t-\ell)`: positive lags use past samples and negative lags use
    future samples. Only reference times available for every requested lag are
    used. Three-dimensional inputs are augmented within each epoch, never
    across epoch boundaries.

    Parameters
    ----------
    data : ndarray
        Continuous ``(n_channels, n_times)`` or epoched
        ``(n_channels, n_times, n_epochs)`` data.
    bias : LinearDenoiser or callable
        Target criterion applied to the lag-augmented data. For classic
        evoked TSDSS, use ``AverageBias(axis='epochs')``.
    lag_samples : sequence of int | None
        Complete explicit lag set in samples. It must contain zero and at
        least one non-zero lag. Exactly one of ``lag_samples`` and
        ``lag_times`` must be supplied.
    lag_times : sequence of float | None
        Complete explicit lag set in seconds. Values must fall on the sampling
        grid, and ``sfreq`` is required.
    sfreq : float | None
        Sampling frequency in Hz. Required for ``lag_times`` and otherwise
        retained only for physical-time diagnostics.
    n_components : int | None
        Number of spatiotemporal DSS components. ``None`` keeps every
        component available at the fitted whitening rank.
    whitening_rank : int | None
        Rank of the lag-augmented covariance used for whitening.
    reg : float
        Relative regularization threshold for DSS whitening.
    normalize_input : bool
        If true, scale each original sensor once before copying it into lag
        blocks. Returned filters and patterns are mapped back to input units.
    cov_method : str
        Covariance method passed to :func:`compute_covariance`.
    cov_kws : dict | None
        Additional covariance keyword arguments.

    Returns
    -------
    fir_filters : ndarray, shape (n_components, n_lags, n_channels)
        Spatiotemporal filters. ``fir_filters[:, i]`` multiplies the block for
        ``diagnostics.lag_samples[i]``.
    fir_patterns : ndarray, shape (n_lags, n_channels, n_components)
        Patterns for every lag block. The zero-lag block maps components back
        to sensor space at the reference time.
    eigenvalues : ndarray, shape (n_components,)
        DSS bias-to-baseline variance ratios.
    diagnostics : TimeShiftDSSDiagnostics
        Immutable lag geometry and valid-region description.
    """
    data = _validate_array(data)
    samples, seconds, sfreq, unit = _resolve_lags(
        lag_samples=lag_samples, lag_times=lag_times, sfreq=sfreq
    )
    n_components = _validate_optional_positive_int(n_components, "n_components")
    whitening_rank = _validate_optional_positive_int(whitening_rank, "whitening_rank")
    if isinstance(reg, bool) or not isinstance(reg, int | float | np.number):
        raise TypeError("reg must be a finite positive number.")
    reg = float(reg)
    if not np.isfinite(reg) or reg <= 0:
        raise ValueError("reg must be finite and positive.")
    if not isinstance(normalize_input, bool):
        raise TypeError("normalize_input must be a bool.")
    if not isinstance(cov_method, str):
        raise TypeError("cov_method must be a string.")
    if cov_kws is not None and not isinstance(cov_kws, dict):
        raise TypeError("cov_kws must be a dict or None.")

    augmented, start, stop = _lag_augment(data, samples)
    n_channels = data.shape[0]
    n_lags = len(samples)
    n_features = augmented.shape[0]
    if whitening_rank is not None and whitening_rank > n_features:
        raise ValueError(
            "whitening_rank cannot exceed the lag-augmented feature count "
            f"({n_features})."
        )

    if normalize_input:
        channel_norms = np.linalg.norm(data.reshape(n_channels, -1), axis=1)
        channel_norms = np.where(channel_norms > 0, channel_norms, 1.0)
    else:
        channel_norms = np.ones(n_channels)
    feature_norms = np.tile(channel_norms, n_lags)
    scale_shape = (n_features,) + (1,) * (augmented.ndim - 1)
    augmented_scaled = augmented / feature_norms.reshape(scale_shape)
    biased_scaled = _apply_bias(bias, augmented_scaled)

    kws = dict(cov_kws or {})
    baseline_cov = compute_covariance(augmented_scaled, method=cov_method, **kws)
    biased_cov = compute_covariance(biased_scaled, method=cov_method, **kws)
    filters_scaled, patterns_scaled, eigenvalues = compute_dss(
        baseline_cov,
        biased_cov,
        n_components=n_components,
        rank=whitening_rank,
        reg=reg,
    )

    # Map the decomposition out of normalized feature coordinates. Every lag
    # copy of a sensor uses the same fitted sensor scale.
    filters = filters_scaled / feature_norms[np.newaxis, :]
    patterns = patterns_scaled * feature_norms[:, np.newaxis]
    fir_filters = filters.reshape(filters.shape[0], n_lags, n_channels)
    fir_patterns = patterns.reshape(n_lags, n_channels, patterns.shape[1])

    n_epochs = data.shape[2] if data.ndim == 3 else 1
    diagnostics = TimeShiftDSSDiagnostics(
        lag_input_unit=unit,
        lag_samples=samples,
        lag_times_seconds=seconds,
        sampling_frequency=sfreq,
        channel_count=n_channels,
        epoch_count=n_epochs,
        lag_count=n_lags,
        augmented_feature_count=n_features,
        input_sample_count_per_epoch=data.shape[1],
        valid_sample_count_per_epoch=stop - start,
        valid_start=start,
        valid_stop=stop,
        left_edge_samples=start,
        right_edge_samples=data.shape[1] - stop,
        requested_whitening_rank=whitening_rank,
        requested_component_count=n_components,
        fitted_component_count=filters.shape[0],
        normalization_applied=normalize_input,
    )
    return fir_filters, fir_patterns, eigenvalues, diagnostics


def _is_mne_instance(value: Any) -> bool:
    """Return whether value is a supported MNE container."""
    return mne is not None and isinstance(value, BaseRaw | BaseEpochs | Evoked)


class TimeShiftDSS(BaseEstimator, TransformerMixin):
    r"""Experimental time-shift DSS with explicit lag and alignment contracts.

    This estimator augments the *data* with delayed sensor copies and learns a
    spatiotemporal DSS/FIR filter. It is not an alias for
    :class:`LagAveragingBias` or ``lag_averaging_dss``.

    Parameters
    ----------
    bias : LinearDenoiser or callable
        Target criterion applied after lag augmentation.
    lag_samples : sequence of int | None
        Complete lag set in samples. It must explicitly include zero and a
        non-zero lag. Exactly one lag representation is required.
    lag_times : sequence of float | None
        Complete lag set in seconds. ``sfreq`` is taken from MNE input or from
        the constructor, and every value must fall on its sampling grid.
    sfreq : float | None
        Sampling frequency for NumPy data with ``lag_times``. When MNE input
        is used, a supplied value must match ``info['sfreq']``.
    n_components : int | None
        Number of spatiotemporal DSS components to fit.
    component_action : {'extract', 'retain', 'subtract'}
        ``'extract'`` returns source time series over the valid interval.
        Sensor actions return the same input shape and container, replacing
        only the valid interval and preserving edge samples unchanged.
    component_selection : int | 'auto' | None
        Leading component count for sensor actions. ``None`` retains every
        fitted component and subtracts none; ``'auto'`` applies the package's
        robust eigenvalue selector.
    whitening_rank : int | None
        Whitening rank in the lag-augmented feature space.
    reg : float
        Relative DSS whitening threshold.
    normalize_input : bool
        Scale each original sensor once before lag augmentation, then map the
        fitted filters and patterns back to the input units.
    cov_method : str
        NumPy covariance estimator.
    cov_kws : dict | None
        Additional covariance estimator keyword arguments.
    selection_threshold : float
        Sigma threshold used by automatic component selection.
    knee_rel_floor : float
        Relative eigenvalue floor used by automatic component selection.
    knee_min_ratio : float
        Minimum eigenvalue drop used by automatic component selection.
    verbose : bool | str | int | None
        Logging verbosity.

    Attributes
    ----------
    filters_ : ndarray, shape (n_components, n_lags, n_channels)
        Fitted FIR filters in input sensor units.
    fir_patterns_ : ndarray, shape (n_lags, n_channels, n_components)
        Fitted patterns for all lag blocks.
    patterns_ : ndarray, shape (n_channels, n_components)
        Zero-lag sensor patterns used for aligned sensor reconstruction.
    eigenvalues_ : ndarray
        Ordered DSS bias scores.
    diagnostics_ : TimeShiftDSSDiagnostics
        Immutable descriptive fit geometry.
    valid_slice_ : slice
        Valid reference-time slice for the fitted input length.
    valid_times_ : ndarray | None
        MNE time values corresponding to ``valid_slice_`` at fit time.

    Notes
    -----
    This is an experimental implementation. Unit tests establish numerical and
    alignment invariants, not neural-signal preservation or artifact-removal
    efficacy. Independently validate it for each scientific regime.

    A positive lag :math:`\ell` contributes :math:`X(t-\ell)`. No circular
    wrapping is used. Epochs are shifted independently, so the end of one
    epoch can never enter the beginning of another.
    """

    def __init__(
        self,
        bias: LinearDenoiser | Callable[[np.ndarray], np.ndarray],
        *,
        lag_samples: Sequence[int] | np.ndarray | None = None,
        lag_times: Sequence[float] | np.ndarray | None = None,
        sfreq: float | None = None,
        n_components: int | None = None,
        component_action: str = "extract",
        component_selection: int | str | None = None,
        whitening_rank: int | None = None,
        reg: float = 1e-9,
        normalize_input: bool = True,
        cov_method: str = "empirical",
        cov_kws: dict[str, Any] | None = None,
        selection_threshold: float = 3.0,
        knee_rel_floor: float = 0.01,
        knee_min_ratio: float = 3.0,
        verbose: bool | str | int | None = None,
    ) -> None:
        self.bias = bias
        self.lag_samples = lag_samples
        self.lag_times = lag_times
        self.sfreq = sfreq
        self.n_components = n_components
        self.component_action = component_action
        self.component_selection = component_selection
        self.whitening_rank = whitening_rank
        self.reg = reg
        self.normalize_input = normalize_input
        self.cov_method = cov_method
        self.cov_kws = cov_kws
        self.selection_threshold = selection_threshold
        self.knee_rel_floor = knee_rel_floor
        self.knee_min_ratio = knee_min_ratio
        self.verbose = verbose

        self.filters_: np.ndarray | None = None
        self.fir_patterns_: np.ndarray | None = None
        self.patterns_: np.ndarray | None = None
        self.mixing_: np.ndarray | None = None
        self.eigenvalues_: np.ndarray | None = None
        self.n_selected_: int | None = None
        self.diagnostics_: TimeShiftDSSDiagnostics | None = None
        self.valid_slice_: slice | None = None
        self.valid_times_: np.ndarray | None = None
        self.channel_norms_: np.ndarray | None = None

    def _validate_operation(self) -> tuple[str, int | str | None]:
        """Validate component operation and selection parameters."""
        if self.component_action not in _COMPONENT_ACTIONS:
            choices = ", ".join(sorted(_COMPONENT_ACTIONS))
            raise ValueError(
                f"component_action must be one of {choices}, "
                f"got {self.component_action!r}."
            )
        selection = self.component_selection
        if selection == "auto" or selection is None:
            return self.component_action, selection
        if isinstance(selection, bool) or not isinstance(selection, int | np.integer):
            raise ValueError("component_selection must be an int, 'auto', or None.")
        if int(selection) < 0:
            raise ValueError("component_selection must be non-negative.")
        return self.component_action, int(selection)

    @staticmethod
    def _channel_signature(
        inst: Any, ch_names: list[str]
    ) -> tuple[tuple[Any, ...], ...]:
        """Build a name/type/unit signature in fitted channel order."""
        signatures = []
        for name in ch_names:
            idx = inst.ch_names.index(name)
            channel = inst.info["chs"][idx]
            signatures.append(
                (
                    name,
                    inst.get_channel_types(picks=[idx])[0],
                    int(channel["unit"]),
                    int(channel["unit_mul"]),
                )
            )
        return tuple(signatures)

    def _fit_input(
        self, X: BaseRaw | BaseEpochs | Evoked | np.ndarray
    ) -> tuple[np.ndarray, float | None, str, np.ndarray | None]:
        """Extract fit data and freeze its feature identity."""
        data, extracted_sfreq, mne_type, orig_inst, picks, ch_names = (
            extract_data_from_mne(X, channel_first_epochs=True)
        )
        self._fit_was_mne_ = orig_inst is not None
        self._fit_mne_type_ = mne_type

        if orig_inst is not None:
            bads = set(orig_inst.info["bads"])
            good_positions = [
                idx for idx, name in enumerate(ch_names) if name not in bads
            ]
            if not good_positions:
                raise ValueError("No good channels remain after excluding MNE bads.")
            if len(good_positions) != len(ch_names):
                data = data[np.asarray(good_positions)]
                ch_names = [ch_names[idx] for idx in good_positions]
            picks = np.asarray([orig_inst.ch_names.index(name) for name in ch_names])
            self._mne_ch_names_ = list(ch_names)
            self._mne_channel_signature_ = self._channel_signature(orig_inst, ch_names)
            self.info_ = orig_inst.info.copy()
        else:
            self._mne_ch_names_ = None
            self._mne_channel_signature_ = None
            self.info_ = None

        declared_sfreq = _validate_sfreq(self.sfreq)
        if extracted_sfreq is not None:
            extracted_sfreq = float(extracted_sfreq)
            if declared_sfreq is not None and not np.isclose(
                declared_sfreq, extracted_sfreq, rtol=0.0, atol=1e-12
            ):
                raise ValueError(
                    f"sfreq={declared_sfreq!r} does not match MNE input "
                    f"sfreq={extracted_sfreq!r}."
                )
            effective_sfreq = extracted_sfreq
        else:
            effective_sfreq = declared_sfreq
        return _validate_array(data), effective_sfreq, mne_type, picks

    def _transform_input(
        self, X: BaseRaw | BaseEpochs | Evoked | np.ndarray
    ) -> tuple[np.ndarray, str, np.ndarray | None]:
        """Extract transform data under the frozen fit feature contract."""
        is_mne = _is_mne_instance(X)
        if is_mne != self._fit_was_mne_:
            expected = "an MNE object" if self._fit_was_mne_ else "a NumPy array"
            raise TypeError(
                f"TimeShiftDSS was fitted on {expected}; container families cannot be mixed."
            )

        data, extracted_sfreq, mne_type, orig_inst, picks, ch_names = (
            extract_data_from_mne(
                X,
                ch_names=self._mne_ch_names_,
                channel_first_epochs=True,
            )
        )
        if orig_inst is not None:
            signature = self._channel_signature(orig_inst, ch_names)
            if signature != self._mne_channel_signature_:
                raise ValueError(
                    "MNE channel names, types, or physical units do not match the fit input."
                )
            if not np.isclose(
                float(extracted_sfreq), self.sfreq_, rtol=0.0, atol=1e-12
            ):
                raise ValueError(
                    f"Transform sfreq={float(extracted_sfreq)!r} does not match "
                    f"fit sfreq={self.sfreq_!r}."
                )
        data = _validate_array(data)
        if data.shape[0] != self.n_channels_in_:
            raise ValueError(
                f"Expected {self.n_channels_in_} fitted channels, got {data.shape[0]}."
            )
        return data, mne_type, picks

    def fit(
        self,
        X: BaseRaw | BaseEpochs | Evoked | np.ndarray,
        y: None = None,
    ) -> TimeShiftDSS:
        """Fit lag-augmented DSS filters."""
        del y
        set_log_level_from_verbose(self.verbose)
        _, selection = self._validate_operation()
        data, effective_sfreq, _, _ = self._fit_input(X)

        filters, fir_patterns, eigenvalues, diagnostics = compute_time_shift_dss(
            data,
            bias=self.bias,
            lag_samples=self.lag_samples,
            lag_times=self.lag_times,
            sfreq=effective_sfreq,
            n_components=self.n_components,
            whitening_rank=self.whitening_rank,
            reg=self.reg,
            normalize_input=self.normalize_input,
            cov_method=self.cov_method,
            cov_kws=self.cov_kws,
        )
        self.filters_ = filters
        self.fir_patterns_ = fir_patterns
        self.eigenvalues_ = eigenvalues
        self.diagnostics_ = diagnostics
        self.lag_samples_ = diagnostics.lag_samples
        self.lag_times_ = diagnostics.lag_times_seconds
        self.sfreq_ = diagnostics.sampling_frequency
        self.n_channels_in_ = diagnostics.channel_count
        self.n_features_in_ = diagnostics.channel_count
        self.n_augmented_features_ = diagnostics.augmented_feature_count
        self.valid_slice_ = slice(diagnostics.valid_start, diagnostics.valid_stop)
        self.valid_times_ = (
            np.asarray(X.times[self.valid_slice_], dtype=float).copy()
            if _is_mne_instance(X)
            else None
        )
        self.fit_first_samp_ = int(X.first_samp) if isinstance(X, BaseRaw) else None

        zero_index = self.lag_samples_.index(0)
        self.patterns_ = self.fir_patterns_[zero_index]
        self.mixing_ = self.patterns_

        data_flat = data.reshape(data.shape[0], -1)
        if self.normalize_input:
            norms = np.linalg.norm(data_flat, axis=1)
            self.channel_norms_ = np.where(norms > 0, norms, 1.0)
        else:
            self.channel_norms_ = np.ones(data.shape[0])

        self.n_selected_ = None
        if selection is not None:
            self.n_selected_ = self.auto_select()
        return self

    def _check_fitted(self) -> None:
        """Raise unless all fitted matrices are available."""
        if (
            self.filters_ is None
            or self.fir_patterns_ is None
            or self.patterns_ is None
            or self.eigenvalues_ is None
            or self.diagnostics_ is None
        ):
            raise RuntimeError("TimeShiftDSS not fitted. Call fit() first.")

    def auto_select(self, threshold: float | None = None) -> int:
        """Resolve the configured leading component count."""
        self._check_fitted()
        _, selection = self._validate_operation()
        if isinstance(selection, int):
            return min(selection, len(self.eigenvalues_))
        threshold = self.selection_threshold if threshold is None else threshold
        return int(
            auto_select_components_robust(
                self.eigenvalues_,
                sigma=threshold,
                knee_rel_floor=self.knee_rel_floor,
                knee_min_ratio=self.knee_min_ratio,
            )
        )

    def _operation_component_count(self, action: str) -> int:
        """Return the leading component count used by a sensor action."""
        if self.component_selection is None:
            return self.filters_.shape[0] if action == "retain" else 0
        if self.n_selected_ is None:
            return self.auto_select()
        return min(max(int(self.n_selected_), 0), self.filters_.shape[0])

    def valid_slice(self, n_times: int) -> slice:
        """Return the valid reference-time slice for an input length."""
        self._check_fitted()
        if isinstance(n_times, bool) or not isinstance(n_times, int | np.integer):
            raise TypeError("n_times must be an integer.")
        start, stop = _valid_interval(int(n_times), self.lag_samples_)
        return slice(start, stop)

    def get_valid_times(self, X: BaseRaw | BaseEpochs | Evoked) -> np.ndarray:
        """Return MNE-relative times corresponding to extracted sources."""
        self._check_fitted()
        if not _is_mne_instance(X):
            raise TypeError(
                "get_valid_times requires an MNE Raw, Epochs, or Evoked object."
            )
        if not np.isclose(float(X.info["sfreq"]), self.sfreq_, rtol=0.0, atol=1e-12):
            raise ValueError("Input sampling frequency does not match the fit input.")
        return np.asarray(X.times[self.valid_slice(len(X.times))], dtype=float).copy()

    def get_diagnostics(self) -> TimeShiftDSSDiagnostics:
        """Return the immutable diagnostics object from the fit."""
        self._check_fitted()
        return self.diagnostics_

    @staticmethod
    def _copy_with_valid_sensor_data(
        X: BaseRaw | BaseEpochs | Evoked,
        valid_data: np.ndarray,
        valid: slice,
        mne_type: str,
        picks: np.ndarray,
    ) -> BaseRaw | BaseEpochs | Evoked:
        """Replace only valid selected samples in an MNE copy."""
        out = X.copy()
        if isinstance(out, BaseRaw | BaseEpochs):
            out.load_data()
        if mne_type == "epochs":
            out._data[:, picks, valid] = np.transpose(valid_data, (2, 0, 1))
        else:
            out._data[picks, valid] = valid_data
        return out

    def transform(
        self, X: BaseRaw | BaseEpochs | Evoked | np.ndarray
    ) -> np.ndarray | BaseRaw | BaseEpochs | Evoked:
        """Apply the fitted spatiotemporal component operation.

        Source extraction returns only the valid overlap shared by every lag.
        Sensor-valued operations retain the input shape and timeline: only
        that valid overlap is replaced, while left and right edge samples are
        copied unchanged.
        """
        self._check_fitted()
        action, _ = self._validate_operation()
        data, mne_type, picks = self._transform_input(X)
        augmented, start, stop = _lag_augment(data, self.lag_samples_)
        n_valid = stop - start
        n_epochs = data.shape[2] if data.ndim == 3 else None

        augmented_2d = augmented.reshape(augmented.shape[0], -1)
        augmented_mean = augmented_2d.mean(axis=1, keepdims=True)
        sources_2d = self.filters_.reshape(self.filters_.shape[0], -1) @ (
            augmented_2d - augmented_mean
        )
        if not np.all(np.isfinite(sources_2d)):
            raise RuntimeError("TimeShiftDSS produced non-finite source values.")

        if n_epochs is None:
            sources = sources_2d
        else:
            sources = sources_2d.reshape(self.filters_.shape[0], n_valid, n_epochs)
        if action == "extract":
            if mne_type == "epochs":
                return np.transpose(sources, (2, 0, 1))
            return sources

        n_action = self._operation_component_count(action)
        selected_2d = self.patterns_[:, :n_action] @ sources_2d[:n_action]
        valid_input = data[:, start:stop, ...]
        valid_input_2d = valid_input.reshape(data.shape[0], -1)
        if action == "subtract":
            valid_output_2d = valid_input_2d - selected_2d
        else:
            zero_index = self.lag_samples_.index(0)
            zero_block_start = zero_index * self.n_channels_in_
            zero_mean = augmented_mean[
                zero_block_start : zero_block_start + self.n_channels_in_
            ]
            valid_output_2d = selected_2d + zero_mean
        if not np.all(np.isfinite(valid_output_2d)):
            raise RuntimeError("TimeShiftDSS produced non-finite sensor values.")

        if n_epochs is None:
            valid_output = valid_output_2d
        else:
            valid_output = valid_output_2d.reshape(
                self.n_channels_in_, n_valid, n_epochs
            )

        if mne_type == "array":
            output = np.array(data, copy=True)
            output[:, start:stop, ...] = valid_output
            return output
        return self._copy_with_valid_sensor_data(
            X, valid_output, slice(start, stop), mne_type, picks
        )

    def fit_transform(
        self,
        X: BaseRaw | BaseEpochs | Evoked | np.ndarray,
        y: None = None,
        **fit_params: Any,
    ) -> np.ndarray | BaseRaw | BaseEpochs | Evoked:
        """Fit and transform with exactly ``fit(X).transform(X)`` semantics."""
        return self.fit(X, y=y, **fit_params).transform(X)


__all__ = [
    "compute_time_shift_dss",
    "TimeShiftDSS",
    "TimeShiftDSSDiagnostics",
]
