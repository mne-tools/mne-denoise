"""Experimental recursive iCanClean for causal reference-based denoising.

This module deliberately does not present recursive iCanClean as a published
algorithm. Published iCanClean uses batch CCA in fixed or moving windows; its
authors identify recursive CCA as future work. :class:`RecursiveICanClean`
combines recursively updated joint covariance statistics with the published
iCanClean component-selection and least-squares subtraction rule. It is an
experimental implementation that requires independent validation.

The recursive state is updated one accepted sample at a time. Consequently,
the sufficient statistics and causal outputs are invariant to transport-block
boundaries for a fixed sample order, adaptation mask, and configuration. With
``forgetting_factor=None`` (the default), the moments equal ordinary batch
moments up to floating-point roundoff. A per-sample forgetting factor or a
physical memory duration can be selected explicitly.

Notes
-----
A public U.S. patent application has been filed for the iCanClean method:
US20230363718A1, "Removing latent noise components from data signals"
(Application 18/245,496). Patent applications, and any resulting patents, may
affect commercial use. Consult a lawyer if necessary.

References
----------
Downey, R. J., & Ferris, D. P. (2022). The iCanClean Algorithm: How to Remove
Artifacts using Reference Noise Recordings. arXiv:2201.11798.

Downey, R. J., & Ferris, D. P. (2023). iCanClean Removes Motion, Muscle, Eye,
and Line-Noise Artifacts from Phantom EEG. Sensors, 23(19), 8214.
https://doi.org/10.3390/s23198214

Zhao, H., Sun, D., & Luo, Z. (2020). Incremental Canonical Correlation
Analysis. Applied Sciences, 10(21), 7827.
https://doi.org/10.3390/app10217827
"""

from __future__ import annotations

import base64
import binascii
import copy
import hashlib
import json
import logging
from numbers import Integral, Real
from typing import Any, Literal

import numpy as np
from scipy import linalg as la
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ..utils import extract_data_from_mne

logger = logging.getLogger(__name__)

CleanWith = Literal["X", "Y", "both"]
AdaptationMode = Literal["adaptive", "frozen"]
UpdateOrder = Literal["after", "before"]

_STATE_VERSION = 1
_MODEL_STATE_NAMES = (
    "x_weights_",
    "y_weights_",
    "correlations_",
    "removed_idx_",
    "n_removed_",
    "regression_",
    "artifact_operator_",
    "cleaning_operator_",
    "model_primary_mean_",
    "model_reference_mean_",
    "model_samples_",
    "rank_primary_",
    "rank_reference_",
    "rank_ceiling_",
)


def _finite_real(
    value: float,
    *,
    name: str,
    minimum: float | None = None,
    strict: bool = False,
) -> float:
    """Return a validated finite real scalar."""
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real number.")
    result = float(value)
    if minimum is not None:
        valid = result > minimum if strict else result >= minimum
        if not valid:
            relation = ">" if strict else ">="
            raise ValueError(f"{name} must be {relation} {minimum}.")
    return result


def _positive_int(value: int, *, name: str, minimum: int = 1) -> int:
    """Return a validated positive integer."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return result


def _as_channel_time(data: Any, *, name: str) -> np.ndarray:
    """Validate channel-by-time data without accepting non-finite values."""
    try:
        array = np.asarray(data, dtype=np.float64)
    except (TypeError, ValueError) as err:
        raise TypeError(f"{name} must be numeric channel-by-time data.") from err
    if array.ndim != 2:
        raise ValueError(
            f"{name} must have shape (n_channels, n_samples), got {array.shape}."
        )
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one channel and sample.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _reref_matrix(n_channels: int, reref: bool | str) -> np.ndarray:
    """Return the same CCA-only re-reference matrix as batch iCanClean."""
    identity = np.eye(n_channels)
    if reref is False:
        return identity
    if reref is True or reref == "fullrank":
        return identity - np.ones((n_channels, n_channels)) / (n_channels + 1)
    if reref == "loserank":
        return identity - np.ones((n_channels, n_channels)) / n_channels
    raise ValueError(
        f"reref must be False, True, 'fullrank', or 'loserank', got {reref!r}."
    )


def _inverse_sqrt(
    covariance: np.ndarray,
    *,
    regularization: float,
    rank_tolerance: float,
) -> tuple[np.ndarray, int, float]:
    """Return a symmetric regularized inverse square root and raw rank."""
    covariance = (covariance + covariance.T) / 2.0
    values, vectors = la.eigh(covariance, check_finite=False)
    scale = float(np.max(np.abs(values)))
    if not np.isfinite(scale) or scale <= np.finfo(float).tiny:
        raise ValueError("covariance has zero numerical energy")
    tolerance = rank_tolerance * scale
    rank = int(np.sum(values > tolerance))
    energy_scale = float(np.trace(covariance) / covariance.shape[0])
    if not np.isfinite(energy_scale) or energy_scale <= np.finfo(float).tiny:
        raise ValueError("covariance has zero numerical energy")
    loading = regularization * energy_scale
    retained = values > tolerance
    loaded_values = values[retained] + loading
    if loaded_values.size and np.any(loaded_values <= np.finfo(float).tiny):
        raise ValueError("regularized covariance is not positive definite")
    inverse_values = np.zeros_like(values)
    inverse_values[retained] = 1.0 / np.sqrt(loaded_values)
    inverse_sqrt = (vectors * inverse_values) @ vectors.T
    return inverse_sqrt, rank, loading


def _resolve_selector(
    selector: list[int] | list[str] | tuple[int, ...] | tuple[str, ...],
    *,
    n_channels: int,
    ch_names: list[str] | None,
    name: str,
) -> np.ndarray:
    """Resolve a homogeneous name or integer channel selector."""
    if not selector:
        raise ValueError(f"{name} must contain at least one channel.")
    if all(isinstance(item, str) for item in selector):
        if ch_names is None:
            raise TypeError(
                f"String {name} selectors require an MNE object with channel names."
            )
        missing = [item for item in selector if item not in ch_names]
        if missing:
            raise ValueError(f"Input is missing {name}: {missing[:5]}.")
        indices = np.array([ch_names.index(item) for item in selector], dtype=int)
    elif all(
        isinstance(item, Integral) and not isinstance(item, bool) for item in selector
    ):
        indices = np.asarray(selector, dtype=int)
        if np.any(indices < 0) or np.any(indices >= n_channels):
            raise ValueError(f"{name} contain an index outside [0, {n_channels}).")
    else:
        raise TypeError(f"{name} must contain only channel names or only integers.")
    if np.unique(indices).size != indices.size:
        raise ValueError(f"{name} must not contain duplicates.")
    return indices


def _copy_array_or_none(value: np.ndarray | None) -> np.ndarray | None:
    """Copy an optional array."""
    return None if value is None else np.array(value, copy=True)


def _channel_signature(
    original: Any,
    indices: np.ndarray | None,
) -> tuple[tuple[str, int, int, int, int], ...] | None:
    """Return the MNE channel identity and physical-unit signature."""
    if original is None:
        return None
    selected = np.arange(len(original.ch_names)) if indices is None else indices
    return tuple(
        (
            original.ch_names[int(index)],
            int(original.info["chs"][int(index)]["kind"]),
            int(original.info["chs"][int(index)]["unit"]),
            int(original.info["chs"][int(index)]["unit_mul"]),
            int(original.info["chs"][int(index)]["coil_type"]),
        )
        for index in selected
    )


def _encode_state_value(value: Any) -> Any:
    """Encode checkpoint values into canonical JSON-compatible objects."""
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        if array.dtype.hasobject:
            raise TypeError("Object arrays are not permitted in serialized state.")
        return {
            "__ndarray__": base64.b64encode(array.tobytes()).decode("ascii"),
            "dtype": array.dtype.str,
            "shape": list(array.shape),
        }
    if isinstance(value, np.generic):
        return _encode_state_value(value.item())
    if isinstance(value, tuple):
        return {"__tuple__": [_encode_state_value(item) for item in value]}
    if isinstance(value, list):
        return [_encode_state_value(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("State dictionaries must use string keys.")
        return {key: _encode_state_value(item) for key, item in value.items()}
    if isinstance(value, float) and not np.isfinite(value):
        if np.isnan(value):
            label = "nan"
        elif value > 0:
            label = "+inf"
        else:
            label = "-inf"
        return {"__float__": label}
    if value is None or isinstance(value, bool | int | float | str):
        return value
    raise TypeError(f"Unsupported state value type: {type(value).__name__}.")


def _decode_state_value(value: Any) -> Any:
    """Decode values emitted by :func:`_encode_state_value`."""
    if isinstance(value, list):
        return [_decode_state_value(item) for item in value]
    if not isinstance(value, dict):
        return value
    if set(value) == {"__tuple__"}:
        items = value["__tuple__"]
        if not isinstance(items, list):
            raise ValueError("Malformed tuple in serialized state.")
        return tuple(_decode_state_value(item) for item in items)
    if set(value) == {"__float__"}:
        labels = {"nan": np.nan, "+inf": np.inf, "-inf": -np.inf}
        try:
            return labels[value["__float__"]]
        except (KeyError, TypeError) as err:
            raise ValueError("Malformed non-finite float in serialized state.") from err
    if set(value) == {"__ndarray__", "dtype", "shape"}:
        try:
            dtype = np.dtype(value["dtype"])
            if dtype.hasobject:
                raise ValueError
            shape = tuple(int(size) for size in value["shape"])
            payload = base64.b64decode(value["__ndarray__"], validate=True)
            array = np.frombuffer(payload, dtype=dtype)
            expected_size = int(np.prod(shape, dtype=np.int64))
            if any(size < 0 for size in shape) or array.size != expected_size:
                raise ValueError
            return array.reshape(shape).copy()
        except (binascii.Error, TypeError, ValueError) as err:
            raise ValueError("Malformed array in serialized state.") from err
    if any(str(key).startswith("__") for key in value):
        raise ValueError("Unknown reserved marker in serialized state.")
    return {key: _decode_state_value(item) for key, item in value.items()}


def _canonical_state_json(state: dict[str, Any]) -> str:
    """Return a deterministic lossless JSON representation of a checkpoint."""
    return json.dumps(
        _encode_state_value(state),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _state_payload_hash(state: dict[str, Any]) -> str:
    """Hash every serialized checkpoint field except the checksum itself."""
    payload = {key: value for key, value in state.items() if key != "state_sha256"}
    return hashlib.sha256(_canonical_state_json(payload).encode("utf-8")).hexdigest()


class RecursiveICanClean(BaseEstimator, TransformerMixin):
    """Experimental stateful recursive iCanClean.

    The estimator recursively updates the mean and joint covariance of primary
    and reference channels, recomputes a regularized CCA model at explicit
    sample boundaries, and applies the iCanClean component-rejection rule. It
    supports both combined recordings (selected by ``ref_channels`` and
    ``primary_channels``) and separate primary/reference recordings.

    This class is not a relabeling of ordinary moving-window iCanClean and is
    not an implementation released by the iCanClean authors. Published
    iCanClean identifies recursive CCA as future work. Treat this API as an
    experimental research prototype.

    Parameters
    ----------
    sfreq : float | None
        Sampling frequency in Hz. MNE metadata is used when available and must
        agree with this value. It is required when any duration in seconds is
        selected.
    ref_channels : sequence of int | sequence of str | None
        Reference channels for a combined input. Leave as ``None`` when passing
        a separate ``reference=`` object.
    primary_channels : sequence of int | sequence of str | None
        Primary channels for a combined input. By default, all channels not in
        ``ref_channels`` are primary.
    clean_with : {'X', 'Y', 'both'}
        Canonical basis used to reconstruct the artifact, following iCanClean
        naming. ``X`` uses primary canonical variates, ``Y`` uses reference
        variates, and ``both`` concatenates them.
    threshold : float
        Squared canonical-correlation threshold in [0, 1].
    max_reject_fraction : float
        Maximum fraction of canonical components removed at an update.
    reref_primary, reref_ref : bool | {'fullrank', 'loserank'}
        Optional average re-references used for CCA estimation only.
    regularization : float
        Positive relative covariance and regression loading.
    rank_tolerance : float
        Positive relative tolerance for unregularized covariance rank.
    forgetting_factor : float | None
        Per-accepted-sample forgetting factor in (0, 1]. ``None`` means no
        forgetting. This is a sample-domain quantity, not a per-chunk value.
    memory_duration_s : float | None
        Alternative physical exponential-memory time constant in seconds. It
        is converted to ``exp(-1 / (sfreq * memory_duration_s))`` and cannot be
        combined with ``forgetting_factor``.
    warmup_samples : int | None
        Accepted samples required before the first model. Cannot be combined
        with ``warmup_duration_s``.
    warmup_duration_s : float | None
        Alternative warm-up duration in seconds.
    update_interval_samples : int | None
        Accepted samples between model updates. Cannot be combined with
        ``update_interval_s``.
    update_interval_s : float | None
        Alternative model-update interval in seconds.
    convergence_tol : float
        Relative artifact-operator change considered stable.
    stable_updates : int
        Consecutive stable updates required for ``converged_``.
    update_order : {'after', 'before'}
        In :meth:`process`, update ``'after'`` is causal: a sample updates the
        model only for future samples. ``'before'`` lets the current sample
        influence its own model and is intended only for controlled parity
        experiments.
    adaptation_mode : {'adaptive', 'frozen'}
        Whether :meth:`process` updates state. ``'frozen'`` provides the
        reference-freeze control and requires prior calibration with
        :meth:`fit`, :meth:`partial_fit`, or :meth:`load_state_dict`.
    verbose : bool | str | int | None
        Log concise model-update summaries when truthy.

    Attributes
    ----------
    correlations_ : ndarray, shape (n_components,)
        Squared canonical correlations for the current model, matching the
        convention of :class:`ICanClean`.
    removed_idx_ : ndarray of int
        Canonical-pair indices selected by the current operating point.
    artifact_operator_ : ndarray, shape (n_primary + n_reference, n_primary)
        Frozen linear map from centered joint data to estimated artifact.
    rank_primary_, rank_reference_, rank_ceiling_ : int
        Unregularized covariance ranks and the resulting CCA dimension.
    update_history_ : list of dict
        Terminal diagnostic record for every attempted model update.
    process_history_ : list of dict
        Transport, latency, gating, and model-version record for each processed
        block.
    converged_ : bool
        Whether the requested number of consecutive operator-change checks has
        passed. This is a numerical stability diagnostic, not evidence of
        denoising effectiveness.

    Notes
    -----
    ``adaptation_mask`` values passed to :meth:`fit`, :meth:`partial_fit`, or
    :meth:`process` gate recursive adaptation. False samples are still cleaned
    and counted as transported samples, but they neither update nor decay the
    covariance state. This makes sustained-contamination freeze controls
    explicit and replayable.

    MNE channel names, types, physical units, and sampling frequency are locked
    when state is initialized. Separate Raw inputs must also share
    ``first_samp``. Successive Raw blocks passed to :meth:`partial_fit` or
    :meth:`process` must be contiguous; NumPy streams have no sample identity,
    so their order remains the caller's responsibility.
    """

    def __init__(
        self,
        *,
        sfreq: float | None = None,
        ref_channels: list[int]
        | list[str]
        | tuple[int, ...]
        | tuple[str, ...]
        | None = None,
        primary_channels: list[int]
        | list[str]
        | tuple[int, ...]
        | tuple[str, ...]
        | None = None,
        clean_with: CleanWith = "X",
        threshold: float = 0.7,
        max_reject_fraction: float = 0.5,
        reref_primary: bool | str = False,
        reref_ref: bool | str = False,
        regularization: float = 1e-6,
        rank_tolerance: float = 1e-10,
        forgetting_factor: float | None = None,
        memory_duration_s: float | None = None,
        warmup_samples: int | None = 256,
        warmup_duration_s: float | None = None,
        update_interval_samples: int | None = 64,
        update_interval_s: float | None = None,
        convergence_tol: float = 1e-3,
        stable_updates: int = 3,
        update_order: UpdateOrder = "after",
        adaptation_mode: AdaptationMode = "adaptive",
        verbose: bool | str | int | None = None,
    ) -> None:
        self.sfreq = sfreq
        self.ref_channels = ref_channels
        self.primary_channels = primary_channels
        self.clean_with = clean_with
        self.threshold = threshold
        self.max_reject_fraction = max_reject_fraction
        self.reref_primary = reref_primary
        self.reref_ref = reref_ref
        self.regularization = regularization
        self.rank_tolerance = rank_tolerance
        self.forgetting_factor = forgetting_factor
        self.memory_duration_s = memory_duration_s
        self.warmup_samples = warmup_samples
        self.warmup_duration_s = warmup_duration_s
        self.update_interval_samples = update_interval_samples
        self.update_interval_s = update_interval_s
        self.convergence_tol = convergence_tol
        self.stable_updates = stable_updates
        self.update_order = update_order
        self.adaptation_mode = adaptation_mode
        self.verbose = verbose
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        """Validate constructor values without data-dependent conversions."""
        if self.sfreq is not None:
            _finite_real(self.sfreq, name="sfreq", minimum=0.0, strict=True)
        if self.clean_with not in {"X", "Y", "both"}:
            raise ValueError("clean_with must be 'X', 'Y', or 'both'.")
        threshold = _finite_real(self.threshold, name="threshold", minimum=0.0)
        if threshold > 1.0:
            raise ValueError("threshold must be <= 1.")
        reject_fraction = _finite_real(
            self.max_reject_fraction,
            name="max_reject_fraction",
            minimum=0.0,
        )
        if reject_fraction > 1.0:
            raise ValueError("max_reject_fraction must be <= 1.")
        _reref_matrix(2, self.reref_primary)
        _reref_matrix(2, self.reref_ref)
        _finite_real(
            self.regularization,
            name="regularization",
            minimum=0.0,
            strict=True,
        )
        rank_tolerance = _finite_real(
            self.rank_tolerance,
            name="rank_tolerance",
            minimum=0.0,
            strict=True,
        )
        if rank_tolerance >= 1.0:
            raise ValueError("rank_tolerance must be < 1.")
        if self.forgetting_factor is not None:
            factor = _finite_real(
                self.forgetting_factor,
                name="forgetting_factor",
                minimum=0.0,
                strict=True,
            )
            if factor > 1.0:
                raise ValueError("forgetting_factor must be <= 1.")
        if self.memory_duration_s is not None:
            _finite_real(
                self.memory_duration_s,
                name="memory_duration_s",
                minimum=0.0,
                strict=True,
            )
        if self.forgetting_factor is not None and self.memory_duration_s is not None:
            raise ValueError(
                "forgetting_factor and memory_duration_s are mutually exclusive."
            )
        self._validate_sample_duration_pair(
            self.warmup_samples,
            self.warmup_duration_s,
            sample_name="warmup_samples",
            duration_name="warmup_duration_s",
            minimum=2,
        )
        self._validate_sample_duration_pair(
            self.update_interval_samples,
            self.update_interval_s,
            sample_name="update_interval_samples",
            duration_name="update_interval_s",
            minimum=1,
        )
        _finite_real(
            self.convergence_tol,
            name="convergence_tol",
            minimum=0.0,
        )
        _positive_int(self.stable_updates, name="stable_updates")
        if self.update_order not in {"after", "before"}:
            raise ValueError("update_order must be 'after' or 'before'.")
        if self.adaptation_mode not in {"adaptive", "frozen"}:
            raise ValueError("adaptation_mode must be 'adaptive' or 'frozen'.")

    @staticmethod
    def _validate_sample_duration_pair(
        samples: int | None,
        duration: float | None,
        *,
        sample_name: str,
        duration_name: str,
        minimum: int,
    ) -> None:
        """Validate an explicitly named sample/second parameter pair."""
        if samples is None and duration is None:
            raise ValueError(f"Set exactly one of {sample_name} or {duration_name}.")
        if samples is not None and duration is not None:
            raise ValueError(
                f"{sample_name} and {duration_name} are mutually exclusive."
            )
        if samples is not None:
            _positive_int(samples, name=sample_name, minimum=minimum)
        else:
            _finite_real(duration, name=duration_name, minimum=0.0, strict=True)

    def _resolve_sfreq(
        self, primary_sfreq: float | None, reference_sfreq: float | None
    ) -> float | None:
        """Resolve configured and metadata sampling frequencies."""
        rates = [rate for rate in (primary_sfreq, reference_sfreq) if rate is not None]
        if len(rates) == 2 and not np.isclose(rates[0], rates[1], rtol=1e-9, atol=0.0):
            raise ValueError("Primary and reference sampling frequencies must match.")
        metadata_sfreq = rates[0] if rates else None
        configured = None
        if self.sfreq is not None:
            configured = _finite_real(
                self.sfreq, name="sfreq", minimum=0.0, strict=True
            )
        if (
            configured is not None
            and metadata_sfreq is not None
            and not np.isclose(configured, metadata_sfreq, rtol=1e-9, atol=0.0)
        ):
            raise ValueError(
                f"Configured sfreq ({configured}) does not match MNE metadata "
                f"({metadata_sfreq})."
            )
        return metadata_sfreq if metadata_sfreq is not None else configured

    def _resolve_time_configuration(self, sfreq: float | None) -> None:
        """Resolve durations and per-sample memory after sfreq is known."""
        requires_sfreq = any(
            value is not None
            for value in (
                self.memory_duration_s,
                self.warmup_duration_s,
                self.update_interval_s,
            )
        )
        if requires_sfreq and sfreq is None:
            raise ValueError(
                "sfreq is required when memory, warm-up, or update interval is "
                "specified in seconds."
            )
        if self.warmup_samples is not None:
            warmup = int(self.warmup_samples)
        else:
            warmup = int(round(float(self.warmup_duration_s) * float(sfreq)))
            if warmup < 2:
                raise ValueError(
                    "warmup_duration_s resolves to fewer than 2 samples at sfreq."
                )
        if self.update_interval_samples is not None:
            interval = int(self.update_interval_samples)
        else:
            interval = int(round(float(self.update_interval_s) * float(sfreq)))
            if interval < 1:
                raise ValueError(
                    "update_interval_s resolves to fewer than 1 sample at sfreq."
                )
        if self.memory_duration_s is not None:
            factor = float(np.exp(-1.0 / (float(sfreq) * self.memory_duration_s)))
            if not np.isfinite(factor) or factor <= 0.0:
                raise ValueError(
                    "memory_duration_s is too short to resolve a positive "
                    "per-sample forgetting factor at sfreq."
                )
        elif self.forgetting_factor is None:
            factor = 1.0
        else:
            factor = float(self.forgetting_factor)
        self._sfreq_ = sfreq
        self._warmup_samples_ = warmup
        self._update_interval_samples_ = interval
        self._forgetting_factor_ = factor

    def _extract_object(
        self,
        X: Any,
        *,
        ch_names: list[str] | None = None,
        auto_pick: bool = True,
        name: str,
    ) -> tuple[np.ndarray, float | None, str, Any, np.ndarray | None, list[str] | None]:
        """Extract a supported continuous object."""
        data, sfreq, mne_type, original, picks, extracted_names = extract_data_from_mne(
            X,
            ch_names=ch_names,
            auto_pick=auto_pick,
        )
        if mne_type not in {"array", "raw"}:
            raise TypeError(
                f"{name} must be a continuous Raw object or a 2D ndarray; "
                f"got {mne_type!r}."
            )
        return (
            _as_channel_time(data, name=name),
            sfreq,
            mne_type,
            original,
            picks,
            extracted_names,
        )

    def _prepare_inputs(
        self,
        X: Any,
        reference: Any | None,
        *,
        initialize: bool,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Extract, align, and validate combined or separate inputs."""
        if reference is None:
            if self.ref_channels is None:
                raise ValueError(
                    "ref_channels are required for combined input; otherwise pass "
                    "a separate reference= object."
                )
            full, sfreq_x, mne_type, original, _, ch_names = self._extract_object(
                X,
                auto_pick=False,
                name="X",
            )
            mode = "combined"
            if hasattr(self, "_input_mode_") and self._input_mode_ != mode:
                raise ValueError(
                    "Cannot switch between combined and separate input modes."
                )
            if (
                hasattr(self, "_primary_ch_names_")
                and self._primary_ch_names_ is not None
            ):
                primary_idx = _resolve_selector(
                    self._primary_ch_names_,
                    n_channels=full.shape[0],
                    ch_names=ch_names,
                    name="fitted primary channels",
                )
                ref_idx = _resolve_selector(
                    self._reference_ch_names_,
                    n_channels=full.shape[0],
                    ch_names=ch_names,
                    name="fitted reference channels",
                )
            else:
                ref_idx = _resolve_selector(
                    self.ref_channels,
                    n_channels=full.shape[0],
                    ch_names=ch_names,
                    name="ref_channels",
                )
                if self.primary_channels is None:
                    primary_idx = np.setdiff1d(
                        np.arange(full.shape[0]), ref_idx, assume_unique=False
                    )
                    if primary_idx.size == 0:
                        raise ValueError(
                            "Combined input must leave at least one primary channel."
                        )
                else:
                    primary_idx = _resolve_selector(
                        self.primary_channels,
                        n_channels=full.shape[0],
                        ch_names=ch_names,
                        name="primary_channels",
                    )
                if np.intersect1d(primary_idx, ref_idx).size:
                    raise ValueError("Primary and reference channels must be disjoint.")
            primary = full[primary_idx]
            ref = full[ref_idx]
            primary_names = (
                None if ch_names is None else [ch_names[index] for index in primary_idx]
            )
            reference_names = (
                None if ch_names is None else [ch_names[index] for index in ref_idx]
            )
            primary_signature = _channel_signature(original, primary_idx)
            reference_signature = _channel_signature(original, ref_idx)
            sfreq = self._resolve_sfreq(sfreq_x, sfreq_x)
            context = {
                "mode": mode,
                "mne_type": mne_type,
                "original": original,
                "full": full,
                "primary_idx": primary_idx,
                "primary_picks": None,
                "primary_names": primary_names,
                "reference_names": reference_names,
                "primary_signature": primary_signature,
                "reference_signature": reference_signature,
                "sfreq": sfreq,
                "first_samp": (
                    None if mne_type == "array" else int(original.first_samp)
                ),
            }
        else:
            mode = "separate"
            if self.ref_channels is not None or self.primary_channels is not None:
                raise ValueError(
                    "ref_channels/primary_channels cannot be combined with a separate "
                    "reference= object."
                )
            if hasattr(self, "_input_mode_") and self._input_mode_ != mode:
                raise ValueError(
                    "Cannot switch between combined and separate input modes."
                )
            primary_order = (
                list(self._primary_ch_names_)
                if hasattr(self, "_primary_ch_names_")
                and self._primary_ch_names_ is not None
                else None
            )
            reference_order = (
                list(self._reference_ch_names_)
                if hasattr(self, "_reference_ch_names_")
                and self._reference_ch_names_ is not None
                else None
            )
            primary, sfreq_x, mne_type, original, picks, primary_names = (
                self._extract_object(
                    X,
                    ch_names=primary_order,
                    auto_pick=True,
                    name="X",
                )
            )
            ref, sfreq_ref, ref_mne_type, ref_original, ref_picks, reference_names = (
                self._extract_object(
                    reference,
                    ch_names=reference_order,
                    auto_pick=False,
                    name="reference",
                )
            )
            if mne_type != ref_mne_type:
                raise TypeError(
                    "Separate primary and reference inputs must both be Raw objects "
                    "or both be NumPy arrays."
                )
            if mne_type == "raw" and original.first_samp != ref_original.first_samp:
                raise ValueError(
                    "Separate Raw primary and reference inputs must have the same "
                    "first_samp so their samples are time-aligned."
                )
            sfreq = self._resolve_sfreq(sfreq_x, sfreq_ref)
            context = {
                "mode": mode,
                "mne_type": mne_type,
                "original": original,
                "full": None,
                "primary_idx": None,
                "primary_picks": picks,
                "primary_names": primary_names,
                "reference_names": reference_names,
                "primary_signature": _channel_signature(original, picks),
                "reference_signature": _channel_signature(ref_original, ref_picks),
                "sfreq": sfreq,
                "first_samp": (
                    None if mne_type == "array" else int(original.first_samp)
                ),
            }

        if primary.shape[1] != ref.shape[1]:
            raise ValueError(
                "Primary and reference data must have the same number of samples."
            )
        if initialize:
            self._initialize_or_validate_geometry(
                primary,
                ref,
                mode=context["mode"],
                primary_names=context["primary_names"],
                reference_names=context["reference_names"],
                primary_signature=context["primary_signature"],
                reference_signature=context["reference_signature"],
                mne_type=context["mne_type"],
                sfreq=context["sfreq"],
            )
        else:
            self._validate_geometry(
                primary,
                ref,
                mode=context["mode"],
                primary_signature=context["primary_signature"],
                reference_signature=context["reference_signature"],
                mne_type=context["mne_type"],
                sfreq=context["sfreq"],
            )
        return primary, ref, context

    def _initialize_or_validate_geometry(
        self,
        primary: np.ndarray,
        reference: np.ndarray,
        *,
        mode: str,
        primary_names: list[str] | None,
        reference_names: list[str] | None,
        primary_signature: tuple[tuple[str, int, int, int, int], ...] | None,
        reference_signature: tuple[tuple[str, int, int, int, int], ...] | None,
        mne_type: str,
        sfreq: float | None,
    ) -> None:
        """Initialize recursive moments or validate existing geometry."""
        if hasattr(self, "_n_primary_"):
            self._validate_geometry(
                primary,
                reference,
                mode=mode,
                primary_signature=primary_signature,
                reference_signature=reference_signature,
                mne_type=mne_type,
                sfreq=sfreq,
            )
            return
        self._resolve_time_configuration(sfreq)
        self._input_mode_ = mode
        self._n_primary_ = primary.shape[0]
        self._n_reference_ = reference.shape[0]
        self._primary_ch_names_ = (
            None if primary_names is None else tuple(primary_names)
        )
        self._reference_ch_names_ = (
            None if reference_names is None else tuple(reference_names)
        )
        self._primary_channel_signature_ = primary_signature
        self._reference_channel_signature_ = reference_signature
        self._mne_type_ = mne_type
        self._samples_processed_ = 0
        self._samples_adapted_ = 0
        self._effective_weight_ = 0.0
        self._mean_primary_ = np.zeros(self._n_primary_)
        self._mean_reference_ = np.zeros(self._n_reference_)
        self._scatter_primary_ = np.zeros((self._n_primary_, self._n_primary_))
        self._scatter_reference_ = np.zeros((self._n_reference_, self._n_reference_))
        self._scatter_cross_ = np.zeros((self._n_primary_, self._n_reference_))
        self._next_update_sample_ = self._warmup_samples_
        self._model_version_ = 0
        self._stable_update_count_ = 0
        self._next_raw_first_samp_ = None
        self.converged_ = False
        self.update_history_ = []
        self.process_history_ = []

    def _validate_geometry(
        self,
        primary: np.ndarray,
        reference: np.ndarray,
        *,
        mode: str,
        primary_signature: tuple[tuple[str, int, int, int, int], ...] | None,
        reference_signature: tuple[tuple[str, int, int, int, int], ...] | None,
        mne_type: str,
        sfreq: float | None,
    ) -> None:
        """Validate data against initialized recursive geometry."""
        if not hasattr(self, "_n_primary_"):
            raise ValueError("Recursive state has not been initialized.")
        if mode != self._input_mode_:
            raise ValueError("Cannot switch between combined and separate input modes.")
        if mne_type != self._mne_type_:
            raise TypeError(
                "Cannot switch between Raw and NumPy inputs after recursive state "
                "initialization."
            )
        if primary.shape[0] != self._n_primary_:
            raise ValueError(
                f"Primary channel count changed ({primary.shape[0]} vs "
                f"{self._n_primary_})."
            )
        if reference.shape[0] != self._n_reference_:
            raise ValueError(
                f"Reference channel count changed ({reference.shape[0]} vs "
                f"{self._n_reference_})."
            )
        if primary_signature != self._primary_channel_signature_:
            raise ValueError(
                "Primary MNE channel names, types, or physical units differ from "
                "the recursive state."
            )
        if reference_signature != self._reference_channel_signature_:
            raise ValueError(
                "Reference MNE channel names, types, or physical units differ from "
                "the recursive state."
            )
        if self._sfreq_ is None and sfreq is not None:
            raise ValueError(
                "Sampling-frequency metadata differs from recursive state."
            )
        if self._sfreq_ is not None and (
            sfreq is None or not np.isclose(self._sfreq_, sfreq, rtol=1e-9, atol=0.0)
        ):
            raise ValueError("Sampling frequency differs from the recursive state.")

    @staticmethod
    def _normalize_adaptation_mask(
        adaptation_mask: np.ndarray | None, n_samples: int
    ) -> np.ndarray:
        """Return a boolean sample-wise adaptation gate."""
        if adaptation_mask is None:
            return np.ones(n_samples, dtype=bool)
        mask = np.asarray(adaptation_mask)
        if mask.ndim != 1 or mask.size != n_samples:
            raise ValueError(
                "adaptation_mask must be one-dimensional with one value per sample."
            )
        if mask.dtype != np.bool_ and (
            not np.issubdtype(mask.dtype, np.number)
            or not np.all(np.isin(mask, (0, 1)))
        ):
            raise ValueError("adaptation_mask must contain only boolean/0/1 values.")
        return mask.astype(bool, copy=False)

    def _validate_stream_timeline(self, context: dict[str, Any]) -> None:
        """Reject reordered, repeated, or gapped Raw transport blocks."""
        first_samp = context["first_samp"]
        if first_samp is None or self._next_raw_first_samp_ is None:
            return
        if first_samp != self._next_raw_first_samp_:
            raise ValueError(
                "Raw streaming blocks must be contiguous and ordered: expected "
                f"first_samp={self._next_raw_first_samp_}, got {first_samp}."
            )

    def _commit_stream_timeline(
        self,
        context: dict[str, Any],
        n_samples: int,
    ) -> None:
        """Record the next expected Raw sample after a successful state update."""
        first_samp = context["first_samp"]
        if first_samp is not None:
            self._next_raw_first_samp_ = int(first_samp + n_samples)

    def _update_one(self, primary: np.ndarray, reference: np.ndarray) -> None:
        """Apply a stable exponentially weighted joint-moment update."""
        factor = self._forgetting_factor_
        old_weight = self._effective_weight_
        new_weight = factor * old_weight + 1.0
        delta_primary = primary - self._mean_primary_
        delta_reference = reference - self._mean_reference_
        self._mean_primary_ += delta_primary / new_weight
        self._mean_reference_ += delta_reference / new_weight
        correction = factor * old_weight / new_weight
        self._scatter_primary_ *= factor
        self._scatter_reference_ *= factor
        self._scatter_cross_ *= factor
        self._scatter_primary_ += correction * np.outer(delta_primary, delta_primary)
        self._scatter_reference_ += correction * np.outer(
            delta_reference, delta_reference
        )
        self._scatter_cross_ += correction * np.outer(delta_primary, delta_reference)
        self._effective_weight_ = new_weight
        self._samples_adapted_ += 1

    def _update_statistics(
        self,
        primary: np.ndarray,
        reference: np.ndarray,
        gate: np.ndarray,
        *,
        update_models: bool,
    ) -> None:
        """Update state in sample order and honor exact model boundaries."""
        for sample in range(primary.shape[1]):
            self._samples_processed_ += 1
            if not gate[sample]:
                continue
            self._update_one(primary[:, sample], reference[:, sample])
            if update_models and self._samples_adapted_ >= self._next_update_sample_:
                self._update_model()
                self._next_update_sample_ += self._update_interval_samples_

    def _covariances(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return consistently scaled joint covariance blocks."""
        denominator = max(self._effective_weight_, np.finfo(float).eps)
        primary = self._scatter_primary_ / denominator
        reference = self._scatter_reference_ / denominator
        cross = self._scatter_cross_ / denominator
        return primary, reference, cross

    def _select_components(self, r2: np.ndarray) -> np.ndarray:
        """Apply the iCanClean threshold and reject cap."""
        selected = np.flatnonzero(r2 >= float(self.threshold))
        if self.max_reject_fraction == 0.0:
            return np.empty(0, dtype=int)
        maximum = max(1, int(float(self.max_reject_fraction) * r2.size))
        if selected.size > maximum:
            selected = np.argsort(r2)[::-1][:maximum]
            selected.sort()
        return selected.astype(int, copy=False)

    def _record_inadmissible_update(self, reason: str) -> bool:
        """Record a failed model update while retaining the last valid model."""
        diagnostic = {
            "status": "INADMISSIBLE",
            "reason": reason,
            "model_version": self._model_version_,
            "samples_processed": self._samples_processed_,
            "samples_adapted": self._samples_adapted_,
            "effective_weight": float(self._effective_weight_),
            "retained_previous_model": hasattr(self, "artifact_operator_"),
        }
        self.last_update_diagnostics_ = diagnostic
        self.update_history_.append(copy.deepcopy(diagnostic))
        return False

    def _update_model(self, *, force: bool = False) -> bool:
        """Recompute CCA and the artifact operator from recursive moments."""
        if self._samples_adapted_ < self._warmup_samples_ and not force:
            return False
        covariance_p, covariance_r, covariance_pr = self._covariances()
        transform_p = _reref_matrix(self._n_primary_, self.reref_primary)
        transform_r = _reref_matrix(self._n_reference_, self.reref_ref)
        covariance_x = transform_p.T @ covariance_p @ transform_p
        covariance_y = transform_r.T @ covariance_r @ transform_r
        covariance_xy = transform_p.T @ covariance_pr @ transform_r
        previous_operator = (
            None
            if not hasattr(self, "artifact_operator_")
            else self.artifact_operator_.copy()
        )

        try:
            whitening_x, rank_x, loading_x = _inverse_sqrt(
                covariance_x,
                regularization=float(self.regularization),
                rank_tolerance=float(self.rank_tolerance),
            )
            whitening_y, rank_y, loading_y = _inverse_sqrt(
                covariance_y,
                regularization=float(self.regularization),
                rank_tolerance=float(self.rank_tolerance),
            )
        except ValueError as err:
            return self._record_inadmissible_update(str(err))
        if rank_x == 0 or rank_y == 0:
            return self._record_inadmissible_update(
                "primary or reference covariance has zero rank"
            )

        coherence = whitening_x @ covariance_xy @ whitening_y
        try:
            left, correlations, right_t = la.svd(
                coherence, full_matrices=False, check_finite=False
            )
        except la.LinAlgError as err:
            return self._record_inadmissible_update(f"CCA SVD failed: {err}")
        rank_ceiling = min(rank_x, rank_y)
        left = left[:, :rank_ceiling]
        correlations = correlations[:rank_ceiling]
        right_t = right_t[:rank_ceiling, :]
        correlations = np.clip(correlations, 0.0, 1.0)
        weights_x = whitening_x @ left
        weights_y = whitening_y @ right_t.T
        r2 = correlations**2
        selected = self._select_components(r2)

        coefficients_x = transform_p @ weights_x[:, selected]
        coefficients_y = transform_r @ weights_y[:, selected]
        zeros_x = np.zeros((self._n_primary_, selected.size))
        zeros_y = np.zeros((self._n_reference_, selected.size))
        if self.clean_with == "X":
            basis = np.vstack((coefficients_x, zeros_y))
        elif self.clean_with == "Y":
            basis = np.vstack((zeros_x, coefficients_y))
        else:
            basis = np.column_stack(
                (
                    np.vstack((coefficients_x, zeros_y)),
                    np.vstack((zeros_x, coefficients_y)),
                )
            )

        joint_covariance = np.block(
            [
                [covariance_p, covariance_pr],
                [covariance_pr.T, covariance_r],
            ]
        )
        covariance_to_primary = np.vstack((covariance_p, covariance_pr.T))
        if basis.shape[1] == 0:
            regression = np.zeros((0, self._n_primary_))
            artifact_operator = np.zeros(
                (self._n_primary_ + self._n_reference_, self._n_primary_)
            )
        else:
            basis_covariance = basis.T @ joint_covariance @ basis
            basis_scale = max(
                float(np.trace(basis_covariance) / basis_covariance.shape[0]),
                np.finfo(float).eps,
            )
            basis_covariance += (
                float(self.regularization)
                * basis_scale
                * np.eye(basis_covariance.shape[0])
            )
            try:
                regression = la.solve(
                    basis_covariance,
                    basis.T @ covariance_to_primary,
                    assume_a="pos",
                    check_finite=False,
                )
            except la.LinAlgError as err:
                return self._record_inadmissible_update(
                    f"artifact regression failed: {err}"
                )
            artifact_operator = basis @ regression
        if not np.all(np.isfinite(artifact_operator)):
            return self._record_inadmissible_update(
                "recursive iCanClean produced a non-finite operator"
            )

        base_operator = np.vstack(
            (
                np.eye(self._n_primary_),
                np.zeros((self._n_reference_, self._n_primary_)),
            )
        )
        self.x_weights_ = weights_x
        self.y_weights_ = weights_y
        self.correlations_ = r2
        self.removed_idx_ = selected
        self.n_removed_ = int(selected.size)
        self.regression_ = regression
        self.artifact_operator_ = artifact_operator
        self.cleaning_operator_ = base_operator - artifact_operator
        self.model_primary_mean_ = self._mean_primary_.copy()
        self.model_reference_mean_ = self._mean_reference_.copy()
        self.model_samples_ = int(self._samples_adapted_)
        self.rank_primary_ = rank_x
        self.rank_reference_ = rank_y
        self.rank_ceiling_ = rank_ceiling
        self._model_version_ += 1

        if previous_operator is None:
            operator_change = np.inf
            self._stable_update_count_ = 0
        else:
            denominator = max(
                la.norm(previous_operator, ord="fro"), np.finfo(float).eps
            )
            operator_change = float(
                la.norm(artifact_operator - previous_operator, ord="fro") / denominator
            )
            if operator_change <= float(self.convergence_tol):
                self._stable_update_count_ += 1
            else:
                self._stable_update_count_ = 0
        self.converged_ = self._stable_update_count_ >= int(self.stable_updates)
        diagnostic = {
            "status": "APPLIED" if selected.size else "NO_OP",
            "model_version": self._model_version_,
            "samples_processed": self._samples_processed_,
            "samples_adapted": self._samples_adapted_,
            "effective_weight": float(self._effective_weight_),
            "rank_primary": rank_x,
            "rank_reference": rank_y,
            "rank_ceiling": rank_ceiling,
            "rank_limited": rank_ceiling < min(self._n_primary_, self._n_reference_),
            "regularization_primary": loading_x,
            "regularization_reference": loading_y,
            "squared_canonical_correlations": r2.copy(),
            "removed_idx": selected.copy(),
            "n_removed": int(selected.size),
            "operator_change": operator_change,
            "stable_update_count": self._stable_update_count_,
            "converged": self.converged_,
            "forgetting_factor_per_sample": self._forgetting_factor_,
        }
        diagnostic["model_state_sha256"] = self._model_state_hash()
        self.last_update_diagnostics_ = diagnostic
        self.update_history_.append(copy.deepcopy(diagnostic))
        if self.verbose:
            logger.info(
                "RecursiveICanClean update %d: %d components, ranks %d/%d, "
                "change %.3g.",
                self._model_version_,
                selected.size,
                rank_x,
                rank_y,
                operator_change,
            )
        return True

    def _model_state_hash(self) -> str:
        """Hash numerical adaptation state without transport counters or logs."""
        digest = hashlib.sha256()
        digest.update(str(_STATE_VERSION).encode("ascii"))
        for value in (
            self._effective_weight_,
            self._samples_adapted_,
            getattr(self, "_model_version_", 0),
        ):
            digest.update(np.asarray(value).tobytes())
        for name in (
            "_mean_primary_",
            "_mean_reference_",
            "_scatter_primary_",
            "_scatter_reference_",
            "_scatter_cross_",
            "x_weights_",
            "y_weights_",
            "correlations_",
            "removed_idx_",
            "regression_",
            "artifact_operator_",
            "cleaning_operator_",
            "model_primary_mean_",
            "model_reference_mean_",
        ):
            if hasattr(self, name):
                array = np.ascontiguousarray(getattr(self, name), dtype=np.float64)
                digest.update(str(array.shape).encode("ascii"))
                digest.update(array.tobytes())
        for name in (
            "n_removed_",
            "model_samples_",
            "rank_primary_",
            "rank_reference_",
            "rank_ceiling_",
        ):
            if hasattr(self, name):
                digest.update(np.asarray(getattr(self, name)).tobytes())
        return digest.hexdigest()

    def _apply_model(self, primary: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Apply the current frozen model, or abstain during warm-up."""
        if not hasattr(self, "artifact_operator_"):
            return primary.copy()
        centered = np.vstack(
            (
                primary - self.model_primary_mean_[:, None],
                reference - self.model_reference_mean_[:, None],
            )
        )
        artifact = self.artifact_operator_.T @ centered
        cleaned = primary - artifact
        if not np.all(np.isfinite(cleaned)):
            raise ValueError("Recursive iCanClean produced non-finite output.")
        return cleaned

    def _restore_output(self, cleaned: np.ndarray, context: dict[str, Any]) -> Any:
        """Restore a NumPy or Raw output without discarding metadata."""
        if context["mode"] == "combined":
            full = context["full"].copy()
            full[context["primary_idx"], :] = cleaned
            if context["mne_type"] == "array":
                return full
            output = context["original"].copy()
            output.load_data()
            output._data[:, :] = full
            return output
        if context["mne_type"] == "array":
            return cleaned
        output = context["original"].copy()
        output.load_data()
        selected = (
            slice(None)
            if context["primary_picks"] is None
            else context["primary_picks"]
        )
        output._data[selected, :] = cleaned
        return output

    def reset(self) -> RecursiveICanClean:
        """Clear all recursive, fitted, diagnostic, and replay state."""
        fitted_names = [
            name
            for name in vars(self)
            if name.endswith("_") and name not in {"reref_primary", "reref_ref"}
        ]
        for name in fitted_names:
            delattr(self, name)
        return self

    def fit(
        self,
        X: Any,
        y=None,
        *,
        reference: Any | None = None,
        adaptation_mask: np.ndarray | None = None,
    ) -> RecursiveICanClean:
        """Reset and calibrate a frozen model from all accepted samples."""
        del y
        self.reset()
        primary, ref, _ = self._prepare_inputs(X, reference, initialize=True)
        gate = self._normalize_adaptation_mask(adaptation_mask, primary.shape[1])
        self._update_statistics(primary, ref, gate, update_models=True)
        if self._samples_adapted_ < self._warmup_samples_:
            raise ValueError(
                f"Calibration supplied {self._samples_adapted_} accepted samples, "
                f"but warm-up requires {self._warmup_samples_}."
            )
        if getattr(self, "model_samples_", None) != self._samples_adapted_:
            self._update_model(force=True)
        if not hasattr(self, "artifact_operator_"):
            reason = getattr(self, "last_update_diagnostics_", {}).get(
                "reason", "CCA model was inadmissible"
            )
            raise ValueError(f"Recursive iCanClean calibration failed: {reason}.")
        self._next_update_sample_ = (
            self._samples_adapted_ + self._update_interval_samples_
        )
        return self

    def partial_fit(
        self,
        X: Any,
        y=None,
        *,
        reference: Any | None = None,
        adaptation_mask: np.ndarray | None = None,
    ) -> RecursiveICanClean:
        """Update recursive state without cleaning or resetting it."""
        del y
        primary, ref, context = self._prepare_inputs(X, reference, initialize=True)
        self._validate_stream_timeline(context)
        gate = self._normalize_adaptation_mask(adaptation_mask, primary.shape[1])
        self._update_statistics(primary, ref, gate, update_models=True)
        self._commit_stream_timeline(context, primary.shape[1])
        return self

    def transform(self, X: Any, *, reference: Any | None = None) -> Any:
        """Apply the frozen current model without changing recursive state."""
        check_is_fitted(self, attributes=["artifact_operator_", "_n_primary_"])
        primary, ref, context = self._prepare_inputs(X, reference, initialize=False)
        cleaned = self._apply_model(primary, ref)
        return self._restore_output(cleaned, context)

    def fit_transform(
        self,
        X: Any,
        y=None,
        *,
        reference: Any | None = None,
        adaptation_mask: np.ndarray | None = None,
        **fit_params,
    ) -> Any:
        """Calibrate on ``X`` and apply the resulting frozen model to ``X``.

        This convenience method intentionally applies a model to its own
        calibration samples. Do not use its output for held-out effectiveness
        evaluation; call :meth:`fit` on calibration data and :meth:`transform`
        on an independent evaluation partition instead.
        """
        if fit_params:
            unknown = ", ".join(sorted(fit_params))
            raise TypeError(f"Unexpected fit parameters: {unknown}.")
        return self.fit(
            X,
            y,
            reference=reference,
            adaptation_mask=adaptation_mask,
        ).transform(X, reference=reference)

    def process(
        self,
        X: Any,
        *,
        reference: Any | None = None,
        adaptation_mask: np.ndarray | None = None,
    ) -> Any:
        """Causally clean one transport block and optionally adapt state.

        With ``update_order='after'``, each accepted sample updates the model
        only after that sample has been cleaned. Model-update boundaries are
        indexed by accepted samples, so splitting the same ordered stream into
        different transport blocks produces the same numerical output up to
        floating-point roundoff.
        """
        if self.adaptation_mode == "frozen" and not hasattr(self, "artifact_operator_"):
            raise ValueError(
                "adaptation_mode='frozen' requires a calibrated or loaded state."
            )
        primary, ref, context = self._prepare_inputs(X, reference, initialize=True)
        self._validate_stream_timeline(context)
        gate = self._normalize_adaptation_mask(adaptation_mask, primary.shape[1])
        start_processed = self._samples_processed_
        start_adapted = self._samples_adapted_
        start_version = self._model_version_
        cleaned = np.empty_like(primary)
        warmup_passthrough_samples = 0
        for sample in range(primary.shape[1]):
            self._samples_processed_ += 1
            should_adapt = self.adaptation_mode == "adaptive" and gate[sample]
            if should_adapt and self.update_order == "before":
                self._update_one(primary[:, sample], ref[:, sample])
                if self._samples_adapted_ >= self._next_update_sample_:
                    self._update_model()
                    self._next_update_sample_ += self._update_interval_samples_
            if not hasattr(self, "artifact_operator_"):
                warmup_passthrough_samples += 1
            cleaned[:, sample : sample + 1] = self._apply_model(
                primary[:, sample : sample + 1], ref[:, sample : sample + 1]
            )
            if should_adapt and self.update_order == "after":
                self._update_one(primary[:, sample], ref[:, sample])
                if self._samples_adapted_ >= self._next_update_sample_:
                    self._update_model()
                    self._next_update_sample_ += self._update_interval_samples_

        block_samples = primary.shape[1]
        duration = None if self._sfreq_ is None else float(block_samples / self._sfreq_)
        process_diagnostic = {
            "status": (
                "WARMING_UP"
                if not hasattr(self, "artifact_operator_")
                else ("APPLIED" if self.n_removed_ else "NO_OP")
            ),
            "start_sample": start_processed,
            "stop_sample": self._samples_processed_,
            "transport_block_samples": block_samples,
            "transport_block_duration_s": duration,
            "algorithmic_lookahead_samples": 0,
            "current_sample_used_for_own_model": (
                self.adaptation_mode == "adaptive" and self.update_order == "before"
            ),
            "adaptation_delay_samples": (
                1
                if self.adaptation_mode == "adaptive" and self.update_order == "after"
                else 0
            ),
            "accepted_adaptation_samples": self._samples_adapted_ - start_adapted,
            "warmup_passthrough_samples": warmup_passthrough_samples,
            "cleaned_with_model_samples": block_samples - warmup_passthrough_samples,
            "gated_samples": int(block_samples - np.sum(gate))
            if self.adaptation_mode == "adaptive"
            else block_samples,
            "model_version_before": start_version,
            "model_version_after": self._model_version_,
            "model_state_sha256": self._model_state_hash(),
            "update_order": self.update_order,
            "adaptation_mode": self.adaptation_mode,
        }
        self.last_process_diagnostics_ = process_diagnostic
        self.process_history_.append(copy.deepcopy(process_diagnostic))
        self._commit_stream_timeline(context, primary.shape[1])
        return self._restore_output(cleaned, context)

    def _configuration_state(self) -> dict[str, Any]:
        """Return constructor configuration relevant to numerical replay."""
        return {
            "sfreq": self.sfreq,
            "ref_channels": None
            if self.ref_channels is None
            else tuple(self.ref_channels),
            "primary_channels": None
            if self.primary_channels is None
            else tuple(self.primary_channels),
            "clean_with": self.clean_with,
            "threshold": self.threshold,
            "max_reject_fraction": self.max_reject_fraction,
            "reref_primary": self.reref_primary,
            "reref_ref": self.reref_ref,
            "regularization": self.regularization,
            "rank_tolerance": self.rank_tolerance,
            "forgetting_factor": self.forgetting_factor,
            "memory_duration_s": self.memory_duration_s,
            "warmup_samples": self.warmup_samples,
            "warmup_duration_s": self.warmup_duration_s,
            "update_interval_samples": self.update_interval_samples,
            "update_interval_s": self.update_interval_s,
            "convergence_tol": self.convergence_tol,
            "stable_updates": self.stable_updates,
            "update_order": self.update_order,
            "adaptation_mode": self.adaptation_mode,
        }

    def state_dict(self) -> dict[str, Any]:
        """Return a deep-copy checkpoint sufficient for exact state replay."""
        if not hasattr(self, "_n_primary_"):
            raise ValueError("Recursive state has not been initialized.")
        state = {
            "version": _STATE_VERSION,
            "configuration": self._configuration_state(),
            "input_mode": self._input_mode_,
            "sfreq_resolved": self._sfreq_,
            "warmup_samples_resolved": self._warmup_samples_,
            "update_interval_samples_resolved": self._update_interval_samples_,
            "forgetting_factor_resolved": self._forgetting_factor_,
            "n_primary": self._n_primary_,
            "n_reference": self._n_reference_,
            "primary_ch_names": self._primary_ch_names_,
            "reference_ch_names": self._reference_ch_names_,
            "primary_channel_signature": self._primary_channel_signature_,
            "reference_channel_signature": self._reference_channel_signature_,
            "mne_type": self._mne_type_,
            "next_raw_first_samp": self._next_raw_first_samp_,
            "samples_processed": self._samples_processed_,
            "samples_adapted": self._samples_adapted_,
            "effective_weight": self._effective_weight_,
            "mean_primary": self._mean_primary_.copy(),
            "mean_reference": self._mean_reference_.copy(),
            "scatter_primary": self._scatter_primary_.copy(),
            "scatter_reference": self._scatter_reference_.copy(),
            "scatter_cross": self._scatter_cross_.copy(),
            "next_update_sample": self._next_update_sample_,
            "model_version": self._model_version_,
            "stable_update_count": self._stable_update_count_,
            "converged": self.converged_,
            "update_history": copy.deepcopy(self.update_history_),
            "process_history": copy.deepcopy(self.process_history_),
            "last_update_diagnostics": copy.deepcopy(
                getattr(self, "last_update_diagnostics_", None)
            ),
            "last_process_diagnostics": copy.deepcopy(
                getattr(self, "last_process_diagnostics_", None)
            ),
        }
        state["model"] = {
            name: copy.deepcopy(getattr(self, name))
            for name in _MODEL_STATE_NAMES
            if hasattr(self, name)
        }
        state["model_state_sha256"] = self._model_state_hash()
        state["state_sha256"] = _state_payload_hash(state)
        return state

    def state_json(self) -> str:
        """Serialize a lossless checkpoint to canonical UTF-8 JSON.

        Unlike a pickle, this representation contains no executable objects.
        NumPy arrays are stored as base64-encoded, dtype- and shape-tagged byte
        sequences, so a round trip retains exact floating-point state.
        """
        return _canonical_state_json(self.state_dict())

    def load_state_json(self, payload: str | bytes) -> RecursiveICanClean:
        """Load a checkpoint emitted by :meth:`state_json`."""
        if isinstance(payload, bytes):
            try:
                payload = payload.decode("utf-8")
            except UnicodeDecodeError as err:
                raise ValueError("Serialized state must be valid UTF-8.") from err
        if not isinstance(payload, str):
            raise TypeError("payload must be a JSON string or UTF-8 bytes.")
        try:
            encoded = json.loads(payload)
        except json.JSONDecodeError as err:
            raise ValueError("Serialized state is not valid JSON.") from err
        decoded = _decode_state_value(encoded)
        if not isinstance(decoded, dict):
            raise ValueError("Serialized state must decode to a dictionary.")
        return self.load_state_dict(decoded)

    def load_state_dict(self, state: dict[str, Any]) -> RecursiveICanClean:
        """Transactionally load a checksum-validated checkpoint."""
        if not isinstance(state, dict):
            raise TypeError("state must be a dictionary returned by state_dict().")
        required_fields = {
            "version",
            "configuration",
            "input_mode",
            "sfreq_resolved",
            "warmup_samples_resolved",
            "update_interval_samples_resolved",
            "forgetting_factor_resolved",
            "n_primary",
            "n_reference",
            "primary_ch_names",
            "reference_ch_names",
            "primary_channel_signature",
            "reference_channel_signature",
            "mne_type",
            "next_raw_first_samp",
            "samples_processed",
            "samples_adapted",
            "effective_weight",
            "mean_primary",
            "mean_reference",
            "scatter_primary",
            "scatter_reference",
            "scatter_cross",
            "next_update_sample",
            "model_version",
            "stable_update_count",
            "converged",
            "update_history",
            "process_history",
            "last_update_diagnostics",
            "last_process_diagnostics",
            "model",
            "model_state_sha256",
            "state_sha256",
        }
        if set(state) != required_fields:
            raise ValueError(
                "State fields do not match the versioned checkpoint schema."
            )
        if state.get("version") != _STATE_VERSION:
            raise ValueError(
                f"Unsupported RecursiveICanClean state version {state.get('version')!r}."
            )
        if state.get("configuration") != self._configuration_state():
            raise ValueError("State configuration does not match this estimator.")
        model = state.get("model")
        if not isinstance(model, dict) or set(model) not in (
            set(),
            set(_MODEL_STATE_NAMES),
        ):
            raise ValueError("State model fields are incomplete or unknown.")
        expected_state_hash = state.get("state_sha256")
        if not isinstance(expected_state_hash, str) or len(expected_state_hash) != 64:
            raise ValueError("State is missing a valid state_sha256 checksum.")
        if _state_payload_hash(state) != expected_state_hash:
            raise ValueError("State checksum does not match its serialized payload.")

        candidate = type(self)(**self.get_params(deep=False))
        candidate._restore_state_payload(state)
        candidate._validate_loaded_shapes()
        expected_model_hash = state.get("model_state_sha256")
        if (
            not isinstance(expected_model_hash, str)
            or candidate._model_state_hash() != expected_model_hash
        ):
            raise ValueError("State checksum does not match its numerical payload.")

        self.reset()
        for name, value in vars(candidate).items():
            if name.endswith("_"):
                setattr(self, name, copy.deepcopy(value))
        return self

    def _restore_state_payload(self, state: dict[str, Any]) -> None:
        """Restore a prevalidated payload into a disposable candidate."""
        self._input_mode_ = state["input_mode"]
        self._sfreq_ = state["sfreq_resolved"]
        self._warmup_samples_ = int(state["warmup_samples_resolved"])
        self._update_interval_samples_ = int(state["update_interval_samples_resolved"])
        self._forgetting_factor_ = float(state["forgetting_factor_resolved"])
        self._n_primary_ = int(state["n_primary"])
        self._n_reference_ = int(state["n_reference"])
        self._primary_ch_names_ = state["primary_ch_names"]
        self._reference_ch_names_ = state["reference_ch_names"]
        self._primary_channel_signature_ = state["primary_channel_signature"]
        self._reference_channel_signature_ = state["reference_channel_signature"]
        self._mne_type_ = state["mne_type"]
        self._next_raw_first_samp_ = state["next_raw_first_samp"]
        self._samples_processed_ = int(state["samples_processed"])
        self._samples_adapted_ = int(state["samples_adapted"])
        self._effective_weight_ = float(state["effective_weight"])
        self._mean_primary_ = np.array(state["mean_primary"], copy=True)
        self._mean_reference_ = np.array(state["mean_reference"], copy=True)
        self._scatter_primary_ = np.array(state["scatter_primary"], copy=True)
        self._scatter_reference_ = np.array(state["scatter_reference"], copy=True)
        self._scatter_cross_ = np.array(state["scatter_cross"], copy=True)
        self._next_update_sample_ = int(state["next_update_sample"])
        self._model_version_ = int(state["model_version"])
        self._stable_update_count_ = int(state["stable_update_count"])
        self.converged_ = bool(state["converged"])
        self.update_history_ = copy.deepcopy(state["update_history"])
        self.process_history_ = copy.deepcopy(state["process_history"])
        if state["last_update_diagnostics"] is not None:
            self.last_update_diagnostics_ = copy.deepcopy(
                state["last_update_diagnostics"]
            )
        if state["last_process_diagnostics"] is not None:
            self.last_process_diagnostics_ = copy.deepcopy(
                state["last_process_diagnostics"]
            )
        for name, value in state["model"].items():
            setattr(self, name, copy.deepcopy(value))

    def _validate_loaded_shapes(self) -> None:
        """Reject malformed or cross-geometry state payloads."""
        if self._input_mode_ not in {"combined", "separate"}:
            raise ValueError("State input_mode is invalid.")
        if self._mne_type_ not in {"array", "raw"}:
            raise ValueError("State mne_type is invalid.")
        if self._n_primary_ < 1 or self._n_reference_ < 1:
            raise ValueError("State channel counts must be positive.")
        if self._samples_processed_ < 0 or not (
            0 <= self._samples_adapted_ <= self._samples_processed_
        ):
            raise ValueError("State sample counters are inconsistent.")
        if (
            not np.isfinite(self._effective_weight_)
            or self._effective_weight_ < 0.0
            or (self._samples_adapted_ == 0) != (self._effective_weight_ == 0.0)
        ):
            raise ValueError("State effective weight is inconsistent.")
        if self._warmup_samples_ < 2 or self._update_interval_samples_ < 1:
            raise ValueError("State resolved update intervals are invalid.")
        if not 0.0 < self._forgetting_factor_ <= 1.0:
            raise ValueError("State forgetting factor is invalid.")
        if self._next_update_sample_ <= self._samples_adapted_:
            raise ValueError("State next model-update boundary is inconsistent.")
        if self._model_version_ < 0 or self._stable_update_count_ < 0:
            raise ValueError("State model counters must be non-negative.")
        if self.converged_ != (self._stable_update_count_ >= int(self.stable_updates)):
            raise ValueError("State convergence flag is inconsistent.")
        if self._sfreq_ is not None and (
            not np.isfinite(self._sfreq_) or self._sfreq_ <= 0.0
        ):
            raise ValueError("State sampling frequency is invalid.")
        if self._next_raw_first_samp_ is not None and self._mne_type_ != "raw":
            raise ValueError("Only Raw state may contain a next sample identity.")
        if self._mne_type_ == "array" and (
            self._primary_channel_signature_ is not None
            or self._reference_channel_signature_ is not None
        ):
            raise ValueError("NumPy state cannot contain MNE channel metadata.")
        if self._mne_type_ == "raw" and (
            self._primary_channel_signature_ is None
            or self._reference_channel_signature_ is None
            or len(self._primary_channel_signature_) != self._n_primary_
            or len(self._reference_channel_signature_) != self._n_reference_
        ):
            raise ValueError("Raw state has incomplete channel metadata.")
        if self._primary_ch_names_ is not None and (
            len(self._primary_ch_names_) != self._n_primary_
        ):
            raise ValueError("State primary channel names have invalid length.")
        if self._reference_ch_names_ is not None and (
            len(self._reference_ch_names_) != self._n_reference_
        ):
            raise ValueError("State reference channel names have invalid length.")
        if not isinstance(self.update_history_, list) or not isinstance(
            self.process_history_, list
        ):
            raise ValueError("State diagnostic histories must be lists.")
        expected = {
            "_mean_primary_": (self._n_primary_,),
            "_mean_reference_": (self._n_reference_,),
            "_scatter_primary_": (self._n_primary_, self._n_primary_),
            "_scatter_reference_": (self._n_reference_, self._n_reference_),
            "_scatter_cross_": (self._n_primary_, self._n_reference_),
        }
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f"State field {name} has invalid shape or values.")
        if not np.allclose(
            self._scatter_primary_, self._scatter_primary_.T, rtol=1e-12, atol=0.0
        ):
            raise ValueError("State primary scatter matrix is not symmetric.")
        if not np.allclose(
            self._scatter_reference_,
            self._scatter_reference_.T,
            rtol=1e-12,
            atol=0.0,
        ):
            raise ValueError("State reference scatter matrix is not symmetric.")
        if hasattr(self, "artifact_operator_"):
            expected_operator = (
                self._n_primary_ + self._n_reference_,
                self._n_primary_,
            )
            if self.artifact_operator_.shape != expected_operator:
                raise ValueError("State artifact_operator_ has invalid shape.")
            model_arrays = (
                "x_weights_",
                "y_weights_",
                "correlations_",
                "removed_idx_",
                "regression_",
                "artifact_operator_",
                "cleaning_operator_",
                "model_primary_mean_",
                "model_reference_mean_",
            )
            for name in model_arrays:
                value = np.asarray(getattr(self, name))
                if not np.all(np.isfinite(value)):
                    raise ValueError(f"State model field {name} is non-finite.")
            if not (
                1 <= int(self.rank_primary_) <= self._n_primary_
                and 1 <= int(self.rank_reference_) <= self._n_reference_
                and int(self.rank_ceiling_)
                == min(int(self.rank_primary_), int(self.rank_reference_))
            ):
                raise ValueError("State covariance ranks are inconsistent.")
            n_components = int(self.rank_ceiling_)
            if self.x_weights_.shape != (self._n_primary_, n_components):
                raise ValueError("State x_weights_ has invalid shape.")
            if self.y_weights_.shape != (self._n_reference_, n_components):
                raise ValueError("State y_weights_ has invalid shape.")
            if self.correlations_.shape != (n_components,):
                raise ValueError("State correlations_ has invalid shape.")
            if np.any(self.correlations_ < 0.0) or np.any(self.correlations_ > 1.0):
                raise ValueError("State correlations_ must lie in [0, 1].")
            removed = np.asarray(self.removed_idx_)
            if (
                removed.ndim != 1
                or not np.issubdtype(removed.dtype, np.integer)
                or np.unique(removed).size != removed.size
                or np.any(removed < 0)
                or np.any(removed >= n_components)
                or int(self.n_removed_) != removed.size
            ):
                raise ValueError("State removed component indices are invalid.")
            n_basis = removed.size * (2 if self.clean_with == "both" else 1)
            if self.regression_.shape != (n_basis, self._n_primary_):
                raise ValueError("State regression_ has invalid shape.")
            if self.cleaning_operator_.shape != expected_operator:
                raise ValueError("State cleaning_operator_ has invalid shape.")
            if self.model_primary_mean_.shape != (self._n_primary_,):
                raise ValueError("State model_primary_mean_ has invalid shape.")
            if self.model_reference_mean_.shape != (self._n_reference_,):
                raise ValueError("State model_reference_mean_ has invalid shape.")
            if not 0 <= int(self.model_samples_) <= self._samples_adapted_:
                raise ValueError("State model sample count is inconsistent.")
        elif self._model_version_ != 0:
            raise ValueError("State has a model version but no model payload.")


__all__ = ["RecursiveICanClean"]
