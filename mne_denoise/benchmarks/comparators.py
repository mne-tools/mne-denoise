"""Comparator contracts for the mne-denoise benchmark suite.

Every benchmarked method *and* every baseline/comparator implements the same
adapter so the runners can treat them uniformly, enforce a train/evaluation
**leakage barrier**, and record reproducibility metadata.

Contract (see ``docs/benchmarks/RECONCILIATION.md``)::

    state  = comparator.fit(train, ctx)          # never sees the eval object
    result = comparator.transform(eval, state, ctx)   # never re-fits

``fit`` returns an opaque *state*; ``transform`` consumes it and returns a
:class:`ComparatorResult`.  A stable string ID maps to exactly one
implementation + parameter policy (never reuse an ID for two behaviours).

This module ships the contract, the ``fit_scope`` taxonomy, a small registry,
and two trivial comparators (``none``, ``pca_reconstruct``) used to exercise the
contract.  Real method wrappers are added in P3, not here.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Literal

import numpy as np

# ---------------------------------------------------------------------------
# Taxonomies
# ---------------------------------------------------------------------------
FitScope = Literal[
    "train_only",                 # DSS, rank-matched PCA, ICA, EOG-DSS
    "calibration_then_transform", # ASR, rASR
    "per_recording_unsupervised", # ZapLine, ZapLine-plus
    "window_local",               # iCanClean
]
VALID_FIT_SCOPES: tuple[str, ...] = (
    "train_only",
    "calibration_then_transform",
    "per_recording_unsupervised",
    "window_local",
)

# Terminal run statuses (also used by runners; see ANALYSIS_PLAN.md §6).
VALID_STATUSES: tuple[str, ...] = (
    "success",
    "unavailable_dependency",
    "skipped_missing_channels",
    "failed_convergence",
    "failed_numerical",
    "timeout",
    "excluded_dataset_qc",
)

_SCALAR_FIELDS = (
    "status", "error", "runtime_seconds", "cpu_seconds", "peak_memory_mb",
    "rank_before", "rank_after", "n_samples_before", "n_samples_after",
    "parameters", "random_seed", "software_versions",
)


@dataclass
class ComparatorResult:
    """Container returned by every ``transform`` (see RECONCILIATION.md)."""

    cleaned: Any = None
    model: Any = None
    diagnostics: dict = field(default_factory=dict)
    status: str = "success"
    error: str | None = None
    runtime_seconds: float | None = None
    cpu_seconds: float | None = None
    peak_memory_mb: float | None = None
    rank_before: int | None = None
    rank_after: int | None = None
    n_samples_before: int | None = None
    n_samples_after: int | None = None
    parameters: dict = field(default_factory=dict)
    random_seed: int | None = None
    software_versions: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            raise ValueError(
                f"status {self.status!r} not in {VALID_STATUSES}"
            )

    def summary(self) -> dict:
        """JSON-serialisable scalar view (drops ``cleaned``/``model``)."""
        out = {k: getattr(self, k) for k in _SCALAR_FIELDS}
        out["diagnostics"] = {
            k: v for k, v in self.diagnostics.items()
            if isinstance(v, (int, float, str, bool, type(None)))
        }
        return out


@dataclass
class ComparatorMeta:
    """Declarative metadata used for fairness tiers and gating."""

    comparator_id: str
    fit_scope: FitScope = "train_only"
    reference_aware: bool = False
    rank_reducing: bool = False
    requires_fit: bool = True
    manual_selection: bool = False
    optional_dependency: str | None = None
    ground_truth_only: bool = False
    required_channels: tuple[str, ...] = ()
    supported_input_types: tuple[str, ...] = ("raw", "epochs", "ndarray")
    deterministic: bool = True
    information_tier: str = "blind"  # "blind" | "reference_aware"

    def __post_init__(self) -> None:
        if self.fit_scope not in VALID_FIT_SCOPES:
            raise ValueError(
                f"fit_scope {self.fit_scope!r} not in {VALID_FIT_SCOPES}"
            )
        if self.reference_aware:
            self.information_tier = "reference_aware"


class _FitState:
    """Opaque marker proving ``fit`` ran before ``transform``."""

    __slots__ = ("payload", "fitted")

    def __init__(self, payload: Any = None) -> None:
        self.payload = payload
        self.fitted = True


class Comparator:
    """Base adapter enforcing the fit→transform leakage barrier.

    Subclasses implement :meth:`_fit` (no access to the eval object) and
    :meth:`_transform`.  Unsupervised methods (``requires_fit=False``) may have a
    no-op ``_fit`` and do all work in ``_transform``.
    """

    def __init__(self, meta: ComparatorMeta, **params: Any) -> None:
        self.meta = meta
        self.params = params

    # -- public API --------------------------------------------------------
    def fit(self, train: Any, ctx: dict | None = None) -> _FitState:
        payload = self._fit(train, ctx or {})
        return _FitState(payload)

    def transform(
        self, evaluation: Any, state: _FitState, ctx: dict | None = None
    ) -> ComparatorResult:
        if not isinstance(state, _FitState) or not state.fitted:
            raise RuntimeError(
                "transform() requires a state object returned by fit()"
            )
        t0, c0 = time.perf_counter(), time.process_time()
        try:
            res = self._transform(evaluation, state.payload, ctx or {})
        except Exception as exc:  # noqa: BLE001 - report, don't crash the sweep
            return ComparatorResult(
                status="failed_numerical",
                error=f"{type(exc).__name__}: {exc}",
                parameters=dict(self.params),
            )
        res.runtime_seconds = round(time.perf_counter() - t0, 6)
        res.cpu_seconds = round(time.process_time() - c0, 6)
        res.parameters = {**dict(self.params), **(res.parameters or {})}
        return res

    # -- to override -------------------------------------------------------
    def _fit(self, train: Any, ctx: dict) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def _transform(self, evaluation: Any, payload: Any, ctx: dict) -> ComparatorResult:  # pragma: no cover
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Registry  (stable ID -> factory)
# ---------------------------------------------------------------------------
_REGISTRY: dict[str, Callable[..., Comparator]] = {}


def register(comparator_id: str, factory: Callable[..., Comparator]) -> None:
    if comparator_id in _REGISTRY:
        raise ValueError(f"comparator id {comparator_id!r} already registered")
    _REGISTRY[comparator_id] = factory


def get(comparator_id: str, **params: Any) -> Comparator:
    if comparator_id not in _REGISTRY:
        raise KeyError(
            f"unknown comparator {comparator_id!r}; available: {available()}"
        )
    return _REGISTRY[comparator_id](**params)


def available() -> list[str]:
    return sorted(_REGISTRY)


# ---------------------------------------------------------------------------
# Trivial comparators to exercise the contract (not the science set)
# ---------------------------------------------------------------------------
def _as_array(obj: Any) -> np.ndarray:
    if hasattr(obj, "get_data"):
        return np.asarray(obj.get_data())
    return np.asarray(obj)


class NoCorrection(Comparator):
    """``none`` baseline — identity passthrough."""

    def __init__(self, **params: Any) -> None:
        super().__init__(ComparatorMeta("none", requires_fit=False), **params)

    def _fit(self, train, ctx):
        return None

    def _transform(self, evaluation, payload, ctx):
        x = _as_array(evaluation)
        # collapse epoched data (n_epochs, n_ch, n_times) to a 2-D channel view for rank
        x2 = x.transpose(1, 0, 2).reshape(x.shape[1], -1) if x.ndim == 3 else x
        rank = int(np.linalg.matrix_rank(x2)) if x2.ndim == 2 else None
        return ComparatorResult(
            cleaned=evaluation,
            status="success",
            rank_before=rank,
            rank_after=rank,
            n_samples_before=int(x.shape[-1]),
            n_samples_after=int(x.shape[-1]),
        )


class PCAReconstruct(Comparator):
    """``pca_reconstruct`` — rank-matched PCA fitted on train, applied to eval.

    Data are ``(n_channels, n_times)``; PCA is taken across channels.  Exercises
    the leakage barrier (components learned on *train* only) and rank reporting.
    """

    def __init__(self, n_components: int = 5, **params: Any) -> None:
        super().__init__(
            ComparatorMeta("pca_reconstruct", rank_reducing=True),
            n_components=n_components, **params,
        )
        self.n_components = int(n_components)

    def _fit(self, train, ctx):
        from sklearn.decomposition import PCA

        x = _as_array(train)                 # (n_channels, n_times)
        k = min(self.n_components, x.shape[0])
        pca = PCA(n_components=k, svd_solver="full")
        pca.fit(x.T)                         # samples x channels
        return pca

    def _transform(self, evaluation, payload, ctx):
        pca = payload
        x = _as_array(evaluation)
        recon = pca.inverse_transform(pca.transform(x.T)).T
        return ComparatorResult(
            cleaned=recon,
            model=pca,
            status="success",
            rank_before=int(np.linalg.matrix_rank(x)),
            rank_after=int(pca.n_components_),
            n_samples_before=int(x.shape[-1]),
            n_samples_after=int(recon.shape[-1]),
            parameters={"n_components": int(pca.n_components_)},
        )


register("none", NoCorrection)
register("pca_reconstruct", PCAReconstruct)
