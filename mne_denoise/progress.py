"""Structured computational progress events and callback helpers.

Structured callbacks provide machine-readable computational progress, while
logging and ``verbose`` provide human-readable diagnostics. The two mechanisms
are independent: setting ``verbose=True`` does not enable callbacks, and
providing a callback does not alter logging level or logging content.

Callbacks are synchronous and are invoked only when an emitter explicitly
emits an event. Callback exceptions propagate unchanged, and callback return
values are ignored.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

__all__ = ["ProgressEvent"]


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """Describe a completed unit of computational progress.

    Parameters
    ----------
    method : str
        Stable string identifier for the algorithm emitting the event. Future
        integrations may use identifiers such as ``"sound"``,
        ``"iterative_dss"``, ``"asr"``, ``"icanclean"``, ``"zapline"``, or
        ``"narrowband_scan"``.
    stage : str
        Semantic phase within the algorithm. Future integrations may use
        stages such as ``"iteration"``, ``"component"``, ``"window"``,
        ``"segment"``, or ``"frequency"``.
    current : int | None
        1-based count of the progress unit that has just completed.
    total : int | None
        Total planned units when known. For an iterative algorithm with early
        convergence, this may be the maximum possible number of iterations
        rather than the number eventually run.
    component : int | None
        Optional 1-based component index for nested component-based
        algorithms, such as deflationary iterative DSS. ``None`` is used when
        components are not meaningful.
    metric : float | None
        Optional scalar diagnostic associated with the event. Its
        interpretation is specific to the method and stage, such as a
        convergence change, relative sigma change, or frequency score.

    Notes
    -----
    Progress events represent completed work. Future algorithm integrations
    should normally emit an event after an iteration, window, frequency, or
    other unit completes and after its diagnostic metric is known. Thus,
    ``current=1`` means that one unit is complete, not that unit 1 is about to
    start.

    Logging and structured callbacks are independent. Setting ``verbose=True``
    does not enable callbacks, and providing a callback does not alter logging
    level or logging content.
    """

    method: str
    stage: str
    current: int | None = None
    total: int | None = None
    component: int | None = None
    metric: float | None = None


_ProgressCallback = Callable[[ProgressEvent], object]


def _validate_callback(callback: object) -> _ProgressCallback | None:
    """Validate a progress callback without inspecting or invoking it."""
    if callback is None:
        return None
    if not callable(callback):
        raise TypeError(
            f"callback must be callable or None, got {type(callback).__name__}."
        )
    return cast(_ProgressCallback, callback)


def _emit_progress(
    callback: _ProgressCallback | None,
    *,
    method: str,
    stage: str,
    current: int | None = None,
    total: int | None = None,
    component: int | None = None,
    metric: float | None = None,
) -> None:
    """Emit one structured progress event to ``callback`` when provided."""
    if callback is None:
        return
    callback(
        ProgressEvent(
            method=method,
            stage=stage,
            current=current,
            total=total,
            component=component,
            metric=metric,
        )
    )
