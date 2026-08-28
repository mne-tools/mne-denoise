"""Structured computational progress events and callback helpers.

The package-wide progress API is a synchronous observer interface. A callback
is a callable receiving one immutable :class:`ProgressEvent`; its return value
is ignored and an exception raised by the callback propagates unchanged.
Callbacks and logging are independent: ``verbose`` controls package logs,
whereas ``callback`` controls structured events. A callback is runtime observer
state, not an estimator hyperparameter, and public APIs expect it to be passed
by keyword.

Events normally report a 1-based count of completed work. ``total`` is the
known number of work units, or ``None`` only when that number is genuinely
unknown. The current method vocabulary includes ``sound``, ``iterative_dss``,
``asr``, ``guided_asr``, ``adaptive_asr``, ``dss``, ``zapline``,
``narrowband_scan``, ``bss_cca``, ``icanclean``, ``basic_ssa``, ``local_ssa``,
and ``sns``. Current stages include ``iteration``, ``calibration``,
``window``, ``epoch``, ``segment``, ``frequency``, ``block``, and ``channel``.
These are documented identifiers, not a closed enum; the contract remains open
to additional semantic strings.
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
        Stable string identifier for the algorithm emitting the event. Current
        identifiers include ``"sound"``, ``"iterative_dss"``, ``"asr"``,
        ``"guided_asr"``, ``"adaptive_asr"``, ``"dss"``, ``"zapline"``,
        ``"narrowband_scan"``, ``"bss_cca"``, ``"icanclean"``,
        ``"basic_ssa"``, ``"local_ssa"``, and ``"sns"``. This is an open
        string contract rather than a Python enum.
    stage : str
        Singular semantic phase within the algorithm. Current stages include
        ``"iteration"``, ``"calibration"``, ``"window"``, ``"epoch"``,
        ``"segment"``, ``"frequency"``, ``"block"``, and ``"channel"``.
        This is an open string contract rather than a Python enum.
    current : int | None
        Normally the 1-based count of the progress unit that has just
        completed. ``None`` is used only when a meaningful count is unavailable.
    total : int | None
        Total planned units when known. ``None`` is used only when the total is
        genuinely unknown. For an iterative algorithm with early convergence,
        this may be the maximum possible number of iterations rather than the
        number eventually run.
    component : int | None
        Optional 1-based component index for nested component-based
        algorithms, such as deflationary iterative DSS. ``None`` is used when
        component identity is not semantically useful.
    metric : float | None
        Optional scalar diagnostic associated with the event. Its
        interpretation is specific to the method and stage, such as a
        convergence change, threshold, rank, component count, or frequency.

    Notes
    -----
    Progress events represent completed work. Emitters report after an
    iteration, calibration unit, window, frequency, block, channel, or other
    meaningful unit completes and after its diagnostic metric is known. Thus,
    ``current=1`` means that one unit is complete, not that unit 1 is about to
    start.

    Logging and structured callbacks are independent. Setting ``verbose=True``
    does not enable callbacks, and providing a callback does not alter logging
    level or logging content. Callback return values are ignored and callback
    exceptions propagate unchanged. Callbacks are synchronous and should be
    supplied by keyword; they are runtime observer state, not estimator
    hyperparameters.
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
