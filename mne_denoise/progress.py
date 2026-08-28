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
from typing import Any, cast

__all__ = ["ProgressEvent", "TqdmProgress"]


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


def _load_tqdm() -> Any:
    """Load the optional tqdm factory on demand."""
    try:
        from tqdm.auto import tqdm
    except ImportError as error:
        raise ImportError(
            "TqdmProgress requires the optional 'tqdm' dependency. "
            'Install it with `pip install "mne-denoise[progress]"`.'
        ) from error
    return tqdm


class TqdmProgress:
    """Render structured progress events with one optional tqdm bar.

    Parameters
    ----------
    leave : bool
        Whether completed bars should remain visible. Defaults to ``False``.
    **tqdm_kwargs
        Additional keyword arguments passed to tqdm. The adapter controls
        ``total`` and ``initial`` from the first event of each stream.
    """

    def __init__(self, *, leave: bool = False, **tqdm_kwargs: Any) -> None:
        if "total" in tqdm_kwargs or "initial" in tqdm_kwargs:
            raise TypeError("TqdmProgress controls tqdm's total and initial")

        self._tqdm = _load_tqdm()
        self._tqdm_kwargs = {**tqdm_kwargs, "leave": leave}
        self._bar: Any | None = None
        self._previous_event: ProgressEvent | None = None

    def __call__(self, event: ProgressEvent) -> None:
        """Render one completed-work progress event."""
        if self._starts_new_stream(event):
            self.close()
            bar_kwargs = dict(self._tqdm_kwargs)
            bar_kwargs["total"] = event.total
            bar_kwargs["initial"] = event.current if event.current is not None else 0
            bar_kwargs.setdefault("desc", f"{event.method}: {event.stage}")
            self._bar = self._tqdm(**bar_kwargs)

        if event.current is None:
            self._bar.update(1)
        else:
            delta = event.current - self._bar.n
            if delta > 0:
                self._bar.update(delta)

        self._previous_event = event

    def _starts_new_stream(self, event: ProgressEvent) -> bool:
        """Return whether ``event`` starts a new semantic stream."""
        if self._bar is None or self._previous_event is None:
            return True

        previous = self._previous_event
        if event.method != previous.method or event.stage != previous.stage:
            return True

        return (
            event.current is not None
            and previous.current is not None
            and event.current <= previous.current
        )

    def __enter__(self) -> "TqdmProgress":
        """Return this callback for use in a context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        """Close the active bar without suppressing exceptions."""
        self.close()
        return False

    def close(self) -> None:
        """Close the active bar and clear its state."""
        if self._bar is not None:
            self._bar.close()
        self._bar = None
        self._previous_event = None


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
