"""Structured progress events and callback helpers."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

__all__ = ["ProgressEvent", "TqdmProgress"]


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """Immutable description of one completed progress unit.

    Parameters
    ----------
    method : str
        Algorithm identifier. The value is an open string.
    stage : str
        Semantic phase identifier. The value is an open string.
    current : int or None, default=None
        Completed-unit count, normally 1-based.
    total : int or None, default=None
        Planned unit count when known.
    component : int or None, default=None
        Optional 1-based component index.
    metric : float or None, default=None
        Optional method- and stage-specific diagnostic.

    Notes
    -----
    Events are emitted after work and its metric complete. Callback return values
    are ignored; callback exceptions propagate unchanged.
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
    """Presentation adapter that renders ProgressEvent objects with tqdm.

    Parameters
    ----------
    leave : bool, default=False
        Keep completed bars visible.
    **tqdm_kwargs : dict
        Additional keyword arguments passed to tqdm. total and initial are controlled
        by the adapter.
    """

    def __init__(self, *, leave: bool = False, **tqdm_kwargs: Any) -> None:
        if "total" in tqdm_kwargs or "initial" in tqdm_kwargs:
            raise TypeError("TqdmProgress controls tqdm's total and initial")

        self._tqdm = _load_tqdm()
        self._tqdm_kwargs = {**tqdm_kwargs, "leave": leave}
        self._bar: Any | None = None
        self._previous_event: ProgressEvent | None = None

    def __call__(self, event: ProgressEvent) -> None:
        """Render one completed progress event."""
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
        """Return this adapter as a callback."""
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        """Close the active bar and propagate exceptions."""
        self.close()
        return False

    def close(self) -> None:
        """Close the active progress bar and clear its state."""
        if self._bar is not None:
            self._bar.close()
        self._bar = None
        self._previous_event = None


_ProgressCallback = Callable[[ProgressEvent], object]


def _validate_callback(callback: object) -> _ProgressCallback | None:
    """Validate a callback without invoking it."""
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
    """Emit one ProgressEvent when callback is provided."""
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
