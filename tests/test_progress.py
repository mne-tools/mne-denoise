"""Tests for the structured progress event contract."""

import pytest

from mne_denoise.progress import (
    ProgressEvent,
    _emit_progress,
    _validate_callback,
)


def test_progress_event_basic_fields():
    """ProgressEvent stores all supplied fields exactly."""
    event = ProgressEvent(
        method="sound",
        stage="iteration",
        current=2,
        total=5,
        component=None,
        metric=0.01,
    )

    assert event.method == "sound"
    assert event.stage == "iteration"
    assert event.current == 2
    assert event.total == 5
    assert event.component is None
    assert event.metric == 0.01


def test_progress_event_defaults():
    """Optional progress fields default to None."""
    event = ProgressEvent(method="sound", stage="iteration")

    assert event.current is None
    assert event.total is None
    assert event.component is None
    assert event.metric is None


def test_progress_event_is_immutable():
    """ProgressEvent fields cannot be assigned after construction."""
    event = ProgressEvent(method="sound", stage="iteration")

    with pytest.raises(AttributeError):
        event.current = 1


def test_validate_callback():
    """Callback validation accepts None and callable objects unchanged."""
    assert _validate_callback(None) is None

    callback = lambda event: None  # noqa: E731
    assert _validate_callback(callback) is callback


@pytest.mark.parametrize("callback", [1, "callback", object()])
def test_validate_callback_rejects_non_callables(callback):
    """Callback validation rejects non-callable values."""
    with pytest.raises(TypeError, match="callback must be callable or None"):
        _validate_callback(callback)


def test_emit_progress():
    """Emission creates one event with the supplied fields."""
    events = []

    _emit_progress(
        events.append,
        method="iterative_dss",
        stage="iteration",
        current=3,
        total=7,
        component=2,
        metric=0.125,
    )

    assert len(events) == 1
    assert isinstance(events[0], ProgressEvent)
    assert events[0].method == "iterative_dss"
    assert events[0].stage == "iteration"
    assert events[0].current == 3
    assert events[0].total == 7
    assert events[0].component == 2
    assert events[0].metric == 0.125


def test_emit_progress_with_none_callback():
    """Emission with no callback does nothing."""
    assert (
        _emit_progress(
            None,
            method="sound",
            stage="iteration",
            current=1,
        )
        is None
    )


def test_emit_progress_propagates_callback_exception():
    """Callback exceptions propagate unchanged."""

    class SentinelError(Exception):
        pass

    error = SentinelError("stop")

    def callback(event):
        raise error

    with pytest.raises(SentinelError) as caught:
        _emit_progress(callback, method="sound", stage="iteration")

    assert caught.value is error


def test_emit_progress_ignores_callback_return_value():
    """Callback return values do not affect successful emission."""
    events = []
    result = object()

    def callback(event):
        events.append(event)
        return result

    assert _emit_progress(callback, method="sound", stage="iteration") is None
    assert len(events) == 1
