"""Tests for the structured progress event contract."""

import pytest

from mne_denoise.progress import ProgressEvent


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
