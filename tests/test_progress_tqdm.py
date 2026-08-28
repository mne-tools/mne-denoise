"""Tests for the optional tqdm progress adapter."""

from __future__ import annotations

import logging

import numpy as np
import pytest

import mne_denoise.progress as progress_module
from mne_denoise._logging import logger
from mne_denoise.progress import ProgressEvent, TqdmProgress
from mne_denoise.sns import compute_sns_weights


class _FakeBar:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.n = kwargs["initial"]
        self.closed = False
        self.close_calls = 0
        self.updates = []

    def update(self, amount=1):
        assert amount >= 0
        self.updates.append(amount)
        self.n += amount

    def close(self):
        self.close_calls += 1
        self.closed = True


class _FakeTqdm:
    def __init__(self):
        self.bars = []

    def __call__(self, **kwargs):
        bar = _FakeBar(**kwargs)
        self.bars.append(bar)
        return bar


@pytest.fixture
def fake_tqdm(monkeypatch):
    """Use deterministic bars without terminal output."""
    factory = _FakeTqdm()
    monkeypatch.setattr(progress_module, "_load_tqdm", lambda: factory)
    return factory


def _event(method, stage, current=None, total=None, component=None):
    return ProgressEvent(
        method=method,
        stage=stage,
        current=current,
        total=total,
        component=component,
    )


def test_tqdm_progress_updates_one_bar(fake_tqdm):
    """A monotonic stream creates one bar and reaches its explicit count."""
    progress = TqdmProgress()
    progress(_event("sound", "iteration", current=1, total=3))
    progress(_event("sound", "iteration", current=2, total=3))

    assert len(fake_tqdm.bars) == 1
    assert fake_tqdm.bars[0].n == 2
    assert fake_tqdm.bars[0].kwargs["total"] == 3
    assert fake_tqdm.bars[0].kwargs["desc"] == "sound: iteration"


def test_tqdm_progress_stage_transition_closes_previous_bar(fake_tqdm):
    """A stage transition replaces the active bar."""
    progress = TqdmProgress()
    progress(_event("asr", "calibration", current=1, total=2))
    progress(_event("asr", "calibration", current=2, total=2))
    progress(_event("asr", "window", current=1, total=4))

    assert len(fake_tqdm.bars) == 2
    assert fake_tqdm.bars[0].closed
    assert fake_tqdm.bars[1].kwargs["total"] == 4
    assert fake_tqdm.bars[1].kwargs["desc"] == "asr: window"


def test_tqdm_progress_method_transition_starts_new_bar(fake_tqdm):
    """A method transition starts a new semantic stream."""
    progress = TqdmProgress()
    progress(_event("asr", "calibration", current=1, total=2))
    progress(_event("guided_asr", "window", current=1, total=4))

    assert len(fake_tqdm.bars) == 2
    assert fake_tqdm.bars[0].closed
    assert fake_tqdm.bars[1].kwargs["desc"] == "guided_asr: window"


def test_tqdm_progress_counter_reset_starts_new_bar(fake_tqdm):
    """A same-method/stage counter reset starts a fresh bar."""
    progress = TqdmProgress()
    progress(_event("iterative_dss", "iteration", 1, 5, component=1))
    progress(_event("iterative_dss", "iteration", 2, 5, component=1))
    progress(_event("iterative_dss", "iteration", 1, 5, component=2))

    assert len(fake_tqdm.bars) == 2
    assert fake_tqdm.bars[0].n == 2
    assert fake_tqdm.bars[0].closed
    assert fake_tqdm.bars[1].n == 1


def test_tqdm_progress_component_does_not_split_monotonic_stream(fake_tqdm):
    """Increasing component metadata does not split one calibration stream."""
    progress = TqdmProgress()
    for component, current in enumerate((1, 2, 3), start=1):
        progress(_event("asr", "calibration", current, 3, component))

    assert len(fake_tqdm.bars) == 1
    assert fake_tqdm.bars[0].n == 3


def test_tqdm_progress_flat_sns_stream(fake_tqdm):
    """A flat SNS channel stream remains one bar."""
    progress = TqdmProgress()
    for current in range(1, 7):
        progress(_event("sns", "channel", current, 6))

    assert len(fake_tqdm.bars) == 1
    assert fake_tqdm.bars[0].n == 6


def test_tqdm_progress_unknown_counter_and_total(fake_tqdm):
    """Unknown counters advance by one and unknown totals remain unknown."""
    progress = TqdmProgress()
    for _ in range(3):
        progress(_event("custom", "work"))

    assert len(fake_tqdm.bars) == 1
    assert fake_tqdm.bars[0].kwargs["total"] is None
    assert fake_tqdm.bars[0].n == 3


def test_tqdm_progress_context_cleanup_and_idempotent_close(fake_tqdm):
    """Context exit closes the bar, and repeated close calls are safe."""
    with TqdmProgress() as progress:
        progress(_event("sns", "channel", 1, 1))
        bar = fake_tqdm.bars[0]
        assert not bar.closed

    assert bar.closed
    progress.close()
    progress.close()
    assert bar.close_calls == 1


def test_tqdm_progress_does_not_suppress_exceptions(fake_tqdm):
    """Context cleanup preserves the exact exception raised by the caller."""

    class SentinelError(Exception):
        pass

    error = SentinelError("sentinel")
    with pytest.raises(SentinelError) as caught:
        with TqdmProgress() as progress:
            progress(_event("sns", "channel", 1, 1))
            raise error

    assert caught.value is error
    assert fake_tqdm.bars[0].closed


def test_tqdm_progress_is_independent_of_package_logging(fake_tqdm, caplog):
    """Rendering events does not change or emit through package logging."""
    previous_level = logger.level
    with caplog.at_level(logging.DEBUG, logger="mne_denoise"):
        progress = TqdmProgress()
        progress(_event("sound", "iteration", 1, 2))
        progress(_event("sound", "iteration", 2, 2))
        progress(_event("sns", "channel", 1, 1))

    assert logger.level == previous_level
    assert not [record for record in caplog.records if record.name == "mne_denoise"]


def test_tqdm_progress_rejects_adapter_owned_kwargs(fake_tqdm):
    """Users cannot override event-owned tqdm state."""
    with pytest.raises(TypeError, match="controls tqdm's total and initial"):
        TqdmProgress(total=10)
    with pytest.raises(TypeError, match="controls tqdm's total and initial"):
        TqdmProgress(initial=10)


def test_compute_sns_weights_with_tqdm_progress():
    """A real callback-aware operation works with the adapter."""
    cov = np.eye(3)
    with TqdmProgress(disable=True) as progress:
        weights, n_neighbors, ranks = compute_sns_weights(
            cov,
            callback=progress,
        )

    assert weights.shape == cov.shape
    assert np.isfinite(weights).all()
    assert n_neighbors == 2
    assert ranks.shape == (3,)
