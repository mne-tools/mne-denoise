"""Tests for the scoped verbose-to-logging contract."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from mne_denoise._logging import (
    _UNSET,
    _active_verbose_scope,
    _level_from_verbose,
    logger,
    use_log_level,
    verbose,
)
from mne_denoise.asr import ASR

SFREQ = 200.0


def _data(seed: int = 0) -> np.ndarray:
    """Small synthetic multichannel array for fit_transform."""
    return np.random.default_rng(seed).standard_normal((6, 2000))


def test_level_from_verbose_values():
    """MNE-style values resolve to standard logging levels."""
    assert _level_from_verbose(True) == logging.INFO
    assert _level_from_verbose(False) == logging.WARNING
    assert _level_from_verbose("debug") == logging.DEBUG
    assert _level_from_verbose(logging.ERROR) == logging.ERROR


def test_level_from_verbose_invalid_values():
    """Invalid values fail with useful exception types."""
    with pytest.raises(ValueError, match="Unknown logging level"):
        _level_from_verbose("not-a-level")
    with pytest.raises(TypeError, match="verbose must be"):
        _level_from_verbose(object())


def test_use_log_level_none_inherits_without_mutating_logger():
    """None is an active inheritance scope but does not change the level."""
    previous = logger.level
    assert _active_verbose_scope.get() is _UNSET
    with use_log_level(None):
        assert logger.level == previous
        assert _active_verbose_scope.get() is None
    assert logger.level == previous
    assert _active_verbose_scope.get() is _UNSET


def test_use_log_level_restores_after_success_and_exception():
    """Concrete scopes restore the logger even when their body raises."""
    previous = logger.level
    try:
        with use_log_level("DEBUG"):
            assert logger.level == logging.DEBUG
        assert logger.level == previous

        with pytest.raises(RuntimeError, match="expected"):
            with use_log_level(True):
                assert logger.level == logging.INFO
                raise RuntimeError("expected")
        assert logger.level == previous
    finally:
        logger.setLevel(previous)


def test_nested_explicit_scopes_restore_in_order():
    """An explicit inner scope temporarily supersedes an outer scope."""
    previous = logger.level
    with use_log_level("DEBUG"):
        assert logger.level == logging.DEBUG
        with use_log_level(False):
            assert logger.level == logging.WARNING
        assert logger.level == logging.DEBUG
    assert logger.level == previous


def test_decorated_nested_call_inherits_outer_scope():
    """An omitted inner override inherits the active outer scope."""

    @verbose
    def inner(*, verbose=None):
        return logger.level

    @verbose
    def outer(*, verbose=None):
        return inner()

    assert outer(verbose="DEBUG") == logging.DEBUG


def test_decorated_nested_explicit_override_restores_outer_scope():
    """An explicit inner override returns control to its outer scope."""

    @verbose
    def inner(*, verbose=None):
        return logger.level

    @verbose
    def outer(*, verbose=None):
        inner_level = inner(verbose=False)
        return inner_level, logger.level

    assert outer(verbose="DEBUG") == (logging.WARNING, logging.DEBUG)


def test_estimator_fallback_only_without_explicit_or_outer_scope():
    """Constructor verbosity is the last internal fallback."""

    class Estimator:
        def __init__(self, value):
            self.verbose = value

        @verbose
        def operation(self, *, verbose=None):
            return logger.level

    previous = logger.level
    try:
        estimator = Estimator(True)
        assert estimator.operation() == logging.INFO
        assert estimator.operation(verbose=False) == logging.WARNING

        estimator.verbose = False
        with use_log_level("DEBUG"):
            assert estimator.operation() == logging.DEBUG
    finally:
        logger.setLevel(previous)


def test_standalone_explicit_verbose_is_resolved():
    """Standalone decorated functions honor their explicit override."""

    @verbose
    def operation(*, verbose=None):
        return logger.level

    assert operation(verbose="debug") == logging.DEBUG


def test_explicit_none_differs_from_omitted_verbose():
    """Explicit None inherits external configuration instead of self.verbose."""

    class Estimator:
        def __init__(self):
            self.verbose = True

        @verbose
        def operation(self, *, verbose=None):
            return logger.level

    previous = logger.level
    try:
        logger.setLevel(logging.ERROR)
        estimator = Estimator()
        assert estimator.operation(verbose=None) == logging.ERROR
        assert estimator.operation() == logging.INFO
    finally:
        logger.setLevel(previous)


def test_unsupported_verbose_is_not_swallowed():
    """Decorating a function does not silently expand its signature."""

    @verbose
    def operation():
        return None

    with pytest.raises(TypeError):
        operation(verbose=True)


def test_verbose_is_not_top_level_public_api():
    """The shared logging decorator remains package-internal."""
    import mne_denoise

    assert "verbose" not in mne_denoise.__all__
    assert not hasattr(mne_denoise, "verbose")


def test_asr_call_level_debug_overrides_constructor_false(caplog):
    """A fit_transform call-level DEBUG controls nested fit and transform."""
    estimator = ASR(
        sfreq=SFREQ,
        cutoff=20.0,
        picks=None,
        calibration="manual",
        filter_kind="none",
        verbose=False,
    )
    with caplog.at_level(logging.DEBUG, logger="mne_denoise"):
        estimator.fit_transform(_data(), verbose="DEBUG")

    package_records = [
        record for record in caplog.records if record.name == "mne_denoise"
    ]
    assert any(record.levelno == logging.DEBUG for record in package_records)
    calibration = next(
        record for record in package_records if "calibrated:" in record.message
    )
    assert any(record.levelno == logging.INFO for record in package_records)
    for token in (
        "method=",
        "channels=",
        "sfreq=",
        "cutoff=",
        "rank=",
        "clean calibration windows=",
    ):
        assert token in calibration.message


def test_asr_call_level_false_overrides_constructor_true(caplog):
    """A fit_transform call-level False suppresses nested normal reports."""
    estimator = ASR(
        sfreq=SFREQ,
        cutoff=20.0,
        picks=None,
        calibration="manual",
        filter_kind="none",
        verbose=True,
    )
    with caplog.at_level(logging.DEBUG, logger="mne_denoise"):
        estimator.fit_transform(_data(), verbose=False)

    assert not any(
        record.name == "mne_denoise" and record.levelno < logging.WARNING
        for record in caplog.records
    )


def test_asr_explicit_none_inherits_external_error_level(caplog):
    """An explicit None prevents constructor verbosity from taking over."""
    estimator = ASR(
        sfreq=SFREQ,
        cutoff=20.0,
        picks=None,
        calibration="manual",
        filter_kind="none",
        verbose=True,
    )
    previous = logger.level
    try:
        logger.setLevel(logging.ERROR)
        with caplog.at_level(logging.DEBUG):
            estimator.fit(_data(), verbose=None)
        assert logger.level == logging.ERROR
        assert not any(
            record.name == "mne_denoise" and record.levelno < logging.ERROR
            for record in caplog.records
        )
    finally:
        logger.setLevel(previous)


def test_callback_progress_is_independent_of_silent_logging(caplog):
    """A silent ASR call still delivers its structured progress events."""
    estimator = ASR(
        sfreq=SFREQ,
        cutoff=20.0,
        picks=None,
        calibration="manual",
        filter_kind="none",
        verbose=False,
    )
    events = []
    with caplog.at_level(logging.DEBUG, logger="mne_denoise"):
        estimator.fit_transform(_data(seed=11), callback=events.append, verbose=False)

    assert events
    assert not any(
        record.name == "mne_denoise" and record.levelno < logging.WARNING
        for record in caplog.records
    )


def test_verbose_does_not_change_asr_progress_or_output(caplog):
    """Changing logging verbosity does not change events or denoising."""
    quiet_events = []
    debug_events = []
    quiet_data = _data(seed=12)
    debug_data = _data(seed=12)
    kwargs = {
        "sfreq": SFREQ,
        "cutoff": 20.0,
        "picks": None,
        "calibration": "manual",
        "filter_kind": "none",
        "verbose": False,
    }

    with caplog.at_level(logging.DEBUG, logger="mne_denoise"):
        quiet = ASR(**kwargs).fit_transform(
            quiet_data, callback=quiet_events.append, verbose=False
        )
        caplog.clear()
        debug = ASR(**kwargs).fit_transform(
            debug_data, callback=debug_events.append, verbose="DEBUG"
        )

    assert quiet_events == debug_events
    np.testing.assert_allclose(quiet, debug)
    assert any(
        record.name == "mne_denoise" and record.levelno == logging.DEBUG
        for record in caplog.records
    )


def test_callback_exception_restores_logging_scope():
    """Callback exceptions propagate while the verbose scope is restored."""

    class SentinelError(RuntimeError):
        pass

    sentinel = SentinelError("stop progress")

    def fail(_event):
        raise sentinel

    previous_level = logger.level
    previous_scope = _active_verbose_scope.get()
    try:
        estimator = ASR(
            sfreq=SFREQ,
            cutoff=20.0,
            picks=None,
            calibration="manual",
            filter_kind="none",
            verbose=False,
        )
        with pytest.raises(SentinelError) as exc_info:
            estimator.fit_transform(_data(seed=13), callback=fail, verbose="DEBUG")

        assert exc_info.value is sentinel
        assert logger.level == previous_level
        assert _active_verbose_scope.get() is previous_scope
    finally:
        logger.setLevel(previous_level)
