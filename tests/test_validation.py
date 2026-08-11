"""Tests for the shared input validation helpers in mne_denoise._validation."""

from __future__ import annotations

import numpy as np
import pytest

from mne_denoise._validation import (
    check_channel_first_data,
    check_channel_layout,
    check_chunk_size,
    check_sfreq,
    resolve_sfreq,
)


@pytest.fixture()
def rng():
    """Shared random generator."""
    return np.random.default_rng(0)


# ---------------------------------------------------------------------------
# check_channel_first_data
# ---------------------------------------------------------------------------


def test_accepts_continuous_and_epoched(rng):
    """Both supported layouts pass and are returned as float64."""
    continuous = check_channel_first_data(rng.standard_normal((4, 100)), name="X")
    epoched = check_channel_first_data(rng.standard_normal((3, 4, 100)), name="X")
    assert continuous.dtype == np.float64
    assert epoched.shape == (3, 4, 100)


def test_converts_integer_input():
    """Integer input is converted rather than rejected."""
    out = check_channel_first_data(np.ones((3, 10), dtype=int), name="X")
    assert out.dtype == np.float64


def test_epochs_can_be_disallowed(rng):
    """allow_epochs=False rejects three-dimensional input."""
    with pytest.raises(ValueError, match="Expected a 2-D channel-first array"):
        check_channel_first_data(
            rng.standard_normal((3, 4, 100)), name="X", allow_epochs=False
        )


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (np.ones(10), "2-D or 3-D"),
        (np.ones((2, 2, 2, 2)), "2-D or 3-D"),
        (np.ones((1, 100)), "at least two channels"),
        (np.ones((4, 1)), "at least two time samples"),
        (np.full((4, 100), np.nan), "finite"),
        (np.full((4, 100), np.inf), "finite"),
    ],
)
def test_rejects_invalid_data(data, message):
    """Shape, size, and finiteness preconditions are enforced."""
    with pytest.raises(ValueError, match=message):
        check_channel_first_data(data, name="X")


def test_error_messages_name_the_algorithm():
    """The caller's name appears in the message, not a hard-coded one."""
    with pytest.raises(ValueError, match="SNS requires at least two channels"):
        check_channel_first_data(np.ones((1, 100)), name="SNS")
    with pytest.raises(ValueError, match="BSS-CCA requires at least two channels"):
        check_channel_first_data(np.ones((1, 100)), name="BSS-CCA")


def test_minimum_sizes_are_configurable(rng):
    """Callers can demand more channels or samples than the default."""
    data = rng.standard_normal((3, 100))
    check_channel_first_data(data, name="X", min_channels=3)
    with pytest.raises(ValueError, match="at least two channels"):
        check_channel_first_data(data, name="X", min_channels=4)


def test_empty_epoch_axis_is_rejected():
    """A zero-epoch array carries no data."""
    with pytest.raises(ValueError, match="at least one epoch"):
        check_channel_first_data(np.ones((0, 4, 100)), name="X")


# ---------------------------------------------------------------------------
# check_sfreq
# ---------------------------------------------------------------------------


def test_returns_float():
    """Valid values are returned as plain floats."""
    assert check_sfreq(250) == 250.0
    assert isinstance(check_sfreq(np.float64(250.0)), float)
    assert check_sfreq(np.int64(500)) == 500.0


def test_missing_sfreq_reports_the_context():
    """The message explains what needed the value."""
    with pytest.raises(ValueError, match="sfreq is required when lag_seconds is used"):
        check_sfreq(None, context="lag_seconds")
    with pytest.raises(ValueError, match="^sfreq is required$"):
        check_sfreq(None)


@pytest.mark.parametrize("value", [True, False, "250", None])
def test_rejects_non_numeric(value):
    """Booleans and non-numbers are not sampling frequencies."""
    with pytest.raises((TypeError, ValueError)):
        check_sfreq(value)


@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_rejects_non_positive_or_non_finite(value):
    """A sampling frequency must be finite and strictly positive."""
    with pytest.raises(ValueError, match="sfreq must be a positive, finite number"):
        check_sfreq(value)


# ---------------------------------------------------------------------------
# check_chunk_size
# ---------------------------------------------------------------------------


def test_chunk_size_accepts_none_and_integers():
    """None means 'all at once'; integers are normalized."""
    assert check_chunk_size(None) is None
    assert check_chunk_size(np.int64(64)) == 64
    assert isinstance(check_chunk_size(64), int)


@pytest.mark.parametrize(
    ("value", "error"),
    [
        (True, TypeError),
        (1.5, TypeError),
        ("64", TypeError),
        (0, ValueError),
        (-1, ValueError),
    ],
)
def test_chunk_size_rejects_invalid(value, error):
    """Booleans, non-integers, and non-positive values are rejected."""
    with pytest.raises(error, match="chunk_size must be a positive integer or None"):
        check_chunk_size(value)


# ---------------------------------------------------------------------------
# resolve_sfreq
# ---------------------------------------------------------------------------


def test_resolve_prefers_the_container_when_they_agree():
    """Agreeing sources collapse to one value."""
    assert resolve_sfreq(250.0, 250.0) == 250.0
    assert resolve_sfreq(None, 250.0) == 250.0
    assert resolve_sfreq(250.0, None) == 250.0


def test_resolve_rejects_disagreement():
    """A declared value is never silently discarded."""
    with pytest.raises(ValueError, match="disagrees with MNE info sfreq"):
        resolve_sfreq(100.0, 250.0)


def test_resolve_reports_a_missing_value():
    """With nothing to go on, the caller learns what needed it."""
    with pytest.raises(ValueError, match="sfreq is required when lag_seconds is used"):
        resolve_sfreq(None, None, context="lag_seconds")


def test_resolve_can_allow_a_missing_value():
    """Optional sampling frequencies return None rather than raising."""
    assert resolve_sfreq(None, None, required=False) is None


def test_resolve_still_validates():
    """A single bad value is rejected as by check_sfreq."""
    with pytest.raises(ValueError, match="positive, finite"):
        resolve_sfreq(-1.0, None)


# ---------------------------------------------------------------------------
# check_channel_layout
# ---------------------------------------------------------------------------


def test_channel_layout_accepts_a_match():
    """Identical names and counts pass."""
    check_channel_layout(
        "X",
        n_channels=2,
        fitted_n_channels=2,
        ch_names=("a", "b"),
        fitted_ch_names=("a", "b"),
    )


def test_channel_layout_rejects_reordering():
    """Order is part of the layout, not just membership."""
    with pytest.raises(ValueError, match="names/order differ from fit"):
        check_channel_layout(
            "SNS",
            n_channels=2,
            fitted_n_channels=2,
            ch_names=("b", "a"),
            fitted_ch_names=("a", "b"),
        )


def test_channel_layout_rejects_a_count_mismatch():
    """Array input has no names, so the count is the only check."""
    with pytest.raises(ValueError, match="X has 3 channels; fitted data had 2"):
        check_channel_layout("X", n_channels=3, fitted_n_channels=2)


def test_channel_layout_skips_names_for_arrays():
    """A fitted-on-array estimator does not demand names."""
    check_channel_layout(
        "X",
        n_channels=2,
        fitted_n_channels=2,
        ch_names=("a", "b"),
        fitted_ch_names=None,
    )
