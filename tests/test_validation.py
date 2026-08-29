"""Tests for the shared input validation helpers in mne_denoise._validation."""

from __future__ import annotations

import numpy as np
import pytest

from mne_denoise._validation import (
    check_channel_first_data,
    check_channel_layout,
    check_chunk_size,
    check_matching_sfreq,
    check_option,
    check_positive_integer,
    check_positive_real,
    resolve_sample_window,
    resolve_sfreq,
)


@pytest.fixture()
def rng():
    """Shared random generator."""
    return np.random.default_rng(0)


def test_positive_integer_contract():
    """Positive integer validation normalizes valid values and rejects the rest."""
    assert check_positive_integer(1, name="count") == 1
    value = check_positive_integer(np.int64(2), name="count")
    assert value == 2
    assert isinstance(value, int)

    for label, value, error in (
        ("bool", True, TypeError),
        ("float", 1.5, TypeError),
        ("zero", 0, ValueError),
        ("negative", -1, ValueError),
    ):
        with pytest.raises(error, match="count must be a positive integer"):
            check_positive_integer(value, name="count")


def test_positive_real_contract():
    """Positive real validation returns floats and rejects invalid values."""
    for value, expected in [(1, 1.0), (1.5, 1.5), (np.float64(0.5), 0.5)]:
        result = check_positive_real(value, name="width")
        assert result == expected
        assert isinstance(result, float)

    for label, value in (("bool", True), ("string", "1.0"), ("none", None)):
        with pytest.raises(TypeError, match="width must be a positive, finite number"):
            check_positive_real(value, name="width")
    for label, value in (
        ("zero", 0),
        ("negative", -1),
        ("nan", np.nan),
        ("positive infinity", np.inf),
        ("negative infinity", -np.inf),
    ):
        with pytest.raises(ValueError, match="width must be a positive, finite number"):
            check_positive_real(value, name="width")


def test_option_contract():
    """Allowed options pass unchanged and invalid options identify their context."""
    value = "auto"
    assert check_option(value, name="blend", allowed=("auto", "constant")) is value

    with pytest.raises(ValueError) as exc_info:
        check_option("invalid", name="blend", allowed=("auto", "constant"))
    message = str(exc_info.value)
    assert all(part in message for part in ("blend", "auto", "constant"))
    assert "received value" in message


def test_matching_sfreq_contract():
    """Sampling-frequency matches tolerate metadata absence and honor tolerance."""
    check_matching_sfreq(250.0, 250.0, name="X")
    check_matching_sfreq(250.0005, 250.0, name="X")
    check_matching_sfreq(None, 250.0, name="X")
    check_matching_sfreq(250.0, None, name="X")
    check_matching_sfreq(100.0 + 5e-13, 100.0, name="X", rtol=0.0, atol=1e-12)

    with pytest.raises(ValueError, match="X.*transform sfreq=251.*fitted sfreq=250"):
        check_matching_sfreq(251.0, 250.0, name="X")
    with pytest.raises(ValueError, match="transform sfreq"):
        check_matching_sfreq(100.0 + 2e-12, 100.0, name="X", rtol=0.0, atol=1e-12)


def test_channel_first_data_contract(rng):
    """Channel-first validation covers supported layouts and meaningful guards."""
    continuous = check_channel_first_data(rng.standard_normal((4, 100)), name="X")
    epoched = check_channel_first_data(rng.standard_normal((3, 4, 100)), name="X")
    assert continuous.dtype == np.float64
    assert epoched.shape == (3, 4, 100)
    assert check_channel_first_data(np.ones((3, 10), dtype=int), name="X").dtype == (
        np.float64
    )

    with pytest.raises(ValueError, match="Expected a 2-D channel-first array"):
        check_channel_first_data(
            rng.standard_normal((3, 4, 100)), name="X", allow_epochs=False
        )
    for label, data, message in (
        ("one-dimensional", np.ones(10), "2-D or 3-D"),
        ("four-dimensional", np.ones((2, 2, 2, 2)), "2-D or 3-D"),
        ("one channel", np.ones((1, 100)), "at least two channels"),
        ("one sample", np.ones((4, 1)), "at least two time samples"),
        ("nan", np.full((4, 100), np.nan), "finite"),
        ("infinity", np.full((4, 100), np.inf), "finite"),
        ("empty epochs", np.ones((0, 4, 100)), "at least one epoch"),
    ):
        with pytest.raises(ValueError, match=message):
            check_channel_first_data(data, name=f"X ({label})")

    data = rng.standard_normal((3, 100))
    check_channel_first_data(data, name="X", min_channels=3)
    with pytest.raises(ValueError, match="at least two channels"):
        check_channel_first_data(data, name="X", min_channels=4)
    for algorithm in ("SNS", "BSS-CCA"):
        with pytest.raises(
            ValueError, match=f"{algorithm} requires at least two channels"
        ):
            check_channel_first_data(np.ones((1, 100)), name=algorithm)


def test_sample_window_contract():
    """Sample windows share one half-open conversion and validation contract."""
    assert resolve_sample_window((-2, 3), unit="samples") == (-2, 3)
    assert resolve_sample_window((-0.015, 0.025), unit="seconds", sfreq=100.0) == (
        -2,
        2,
    )

    for label, sfreq in (
        ("zero", 0.0),
        ("negative", -1.0),
        ("nan", np.nan),
        ("infinity", np.inf),
    ):
        with pytest.raises(ValueError, match="sfreq must be a positive, finite number"):
            resolve_sample_window((0, 1), unit="samples", sfreq=sfreq)
    for label, window, unit, sfreq, match in (
        ("fractional samples", (-1.5, 2), "samples", None, "must be integers"),
        ("reversed", (2, -1), "samples", None, "strictly less"),
        ("empty seconds", (0, 0.001), "seconds", 100.0, "empty or reversed"),
        ("missing sfreq", (-1, 2), "seconds", None, "sfreq is required"),
        ("invalid unit", (-1, 2), "minutes", None, "window_unit"),
        ("integer overflow", (-(2**63) - 1, 2), "samples", None, "signed 64-bit"),
        ("finite range", (-1e308, 1e308), "seconds", 1e308, "finite sample range"),
    ):
        with pytest.raises((TypeError, ValueError), match=match):
            resolve_sample_window(window, unit=unit, sfreq=sfreq)


def test_chunk_size_contract():
    """Chunk sizes accept an unlimited mode or positive integer values only."""
    assert check_chunk_size(None) is None
    assert check_chunk_size(np.int64(64)) == 64
    assert isinstance(check_chunk_size(64), int)
    for label, value, error in (
        ("bool", True, TypeError),
        ("float", 1.5, TypeError),
        ("string", "64", TypeError),
        ("zero", 0, ValueError),
        ("negative", -1, ValueError),
    ):
        with pytest.raises(
            error, match="chunk_size must be a positive integer or None"
        ):
            check_chunk_size(value)


def test_resolve_sfreq_contract():
    """Sampling-frequency resolution covers every source and error state."""
    assert resolve_sfreq(250.0, None) == 250.0
    assert resolve_sfreq(None, 250.0) == 250.0
    assert resolve_sfreq(250.0, 250.0) == 250.0
    with pytest.raises(ValueError, match="disagrees with MNE info sfreq"):
        resolve_sfreq(100.0, 250.0)
    with pytest.raises(ValueError, match="sfreq is required when lag_seconds is used"):
        resolve_sfreq(None, None, context="lag_seconds")
    with pytest.raises(ValueError, match="^sfreq is required$"):
        resolve_sfreq(None, None)
    assert resolve_sfreq(None, None, required=False) is None

    invalid = [-1.0, np.nan, np.inf, True, "250"]
    for value in invalid:
        error = TypeError if isinstance(value, (bool, str)) else ValueError
        with pytest.raises(error, match="sfreq must be a positive, finite number"):
            resolve_sfreq(value, None)
        with pytest.raises(error, match="sfreq must be a positive, finite number"):
            resolve_sfreq(None, value)


def test_channel_layout_contract():
    """Layout validation covers matching arrays, named order, and count-only paths."""
    check_channel_layout(
        "X",
        n_channels=2,
        fitted_n_channels=2,
        ch_names=("a", "b"),
        fitted_ch_names=("a", "b"),
    )
    with pytest.raises(ValueError, match="names/order differ from fit"):
        check_channel_layout(
            "SNS",
            n_channels=2,
            fitted_n_channels=2,
            ch_names=("b", "a"),
            fitted_ch_names=("a", "b"),
        )
    with pytest.raises(ValueError, match="X: X has 3 channels; fitted data had 2"):
        check_channel_layout("X", n_channels=3, fitted_n_channels=2)
    check_channel_layout("X", n_channels=2, fitted_n_channels=2)
    check_channel_layout(
        "X",
        n_channels=2,
        fitted_n_channels=2,
        ch_names=("a", "b"),
        fitted_ch_names=None,
    )
    check_channel_layout(
        "X",
        n_channels=2,
        fitted_n_channels=2,
        ch_names=None,
        fitted_ch_names=("a", "b"),
    )
