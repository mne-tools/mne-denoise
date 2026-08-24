"""Adversarial contract tests for :class:`CycleAverageBias`."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mne_denoise.dss.denoisers.artifact import CycleAverageBias


def _bias(events, window=(-2, 3), **kwargs):
    """Construct a concise data-relative sample-window bias."""
    return CycleAverageBias(
        event_samples=events,
        window=window,
        window_unit="samples",
        event_origin="data",
        **kwargs,
    )


def test_cycle_average_estimates_repeated_morphology_without_mutation():
    """The fixed-window estimate preserves shape and leaves input untouched."""
    rng = np.random.default_rng(42)
    events = np.arange(100, 900, 200)
    template = np.hanning(50)
    artifact = np.zeros(1_000)
    for event in events:
        artifact[event - 25 : event + 25] = template
    clean = np.outer([1.0, 0.5, -0.5], artifact)
    data = clean + rng.normal(0, 0.1, clean.shape)
    original = data.copy()

    observed = _bias(events, window=(-25, 25)).apply(data)

    mask = np.ones(data.shape[1], dtype=bool)
    for event in events:
        mask[event - 25 : event + 25] = False
    assert_allclose(observed[:, mask], 0)
    assert (
        np.corrcoef(observed[:, 75:125].ravel(), clean[:, 75:125].ravel())[0, 1] > 0.95
    )
    assert observed.shape == data.shape
    assert_array_equal(data, original)
    assert not np.shares_memory(observed, data)


def test_cycle_average_seconds_have_explicit_nearest_sample_contract():
    """Second boundaries are converted once with ties-to-even rounding."""
    bias = CycleAverageBias(
        event_samples=[20, 40],
        window=(-0.015, 0.025),
        window_unit="seconds",
        sfreq=100.0,
        event_origin="data",
    )

    assert bias.window_input == (-0.015, 0.025)
    assert bias.window == (-2, 2)


def test_cycle_average_epoch_coordinates_never_join_epochs():
    """Per-epoch extraction cannot borrow samples from a neighboring epoch."""
    data = np.zeros((1, 6, 2))
    data[0, :, 0] = np.arange(1.0, 7.0)
    data[0, :, 1] = np.arange(101.0, 107.0)
    bias = _bias([[0, 2], [1, 2]], window=(-1, 2))

    observed = bias.apply(data)

    expected_template = np.array([52.0, 53.0, 54.0])
    assert_allclose(observed[0, 1:4, 0], expected_template)
    assert_allclose(observed[0, 1:4, 1], expected_template)
    assert_allclose(observed[0, [0, 4, 5], :], 0)


def test_cycle_average_rejects_flat_coordinates_for_epochs():
    """The stale global-index API cannot silently flatten independent epochs."""
    data = np.ones((2, 10, 2))

    with pytest.raises(ValueError, match="requires .*epoch_index, sample_index"):
        _bias([8, 10], window=(-1, 2)).apply(data)


def test_cycle_average_rejects_epoch_coordinates_for_continuous_data():
    """Coordinate dimensionality must match the data dimensionality."""
    with pytest.raises(ValueError, match="2-D data requires one-dimensional"):
        _bias([[0, 4], [1, 4]], window=(-1, 2)).apply(np.ones((2, 10)))


def test_overlapping_events_are_permutation_invariant_and_commutative():
    """Overlaps average shifted contributions instead of last-write winning."""
    data = np.arange(12.0)[np.newaxis, :]
    forward = _bias([4, 6], window=(-2, 3)).apply(data)
    reverse = _bias([6, 4], window=(-2, 3)).apply(data)

    template = (data[:, 2:7] + data[:, 4:9]) / 2
    expected = np.zeros_like(data)
    counts = np.zeros(data.shape[1])
    for event in (4, 6):
        expected[:, event - 2 : event + 3] += template
        counts[event - 2 : event + 3] += 1
    expected /= np.where(counts == 0, 1, counts)

    assert_array_equal(forward, reverse)
    assert_allclose(forward, expected)


def test_duplicate_events_are_deduplicated_deterministically():
    """Duplicates neither reweight the template nor satisfy event count."""
    data = np.arange(20.0)[np.newaxis, :]
    unique = _bias([5, 12]).apply(data)
    duplicated = _bias([12, 5, 5, 12]).apply(data)

    assert_array_equal(duplicated, unique)
    assert duplicated.dtype == np.float64
    with pytest.raises(ValueError, match="received 1 after deduplication"):
        _bias([5, 5]).apply(data)


@pytest.mark.parametrize("events", [[], [5]])
def test_cycle_average_requires_multiple_unique_events(events):
    """Empty and one-event inputs cannot define an event repeatability bias."""
    with pytest.raises(ValueError, match="at least 2 unique complete events"):
        _bias(events).apply(np.ones((2, 20)))


def test_configured_minimum_may_be_stricter_but_never_one():
    """The public minimum is explicit and cannot erase repeatability."""
    with pytest.raises(ValueError, match="greater than or equal to 2"):
        _bias([5, 10], min_events=1)
    with pytest.raises(ValueError, match="at least 3"):
        _bias([5, 10], min_events=3).apply(np.ones((1, 20)))


def test_integer_input_is_floating_and_never_truncates_cycle_average():
    """A half-integer template survives integer public input."""
    data = np.zeros((1, 8), dtype=np.int16)
    data[:, 1:3] = [0, 1]
    data[:, 5:7] = [1, 2]

    observed = _bias([1, 5], window=(0, 2)).apply(data)

    assert observed.dtype == np.float64
    assert_allclose(observed[:, 1:3], [[0.5, 1.5]])
    assert_allclose(observed[:, 5:7], [[0.5, 1.5]])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_supported_float_dtype_is_preserved(dtype):
    """The supported float pathways retain numerical precision contracts."""
    data = np.arange(20, dtype=dtype)[np.newaxis, :]

    observed = _bias([5, 12]).apply(data)

    assert observed.dtype == dtype


@pytest.mark.parametrize(
    ("events", "match"),
    [
        ([1.5, 4], "integer coordinates"),
        ([True, 4], "integer coordinates"),
        ([[0, 1, 2], [1, 2, 3]], r"shape \(n_events, 2\)"),
        ([2**63, 4], "signed 64-bit"),
        ([-(2**63) - 1, 4], "signed 64-bit"),
    ],
)
def test_cycle_average_rejects_invalid_event_coordinates(events, match):
    """Coordinates cannot truncate, wrap, or use an ambiguous shape."""
    with pytest.raises(ValueError, match=match):
        _bias(events)


@pytest.mark.parametrize(
    ("window", "unit", "sfreq", "match"),
    [
        ((-1.5, 2), "samples", None, "must be integers"),
        ((2, -1), "samples", None, "strictly less"),
        ((0, 0.001), "seconds", 100.0, "empty or reversed"),
        ((-1, 2), "seconds", None, "sfreq is required"),
        ((-1, 2), "minutes", None, "window_unit"),
        ((-(2**63) - 1, 2), "samples", None, "signed 64-bit"),
        ((-1e308, 1e308), "seconds", 1e308, "finite sample range"),
    ],
)
def test_cycle_average_rejects_invalid_window_contracts(window, unit, sfreq, match):
    """Window conversion fails before unsafe sample arithmetic."""
    with pytest.raises(ValueError, match=match):
        CycleAverageBias(
            [5, 10],
            window,
            window_unit=unit,
            sfreq=sfreq,
            event_origin="data",
        )


def test_out_of_range_and_boundary_crossing_windows_are_actionable():
    """No event is silently filtered when its complete window is unavailable."""
    data = np.ones((2, 20))
    with pytest.raises(ValueError, match=r"event sample 1.*\[-1, 4\).*boundary"):
        _bias([1, 10]).apply(data)
    with pytest.raises(ValueError, match=r"event sample 19.*\[17, 22\).*boundary"):
        _bias([5, 19]).apply(data)
    with pytest.raises(ValueError, match="out of range for data with 2 epochs"):
        _bias([[0, 5], [2, 5]]).apply(np.ones((2, 20, 2)))


@pytest.mark.parametrize(
    ("events", "window"),
    [([-1, 10], (1, 3)), ([5, 20], (-3, -1))],
)
def test_event_itself_must_be_in_range_even_if_offset_window_would_fit(events, window):
    """An inward-offset window cannot legitimize an invalid event coordinate."""
    with pytest.raises(ValueError, match=r"out of range.*\[0, 20\)"):
        _bias(events, window=window).apply(np.ones((2, 20)))


def test_exact_boundary_windows_are_valid():
    """The half-open convention admits a stop exactly at n_times."""
    data = np.arange(20.0)[np.newaxis, :]

    observed = _bias([2, 17]).apply(data)

    assert np.count_nonzero(observed[:, :5]) > 0
    assert np.count_nonzero(observed[:, 15:]) > 0


def test_raw_origin_subtracts_nonzero_first_samp_exactly_once():
    """MNE acquisition event samples map into Raw data coordinates."""
    data = np.arange(40.0)[np.newaxis, :]
    bias = CycleAverageBias(
        event_samples=[1_010, 1_020],
        window=(-2, 3),
        window_unit="samples",
        event_origin="raw",
        first_samp=1_000,
    )

    first = bias.apply(data)
    second = bias.apply(data)

    assert bias.event_samples.tolist() == [1010, 1020]
    assert bias.event_samples_data_.tolist() == [10, 20]
    assert_array_equal(first, second)


def test_event_origin_contract_rejects_ambiguous_offsets_and_overflow():
    """The Raw offset is required, exclusive, and overflow-safe."""
    with pytest.raises(ValueError, match="first_samp is required"):
        CycleAverageBias([10, 20], event_origin="raw")
    with pytest.raises(ValueError, match="must be omitted"):
        CycleAverageBias([10, 20], event_origin="data", first_samp=1)
    with pytest.raises(ValueError, match="not defined for per-epoch"):
        CycleAverageBias([[0, 5], [1, 5]], event_origin="raw", first_samp=0)
    with pytest.raises(ValueError, match="signed 64-bit"):
        CycleAverageBias([np.iinfo(np.int64).min, 1], event_origin="raw", first_samp=1)


@pytest.mark.parametrize(
    ("data", "error", "match"),
    [
        (np.ones(10), ValueError, "2D or 3D"),
        (np.array([["x", "y"]]), TypeError, "real numerical"),
        (np.ones((1, 10), dtype=complex), TypeError, "real-valued"),
        (np.array([[1.0, np.nan] * 5]), ValueError, "finite values"),
    ],
)
def test_cycle_average_validates_public_numerical_input(data, error, match):
    """Unsupported dimensionality, dtype, and nonfinite values fail clearly."""
    with pytest.raises(error, match=match):
        _bias([2, 7], window=(-1, 2)).apply(data)
