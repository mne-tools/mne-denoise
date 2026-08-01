"""Unit tests for artifact denoisers (CycleAverageBias)."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mne_denoise.dss.denoisers.artifact import CycleAverageBias


def test_cycle_average_bias():
    """Test CycleAverageBias on synthetic data."""
    rng = np.random.default_rng(42)
    n_channels = 3
    n_times = 1000

    # 1. Create synthetic periodic artifact (e.g., heartbeat)
    events = np.arange(100, n_times - 100, 200)
    artifact_signal = np.zeros(n_times)

    template_len = 50
    template = np.hanning(template_len)

    for event in events:
        start = event - template_len // 2
        end = start + template_len
        if start >= 0 and end <= n_times:
            artifact_signal[start:end] += template

    artifact_data = np.outer([1.0, 0.5, -0.5], artifact_signal)

    # 2. Add asynchronous noise (simulated brain activity)
    noise = rng.normal(0, 0.1, (n_channels, n_times))
    data = artifact_data + noise

    # 3. Apply CycleAverageBias
    window = (-25, 25)
    bias = CycleAverageBias(
        event_samples=events,
        window=window,
        window_unit="samples",
        event_origin="data",
    )
    biased_data = bias.apply(data)

    # 4. Verification
    mask = np.ones(n_times, dtype=bool)
    for event in events:
        mask[event + window[0] : event + window[1]] = False

    assert_allclose(
        biased_data[:, mask],
        0,
        atol=1e-10,
        err_msg="Biased data should be zero outside windows",
    )

    # Extract one window from biased data
    event = events[0]
    biased_epoch = biased_data[:, event + window[0] : event + window[1]]

    # Theoretical clean epoch
    clean_epoch = artifact_data[:, event + window[0] : event + window[1]]

    # Correlation should be high
    corr = np.corrcoef(biased_epoch.ravel(), clean_epoch.ravel())[0, 1]
    assert corr > 0.95, (
        f"Biased data should correlate with clean artifact (got {corr:.3f})"
    )

    # Check shape preservation
    assert biased_data.shape == data.shape


def test_cycle_average_bias_sfreq():
    """Test CycleAverageBias with second-based window."""
    events = [100, 200]
    sfreq = 100.0

    # Window: -0.1s to +0.2s -> -10 to +20 samples
    bias = CycleAverageBias(
        event_samples=events,
        window=(-0.1, 0.2),
        window_unit="seconds",
        sfreq=sfreq,
        event_origin="data",
    )

    assert bias.window == (-10, 20)
    assert bias.window_input == (-0.1, 0.2)
    assert bias.window_unit == "seconds"


def test_cycle_average_bias_3d_data():
    """Test CycleAverageBias with 3D epoched data."""
    rng = np.random.default_rng(42)
    n_channels, n_times, n_epochs = 3, 100, 5

    # Create 3D data
    data = rng.normal(0, 1, (n_channels, n_times, n_epochs))

    # Add a periodic artifact at known locations
    events = np.array([20, 50, 80])  # Within first epoch

    bias = CycleAverageBias(
        event_samples=events,
        window=(-5, 5),
        window_unit="samples",
        event_origin="data",
    )
    biased = bias.apply(data)

    # Output should be 3D with same shape
    assert biased.shape == data.shape
    assert biased.ndim == 3


def test_cycle_average_bias_empty_events():
    """Test CycleAverageBias with no valid events returns zeros."""
    rng = np.random.default_rng(42)
    n_channels, n_times = 3, 100
    data = rng.normal(0, 1, (n_channels, n_times))

    # Events outside data bounds
    events = np.array([1000, 2000])  # Way outside

    bias = CycleAverageBias(
        event_samples=events,
        window=(-10, 10),
        window_unit="samples",
        event_origin="data",
    )
    biased = bias.apply(data)

    # Should return zeros when no valid events
    assert_allclose(biased, 0)
    assert biased.shape == data.shape


def test_cycle_average_bias_invalid_ndim():
    """Test CycleAverageBias raises error for invalid dimensions."""
    events = [50]
    bias = CycleAverageBias(
        event_samples=events,
        window=(-5, 5),
        window_unit="samples",
        event_origin="data",
    )

    # 1D data should raise ValueError
    data_1d = np.array([1, 2, 3, 4, 5])
    with pytest.raises(ValueError, match="Data must be 2D or 3D"):
        bias.apply(data_1d)


def test_cycle_average_bias_partial_valid_events():
    """Test CycleAverageBias handles mix of valid and invalid events."""
    rng = np.random.default_rng(42)
    n_channels, n_times = 2, 200
    data = rng.normal(0, 1, (n_channels, n_times))

    # Mix of valid and invalid events (some at edges)
    events = np.array([5, 50, 100, 195])  # 5 and 195 may be clipped by window

    bias = CycleAverageBias(
        event_samples=events,
        window=(-10, 10),
        window_unit="samples",
        event_origin="data",
    )
    biased = bias.apply(data)

    # Should still work with the valid events in the middle
    assert biased.shape == data.shape
    # Middle event (50) window should be non-zero
    assert np.any(biased[:, 40:60] != 0)


@pytest.mark.parametrize(
    "sfreq, expected_unit, expected_window",
    [
        (None, "samples", (-1, 2)),
        (10.0, "seconds", (-10, 20)),
    ],
)
def test_cycle_average_legacy_unit_inference(sfreq, expected_unit, expected_window):
    """Released implicit units warn and retain their 0.x conversion."""
    with pytest.warns(FutureWarning) as records:
        bias = CycleAverageBias(event_samples=[20], window=(-1, 2), sfreq=sfreq)

    messages = [str(record.message) for record in records]
    assert any("window_unit" in message for message in messages)
    assert any("event_origin" in message for message in messages)
    assert bias.window_unit == expected_unit
    assert bias.event_origin == "data"
    assert bias.window == expected_window


def test_cycle_average_seconds_use_documented_nearest_sample_rounding():
    """Second boundaries are converted once using NumPy nearest rounding."""
    bias = CycleAverageBias(
        event_samples=[20],
        window=(-0.015, 0.015),
        window_unit="seconds",
        sfreq=100.0,
        event_origin="data",
    )

    assert bias.window == (-2, 2)


def test_cycle_average_raw_event_origin_subtracts_first_samp_once():
    """MNE acquisition samples map to data coordinates exactly once."""
    data = np.arange(40.0)[np.newaxis, :]
    raw_numbered_event = 1_020
    bias = CycleAverageBias(
        event_samples=[raw_numbered_event],
        window=(-2, 3),
        window_unit="samples",
        event_origin="raw",
        first_samp=1_000,
    )

    first = bias.apply(data)
    second = bias.apply(data)

    assert bias.event_samples.tolist() == [raw_numbered_event]
    assert bias.event_offset_samples_ == 1_000
    assert bias.event_samples_data_.tolist() == [20]
    assert_allclose(first, second)
    assert_allclose(first[:, 18:23], data[:, 18:23])
    assert np.count_nonzero(first) == 5


def test_cycle_average_data_origin_does_not_accept_offset():
    """An offset cannot be accidentally applied to data-relative events."""
    with pytest.raises(ValueError, match="must be omitted"):
        CycleAverageBias(
            event_samples=[20],
            window=(-2, 3),
            window_unit="samples",
            event_origin="data",
            first_samp=1_000,
        )


def test_cycle_average_raw_origin_requires_first_samp():
    """Raw-numbered events require an explicit MNE first sample."""
    with pytest.raises(ValueError, match="first_samp is required"):
        CycleAverageBias(
            event_samples=[20],
            window=(-2, 3),
            window_unit="samples",
            event_origin="raw",
        )


def test_cycle_average_rejects_unknown_event_origin():
    """Only data-relative and MNE Raw acquisition coordinates are supported."""
    with pytest.raises(ValueError, match="event_origin"):
        CycleAverageBias(
            event_samples=[20],
            window=(-2, 3),
            window_unit="samples",
            event_origin="epochs",
        )


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"window": (-1.5, 2), "window_unit": "samples"}, "integers"),
        ({"window": (-1, 2), "window_unit": "seconds"}, "sfreq is required"),
        ({"window": (2, -1), "window_unit": "samples"}, "strictly less"),
        ({"window": (-1, 2), "window_unit": "minutes"}, "window_unit"),
        ({"window": (-1, 2), "window_unit": "samples", "sfreq": 0}, "positive"),
    ],
)
def test_cycle_average_rejects_ambiguous_window_contracts(kwargs, match):
    """Invalid or ambiguous window contracts fail before data processing."""
    with pytest.raises(ValueError, match=match):
        CycleAverageBias(event_samples=[20], event_origin="data", **kwargs)


@pytest.mark.parametrize("events", [[1.5], [[1, 2]], [np.nan]])
def test_cycle_average_rejects_noninteger_or_nonvector_events(events):
    """Event positions are a finite one-dimensional integer sequence."""
    with pytest.raises(ValueError, match="event_samples"):
        CycleAverageBias(
            event_samples=events,
            window=(-1, 2),
            window_unit="samples",
            event_origin="data",
        )


@pytest.mark.parametrize(
    "events",
    [
        np.array([2**63], dtype=np.uint64),
        [2**63],
        [-(2**63) - 1],
    ],
)
def test_cycle_average_rejects_event_coordinates_outside_int64(events):
    """Event conversion rejects values that would wrap signed sample indices."""
    with pytest.raises(ValueError, match="signed 64-bit"):
        CycleAverageBias(
            event_samples=events,
            window=(-1, 2),
            window_unit="samples",
            event_origin="data",
        )


def test_cycle_average_extreme_int64_events_remain_invalid_without_wraparound():
    """Extreme signed events cannot overflow into the valid data interval."""
    bounds = np.iinfo(np.int64)
    bias = CycleAverageBias(
        event_samples=np.array([bounds.min, bounds.max], dtype=np.int64),
        window=(-1, 2),
        window_unit="samples",
        event_origin="data",
    )

    observed = bias.apply(np.ones((2, 16)))

    assert_allclose(observed, 0)


@pytest.mark.parametrize(
    "window, unit, sfreq",
    [
        ((-(2**63) - 1, 1), "samples", None),
        ((-1, 2**63), "samples", None),
        ((-1e308, 1e308), "seconds", 1e308),
    ],
)
def test_cycle_average_rejects_window_sample_overflow(window, unit, sfreq):
    """Window conversion fails explicitly instead of wrapping through int64."""
    with pytest.raises(ValueError, match="sample range|signed 64-bit"):
        CycleAverageBias(
            event_samples=[4],
            window=window,
            window_unit=unit,
            sfreq=sfreq,
            event_origin="data",
        )


def test_cycle_average_rejects_raw_origin_subtraction_overflow():
    """Mapping acquisition events to data coordinates is exact and range-safe."""
    with pytest.raises(ValueError, match="signed 64-bit"):
        CycleAverageBias(
            event_samples=[np.iinfo(np.int64).min],
            window=(-1, 2),
            window_unit="samples",
            event_origin="raw",
            first_samp=1,
        )


def test_cycle_average_3d_concatenation_is_epoch_major():
    """Concatenated event coordinates traverse complete epochs in order."""
    data = np.zeros((1, 6, 2))
    data[0, :, 0] = np.arange(1, 7)
    data[0, :, 1] = np.arange(101, 107)
    bias = CycleAverageBias(
        event_samples=[2, 8],
        window=(-1, 2),
        window_unit="samples",
        event_origin="data",
    )

    observed = bias.apply(data)

    expected_template = np.array([52.0, 53.0, 54.0])
    assert_allclose(observed[0, 1:4, 0], expected_template)
    assert_allclose(observed[0, 1:4, 1], expected_template)
