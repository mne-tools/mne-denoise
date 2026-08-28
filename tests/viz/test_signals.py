"""Public contracts for signal-domain visualization helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import mne
import numpy as np
import pytest

from mne_denoise.viz import (
    plot_channel_time_course_comparison,
    plot_evoked_gfp_comparison,
    plot_grand_average_evokeds,
    plot_power_ratio_map,
    plot_signal_overlay,
)


@pytest.fixture(scope="module")
def signal_evokeds(synthetic_data):
    """Build small subject-level evoked groups for aggregation checks."""
    base = synthetic_data.average()
    groups = {}
    for group_idx, group in enumerate(["before", "after"]):
        evoked_list = []
        for subject_idx in range(3):
            evoked = base.copy()
            evoked.data = evoked.data.copy() + 0.05 * group_idx + 0.01 * subject_idx
            evoked_list.append(evoked)
        groups[group] = evoked_list
    return groups


def test_plot_evoked_gfp_comparison_preserves_time_axis_and_confidence_band():
    """GFP plots map before/after traces to times and show epoched uncertainty."""
    times = np.arange(4) / 100.0
    before = np.array(
        [
            [[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 0.0, -1.0]],
            [[1.5, 2.5, 3.5, 4.5], [2.5, 1.5, 0.5, -0.5]],
        ]
    )
    after = 0.5 * before
    fig = plot_evoked_gfp_comparison(before, after, times=times, n_boot=20, show=False)

    assert isinstance(fig, plt.Figure)
    ax = next(ax for ax in fig.axes if ax.get_ylabel() == "Global Field Power")
    lines = {line.get_label(): line for line in ax.lines}
    np.testing.assert_allclose(lines["Before"].get_xdata(), times)
    np.testing.assert_allclose(
        lines["After"].get_ydata(), 0.5 * lines["Before"].get_ydata()
    )
    assert ax.get_xlabel() == "Time"
    assert ax.get_ylabel() == "Global Field Power"
    assert ax.collections


def test_plot_channel_time_course_maps_requested_channel_and_traces():
    """Channel picks select matching before/after traces and preserve samples."""
    times = np.arange(5) / 100.0
    before = np.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]], dtype=float)
    after = before - 1.0
    fig = plot_channel_time_course_comparison(
        before, after, picks=[1], times=times, show=False
    )

    assert isinstance(fig, plt.Figure)
    ax = next(ax for ax in fig.axes if ax.get_ylabel() == "ch1")
    lines = {line.get_label(): line for line in ax.lines}
    np.testing.assert_allclose(lines["Before"].get_xdata(), times)
    np.testing.assert_allclose(lines["Before"].get_ydata(), before[1])
    np.testing.assert_allclose(lines["After"].get_ydata(), after[1])
    assert ax.get_ylabel() == "ch1"


def test_plot_power_ratio_map_uses_after_over_before_and_channel_metadata():
    """Power-ratio maps expose the documented ratio label and MNE metadata path."""
    info = mne.create_info(["Fz", "Cz", "Pz"], 100.0, ch_types="eeg")
    info.set_montage("standard_1020")
    fig = plot_power_ratio_map(
        np.array([1.0, 2.0, 4.0]),
        np.array([0.5, 1.0, 8.0]),
        info=info,
        show=False,
    )

    assert isinstance(fig, plt.Figure)
    assert any(ax.get_title() == "Power Ratio Map" for ax in fig.axes)
    assert any(ax.get_ylabel() == "Power Ratio (After / Before)" for ax in fig.axes)


def test_plot_signal_overlay_scales_and_adds_public_annotations(tmp_path):
    """Overlay scaling, reference traces, highlights, and file output are usable."""
    times = np.arange(5, dtype=float)
    before = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    after = 2.0 * before
    reference = before - 0.25
    path = tmp_path / "overlay.png"

    fig = plot_signal_overlay(
        before,
        after,
        times=times,
        reference=reference,
        highlight_mask=np.array([False, True, False, False, True]),
        highlight_spans=[{"onset": 1.0, "duration": 0.5, "label": "window"}],
        show=False,
        fname=path,
    )
    assert isinstance(fig, plt.Figure)
    assert path.exists()
    ax = next(ax for ax in fig.axes if ax.get_ylabel() == "Amplitude")
    lines = {line.get_label(): line for line in ax.lines}
    np.testing.assert_allclose(lines["Before"].get_xdata(), times)
    np.testing.assert_allclose(lines["After"].get_ydata(), before)
    np.testing.assert_allclose(lines["Reference"].get_ydata(), reference)
    assert {"Before", "After", "Reference"} <= set(lines)
    assert ax.get_xlabel() == "Time"
    assert ax.get_ylabel() == "Amplitude"
    assert ax.collections

    unscaled = plot_signal_overlay(
        before, after, times=times, scale_after=False, show=False
    )
    unscaled_axis = next(ax for ax in unscaled.axes if ax.get_ylabel() == "Amplitude")
    unscaled_lines = {line.get_label(): line for line in unscaled_axis.lines}
    np.testing.assert_allclose(unscaled_lines["After"].get_ydata(), after)


def test_plot_grand_average_evokeds_aggregates_groups_and_channels(signal_evokeds):
    """Grand averages expose group means, SEM bands, channels, and windows."""
    fig = plot_grand_average_evokeds(
        signal_evokeds,
        channels=("Cz", "Pz"),
        group_order=["before", "after"],
        group_labels={"before": "Before", "after": "After"},
        time_windows={"response": (0.1, 0.3)},
        amplitude_scale=2.0,
        y_label="Scaled amplitude",
        show=False,
    )

    assert isinstance(fig, plt.Figure)
    assert {ax.get_title() for ax in fig.axes} == {
        "Grand Average at Cz",
        "Grand Average at Pz",
    }
    for ax in fig.axes:
        assert ax.get_xlabel() == "Time"
        assert ax.get_ylabel() == "Scaled amplitude"
        assert {line.get_label() for line in ax.lines} >= {"Before", "After"}
        assert ax.collections  # SEM bands or the named time window.


def test_signal_plot_can_show_in_headless_backend(monkeypatch):
    """The public show path works with a non-interactive Matplotlib backend."""
    monkeypatch.setattr(plt, "show", lambda: None)
    fig = plot_signal_overlay(
        np.array([1.0, 2.0]), np.array([1.0, 2.0]), times=np.arange(2), show=True
    )
    assert isinstance(fig, plt.Figure)


def test_signal_plot_validation_is_explicit(synthetic_data):
    """Representative invalid combinations fail with actionable errors."""
    data = synthetic_data.get_data()
    times = synthetic_data.times

    with pytest.raises(ValueError, match="times must be a 1D array"):
        plot_evoked_gfp_comparison(data, data, times=times[:-1], show=False)
    with pytest.raises(ValueError, match="picks must be provided explicitly"):
        plot_channel_time_course_comparison(
            data, data, picks=None, times=times, show=False
        )
    with pytest.raises(ValueError, match="info must be provided explicitly"):
        plot_power_ratio_map(data, data, info=None, show=False)
    with pytest.raises(ValueError, match="pick must be provided"):
        plot_signal_overlay(data[0], data[0], times=times, show=False)
    with pytest.raises(ValueError, match="channels cannot be empty"):
        plot_grand_average_evokeds({}, channels=[], show=False)


def test_signal_overlay_aligns_unequal_lengths_before_windowing():
    """The documented common-prefix alignment prevents unequal-length crashes."""
    before = np.arange(100.0)
    after = np.arange(200.0)
    fig = plot_signal_overlay(
        before, after, times=np.arange(100.0), start=10.0, stop=50.0, show=False
    )
    assert isinstance(fig, plt.Figure)
    ax = next(ax for ax in fig.axes if ax.get_ylabel() == "Amplitude")
    lines = {line.get_label(): line for line in ax.lines}
    np.testing.assert_array_equal(lines["Before"].get_xdata(), np.arange(10.0, 51.0))
