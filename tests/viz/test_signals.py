"""Tests for signal-domain visualization helpers and plots."""

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
    """Build a small grouped-evoked dict for generic group plotting tests."""
    base = synthetic_data.average()
    groups = {}
    for group_idx, group in enumerate(["C0", "C1", "C2"]):
        evoked_list = []
        for sub_idx in range(4):
            evoked = base.copy()
            evoked.data = evoked.data.copy() + 0.05 * group_idx + 0.01 * sub_idx
            evoked_list.append(evoked)
        groups[group] = evoked_list
    return groups


def test_signal_viz_show(fitted_dss, synthetic_data):
    """Test show=True code paths for signal plots."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
        epochs_clean = fitted_dss.transform(synthetic_data)
        plot_evoked_gfp_comparison(
            synthetic_data,
            epochs_clean,
            times=synthetic_data.times,
            show=True,
        )
        plot_channel_time_course_comparison(
            synthetic_data,
            epochs_clean,
            picks=[0],
            times=synthetic_data.times,
            show=True,
        )
        plot_power_ratio_map(
            synthetic_data,
            epochs_clean,
            info=synthetic_data.info,
            show=True,
        )
        plot_signal_overlay(
            synthetic_data,
            epochs_clean,
            pick=0,
            times=synthetic_data.times,
            show=True,
        )


def test_signal_comparisons_mne(fitted_dss, synthetic_data):
    """Test public signal primitives on MNE inputs."""
    epochs_clean = fitted_dss.transform(synthetic_data)
    raw_orig = mne.io.RawArray(synthetic_data.get_data()[0], synthetic_data.info)
    raw_clean = mne.io.RawArray(epochs_clean.get_data()[0], synthetic_data.info)

    fig = plot_channel_time_course_comparison(
        synthetic_data,
        epochs_clean,
        picks=[0],
        times=synthetic_data.times,
        show=False,
    )
    assert isinstance(fig, plt.Figure)

    fig = plot_channel_time_course_comparison(
        raw_orig,
        raw_clean,
        picks=[0],
        start=10,
        stop=50,
        times=raw_orig.times,
        show=False,
    )
    assert isinstance(fig, plt.Figure)

    fig = plot_channel_time_course_comparison(
        synthetic_data,
        epochs_clean,
        picks=[0, 1],
        times=synthetic_data.times,
        show=False,
    )
    assert isinstance(fig, plt.Figure)

    fig = plot_power_ratio_map(
        synthetic_data,
        epochs_clean,
        info=synthetic_data.info,
        show=False,
    )
    assert isinstance(fig, plt.Figure)

    fig, ax = plt.subplots()
    plot_power_ratio_map(raw_orig, raw_clean, info=raw_orig.info, ax=ax, show=False)

    fig = plot_signal_overlay(
        synthetic_data,
        epochs_clean,
        pick=0,
        times=synthetic_data.times,
        show=False,
    )
    assert isinstance(fig, plt.Figure)

    fig = plot_signal_overlay(
        synthetic_data,
        epochs_clean,
        pick=0,
        times=synthetic_data.times,
        scale_after=True,
        show=False,
    )
    assert isinstance(fig, plt.Figure)

    fig = plot_signal_overlay(
        synthetic_data,
        epochs_clean,
        pick=0,
        times=synthetic_data.times,
        start=0.1,
        stop=0.5,
        show=False,
    )
    assert isinstance(fig, plt.Figure)

    fig = plot_signal_overlay(
        raw_orig, raw_clean, pick=0, times=raw_orig.times, show=False
    )
    assert isinstance(fig, plt.Figure)


def test_signal_comparisons_array():
    """Test array compatibility for 2D/3D canonical array shapes."""
    rng = np.random.default_rng(42)
    before_2d = rng.standard_normal((4, 200))
    after_2d = before_2d * 0.7
    before_3d = rng.standard_normal((10, 4, 200))
    after_3d = before_3d * 0.8
    times = np.arange(200) / 100.0

    fig = plot_evoked_gfp_comparison(before_2d, after_2d, times=times, show=False)
    assert isinstance(fig, plt.Figure)

    fig = plot_evoked_gfp_comparison(
        before_3d, after_3d, times=times, n_boot=20, show=False
    )
    assert isinstance(fig, plt.Figure)

    fig = plot_channel_time_course_comparison(
        before_2d,
        after_2d,
        picks=[0, 2],
        times=times,
        show=False,
    )
    assert isinstance(fig, plt.Figure)

    fig = plot_channel_time_course_comparison(
        before_3d,
        after_3d,
        picks=[1],
        times=times,
        show=False,
    )
    assert isinstance(fig, plt.Figure)

    info = mne.create_info(["Fz", "Cz", "Pz", "Oz"], 100.0, ch_types="eeg")
    info.set_montage("standard_1020")
    fig = plot_power_ratio_map(before_2d, after_2d, info=info, show=False)
    assert isinstance(fig, plt.Figure)

    var_before = np.var(before_2d, axis=1)
    var_after = np.var(after_2d, axis=1)
    fig = plot_power_ratio_map(var_before, var_after, info=info, show=False)
    assert isinstance(fig, plt.Figure)

    fig = plot_signal_overlay(
        before_2d,
        after_2d,
        pick=0,
        times=times,
        show=False,
    )
    assert isinstance(fig, plt.Figure)


def test_signal_overlay_length_regression():
    """Unequal-length overlays should not crash after alignment."""
    before = np.random.randn(1, 100)
    after = np.random.randn(1, 200)
    times = np.arange(100)

    fig = plot_signal_overlay(
        before,
        after,
        times=times,
        show=False,
    )
    assert isinstance(fig, plt.Figure)

    fig = plot_signal_overlay(
        before,
        after,
        times=times,
        start=10,
        stop=50,
        show=False,
    )
    assert isinstance(fig, plt.Figure)


def test_signal_explicit_contract_errors(fitted_dss, synthetic_data):
    """Test strict explicit error paths for required args."""
    epochs_clean = fitted_dss.transform(synthetic_data)
    raw_orig = mne.io.RawArray(synthetic_data.get_data()[0], synthetic_data.info)
    raw_clean = mne.io.RawArray(epochs_clean.get_data()[0], synthetic_data.info)

    with pytest.raises(TypeError):
        plot_channel_time_course_comparison(synthetic_data, epochs_clean, show=False)

    with pytest.raises(ValueError, match="Unknown channel name"):
        plot_channel_time_course_comparison(
            synthetic_data,
            epochs_clean,
            picks=["DOES_NOT_EXIST"],
            times=synthetic_data.times,
            show=False,
        )

    with pytest.raises(ValueError, match="times must be a 1D array"):
        plot_channel_time_course_comparison(
            synthetic_data,
            epochs_clean,
            picks=[0],
            times=np.arange(len(synthetic_data.times) - 1),
            show=False,
        )

    with pytest.raises(TypeError):
        plot_power_ratio_map(raw_orig, raw_clean, show=False)  # type: ignore[call-arg]

    with pytest.raises(ValueError, match="pick must be provided"):
        plot_signal_overlay(raw_orig, raw_clean, times=raw_orig.times, show=False)

    with pytest.raises(ValueError, match="times must be a 1D array"):
        plot_signal_overlay(
            raw_orig,
            raw_clean,
            pick=0,
            times=np.arange(raw_orig.n_times - 1),
            show=False,
        )

    with pytest.raises(ValueError, match="must share the same n_times"):
        plot_evoked_gfp_comparison(
            np.random.randn(3, 100),
            np.random.randn(3, 120),
            times=np.arange(100),
            show=False,
        )


def test_plot_grand_average_evokeds_basic(signal_evokeds):
    """Test basic grand average plotting."""
    fig = plot_grand_average_evokeds(
        signal_evokeds,
        channels=("Cz", "Pz"),
        show=False,
    )
    assert isinstance(fig, plt.Figure)


def test_plot_grand_average_evokeds_single_channel(signal_evokeds):
    """Test single channel grand average."""
    fig = plot_grand_average_evokeds(
        signal_evokeds,
        channels=("Cz",),
        show=False,
    )
    assert isinstance(fig, plt.Figure)


def test_plot_grand_average_evokeds_single_subject(synthetic_data):
    """Test grand average with single subject per group."""
    single = {group: [synthetic_data.average()] for group in ["C0", "C1", "C2"]}
    fig = plot_grand_average_evokeds(single, channels=("Cz",), show=False)
    assert isinstance(fig, plt.Figure)


def test_plot_grand_average_evokeds_custom_options(signal_evokeds):
    """Test grand average with custom theme options."""
    fig = plot_grand_average_evokeds(
        signal_evokeds,
        channels=("Pz",),
        time_windows={"P300": (0.25, 0.5)},
        suptitle="Custom Grand Average",
        group_order=["C2", "C0"],
        group_colors={"C0": "#aaa", "C2": "#bbb"},
        group_labels={"C0": "Base", "C2": "DSS"},
        amplitude_scale=1.0,
        y_label="Amplitude",
        figsize=(10, 6),
        show=False,
    )
    assert isinstance(fig, plt.Figure)


def test_plot_grand_average_evokeds_fname(signal_evokeds, tmp_path):
    """Test saving grand average to file."""
    fpath = tmp_path / "grand_avg.png"
    plot_grand_average_evokeds(
        signal_evokeds,
        channels=("Cz",),
        show=False,
        fname=str(fpath),
    )
    assert fpath.exists()


def test_signals_internal_validation_errors(synthetic_data):
    """Test internal validation helpers for signals."""
    from mne_denoise.viz.signals import (
        _as_channel_variance,
        _as_signal_array,
        _extract_overlay_trace,
    )

    # _as_signal_array dim checks
    with pytest.raises(ValueError, match="Input must be 2D"):
        _as_signal_array(np.zeros((1, 1, 1, 1)))

    # _as_channel_variance dim checks
    with pytest.raises(ValueError, match="Input must be 1D variances"):
        _as_channel_variance(np.zeros((1, 1, 1, 1)))

    # _extract_overlay_trace edge cases
    with pytest.raises(ValueError, match="String picks require channel names"):
        _extract_overlay_trace(np.zeros((2, 100)), pick="Cz")

    with pytest.raises(ValueError, match="Channel index 5 is out of range"):
        _extract_overlay_trace(np.zeros((2, 100)), pick=5)

    with pytest.raises(ValueError, match="Unknown channel name"):
        _extract_overlay_trace(synthetic_data, pick="NOT_A_CHAN")

    # Successful string pick
    idx = _extract_overlay_trace(synthetic_data, pick="Cz")
    assert isinstance(idx, np.ndarray)


def test_plot_evoked_gfp_comparison_edges(synthetic_data):
    """Test edge cases for GFP comparison."""
    data = synthetic_data.get_data()
    times = synthetic_data.times

    # Existing ax branch
    fig, ax = plt.subplots()
    ret_fig = plot_evoked_gfp_comparison(data, data, times=times, ax=ax, show=False)
    assert ret_fig is fig

    # Invalid times length
    with pytest.raises(ValueError, match="times must be a 1D array"):
        plot_evoked_gfp_comparison(data, data, times=times[:-1], show=False)


def test_plot_channel_time_course_comparison_edges(synthetic_data):
    """Test edge cases for channel time course comparison."""
    data = synthetic_data.get_data()
    times = synthetic_data.times

    # Shape mismatch (time)
    with pytest.raises(ValueError, match="must share the same channel/time dimensions"):
        plot_channel_time_course_comparison(
            data, data[..., :-1], picks=[0], times=times, show=False
        )

    # Shape mismatch (channels)
    with pytest.raises(ValueError, match="must share the same channel/time dimensions"):
        plot_channel_time_course_comparison(
            data, data[:, :-1, :], picks=[0], times=times, show=False
        )

    # picks is None
    with pytest.raises(ValueError, match="picks must be provided explicitly"):
        plot_channel_time_course_comparison(
            data, data, picks=None, times=times, show=False
        )  # type: ignore

    # empty picks
    with pytest.raises(ValueError, match="picks cannot be empty"):
        plot_channel_time_course_comparison(
            data, data, picks=[], times=times, show=False
        )

    # String picks without ch_names
    with pytest.raises(ValueError, match="String picks require channel names"):
        plot_channel_time_course_comparison(
            data, data, picks=["Cz"], times=times, show=False
        )

    # Unknown channel name (MNE)
    with pytest.raises(ValueError, match="Unknown channel name"):
        plot_channel_time_course_comparison(
            synthetic_data,
            synthetic_data,
            picks=["NOT_A_CHAN"],
            times=times,
            show=False,
        )

    # Index out of range
    with pytest.raises(ValueError, match="is out of range"):
        plot_channel_time_course_comparison(
            data, data, picks=[1000], times=times, show=False
        )

    # Successful string pick
    fig = plot_channel_time_course_comparison(
        synthetic_data, synthetic_data, picks=["Cz"], times=times, show=False
    )
    assert isinstance(fig, plt.Figure)


def test_plot_power_ratio_map_edges(synthetic_data):
    """Test edge cases for power ratio map."""
    # info is None
    with pytest.raises(ValueError, match="info must be provided explicitly"):
        plot_power_ratio_map(synthetic_data, synthetic_data, info=None, show=False)  # type: ignore

    # channel count mismatch
    var = np.random.randn(2)
    with pytest.raises(ValueError, match="must provide matching channels"):
        plot_power_ratio_map(
            var, np.random.randn(3), info=synthetic_data.info, show=False
        )

    with pytest.raises(ValueError, match="must match info channel count"):
        plot_power_ratio_map(var, var, info=synthetic_data.info, show=False)


def test_plot_signal_overlay_scaling(synthetic_data):
    """Test scaling logic in signal overlay."""
    fig = plot_signal_overlay(
        synthetic_data,
        synthetic_data,
        pick=0,
        times=synthetic_data.times,
        scale_after=True,
        show=False,
    )
    assert isinstance(fig, plt.Figure)

    fig = plot_signal_overlay(
        synthetic_data,
        synthetic_data,
        pick=0,
        times=synthetic_data.times,
        scale_after=False,
        show=False,
    )
    assert isinstance(fig, plt.Figure)


def test_plot_grand_average_evokeds_errors(signal_evokeds):
    """Test error paths for grand average plotting."""
    # empty channels
    with pytest.raises(ValueError, match="channels cannot be empty"):
        plot_grand_average_evokeds(signal_evokeds, channels=[], show=False)

    # unknown group
    with pytest.raises(ValueError, match="was not found in all_evokeds"):
        plot_grand_average_evokeds(
            signal_evokeds, channels=("Cz",), group_order=["NON_EXISTENT"], show=False
        )

    # empty group
    bad_groups = {"C0": []}
    with pytest.raises(ValueError, match="has no evoked entries"):
        plot_grand_average_evokeds(bad_groups, channels=("Cz",), show=False)

    # missing channel
    with pytest.raises(ValueError, match="was not found in group"):
        plot_grand_average_evokeds(
            signal_evokeds, channels=("NON_EXISTENT",), show=False
        )
