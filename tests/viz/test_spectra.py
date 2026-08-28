"""Public contracts for spectral and time-frequency plots."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from mne_denoise.viz import (
    plot_component_psd_comparison,
    plot_narrowband_score_scan,
    plot_psd_comparison,
    plot_psd_gallery,
    plot_psd_overlay,
    plot_psd_zoom_comparison,
    plot_spectrogram_comparison,
    plot_time_frequency_mask,
)


def test_plot_narrowband_score_scan_maps_frequency_scores_and_annotations():
    """Narrowband scans preserve the frequency grid and optional markers."""
    frequencies = np.arange(5.0, 30.0, 0.5)
    scores = np.exp(-0.5 * ((frequencies - 10.0) / 1.5) ** 2)
    fig = plot_narrowband_score_scan(
        frequencies,
        scores,
        peak_freq=10.0,
        true_freqs=[10.0, 20.0],
        show=False,
    )

    assert isinstance(fig, plt.Figure)
    ax = next(ax for ax in fig.axes if ax.get_ylabel() == "Score / Eigenvalue")
    dominant = next(
        line for line in ax.lines if line.get_label() == "Dominant component"
    )
    np.testing.assert_allclose(dominant.get_xdata(), frequencies)
    np.testing.assert_allclose(dominant.get_ydata(), scores)
    assert ax.get_xlabel() == "Frequency (Hz)"
    assert ax.get_ylabel() == "Score / Eigenvalue"
    assert {"Peak: 10.0 Hz", "True: 10 Hz", "True: 20 Hz"} <= {
        line.get_label() for line in ax.lines
    }

    with pytest.raises(ValueError, match="matching first dimensions"):
        plot_narrowband_score_scan(frequencies, scores[:-1], show=False)


def test_plot_time_frequency_mask_preserves_axes_and_mask_section():
    """Time-frequency masks expose their supplied coordinates and weights."""
    times = np.linspace(0.0, 1.0, 20)
    freqs = np.linspace(5.0, 25.0, 10)
    mask = np.zeros((freqs.size, times.size))
    mask[3:5, 8:12] = 1.0
    fig = plot_time_frequency_mask(mask, times, freqs, show=False)

    assert isinstance(fig, plt.Figure)
    ax = next(ax for ax in fig.axes if ax.get_ylabel() == "Frequency (Hz)")
    assert ax.get_xlabel() == "Time (s)"
    assert ax.get_ylabel() == "Frequency (Hz)"
    assert ax.collections
    assert any(ax.get_ylabel() == "Mask Weight" for ax in fig.axes)

    with pytest.raises(ValueError, match="mask shape must match"):
        plot_time_frequency_mask(mask[:, :-1], times, freqs, show=False)


def test_plot_psd_comparison_maps_before_after_power_and_line_frequency():
    """PSD comparisons preserve a known power scaling and line marker."""
    sfreq = 100.0
    times = np.arange(400) / sfreq
    before = np.sin(2.0 * np.pi * 10.0 * times)[np.newaxis, :]
    after = 0.5 * before
    fig = plot_psd_comparison(
        before,
        after,
        sfreq=sfreq,
        fmin=1.0,
        fmax=30.0,
        line_freq=10.0,
        show=False,
    )

    assert isinstance(fig, plt.Figure)
    ax = next(ax for ax in fig.axes if ax.get_title() == "PSD Comparison")
    lines = {line.get_label(): line for line in ax.lines}
    before_line = lines["Before"]
    after_line = lines["After"]
    peak = int(np.argmax(before_line.get_ydata()))
    assert before_line.get_xdata()[peak] == pytest.approx(10.0)
    assert after_line.get_ydata()[peak] / before_line.get_ydata()[
        peak
    ] == pytest.approx(0.25)
    assert "10 Hz" in lines
    assert ax.get_xlabel() == "Frequency (Hz)"
    assert ax.get_ylabel() == "Power Spectral Density"


def test_plot_psd_zoom_comparison_preserves_zoom_frequency_and_series_labels():
    """Zoom panels retain frequency centers, annotations, and before/after labels."""
    freqs = np.arange(0.0, 121.0, 0.5)
    before = np.exp(-freqs / 40.0)
    after = 0.5 * before
    fig = plot_psd_zoom_comparison(
        freqs,
        before,
        freqs,
        after,
        series_name="DSS",
        series_labels={"DSS": "Cleaned"},
        zoom_freqs=[50.0],
        zoom_annotations=["attenuation"],
        show=False,
    )

    assert isinstance(fig, plt.Figure)
    assert any(
        "50 Hz" in ax.get_title() and "attenuation" in ax.get_title() for ax in fig.axes
    )
    full_axis = next(ax for ax in fig.axes if ax.get_title() == "PSD Comparison")
    assert {"Before", "Cleaned"} <= {line.get_label() for line in full_axis.lines}
    assert full_axis.get_xlabel() == "Frequency (Hz)"


def test_plot_psd_gallery_represents_each_requested_series():
    """The gallery has a full-spectrum row for each ordered series."""
    freqs = np.arange(0.0, 121.0, 0.5)
    reference = np.exp(-freqs / 40.0)
    series = {"M1": (freqs, 0.8 * reference), "M2": (freqs, 0.6 * reference)}
    fig = plot_psd_gallery(
        freqs,
        reference,
        series,
        zoom_freqs=[50.0],
        series_order=["M2", "M1"],
        series_labels={"M2": "Cleaned"},
        show=False,
    )

    assert isinstance(fig, plt.Figure)
    full_axis = next(ax for ax in fig.axes if ax.get_title() == "Full PSD")
    assert any(text.get_text() == "Cleaned" for text in full_axis.texts)
    assert all(
        ax.get_xlabel() == "Frequency (Hz)"
        for ax in fig.axes
        if ax.get_title() == "50 Hz"
    )


def test_plot_psd_overlay_marks_focus_and_harmonic_frequency():
    """Overlay plots keep the reference/series mapping and focus marker."""
    freqs = np.arange(0.0, 121.0, 0.5)
    reference = np.exp(-freqs / 40.0)
    series = {"DSS": (freqs, 0.7 * reference)}
    fig = plot_psd_overlay(
        freqs,
        reference,
        series,
        focus_freq=50.0,
        n_harmonics=2,
        series_labels={"DSS": "Cleaned"},
        show=False,
    )

    assert isinstance(fig, plt.Figure)
    zoom_axis = next(ax for ax in fig.axes if ax.get_title() == "Zoom at 50 Hz")
    assert {"Before", "Cleaned"} <= {line.get_label() for line in zoom_axis.lines}
    assert any(
        len(line.get_xdata()) == 2 and np.allclose(line.get_xdata(), [50.0, 50.0])
        for line in zoom_axis.lines
    )


def test_plot_component_psd_comparison_maps_requested_components():
    """Component PSDs retain selected indices, frequency bounds, and markers."""
    sfreq = 100.0
    times = np.arange(400) / sfreq
    signal = np.vstack(
        [
            np.sin(2.0 * np.pi * 10.0 * times),
            np.sin(2.0 * np.pi * 20.0 * times),
            np.cos(2.0 * np.pi * 5.0 * times),
        ]
    )
    components = np.vstack(
        [
            np.sin(2.0 * np.pi * 10.0 * times),
            np.sin(2.0 * np.pi * 20.0 * times),
        ]
    )
    fig = plot_component_psd_comparison(
        signal,
        components,
        component_indices=[1],
        sfreq=sfreq,
        peak_freq=20.0,
        fmin=1.0,
        fmax=30.0,
        show=False,
    )

    assert isinstance(fig, plt.Figure)
    axes_by_title = {ax.get_title(): ax for ax in fig.axes}
    assert {"Original Data PSD", "Component PSD"} <= axes_by_title.keys()
    assert any(
        line.get_label() == "Component 1"
        for line in axes_by_title["Component PSD"].lines
    )
    assert all(
        axes_by_title[title].get_xlabel() == "Frequency (Hz)"
        for title in ("Original Data PSD", "Component PSD")
    )


def test_plot_spectrogram_comparison_preserves_before_after_difference():
    """Spectrogram panels preserve time/frequency bounds and difference semantics."""
    sfreq = 100.0
    times = np.arange(200) / sfreq
    rng = np.random.default_rng(0)
    before = rng.standard_normal((2, times.size))
    after = 0.5 * before
    fig = plot_spectrogram_comparison(
        before,
        after,
        picks=[0, 1],
        times=times,
        sfreq=sfreq,
        fmin=1.0,
        fmax=20.0,
        n_freqs=6,
        show=False,
    )

    assert isinstance(fig, plt.Figure)
    panels = {
        ax.get_title(): ax
        for ax in fig.axes
        if ax.get_title() in {"Before", "After", "Before - After"}
    }
    assert set(panels) == {"Before", "After", "Before - After"}
    before_image = next(iter(panels["Before"].images))
    after_image = next(iter(panels["After"].images))
    diff_image = next(iter(panels["Before - After"].images))
    np.testing.assert_allclose(
        diff_image.get_array(), before_image.get_array() - after_image.get_array()
    )
    assert before_image.get_extent() == pytest.approx([times[0], times[-1], 1.0, 20.0])
    assert panels["Before"].get_xlabel() == "Time (s)"
    assert panels["Before"].get_ylabel() == "Frequency (Hz)"


def test_spectral_plot_validation_is_explicit():
    """Representative unsupported combinations fail with clear errors."""
    data = np.ones((2, 100))
    with pytest.raises(ValueError, match="sfreq must be provided"):
        plot_psd_comparison(data, data, show=False)
    with pytest.raises(ValueError, match="zoom_freqs must be a non-empty"):
        plot_psd_zoom_comparison(
            np.arange(10),
            np.ones(10),
            np.arange(10),
            np.ones(10),
            zoom_freqs=[],
            show=False,
        )
    with pytest.raises(ValueError, match="component_indices cannot be empty"):
        plot_component_psd_comparison(
            data, data, component_indices=[], sfreq=100.0, show=False
        )
    with pytest.raises(ValueError, match="picks must be provided explicitly"):
        plot_spectrogram_comparison(
            data, data, picks=None, times=np.arange(100), sfreq=100.0, show=False
        )
