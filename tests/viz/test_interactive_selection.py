"""Public state-machine contracts for interactive component selection."""

from __future__ import annotations

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.backend_bases import MouseEvent

from mne_denoise.dss import DSS, IterativeDSS
from mne_denoise.dss.denoisers import VarianceMaskDenoiser
from mne_denoise.viz import ComponentSelector, plot_component_selector
from mne_denoise.zapline import ZapLine


def _fake_click(fig, ax, point=(0.5, 0.5)):
    """Dispatch a headless left click at a relative point inside an axes."""
    fig.canvas.draw()
    x, y = ax.transAxes.transform(point)
    event = MouseEvent("button_press_event", fig.canvas, x, y, button=1)
    fig.canvas.callbacks.process("button_press_event", event)


def _axes_for(selector, comp_idx):
    """Return the visible time-course axes for a component row."""
    row = selector._row_for_comp(comp_idx)
    assert row is not None, f"component {comp_idx} is not on the current page"
    return selector._rows[row].ax_time


def _preview_axis(selector, prefix):
    """Find a visible preview panel by its documented title prefix."""
    return next(ax for ax in selector.fig.axes if ax.get_title().startswith(prefix))


def _paged_selector(*, rows_per_page=2, n_components=6):
    """Build a small multi-page DSS selector for navigation scenarios."""
    rng = np.random.default_rng(5)
    data = rng.standard_normal((n_components, 900))
    dss = DSS(
        n_components=n_components,
        bias=lambda x: x,
        normalize_input=False,
    ).fit(data)
    return plot_component_selector(
        dss,
        data,
        sfreq=100.0,
        rows_per_page=rows_per_page,
        show=False,
    )


def test_selector_builds_for_dss_and_returns_a_usable_figure(
    fitted_dss, synthetic_data
):
    """DSS selection returns the documented controller and figure objects."""
    selector = plot_component_selector(
        fitted_dss,
        synthetic_data,
        info=synthetic_data.info,
        picks=[0, 1, 2, 3, 4],
        show=False,
    )

    assert isinstance(selector, ComponentSelector)
    assert isinstance(selector.fig, plt.Figure)
    assert selector.excluded == []
    assert any(ax.get_title() == "Time course" for ax in selector.fig.axes)
    assert any(ax.get_title() == "PSD" for ax in selector.fig.axes)


def test_selector_mouse_state_machine_supports_select_deselect_and_multi_select(
    fitted_dss, synthetic_data
):
    """Mouse selection supports selecting, deselecting, and multi-selecting rows."""
    selector = plot_component_selector(
        fitted_dss,
        synthetic_data,
        info=synthetic_data.info,
        picks=[0, 1, 2, 3, 4],
        show=False,
    )

    _fake_click(selector.fig, _axes_for(selector, 0))
    _fake_click(selector.fig, _axes_for(selector, 1))
    assert selector.excluded == [0, 1]

    _fake_click(selector.fig, _axes_for(selector, 0))
    assert selector.excluded == [1]


def test_selector_preview_compares_no_exclusion_reference_and_updates():
    """The live preview represents the reference and current reconstruction."""
    rng = np.random.default_rng(8)
    data = rng.standard_normal((8, 900))
    dss = DSS(n_components=3, bias=lambda x: x, normalize_input=False).fit(data)
    selector = plot_component_selector(dss, data, sfreq=100.0, show=False)

    gfp_axis = _preview_axis(selector, "Preview: global field power")
    reference = next(
        line for line in gfp_axis.lines if line.get_label() == "all components kept"
    )
    current = next(
        line for line in gfp_axis.lines if line.get_label() == "current selection"
    )
    reference_data = reference.get_ydata().copy()
    assert np.allclose(reference_data, current.get_ydata(), atol=1e-12)

    _fake_click(selector.fig, _axes_for(selector, 0))
    assert not np.allclose(reference_data, current.get_ydata())
    assert _preview_axis(selector, "Preview: PSD").get_xlabel() == "Frequency (Hz)"


def test_selector_apply_has_identity_baseline_and_excludes_selected_dss_component():
    """Applying no selection is unchanged; selecting a component changes output."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal((4, 600)) + np.arange(4)[:, np.newaxis]
    dss = DSS(
        n_components=4,
        bias=lambda x: x,
        normalize_input=False,
        reg=0,
    ).fit(data)
    selector = plot_component_selector(dss, data, sfreq=100.0, show=False)

    baseline = selector.apply()
    assert baseline.shape == data.shape
    assert np.allclose(baseline, data, atol=1e-10)

    _fake_click(selector.fig, _axes_for(selector, 0))
    cleaned = selector.apply()
    assert cleaned.shape == data.shape
    assert not np.allclose(cleaned, data)


def test_selector_iterative_dss_supports_public_selection_output():
    """Iterative DSS follows the same selection controller contract."""
    rng = np.random.default_rng(1)
    data = rng.standard_normal((5, 800)) + 2.0
    estimator = IterativeDSS(
        VarianceMaskDenoiser(),
        n_components=3,
        max_iter=3,
        random_state=0,
    ).fit(data)

    selector = plot_component_selector(estimator, data, sfreq=200.0, show=False)
    assert isinstance(selector, ComponentSelector)
    assert selector.excluded == []
    _fake_click(selector.fig, _axes_for(selector, 0))
    assert selector.apply().shape == data.shape


def test_selector_zapline_starts_with_removed_components_and_can_restore(
    fitted_zapline, zapline_data
):
    """ZapLine starts with fitted removals active and can restore all of them."""
    data, _ = zapline_data
    selector = plot_component_selector(fitted_zapline, data, show=False)

    assert selector.excluded == [0, 1]
    _fake_click(selector.fig, _axes_for(selector, 0))
    assert selector.excluded == [1]
    _fake_click(selector.fig, _axes_for(selector, 1))
    assert selector.excluded == []
    assert np.allclose(selector.apply(), data, atol=1e-9)


def test_selector_preview_option_is_optional_and_selection_still_works(
    fitted_dss, synthetic_data
):
    """Disabling the optional preview does not disable component selection."""
    selector = plot_component_selector(
        fitted_dss,
        synthetic_data,
        info=synthetic_data.info,
        picks=[0, 1, 2, 3, 4],
        preview=False,
        show=False,
    )

    assert not any(ax.get_title().startswith("Preview:") for ax in selector.fig.axes)
    _fake_click(selector.fig, _axes_for(selector, 1))
    assert selector.excluded == [1]


def test_selector_subset_and_paging_preserve_requested_indices():
    """Explicit component subsets remain selectable across public page changes."""
    rng = np.random.default_rng(14)
    data = rng.standard_normal((6, 900))
    estimator = DSS(
        n_components=6,
        bias=lambda x: x,
        normalize_input=False,
    ).fit(data)
    selector = plot_component_selector(
        estimator,
        data,
        sfreq=100.0,
        n_components=[0, 2, 4],
        rows_per_page=1,
        show=False,
    )

    assert selector.n_pages == 3
    assert selector.page == 0
    _fake_click(selector.fig, _axes_for(selector, 0))
    selector.set_page(2)
    _fake_click(selector.fig, _axes_for(selector, 4))
    assert selector.excluded == [0, 4]


def test_selector_supported_scroll_and_keyboard_navigation_changes_page():
    """Documented scroll and keyboard controls navigate without losing state."""
    selector = _paged_selector()
    row_axis = selector._rows[0].ax_time

    selector._on_scroll(SimpleNamespace(inaxes=row_axis, step=-1))
    assert selector.page == 1
    selector._on_scroll(SimpleNamespace(inaxes=row_axis, step=1))
    assert selector.page == 0

    selector._on_key(SimpleNamespace(key="pagedown"))
    assert selector.page == 1
    selector._on_key(SimpleNamespace(key="end"))
    assert selector.page == selector.n_pages - 1
    selector._on_key(SimpleNamespace(key="home"))
    assert selector.page == 0


def _click_display(fig, x, y):
    """Dispatch a click at absolute canvas coordinates."""
    fig.canvas.draw()
    event = MouseEvent("button_press_event", fig.canvas, x, y, button=1)
    fig.canvas.callbacks.process("button_press_event", event)


def test_selector_row_margin_is_a_real_click_target(fitted_dss, synthetic_data):
    """The documented full-row hit area handles clicks outside a topomap."""
    selector = plot_component_selector(
        fitted_dss,
        synthetic_data,
        info=synthetic_data.info,
        picks=[0, 1, 2, 3, 4],
        show=False,
    )
    selector.fig.canvas.draw()
    row = selector._rows[0]
    mid_y = (row.ax_topo.bbox.y0 + row.ax_topo.bbox.y1) / 2.0

    _click_display(selector.fig, 5, mid_y)
    assert selector.excluded == [0]


def test_selector_click_outside_component_rows_is_inert(fitted_dss, synthetic_data):
    """Clicks on the preview do not accidentally select a component."""
    selector = plot_component_selector(
        fitted_dss,
        synthetic_data,
        info=synthetic_data.info,
        picks=[0, 1, 2, 3, 4],
        show=False,
    )
    preview = _preview_axis(selector, "Preview: global field power")
    _fake_click(selector.fig, preview)
    assert selector.excluded == []


def test_selector_snapshots_array_data_for_apply():
    """The cached reconstruction is stable if the caller mutates its array."""
    rng = np.random.default_rng(11)
    data = rng.standard_normal((4, 500))
    estimator = DSS(n_components=4, bias=lambda x: x, normalize_input=False).fit(data)
    selector = plot_component_selector(estimator, data, sfreq=100.0, show=False)

    expected = selector.apply().copy()
    data[:] = 0.0
    assert np.array_equal(selector.apply(), expected)


def test_selector_requires_sampling_context_and_clamps_psd_frequency():
    """Array inputs need sfreq, while excessive PSD limits clamp to Nyquist."""
    rng = np.random.default_rng(15)
    data = rng.standard_normal((4, 600))
    estimator = DSS(n_components=2, bias=lambda x: x).fit(data)

    with pytest.raises(ValueError, match="sfreq is required"):
        plot_component_selector(estimator, data, show=False)

    with pytest.warns(UserWarning, match="exceeds the Nyquist frequency"):
        selector = plot_component_selector(
            estimator,
            data,
            sfreq=100.0,
            psd_fmax=1000.0,
            show=False,
        )
    assert _preview_axis(selector, "Preview: PSD").get_xlim()[1] == pytest.approx(50.0)


def test_selector_rejects_invalid_documented_combinations(fitted_dss, synthetic_data):
    """Unsupported estimator modes and ambiguous plotting inputs fail clearly."""
    with pytest.raises(ValueError, match="info is required"):
        plot_component_selector(fitted_dss, synthetic_data, picks=[0, 1], show=False)

    with pytest.raises(ValueError, match="No components available"):
        plot_component_selector(
            fitted_dss,
            synthetic_data,
            info=synthetic_data.info,
            picks=[0, 1, 2, 3, 4],
            n_components=[],
            show=False,
        )

    with pytest.raises(NotImplementedError, match="adaptive ZapLine"):
        plot_component_selector(
            ZapLine(sfreq=500.0, line_freq=50.0, adaptive=True),
            np.zeros((4, 1000)),
            show=False,
        )


def test_selector_rejects_invalid_times_and_unfitted_estimators(
    fitted_dss, synthetic_data
):
    """Invalid coordinates and an unfitted estimator produce public errors."""
    with pytest.raises(ValueError, match="times must be"):
        plot_component_selector(
            fitted_dss,
            synthetic_data,
            times=np.arange(5),
            show=False,
        )

    with pytest.raises(RuntimeError, match="not fitted"):
        plot_component_selector(
            DSS(n_components=2, bias=lambda x: x),
            np.zeros((2, 100)),
            sfreq=100.0,
            show=False,
        )
