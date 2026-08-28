"""Tests for the public plotting-theme API."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest

from mne_denoise.viz import (
    get_color,
    get_series_color,
    get_theme_rc,
    set_theme,
    style_axes,
    themed_figure,
    themed_legend,
    use_theme,
)


def test_use_theme_restores_rcparams_and_merges_overrides():
    """Temporary theme options apply inside the context and restore state."""
    before_top = plt.rcParams["axes.spines.top"]
    before_edge = plt.rcParams["axes.edgecolor"]
    custom_edge = "#444444"

    with use_theme(rc={"axes.edgecolor": custom_edge}):
        assert plt.rcParams["axes.spines.top"] is False
        assert mpl.colors.to_hex(plt.rcParams["axes.edgecolor"]) == custom_edge

    assert plt.rcParams["axes.spines.top"] == before_top
    assert plt.rcParams["axes.edgecolor"] == before_edge
    assert get_theme_rc({"axes.edgecolor": custom_edge})["axes.edgecolor"] == (
        custom_edge
    )


def test_color_helpers_support_method_fallbacks_and_palette_cycling():
    """Public color helpers resolve semantic keys and cycle custom palettes."""
    assert get_color("dss") == get_color("primary")
    assert get_color("missing", fallback="#aabbcc") == "#aabbcc"
    assert get_series_color(0, colors=["#000000", "#ffffff"]) == "#000000"
    assert get_series_color(2, colors=["#000000", "#ffffff"]) == "#000000"

    with pytest.raises(ValueError, match="at least one color"):
        get_series_color(0, colors=[])


def test_themed_figure_and_set_theme_apply_supported_options():
    """Figure creation and global theme application honor documented overrides."""
    custom_edge = "#223344"
    with mpl.rc_context():
        set_theme(rc={"axes.edgecolor": custom_edge})
        assert mpl.colors.to_hex(plt.rcParams["axes.edgecolor"]) == custom_edge

    fig, axes = themed_figure(1, 2, rc={"axes.edgecolor": custom_edge})
    assert isinstance(fig, plt.Figure)
    assert len(axes) == 2
    assert all(
        mpl.colors.to_hex(axis.spines["bottom"].get_edgecolor()) == custom_edge
        for axis in axes
    )


def test_themed_legend_and_grid_are_usable_on_existing_axes():
    """Axes helpers return a normal legend and enable the documented grid option."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="trace")
    legend = themed_legend(ax, title="Signals")
    style_axes(ax, grid=True)

    assert isinstance(legend, mpl.legend.Legend)
    assert legend.get_title().get_text() == "Signals"
    assert ax.get_axisbelow() is True
    plt.close(fig)


def test_use_theme_rejects_unknown_theme_name():
    """Unsupported theme names fail clearly instead of changing global state."""
    with pytest.raises(ValueError, match="Unknown theme 'nonexistent'"):
        with use_theme(name="nonexistent"):
            pass
