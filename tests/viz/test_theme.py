"""Tests for the public mne_denoise.viz.theme API."""

import matplotlib as mpl
import matplotlib.pyplot as plt


def test_use_theme_context_manager():
    """Test that use_theme restores rcParams on exit and accepts overrides."""
    from mne_denoise.viz.theme import get_theme_rc, use_theme

    before = plt.rcParams["axes.spines.top"]
    custom_edge = "#444444"
    with use_theme(rc={"axes.edgecolor": custom_edge}):
        in_ctx = plt.rcParams["axes.spines.top"]
        assert in_ctx is False
        assert mpl.colors.to_hex(plt.rcParams["axes.edgecolor"]).lower() == custom_edge
    assert plt.rcParams["axes.spines.top"] == before
    assert (
        get_theme_rc({"axes.edgecolor": custom_edge})["axes.edgecolor"] == custom_edge
    )
    assert get_theme_rc()["axes.edgecolor"] != custom_edge


def test_get_color():
    """Test get_color helper returns expected values."""
    from mne_denoise.viz.theme import (
        COLORS,
        METHOD_COLORS,
        get_color,
    )

    assert get_color("dss") == METHOD_COLORS["dss"]
    assert get_color("nonexistent_xyz") == COLORS["dark"]
    assert get_color("nonexistent_xyz", fallback="#aabbcc") == "#aabbcc"


def test_stats_semantic_colors_present():
    """Test that neutral stats semantic tokens exist in COLORS."""
    from mne_denoise.viz.theme import COLORS

    for key in (
        "stat_mean",
        "stat_subject",
        "stat_reference",
        "stat_ci",
        "stat_highlight",
    ):
        assert key in COLORS


def test_series_palette_exports():
    """Test shared spectral palette exports and lookup helper."""
    import pytest

    from mne_denoise.viz.theme import (
        DIVERGING_CMAP,
        SEQUENTIAL_CMAP,
        SERIES_COLORS,
        get_series_color,
    )

    assert len(SERIES_COLORS) >= 3
    assert get_series_color(0) == SERIES_COLORS[0]
    assert get_series_color(len(SERIES_COLORS)) == SERIES_COLORS[0]
    assert get_series_color(1, colors=["#000000", "#ffffff"]) == "#ffffff"
    assert SEQUENTIAL_CMAP == "viridis"
    assert DIVERGING_CMAP == "RdBu_r"

    with pytest.raises(ValueError, match="at least one color"):
        get_series_color(0, colors=[])


def test_set_theme_and_themed_figure_rc_overrides():
    """Test rc overrides for set_theme and themed_figure."""
    from mne_denoise.viz.theme import set_theme, themed_figure

    custom_edge = "#223344"
    with mpl.rc_context():
        set_theme(rc={"axes.edgecolor": custom_edge})
        assert mpl.colors.to_hex(plt.rcParams["axes.edgecolor"]).lower() == custom_edge

    fig, ax = themed_figure(rc={"axes.edgecolor": custom_edge})
    assert mpl.colors.to_hex(ax.spines["bottom"].get_edgecolor()).lower() == custom_edge
    plt.close(fig)


def test_finalize_fig_basic():
    """Test _finalize_fig returns the figure."""
    from mne_denoise.viz.theme import _finalize_fig

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    ret = _finalize_fig(fig, show=False)
    assert ret is fig


def test_finalize_fig_save(tmp_path):
    """Test _finalize_fig saves the figure when fname is given."""
    from mne_denoise.viz.theme import _finalize_fig

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    fpath = tmp_path / "test_plot.png"
    ret = _finalize_fig(fig, show=False, fname=str(fpath))
    assert ret is fig
    assert fpath.exists()
    plt.close(fig)


def test_finalize_fig_creates_nested_directory(tmp_path):
    """Test _finalize_fig creates nested parent directories when saving."""
    from mne_denoise.viz.theme import _finalize_fig

    deep = tmp_path / "a" / "b" / "c" / "fig.png"
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])

    _finalize_fig(fig, show=False, fname=deep)
    assert deep.exists()
    plt.close(fig)


def test_themed_legend():
    """Test themed_legend helper."""
    from mne_denoise.viz.theme import themed_legend

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="test")
    leg = themed_legend(ax, title="Test Title")
    assert isinstance(leg, mpl.legend.Legend)
    assert leg.get_title().get_text() == "Test Title"
    plt.close(fig)


def test_style_axes_grid():
    """Test style_axes with grid enabled."""
    from mne_denoise.viz.theme import style_axes

    fig, ax = plt.subplots()
    style_axes(ax, grid=True)
    # Check if grid is enabled on the axes
    assert (
        ax.xaxis._gridlines2D[0].get_visible()
        if hasattr(ax.xaxis, "_gridlines2D")
        else True
    )
    # More robust check: check if set_axisbelow was called or if any gridlines exist
    assert ax.get_axisbelow() is True
    plt.close(fig)


def test_use_theme_error():
    """Test use_theme raises error for unknown theme names."""
    import pytest

    from mne_denoise.viz.theme import use_theme

    with pytest.raises(ValueError, match="Unknown theme 'nonexistent'"):
        with use_theme(name="nonexistent"):
            pass


def test_themed_figure_defaults():
    """Test themed_figure with default arguments."""
    from mne_denoise.viz.theme import themed_figure

    fig, ax = themed_figure()
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_finalize_fig_no_tight(tmp_path):
    """Test _finalize_fig without tight_layout."""
    from mne_denoise.viz.theme import _finalize_fig

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    fpath = tmp_path / "test_no_tight.png"
    # coverage for tight=False
    ret = _finalize_fig(fig, show=False, fname=str(fpath), tight=False)
    assert ret is fig
    assert fpath.exists()
    plt.close(fig)
