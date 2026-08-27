"""Interactive component selection for DSS and standard ZapLine estimators.

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

from __future__ import annotations

import warnings
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.gridspec import GridSpec

from .. import _mne
from .._data import (
    continuous_to_epochs,
    epochs_to_continuous,
    extract_data_from_mne,
    reconstruct_mne_object,
)
from ..dss.linear import DSS
from ..dss.nonlinear import IterativeDSS
from ..zapline.core import ZapLine
from ._utils import _compute_gfp, _get_info, _get_patterns
from .components import _resolve_component_indices
from .theme import COLORS, _finalize_fig, style_axes, themed_figure

if TYPE_CHECKING:
    import mne


def psd_array_welch(*args: Any, **kwargs: Any) -> Any:
    """Call MNE's Welch PSD implementation without an eager MNE import."""
    _mne.require_mne("interactive component selection")
    from mne.time_frequency import psd_array_welch as _psd_array_welch

    return _psd_array_welch(*args, **kwargs)


def _require_fitted(estimator: Any, name: str, *attrs: str) -> None:
    """Raise ``RuntimeError`` unless every fitted attribute is populated."""
    if any(getattr(estimator, attr, None) is None for attr in attrs):
        raise RuntimeError(f"{name} is not fitted. Call fit() first.")


def _estimator_kind(estimator: Any) -> str:
    """Validate an estimator and return its component-selection semantics."""
    # ZapLine subclasses DSS, so it must be checked before the DSS branch.
    if isinstance(estimator, ZapLine):
        if estimator.adaptive:
            raise NotImplementedError(
                "Interactive component selection does not support adaptive "
                "ZapLine because its components vary between data segments."
            )
        _require_fitted(
            estimator, "ZapLine", "filters_", "_artifact_mixing_", "n_removed_"
        )
        return "zapline"

    if isinstance(estimator, IterativeDSS):
        _require_fitted(estimator, "IterativeDSS", "filters_", "patterns_")
        return "iterative_dss"

    if isinstance(estimator, DSS):
        _require_fitted(estimator, "DSS", "filters_", "patterns_")
        return "linear_dss"

    raise TypeError(
        "estimator must be a fitted DSS, IterativeDSS, or standard ZapLine instance."
    )


def _flatten_with_layout(
    data: np.ndarray,
    *,
    n_channels: int,
    kind: str,
    mne_type: str,
) -> tuple[np.ndarray, str, tuple[int, ...]]:
    """Flatten estimator input to continuous channel-first data.

    Returns the flattened data, a tag naming the layout it came from, and
    the original shape, so :func:`_restore_layout` can invert it. Unlike
    :func:`mne_denoise._data.epochs_to_continuous`, this also detects
    which of three layouts it was handed and validates the channel count
    against the fitted estimator.
    """
    data = np.asarray(data, dtype=float)
    shape = data.shape
    if data.ndim == 2:
        if data.shape[0] != n_channels:
            raise ValueError(
                f"Input has {data.shape[0]} channels; estimator expects {n_channels}."
            )
        continuous = data.copy() if mne_type == "array" else data
        return continuous, "continuous", shape

    epochs_first = mne_type == "epochs" or kind in {"iterative_dss", "zapline"}
    if epochs_first:
        if data.shape[1] != n_channels:
            raise ValueError(
                "Epoched input must have shape (n_epochs, n_channels, n_times); "
                f"got {shape} for an estimator with {n_channels} channels."
            )
        # transpose() + reshape() of a non-contiguous view already returns a fresh
        # array, so no extra copy is needed to decouple from the caller's input.
        continuous = epochs_to_continuous(data)
        return continuous, "epochs_first", shape

    if data.shape[0] != n_channels:
        raise ValueError(
            "Linear DSS array input must have shape "
            "(n_channels, n_times, n_epochs); "
            f"got {shape} for an estimator with {n_channels} channels."
        )
    return data.reshape(n_channels, -1).copy(), "channels_first_epochs", shape


def _restore_layout(
    continuous: np.ndarray,
    layout: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    """Restore a continuous channel-first array to its input layout."""
    if layout == "continuous":
        return continuous
    if layout == "epochs_first":
        return continuous_to_epochs(continuous, shape)
    return continuous.reshape(shape)


def _sources_for_plot(
    sources: np.ndarray,
    layout: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    """Reshape continuous sources into a canonical per-component plot layout.

    Both epoched conventions are normalized to ``(n_components, n_times,
    n_epochs)`` so the plotting code only branches on ``ndim`` (2-D continuous
    vs. 3-D epoched), not on the estimator's layout. All reshapes return views.
    """
    if layout == "continuous":
        return sources
    if layout == "epochs_first":
        n_epochs, _, n_times = shape
        return sources.reshape(sources.shape[0], n_epochs, n_times).transpose(0, 2, 1)
    # Channels-first was flattened time-major, so it recovers directly.
    _, n_times, n_epochs = shape
    return sources.reshape(sources.shape[0], n_times, n_epochs)


@dataclass
class _PreviewHandles:
    """Matplotlib artists and PSD parameters for the live before/after preview."""

    ax_gfp: Any
    gfp_after: Any
    ax_psd: Any
    psd_after: Any
    sfreq: float
    psd_fmax: float
    n_fft: int


@dataclass
class _RowArtists:
    """Reusable matplotlib artists for one component row of the selector.

    Rows are created once and re-pointed at a different component when the
    user pages through the fitted decomposition, so paging never rebuilds
    subplots.
    """

    ax_hit: Any
    ax_topo: Any
    ax_time: Any
    ax_psd: Any
    time_line: Any
    psd_line: Any

    @property
    def axes(self) -> tuple[Any, Any, Any]:
        """The three visible panel axes, in left-to-right display order."""
        return (self.ax_topo, self.ax_time, self.ax_psd)

    @property
    def all_axes(self) -> tuple[Any, Any, Any, Any]:
        """Every axes of the row, including the invisible click target."""
        return (self.ax_hit, self.ax_topo, self.ax_time, self.ax_psd)


@dataclass
class _ComponentPanels:
    """Per-component display data precomputed once at build time.

    Component traces, spectra, and topomap values do not depend on the
    exclusion set, so they are computed for every selectable component up
    front. Paging then only swaps artist data.
    """

    indices: list[int]
    traces: np.ndarray
    psds: np.ndarray
    freqs: np.ndarray
    labels: list[str]
    topo_data: list[np.ndarray | None]
    topo_info: Any


@dataclass
class _SelectionState:
    """Cached estimator data required for selection and reconstruction."""

    estimator: Any
    kind: str
    continuous: np.ndarray
    sources: np.ndarray
    plot_sources: np.ndarray
    patterns: np.ndarray
    layout: str
    input_shape: tuple[int, ...]
    mne_type: str
    orig_inst: Any
    data_picks: np.ndarray | None
    sfreq: float | None
    channel_mean: np.ndarray | None = None

    @property
    def n_channels(self) -> int:
        """Number of fitted sensor channels."""
        return self.patterns.shape[0]

    @property
    def n_components(self) -> int:
        """Number of selectable fitted components."""
        return self.patterns.shape[1]

    def reconstruct_continuous(self, excluded: set[int] | list[int]) -> np.ndarray:
        """Reconstruct continuous sensor data for an exclusion set."""
        excluded_array = np.asarray(sorted({int(i) for i in excluded}), dtype=int)
        if excluded_array.size and (
            excluded_array[0] < 0 or excluded_array[-1] >= self.n_components
        ):
            raise ValueError(
                f"Excluded component indices must be between 0 and "
                f"{self.n_components - 1}."
            )

        if self.kind == "zapline":
            if excluded_array.size == 0:
                return self.continuous.copy()
            # ZapLine has no public partial-removal method, so subtract the
            # selected artifact components directly using the fitted artifact
            # mixing matrix (patterns_ is an alias of this for ZapLine).
            artifact_mixing = self.estimator._artifact_mixing_
            artifact = artifact_mixing[:, excluded_array] @ self.sources[excluded_array]
            return self.continuous - artifact

        # DSS / IterativeDSS: zero the excluded sources and delegate mixing and
        # de-normalization to the estimator's own inverse_transform, then restore
        # the input channel means (which inverse_transform does not add back).
        kept_sources = self.sources.copy()
        kept_sources[excluded_array] = 0.0
        reconstructed = np.asarray(
            self.estimator.inverse_transform(kept_sources), dtype=float
        )
        reconstructed += self.channel_mean[:, np.newaxis]
        return reconstructed

    def reconstruct(self, excluded: set[int] | list[int]) -> Any:
        """Reconstruct data and preserve the original input type and layout."""
        continuous = self.reconstruct_continuous(excluded)
        data = _restore_layout(continuous, self.layout, self.input_shape)
        return reconstruct_mne_object(
            data,
            self.orig_inst,
            self.mne_type,
            picks=self.data_picks,
        )


def _prepare_selection_state(estimator: Any, data: Any) -> _SelectionState:
    """Validate and cache component-selection data for one input."""
    kind = _estimator_kind(estimator)
    patterns = np.asarray(_get_patterns(estimator), dtype=float)
    filters = np.asarray(estimator.filters_, dtype=float)
    if patterns.ndim != 2 or filters.ndim != 2:
        raise ValueError(
            "Estimator patterns_ and filters_ must both be two-dimensional."
        )
    if filters.shape[0] != patterns.shape[1]:
        raise ValueError(
            "Estimator filters_ and patterns_ expose different component counts."
        )

    extracted, input_sfreq, mne_type, orig_inst, data_picks, _ = extract_data_from_mne(
        data,
        ch_names=getattr(estimator, "_mne_ch_names_", None),
        auto_pick=not getattr(estimator, "whiten", False),
    )
    continuous, layout, input_shape = _flatten_with_layout(
        extracted,
        n_channels=patterns.shape[0],
        kind=kind,
        mne_type=mne_type,
    )

    channel_mean = None
    if kind == "zapline":
        if input_sfreq is not None and not np.isclose(input_sfreq, estimator.sfreq):
            warnings.warn(
                f"Input data sfreq ({input_sfreq}) differs from the fitted "
                f"ZapLine sfreq ({estimator.sfreq}). Using the fitted sfreq.",
                UserWarning,
                stacklevel=3,
            )
        _, residual = estimator._get_smooth_residual(continuous, warn=False)
        sources = filters @ residual
        if int(estimator.n_removed_) != patterns.shape[1]:
            raise ValueError(
                "ZapLine's n_removed_ does not match its fitted component matrices."
            )
    else:
        working = continuous
        if estimator.normalize_input and not getattr(estimator, "whiten", False):
            channel_scale = np.asarray(estimator.channel_norms_, dtype=float)
            if channel_scale.shape != (patterns.shape[0],):
                raise ValueError(
                    "Estimator channel_norms_ does not match the fitted channels."
                )
            # Sources must be computed in the same normalized space the estimator
            # uses internally; inverse_transform re-applies channel_norms_ on the
            # way back, so no de-normalization is cached here.
            working = working / channel_scale[:, np.newaxis]
        channel_mean = continuous.mean(axis=1)
        working = working - working.mean(axis=1, keepdims=True)
        sources = filters @ working

    return _SelectionState(
        estimator=estimator,
        kind=kind,
        continuous=continuous,
        sources=sources,
        plot_sources=_sources_for_plot(sources, layout, input_shape),
        patterns=patterns,
        layout=layout,
        input_shape=input_shape,
        mne_type=mne_type,
        orig_inst=orig_inst,
        data_picks=data_picks,
        sfreq=input_sfreq,
        channel_mean=channel_mean,
    )


def _resolve_sfreq(estimator: Any, info: mne.Info | None, sfreq: float | None) -> float:
    """Resolve and validate the sampling frequency."""
    if info is not None:
        value = float(info["sfreq"])
    elif sfreq is not None:
        value = float(sfreq)
    elif getattr(estimator, "sfreq", None) is not None:
        value = float(estimator.sfreq)
    else:
        raise ValueError("sfreq is required when info is not available.")
    if not np.isfinite(value) or value <= 0:
        raise ValueError("sfreq must be a finite, strictly positive number.")
    return value


class ComponentSelector:
    """Selection controller returned by :func:`plot_component_selector`.

    The controller stores the current component selection and the cached data
    needed by the live preview. Use :attr:`excluded` to inspect the selection
    and :meth:`apply` to reconstruct the selected signal.

    Notes
    -----
    This class is a public return type, but direct construction is not part of
    the public API. Create instances with :func:`plot_component_selector`.
    """

    def __init__(
        self,
        *,
        estimator: Any,
        data: Any,
        fig: Any,
        state: _SelectionState,
        excluded: list[int],
    ) -> None:
        self.estimator = estimator
        self.data = data
        self.fig = fig
        self._state = state
        self._excluded = {int(i) for i in excluded}
        self._axes_to_row: dict[Any, int] = {}
        self._rows: list[_RowArtists] = []
        self._panels: _ComponentPanels | None = None
        self._page = 0
        self._suptitle: Any = None
        self._ax_pager: Any = None
        self._page_labels: list[tuple[Any, int]] = []
        self._preview: _PreviewHandles | None = None
        self._preview_cache: OrderedDict[
            frozenset[int], tuple[np.ndarray, np.ndarray]
        ] = OrderedDict()
        self._cids: list[int] = []

    @property
    def excluded(self) -> list[int]:
        """Return sorted component indices excluded from the clean output."""
        return sorted(self._excluded)

    @property
    def n_pages(self) -> int:
        """Number of pages needed to display every selectable component."""
        if self._panels is None or not self._rows:
            return 1
        n_selectable = len(self._panels.indices)
        return max(1, -(-n_selectable // len(self._rows)))

    @property
    def page(self) -> int:
        """Zero-based index of the currently displayed page of components."""
        return self._page

    def set_page(self, page: int) -> None:
        """Display a page of components, clamped to the available range.

        Parameters
        ----------
        page : int
            Zero-based page index. Values outside ``[0, n_pages - 1]`` are
            clamped rather than raising, so scroll and key handlers can call
            this without bounds checks.
        """
        page = int(np.clip(page, 0, self.n_pages - 1))
        if page == self._page:
            return
        self._page = page
        self._render_page()
        self.fig.canvas.draw_idle()

    def _comp_for_row(self, row: int) -> int | None:
        """Return the component displayed in a row slot, or None if empty."""
        if self._panels is None:
            return None
        position = self._page * len(self._rows) + row
        if position >= len(self._panels.indices):
            return None
        return self._panels.indices[position]

    def _row_for_comp(self, comp_idx: int) -> int | None:
        """Return the row slot showing a component, or None if it is off-page."""
        for row in range(len(self._rows)):
            if self._comp_for_row(row) == comp_idx:
                return row
        return None

    def apply(self, data: Any = None) -> Any:
        """Apply the current selection while preserving the input data type.

        Parameters
        ----------
        data : Raw | Epochs | Evoked | ndarray | None, default=None
            Data to reconstruct. If None, use the cached snapshot supplied to
            :func:`plot_component_selector`. Passing data explicitly computes
            fresh component sources for that input.

        Returns
        -------
        cleaned : Raw | Epochs | Evoked | ndarray
            Reconstructed data with the same type and layout as the input.
            Channels not used by the fitted estimator are preserved for MNE
            objects.
        """
        if data is None:
            if self.data is None:
                raise ValueError("No data available; pass data=... to apply().")
            state = self._state
        else:
            state = _prepare_selection_state(self.estimator, data)
        return state.reconstruct(self._excluded)

    def _toggle(self, comp_idx: int) -> None:
        if comp_idx in self._excluded:
            self._excluded.remove(comp_idx)
        else:
            self._excluded.add(comp_idx)
        # Components toggled while off-page still count; only the visible row
        # needs restyling, and paging back to it re-applies the styling.
        row = self._row_for_comp(comp_idx)
        if row is not None:
            self._style_row(row, comp_idx)
        self._update_status()
        self._update_preview()
        self.fig.canvas.draw_idle()

    def _style_row(self, row: int, comp_idx: int) -> None:
        """Apply kept/excluded styling to every artist of one row.

        The whole row is tinted rather than only the label, so the current
        selection stays readable at a glance instead of needing a per-title
        color check.
        """
        artists = self._rows[row]
        excluded = comp_idx in self._excluded
        color = COLORS["excluded"] if excluded else COLORS["kept"]
        # A faint wash rather than a solid fill, so the traces stay legible.
        tint = to_rgba(COLORS["excluded"], 0.10) if excluded else "none"

        title = artists.ax_topo.title
        title.set_color(color)
        if self._panels is not None:
            position = self._panels.indices.index(comp_idx)
            label = self._panels.labels[position]
            title.set_text(f"✕ {label}" if excluded else label)

        # The hit area spans the full row width, so tinting it (rather than only
        # the three panels) is what makes the whole band read as excluded.
        for ax in artists.all_axes:
            ax.patch.set_facecolor(tint)
            ax.patch.set_visible(True)
        for line in (artists.time_line, artists.psd_line):
            line.set_alpha(0.45 if excluded else 1.0)

    def _render_page(self) -> None:
        """Point every row slot at the components of the current page."""
        panels = self._panels
        if panels is None:
            return

        for row, artists in enumerate(self._rows):
            comp_idx = self._comp_for_row(row)
            if comp_idx is None:
                # Trailing slots on the last page have no component to show.
                for ax in artists.all_axes:
                    ax.set_visible(False)
                continue
            for ax in artists.all_axes:
                ax.set_visible(True)

            position = panels.indices.index(comp_idx)
            _draw_topomap(artists.ax_topo, panels.topo_data[position], panels.topo_info)

            artists.time_line.set_ydata(panels.traces[position])
            artists.ax_time.relim()
            artists.ax_time.autoscale_view()

            artists.psd_line.set_ydata(panels.psds[position])
            artists.ax_psd.relim()
            artists.ax_psd.autoscale_view()
            artists.ax_psd.set_xlim(panels.freqs[0], panels.freqs[-1])

            self._style_row(row, comp_idx)

        self._update_status()
        self._render_pager()

    def _update_status(self) -> None:
        """Refresh the one-line header with the current exclusion count."""
        if self._suptitle is None or self._panels is None:
            return
        self._suptitle.set_text(
            f"Click a component to exclude it (red)  ·  "
            f"{len(self._excluded)} of {self._state.n_components} excluded"
        )

    def _render_pager(self) -> None:
        """Draw the clickable page selector above the component rows."""
        ax = self._ax_pager
        if ax is None:
            return
        n_pages = self.n_pages
        if n_pages <= 1:
            ax.set_visible(False)
            return
        ax.set_visible(True)

        for text, _ in self._page_labels:
            text.remove()
        self._page_labels = []

        # Individual page numbers stay clickable while they fit; beyond eight a
        # decomposition needs arrows plus a counter to avoid an unreadable strip.
        if n_pages <= 8:
            entries = [(f"Page {i + 1}", i) for i in range(n_pages)]
        else:
            entries = [
                ("◀", self._page - 1),
                (f"Page {self._page + 1} of {n_pages}", self._page),
                ("▶", self._page + 1),
            ]

        spacing = min(0.11, 0.92 / len(entries))
        start = 0.5 - spacing * (len(entries) - 1) / 2.0
        for position, (label, target) in enumerate(entries):
            current = target == self._page
            disabled = not 0 <= target < n_pages
            text = ax.text(
                start + position * spacing,
                0.5,
                label,
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold" if current else "normal",
                color=(
                    COLORS["no_artifact"]
                    if disabled
                    else (COLORS["primary"] if current else COLORS["text"])
                ),
                bbox={
                    "boxstyle": "round,pad=0.35",
                    "facecolor": to_rgba(COLORS["primary"], 0.14)
                    if current
                    else "none",
                    "edgecolor": COLORS["primary"] if current else "none",
                    "linewidth": 0.8,
                },
            )
            self._page_labels.append((text, target))

    def _on_click(self, event: Any) -> None:
        if event.button != 1:
            return
        if self._ax_pager is not None and event.inaxes is self._ax_pager:
            self._click_pager(event)
            return
        row = self._axes_to_row.get(event.inaxes)
        if row is None:
            row = self._row_at_position(event)
        if row is None:
            return
        comp_idx = self._comp_for_row(row)
        if comp_idx is not None:
            self._toggle(comp_idx)

    def _row_at_position(self, event: Any) -> int | None:
        """Resolve a click that landed in a row's margins to that row.

        Constrained layout insets every axes, so the label strip and the
        figure-edge margins belong to no axes even with the full-width hit area
        in place. Anything between the pager and the preview is therefore
        attributed to the vertically nearest row.
        """
        y = getattr(event, "y", None)
        if y is None:
            return None
        visible = [
            (row, artists.ax_hit.bbox)
            for row, artists in enumerate(self._rows)
            if artists.ax_hit.get_visible()
        ]
        if not visible:
            return None

        centers = [(row, (bbox.y0 + bbox.y1) / 2.0) for row, bbox in visible]
        # One row's vertical pitch is its own band: half a pitch above the top
        # row covers its label strip, and half a pitch below the bottom row
        # stops short of the preview, so the header and preview stay inert
        # without needing to know where either one sits.
        if len(centers) > 1:
            pitch = abs(centers[0][1] - centers[1][1])
        else:
            pitch = visible[0][1].height * 1.6

        row, center = min(centers, key=lambda item: abs(item[1] - y))
        if abs(y - center) > pitch / 2.0:
            return None
        return row

    def _click_pager(self, event: Any) -> None:
        """Jump to whichever page label was clicked, if any."""
        for text, target in self._page_labels:
            contains, _ = text.contains(event)
            if contains and 0 <= target < self.n_pages:
                self.set_page(target)
                return

    def _on_scroll(self, event: Any) -> None:
        if event.inaxes is not None and event.inaxes not in self._axes_to_row:
            # Let the preview panels keep the wheel for their own use.
            return
        step = getattr(event, "step", 0)
        if step > 0:
            self.set_page(self._page - 1)
        elif step < 0:
            self.set_page(self._page + 1)

    def _on_key(self, event: Any) -> None:
        key = getattr(event, "key", None)
        if key in {"pagedown", "down", "right"}:
            self.set_page(self._page + 1)
        elif key in {"pageup", "up", "left"}:
            self.set_page(self._page - 1)
        elif key == "home":
            self.set_page(0)
        elif key == "end":
            self.set_page(self.n_pages - 1)

    def _update_preview(self) -> None:
        if self._preview is None:
            return
        _mne.require_mne("interactive component selection")
        preview = self._preview
        key = frozenset(self._excluded)
        cached = self._preview_cache.get(key)
        if cached is None:
            after_2d = self._state.reconstruct_continuous(self._excluded)
            gfp_after = _compute_gfp(after_2d)
            psd_after, _ = psd_array_welch(
                after_2d,
                sfreq=preview.sfreq,
                fmin=0,
                fmax=preview.psd_fmax,
                n_fft=preview.n_fft,
                verbose=False,
            )
            psd_after = np.clip(psd_after.mean(axis=0), a_min=1e-30, a_max=None)
            self._preview_cache[key] = (gfp_after, psd_after)
            if len(self._preview_cache) > 16:
                self._preview_cache.popitem(last=False)
        else:
            self._preview_cache.move_to_end(key)
            gfp_after, psd_after = cached

        preview.gfp_after.set_ydata(gfp_after)
        preview.ax_gfp.relim()
        preview.ax_gfp.autoscale_view()

        preview.psd_after.set_ydata(psd_after)
        preview.ax_psd.relim()
        preview.ax_psd.autoscale_view()


def plot_component_selector(
    estimator: DSS | IterativeDSS | ZapLine,
    data: Any,
    *,
    info: mne.Info | None = None,
    picks: Sequence[int] | np.ndarray | None = None,
    times: np.ndarray | None = None,
    n_components: int | Sequence[int] | np.ndarray | None = None,
    rows_per_page: int = 3,
    preview: bool = True,
    sfreq: float | None = None,
    psd_fmax: float | None = None,
    show: bool = True,
    fname: Any = None,
) -> ComponentSelector:
    """Plot an interactive component selector for DSS or standard ZapLine.

    Each component row contains its spatial pattern, time course, and power
    spectrum. Left-clicking a row toggles whether the component is excluded and
    updates the optional reconstruction preview. Rows excluded from the clean
    output are tinted red.

    Anywhere in a component's row responds to the click, including the margins
    around the topomap and the label above it.

    Matplotlib canvases cannot scroll, so decompositions with more components
    than ``rows_per_page`` are paged rather than stretched off-screen. Click a
    page button above the rows to jump to that page; the scroll wheel and
    ``PageUp``/``PageDown`` (also ``Home``/``End``) work too. Past eight pages
    the buttons compact to prev/next arrows with a counter. The selection is
    global: paging never discards toggles made on another page.

    Parameters
    ----------
    estimator : DSS | IterativeDSS | ZapLine
        Fitted estimator. ZapLine must use standard mode
        (``adaptive=False``).
    data : Raw | Epochs | Evoked | ndarray
        Data used to compute component sources and the live preview.
    info : mne.Info | None, default=None
        Sensor metadata. This can be inferred from a fitted estimator for the
        sampling frequency, but topomaps require both explicit ``info`` and
        explicit ``picks``.
    picks : array-like of int | None, default=None
        Channel indices used for topomaps. Requires an explicitly supplied
        ``info``. If None, the spatial-pattern panels contain a placeholder.
    times : array | None, default=None
        Time coordinates for component traces. The length must match the
        per-epoch time dimension, or the continuous sample count for Raw,
        Evoked, and two-dimensional arrays. If None, sample indices are used.
    n_components : int | array-like of int | None, default=None
        Components displayed and made clickable. An integer selects the first
        requested components; a sequence selects explicit component indices.
        If None, all fitted components are selectable, spread over pages.
    rows_per_page : int, default=3
        Component rows drawn per page. This fixes the figure height regardless
        of how many components are selectable.
    preview : bool, default=True
        Add a live reference/selection GFP and PSD preview.
    sfreq : float | None, default=None
        Sampling frequency used for PSDs when ``info`` is unavailable.
    psd_fmax : float | None, default=None
        Maximum PSD frequency. If None, use ``min(100, sfreq / 2)``.
    show : bool, default=True
        Show the figure.
    fname : path-like | None, default=None
        Optional path at which to save the figure.

    Returns
    -------
    selector : ComponentSelector
        Selection controller. Read ``selector.excluded`` and call
        ``selector.apply()`` to obtain the current reconstruction.

    Raises
    ------
    TypeError
        If ``estimator`` is not a supported estimator.
    RuntimeError
        If the estimator is not fitted.
    NotImplementedError
        If adaptive ZapLine is supplied.
    ValueError
        If the data, plotting coordinates, or frequency parameters are invalid.

    Notes
    -----
    For DSS and IterativeDSS, red components are omitted from the reconstruction.
    With no exclusions, the output is the fitted DSS-subspace reconstruction
    with the input channel means restored; it equals the input only for a
    complete reconstruction. For standard ZapLine, red components are noise
    components subtracted from the input, and all fitted noise components start
    excluded.

    The preview therefore compares the current selection against the
    *no-exclusion reconstruction*, not against the raw input. For a rank-reduced
    DSS fit those two differ by the discarded subspace, which would otherwise
    swamp the effect of the toggles being previewed; the panel titles report
    that rank. For ZapLine the no-exclusion reconstruction is the input itself,
    so the comparison is the familiar before/after.

    ``n_components`` controls only which already-fitted components are displayed;
    it does not refit an estimator. In particular, the selector can restore
    fitted ZapLine components but cannot expose components discarded during fit.

    NumPy layout follows the estimator APIs: Linear DSS accepts channel-first
    epoched arrays ``(n_channels, n_times, n_epochs)``; IterativeDSS and ZapLine
    accept MNE-style arrays ``(n_epochs, n_channels, n_times)``.

    Examples
    --------
    >>> from mne_denoise.viz import plot_component_selector
    >>> selector = plot_component_selector(
    ...     dss,
    ...     epochs,
    ...     info=epochs.info,
    ...     picks=[0, 1, 2, 3],
    ...     times=epochs.times,
    ... )
    >>> cleaned = selector.apply()
    """
    _mne.require_mne("interactive component selection")
    if picks is not None and info is None:
        raise ValueError("info is required when picks is provided.")

    state = _prepare_selection_state(estimator, data)
    resolved_info = _get_info(estimator, info)
    sfreq_eff = _resolve_sfreq(estimator, resolved_info, sfreq)

    indices = _resolve_component_indices(
        n_components,
        state.n_components,
        default_max=state.n_components,
    )
    if not indices:
        raise ValueError("No components available for selection.")

    rows_per_page = int(rows_per_page)
    if rows_per_page < 1:
        raise ValueError("rows_per_page must be a positive integer.")
    rows_per_page = min(rows_per_page, len(indices))

    if psd_fmax is None:
        psd_fmax = min(100.0, sfreq_eff / 2.0)
    psd_fmax = float(psd_fmax)
    if not np.isfinite(psd_fmax) or psd_fmax <= 0:
        raise ValueError("psd_fmax must be a finite, strictly positive number.")
    nyquist = sfreq_eff / 2.0
    if psd_fmax > nyquist:
        warnings.warn(
            f"psd_fmax ({psd_fmax}) exceeds the Nyquist frequency ({nyquist}); "
            "clamping to the Nyquist frequency.",
            UserWarning,
            stacklevel=2,
        )
        psd_fmax = nyquist

    n_times = state.plot_sources.shape[1]
    if times is None:
        time_values = np.arange(n_times)
        time_label = "Time (samples)"
    else:
        time_values = np.asarray(times, dtype=float)
        if time_values.ndim != 1 or time_values.shape[0] != n_times:
            raise ValueError(f"times must be one-dimensional with length {n_times}.")
        if not np.all(np.isfinite(time_values)):
            raise ValueError("times must contain only finite values.")
        time_label = "Time"

    topo_picks = None
    if picks is not None:
        topo_picks = np.asarray(picks, dtype=int)
        if topo_picks.ndim != 1 or topo_picks.size == 0:
            raise ValueError("picks must be a non-empty one-dimensional sequence.")
        if topo_picks.min() < 0 or topo_picks.max() >= len(info["ch_names"]):
            raise ValueError("picks contains an index outside info.")
        if state.n_channels not in {len(info["ch_names"]), topo_picks.size}:
            raise ValueError(
                "Estimator patterns must match either all info channels or "
                "the explicitly picked channels."
            )

    panels = _compute_panels(
        state,
        indices,
        topo_picks=topo_picks,
        info=info,
        sfreq=sfreq_eff,
        psd_fmax=psd_fmax,
    )

    excluded0 = list(range(state.n_components)) if state.kind == "zapline" else []
    # Row 0 holds the page selector; component rows follow, then the preview.
    n_rows = 1 + rows_per_page + int(preview)
    # Constrained layout reconciles titles, labels, and the header for us, so
    # the figure only needs a height budget per row rather than hand-tuned
    # spacing. Height stays bounded because extra components are paged.
    height = 0.45 + 1.25 * rows_per_page + (2.25 if preview else 0.8)
    fig, root_ax = themed_figure(figsize=(12, height), layout="constrained")
    root_ax.remove()
    height_ratios = [0.28] + [1.0] * rows_per_page + ([1.35] if preview else [])
    gs = GridSpec(
        n_rows,
        3,
        figure=fig,
        width_ratios=[0.8, 2.4, 1.2],
        height_ratios=height_ratios,
    )
    selector = ComponentSelector(
        estimator=estimator,
        data=data,
        fig=fig,
        state=state,
        excluded=excluded0,
    )
    selector._panels = panels

    selector._ax_pager = fig.add_subplot(gs[0, :])
    selector._ax_pager.set_axis_off()

    last_row = rows_per_page - 1
    share_time = share_psd = None
    for row_idx in range(rows_per_page):
        grid_row = row_idx + 1
        # A full-width axes behind each row makes the entire band clickable.
        # mne.viz.plot_topomap forces an equal aspect, which shrinks the topomap
        # axes to a small square and leaves most of the row belonging to no axes
        # at all, so without this the row is mostly dead to clicks.
        ax_hit = fig.add_subplot(gs[grid_row, :], zorder=-1)
        ax_hit.set_axis_off()

        ax_topo = fig.add_subplot(gs[grid_row, 0])
        ax_topo.set_title("", fontweight="bold")

        # Columns share an x-axis so paging cannot desynchronize the rows and
        # only the bottom row needs tick labels.
        ax_time = fig.add_subplot(gs[grid_row, 1], sharex=share_time)
        (time_line,) = ax_time.plot(
            time_values, panels.traces[0], color=COLORS["before"]
        )
        style_axes(ax_time, grid=True)

        ax_psd = fig.add_subplot(gs[grid_row, 2], sharex=share_psd)
        (psd_line,) = ax_psd.semilogy(
            panels.freqs, panels.psds[0], color=COLORS["primary"]
        )
        ax_psd.set_xlim(0, psd_fmax)
        style_axes(ax_psd, grid=True)

        if share_time is None:
            share_time, share_psd = ax_time, ax_psd
        # Column headings belong to the grid, not to each row; repeating them
        # per row is what made the panel look crowded.
        if row_idx == 0:
            ax_time.set_title("Time course")
            ax_psd.set_title("PSD")
        if row_idx == last_row:
            ax_time.set_xlabel(time_label)
            ax_psd.set_xlabel("Frequency (Hz)")
        else:
            ax_time.tick_params(labelbottom=False)
            ax_psd.tick_params(labelbottom=False)

        artists = _RowArtists(
            ax_hit=ax_hit,
            ax_topo=ax_topo,
            ax_time=ax_time,
            ax_psd=ax_psd,
            time_line=time_line,
            psd_line=psd_line,
        )
        selector._rows.append(artists)
        for ax in artists.all_axes:
            selector._axes_to_row[ax] = row_idx

    if preview:
        _build_preview(
            fig,
            gs,
            rows_per_page + 1,
            selector,
            sfreq_eff,
            psd_fmax,
        )

    selector._suptitle = fig.suptitle("", fontsize=11)
    selector._render_page()

    for event_name, handler in (
        ("button_press_event", selector._on_click),
        ("scroll_event", selector._on_scroll),
        ("key_press_event", selector._on_key),
    ):
        selector._cids.append(fig.canvas.mpl_connect(event_name, handler))

    _finalize_fig(fig, show=show, fname=fname, tight=False)
    return selector


def _draw_topomap(ax: Any, topo_data: np.ndarray | None, topo_info: Any) -> None:
    """Draw one component topomap, or a placeholder when picks are missing."""
    title = ax.get_title()
    color = ax.title.get_color()
    ax.clear()
    if topo_data is None:
        ax.text(
            0.5,
            0.5,
            "topomap requires\ninfo and picks",
            ha="center",
            va="center",
        )
    else:
        _mne.require_mne("interactive component topomap visualization")
        _mne.mne.viz.plot_topomap(topo_data, topo_info, axes=ax, show=False)
    ax.set_axis_off()
    ax.set_title(title, fontweight="bold", color=color)


def _component_labels(state: _SelectionState, indices: Sequence[int]) -> list[str]:
    """Build row labels, annotated with the estimator's own component metric.

    A bare index carries no information about why a component matters, so the
    fitted eigenvalue is appended whenever the estimator exposes one that lines
    up with its component count.
    """
    eigenvalues = getattr(state.estimator, "eigenvalues_", None)
    values = None
    if eigenvalues is not None:
        eigenvalues = np.asarray(eigenvalues, dtype=float).ravel()
        if eigenvalues.size >= state.n_components:
            values = eigenvalues[: state.n_components]

    labels = []
    for comp_idx in indices:
        label = f"Comp {comp_idx}"
        if values is not None and np.isfinite(values[comp_idx]):
            label += f"\nλ = {values[comp_idx]:.3g}"
        labels.append(label)
    return labels


def _compute_panels(
    state: _SelectionState,
    indices: Sequence[int],
    *,
    topo_picks: np.ndarray | None,
    info: mne.Info | None,
    sfreq: float,
    psd_fmax: float,
) -> _ComponentPanels:
    """Precompute the display data for every selectable component.

    None of this depends on the exclusion set, so computing it once keeps
    paging free of PSD recomputation.
    """
    _mne.require_mne("interactive component selection")
    indices = list(indices)
    sources = state.plot_sources[indices]
    # Epoched sources are (n_sel, n_times, n_epochs): average epochs for the
    # trace, but keep single epochs for the PSD so Welch sees real segments.
    if sources.ndim == 3:
        traces = sources.mean(axis=2)
        psd_input = np.transpose(sources, (0, 2, 1))
    else:
        traces = sources
        psd_input = sources[:, np.newaxis, :]

    psds, freqs = psd_array_welch(
        psd_input,
        sfreq=sfreq,
        fmin=0,
        fmax=psd_fmax,
        n_fft=min(2048, psd_input.shape[-1]),
        verbose=False,
    )
    psds = np.clip(psds.mean(axis=1), 1e-30, None)

    topo_info = None
    topo_data: list[np.ndarray | None] = [None] * len(indices)
    if topo_picks is not None:
        topo_info = _mne.mne.pick_info(info, topo_picks)
        rows = topo_picks if state.n_channels == len(info["ch_names"]) else slice(None)
        topo_data = [state.patterns[rows, comp_idx] for comp_idx in indices]

    return _ComponentPanels(
        indices=indices,
        traces=traces,
        psds=psds,
        freqs=freqs,
        labels=_component_labels(state, indices),
        topo_data=topo_data,
        topo_info=topo_info,
    )


def _build_preview(
    fig: Any,
    gs: GridSpec,
    preview_row: int,
    selector: ComponentSelector,
    sfreq: float,
    psd_fmax: float,
) -> None:
    """Add the reference/selection GFP and PSD preview panels."""
    _mne.require_mne("interactive component selection")
    state = selector._state
    # The reference is the no-exclusion reconstruction, not the raw input. For a
    # rank-reduced DSS fit the raw input also carries the discarded subspace, so
    # comparing against it would show a large gap even with nothing excluded and
    # hide the actual effect of the user's toggles. ZapLine's no-exclusion
    # reconstruction is the input itself, so both estimators stay comparable.
    before_2d = state.reconstruct_continuous([])
    after_2d = state.reconstruct_continuous(selector._excluded)
    n_fft = min(2048, before_2d.shape[-1])

    if state.kind == "zapline":
        ref_label, sel_label, rank_note = "input", "cleaned", ""
    else:
        ref_label, sel_label = "all components kept", "current selection"
        rank_note = (
            f" — reference = DSS subspace ({state.n_components}/"
            f"{state.n_channels} comps)"
            if state.n_components < state.n_channels
            else ""
        )

    ax_gfp = fig.add_subplot(gs[preview_row, 0:2])
    samples = np.arange(before_2d.shape[1])
    # The two traces now share a scale and overlap heavily when little is
    # excluded, so the reference is drawn wide and faded underneath.
    ax_gfp.plot(
        samples,
        _compute_gfp(before_2d),
        color=COLORS["before"],
        label=ref_label,
        linewidth=2.0,
        alpha=0.45,
    )
    gfp_after_vals = _compute_gfp(after_2d)
    (gfp_after,) = ax_gfp.plot(
        samples,
        gfp_after_vals,
        color=COLORS["after"],
        label=sel_label,
        linewidth=0.9,
    )
    ax_gfp.set_title(f"Preview: global field power{rank_note}")
    ax_gfp.set_xlabel("Time (concatenated samples)")
    ax_gfp.legend(loc="upper right", fontsize=8)
    style_axes(ax_gfp, grid=True)

    ax_psd = fig.add_subplot(gs[preview_row, 2])
    psd_before, freqs = psd_array_welch(
        before_2d,
        sfreq=sfreq,
        fmin=0,
        fmax=psd_fmax,
        n_fft=n_fft,
        verbose=False,
    )
    psd_after_data, _ = psd_array_welch(
        after_2d,
        sfreq=sfreq,
        fmin=0,
        fmax=psd_fmax,
        n_fft=n_fft,
        verbose=False,
    )
    ax_psd.semilogy(
        freqs,
        np.clip(psd_before.mean(axis=0), 1e-30, None),
        color=COLORS["before"],
        linewidth=2.0,
        alpha=0.45,
    )
    psd_after_vals = np.clip(psd_after_data.mean(axis=0), 1e-30, None)
    (psd_after,) = ax_psd.semilogy(
        freqs,
        psd_after_vals,
        color=COLORS["after"],
        linewidth=0.9,
    )
    ax_psd.set_title("Preview: PSD")
    ax_psd.set_xlabel("Frequency (Hz)")
    ax_psd.set_xlim(0, psd_fmax)
    style_axes(ax_psd, grid=True)

    selector._preview = _PreviewHandles(
        ax_gfp=ax_gfp,
        gfp_after=gfp_after,
        ax_psd=ax_psd,
        psd_after=psd_after,
        sfreq=sfreq,
        psd_fmax=psd_fmax,
        n_fft=n_fft,
    )
    selector._preview_cache[frozenset(selector._excluded)] = (
        gfp_after_vals,
        psd_after_vals,
    )
