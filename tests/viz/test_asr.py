"""Smoke + contract tests for the ASR-specific visualization diagnostics.

After the viz API was made generic, ``mne_denoise.viz.asr`` keeps only the
three diagnostics that have no generic equivalent: the per-window repair
timeline, the calibration / reference fraction, and the component
variance-vs-threshold map. Generic before/after plots (overlay, PSD,
power-ratio topomap, grand average, scatter) are covered by the generic viz
tests and are reused directly by the ASR examples.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from mne_denoise.asr import ASR, JugglerASR
from mne_denoise.viz import (
    plot_asr_calibration_fraction,
    plot_asr_component_reconstruction,
    plot_asr_repair_timeline,
)

SFREQ = 250.0


@pytest.fixture()
def fitted_asr():
    """Return an ASR estimator fitted + transformed on synthetic burst data."""
    rng = np.random.default_rng(3)
    n = 8000
    t = np.arange(n) / SFREQ
    X = np.zeros((8, n))
    for c in range(8):
        X[c] = 0.6 * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 6.28)) + (
            0.05 * rng.standard_normal(n)
        )
    for s in np.linspace(800, n - 600, 6).astype(int):
        spatial = rng.standard_normal(8)
        spatial /= np.linalg.norm(spatial)
        X[:, s : s + 150] += 10.0 * np.outer(spatial, rng.standard_normal(150))
    asr = ASR(sfreq=SFREQ, cutoff=10.0, verbose=False)
    asr.fit_transform(X)
    return asr


def _assert_fig_ax(ret):
    fig, ax = ret
    assert isinstance(fig, Figure)
    first = ax.ravel()[0] if hasattr(ax, "ravel") else ax
    assert isinstance(first, Axes)
    plt.close(fig)


def test_plot_asr_repair_timeline(fitted_asr):
    _assert_fig_ax(plot_asr_repair_timeline(fitted_asr, show=False))


def test_plot_asr_repair_timeline_ax_reuse(fitted_asr):
    fig, ax = plt.subplots()
    out_fig, out_ax = plot_asr_repair_timeline(fitted_asr, ax=ax, show=False)
    assert out_ax is ax and out_fig is fig
    plt.close(fig)


def test_plot_asr_component_reconstruction(fitted_asr):
    _assert_fig_ax(plot_asr_component_reconstruction(fitted_asr, show=False))


def test_plot_asr_calibration_fraction(fitted_asr):
    rng = np.random.default_rng(4)
    juggler = JugglerASR(sfreq=SFREQ, cutoff=10.0, verbose=False)
    juggler.fit_transform(0.5 * rng.standard_normal((8, 8000)))
    _assert_fig_ax(
        plot_asr_calibration_fraction(
            [fitted_asr, juggler], labels=["standard", "juggler"], show=False
        )
    )


def test_calibration_fraction_singlecore(fitted_asr):
    _assert_fig_ax(plot_asr_calibration_fraction(fitted_asr, show=False))


def test_repair_timeline_unfitted_raises():
    fresh = ASR(sfreq=SFREQ, verbose=False)
    with pytest.raises(ValueError, match="diagnostics"):
        plot_asr_repair_timeline(fresh, show=False)


def test_component_reconstruction_unfitted_raises():
    fresh = ASR(sfreq=SFREQ, verbose=False)
    with pytest.raises(ValueError, match="diagnostics"):
        plot_asr_component_reconstruction(fresh, show=False)


def test_repair_timeline_show_true_under_agg(fitted_asr):
    fig, _ = plot_asr_repair_timeline(fitted_asr, show=True)
    plt.close(fig)
