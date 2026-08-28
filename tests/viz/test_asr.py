"""Public contracts for ASR-specific diagnostic plots."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from mne_denoise.asr import ASR, GuidedASR, JugglerASR
from mne_denoise.viz import (
    plot_asr_calibration_fraction,
    plot_asr_component_reconstruction,
    plot_asr_repair_timeline,
    plot_guided_asr_weights,
)

SFREQ = 250.0


@pytest.fixture()
def fitted_asr():
    """Return an ASR estimator fitted and transformed on burst data."""
    rng = np.random.default_rng(3)
    n_times = 8000
    times = np.arange(n_times) / SFREQ
    data = np.zeros((8, n_times))
    for channel in range(8):
        data[channel] = 0.6 * np.sin(
            2 * np.pi * 10 * times + rng.uniform(0, 6.28)
        ) + 0.05 * rng.standard_normal(n_times)
    for start in np.linspace(800, n_times - 600, 6).astype(int):
        spatial = rng.standard_normal(8)
        spatial /= np.linalg.norm(spatial)
        data[:, start : start + 150] += 10.0 * np.outer(
            spatial, rng.standard_normal(150)
        )
    estimator = ASR(sfreq=SFREQ, cutoff=10.0, verbose=False)
    estimator.fit_transform(data)
    return estimator


def test_asr_diagnostic_plots_expose_documented_axes(fitted_asr):
    """ASR diagnostics return usable ``(Figure, Axes)`` pairs with units."""
    fig, ax = plot_asr_repair_timeline(fitted_asr, show=False)
    assert isinstance(fig, plt.Figure)
    assert ax.get_xlabel() == "Time (s)"
    assert ax.get_ylabel() == "Components reconstructed"

    fig, ax = plot_asr_component_reconstruction(fitted_asr, show=False)
    assert isinstance(fig, plt.Figure)
    assert ax.get_xlabel() == "Time (s)"
    assert ax.get_ylabel() == "Component"


def test_plot_asr_repair_timeline_reuses_existing_axis(fitted_asr):
    """The documented ``ax`` argument draws into and returns the same axes."""
    fig, ax = plt.subplots()
    returned_fig, returned_ax = plot_asr_repair_timeline(fitted_asr, ax=ax, show=False)
    assert returned_fig is fig
    assert returned_ax is ax
    plt.close(fig)


def test_plot_asr_calibration_fraction_aggregates_estimators(fitted_asr):
    """Calibration fractions include one labeled value per fitted estimator."""
    rng = np.random.default_rng(4)
    juggler = JugglerASR(sfreq=SFREQ, cutoff=10.0, verbose=False)
    juggler.fit_transform(0.5 * rng.standard_normal((8, 8000)))

    fig, ax = plot_asr_calibration_fraction(
        [fitted_asr, juggler], labels=["standard", "juggler"], show=False
    )
    assert isinstance(fig, plt.Figure)
    assert ax.get_ylabel() == "Calibration fraction (%)"
    assert [tick.get_text() for tick in ax.get_xticklabels()] == [
        "standard",
        "juggler",
    ]


def test_plot_guided_asr_weights_returns_weighted_component_map():
    """The experimental GuidedASR diagnostic remains headless and usable."""
    rng = np.random.default_rng(9)
    guided = GuidedASR(
        sfreq=SFREQ,
        reconstruction="soft",
        experimental=True,
        max_dims=0,
        picks=None,
        verbose=False,
    )
    with pytest.warns(UserWarning, match="unpublished, unvalidated"):
        guided.fit_transform(rng.standard_normal((8, 2000)))

    fig, ax = plot_guided_asr_weights(guided, show=False)
    assert isinstance(fig, plt.Figure)
    assert ax.get_xlabel() == "Time (s)"
    assert ax.get_ylabel() == "Component"


@pytest.mark.parametrize(
    "plotter",
    [plot_asr_repair_timeline, plot_asr_component_reconstruction],
    ids=["timeline", "component_reconstruction"],
)
def test_asr_diagnostics_reject_unfitted_estimators(plotter):
    """Diagnostics clearly require a completed estimator transform."""
    with pytest.raises(ValueError, match="diagnostics"):
        plotter(ASR(sfreq=SFREQ, verbose=False), show=False)


def test_asr_plot_can_show_in_headless_backend(fitted_asr, monkeypatch):
    """The public show path does not require an interactive display."""
    monkeypatch.setattr(plt, "show", lambda: None)
    fig, _ = plot_asr_repair_timeline(fitted_asr, show=True)
    assert isinstance(fig, plt.Figure)
