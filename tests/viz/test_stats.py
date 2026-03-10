"""Tests for grouped metric and statistical visualization helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from mne_denoise.viz import (
    plot_forest,
    plot_harmonic_attenuation,
    plot_metric_bars,
    plot_metric_comparison,
    plot_metric_slopes,
    plot_metric_tradeoff_summary,
    plot_metric_violins,
    plot_null_distribution,
    plot_tradeoff_scatter,
)


@pytest.fixture
def freqs():
    """Return a synthetic frequency vector."""
    return np.arange(0, 200, 0.5)


@pytest.fixture
def gm_psd(freqs):
    """Synthetic geometric-mean PSD with line-noise peaks."""
    psd = np.ones_like(freqs) * 1e-6
    for harmonic in [50, 100, 150]:
        mask = (freqs >= harmonic - 1) & (freqs <= harmonic + 1)
        psd[mask] = 1e-3
    return psd


@pytest.fixture
def gm_psd_clean(freqs):
    """Synthetic cleaned PSD with reduced line-noise peaks."""
    psd = np.ones_like(freqs) * 1e-6
    for harmonic in [50, 100, 150]:
        mask = (freqs >= harmonic - 1) & (freqs <= harmonic + 1)
        psd[mask] = 1e-5
    return psd


@pytest.fixture
def cleaned_psds(freqs, gm_psd_clean):
    """Return cleaned PSD mapping for two methods."""
    return {
        "M1": (freqs, gm_psd_clean),
        "M2": (freqs, gm_psd_clean * 1.1),
    }


@pytest.fixture
def single_subject_data():
    """Return columnar mapping for one subject across methods."""
    rng = np.random.default_rng(0)
    methods = np.array(["M0", "M1", "M2"], dtype=object)
    return {
        "subject": np.array(["sub-01", "sub-01", "sub-01"], dtype=object),
        "method": methods,
        "R_f0": rng.uniform(0.7, 1.2, size=3),
        "peak_attenuation_db": rng.uniform(5, 20, size=3),
        "below_noise_distortion_db": rng.uniform(-5, 5, size=3),
        "overclean_proportion": rng.uniform(0, 0.3, size=3),
        "underclean_proportion": rng.uniform(0, 0.3, size=3),
    }


@pytest.fixture
def multi_subject_data():
    """Return columnar mapping for three subjects across methods."""
    rng = np.random.default_rng(42)
    subjects = []
    methods = []
    for sub in ["sub-01", "sub-02", "sub-03"]:
        for method in ["M0", "M1", "M2"]:
            subjects.append(sub)
            methods.append(method)
    n_rows = len(subjects)
    return {
        "subject": np.asarray(subjects, dtype=object),
        "method": np.asarray(methods, dtype=object),
        "R_f0": rng.uniform(0.7, 1.2, size=n_rows),
        "peak_attenuation_db": rng.uniform(5, 20, size=n_rows),
        "below_noise_distortion_db": rng.uniform(-5, 5, size=n_rows),
        "overclean_proportion": rng.uniform(0, 0.3, size=n_rows),
        "underclean_proportion": rng.uniform(0, 0.3, size=n_rows),
    }


def test_plot_metric_bars_basic(single_subject_data):
    """Tests for grouped bar plots."""
    fig = plot_metric_bars(
        single_subject_data,
        metric_cols=["R_f0"],
        group_col="method",
        show=False,
    )
    assert isinstance(fig, plt.Figure)


def test_plot_metric_bars_custom_order(single_subject_data):
    fig = plot_metric_bars(
        single_subject_data,
        metric_cols=["R_f0", "peak_attenuation_db"],
        group_col="method",
        group_order=["M0", "M1", "M2"],
        show=False,
    )
    assert isinstance(fig, plt.Figure)


def test_plot_tradeoff_scatter_single_subject(single_subject_data):
    """Tests for grouped trade-off scatter plots."""
    fig = plot_tradeoff_scatter(
        single_subject_data,
        group_col="method",
        x_col="below_noise_distortion_db",
        y_col="peak_attenuation_db",
        show=False,
    )
    assert isinstance(fig, plt.Figure)


def test_plot_tradeoff_scatter_multi_subject(multi_subject_data):
    fig = plot_tradeoff_scatter(
        multi_subject_data,
        group_col="method",
        group_order=["M0", "M1", "M2"],
        x_col="below_noise_distortion_db",
        y_col="peak_attenuation_db",
        show=False,
    )
    assert isinstance(fig, plt.Figure)


def test_plot_metric_comparison_single_subject_bar(single_subject_data):
    """Tests for single-metric group comparison plots."""
    fig = plot_metric_comparison(
        single_subject_data,
        metric_col="R_f0",
        group_col="method",
        group_order=["M0", "M1", "M2"],
        show=False,
    )
    assert isinstance(fig, plt.Figure)


def test_plot_metric_comparison_multi_subject_paired(multi_subject_data):
    fig = plot_metric_comparison(
        multi_subject_data,
        metric_col="R_f0",
        group_col="method",
        group_order=["M0", "M1", "M2"],
        show=False,
    )
    assert isinstance(fig, plt.Figure)


def test_plot_harmonic_attenuation_basic(freqs, gm_psd, cleaned_psds):
    """Tests for per-harmonic attenuation bars."""
    fig = plot_harmonic_attenuation(
        freqs,
        gm_psd,
        cleaned_psds,
        harmonics_hz=[50.0, 100.0],
        series_order=["M1", "M2"],
        show=False,
    )
    assert isinstance(fig, plt.Figure)


def test_plot_harmonic_attenuation_with_subject(freqs, gm_psd, cleaned_psds):
    fig = plot_harmonic_attenuation(
        freqs,
        gm_psd,
        cleaned_psds,
        harmonics_hz=[50.0],
        subject="sub-01",
        series_order=["M1", "M2"],
        show=False,
    )
    assert isinstance(fig, plt.Figure)


def test_plot_metric_slopes_multi_subject(multi_subject_data):
    """Tests for paired subject trajectory plots."""
    fig = plot_metric_slopes(
        multi_subject_data,
        metric_cols=["R_f0"],
        group_col="method",
        group_order=["M0", "M1", "M2"],
        show=False,
    )
    assert isinstance(fig, plt.Figure)


def test_plot_metric_slopes_single_subject(single_subject_data):
    fig = plot_metric_slopes(
        single_subject_data,
        metric_cols=["R_f0"],
        group_col="method",
        show=False,
    )
    assert isinstance(fig, plt.Figure)


def test_plot_tradeoff_scatter_explicit_xy_on_wide_table(multi_subject_data):
    """Tests for trade-off x/y requirements."""
    data = dict(multi_subject_data)
    data["subject_id"] = np.arange(len(data["subject"]))
    data["run"] = np.ones(len(data["subject"]), dtype=int)
    fig = plot_tradeoff_scatter(
        data,
        group_col="method",
        x_col="below_noise_distortion_db",
        y_col="peak_attenuation_db",
        show=False,
    )
    assert isinstance(fig, plt.Figure)


def test_plot_tradeoff_scatter_raises_when_xy_not_provided():
    data = {
        "method": np.array(["M0", "M1"], dtype=object),
        "metric_a": np.array([1.0, 2.0]),
        "metric_b": np.array([3.0, 4.0]),
        "metric_c": np.array([5.0, 6.0]),
    }
    with pytest.raises(TypeError, match="missing 2 required positional arguments"):
        plot_tradeoff_scatter(data, group_col="method", show=False)


def test_plot_metric_tradeoff_summary_single_subject(single_subject_data):
    """Tests for composed tradeoff summary plot."""
    fig = plot_metric_tradeoff_summary(
        single_subject_data,
        group_col="method",
        subject_col="subject",
        x_col="below_noise_distortion_db",
        y_col="peak_attenuation_db",
        metric_col="R_f0",
        show=False,
    )
    assert isinstance(fig, plt.Figure)


def test_plot_metric_tradeoff_summary_multi_subject(multi_subject_data):
    fig = plot_metric_tradeoff_summary(
        multi_subject_data,
        group_col="method",
        subject_col="subject",
        x_col="below_noise_distortion_db",
        y_col="peak_attenuation_db",
        metric_col="R_f0",
        show=False,
    )
    assert isinstance(fig, plt.Figure)


def test_plot_metric_violins_basic(multi_subject_data):
    """Tests for distribution plots."""
    fig = plot_metric_violins(
        multi_subject_data,
        metric_cols=["R_f0"],
        group_col="method",
        subject_col="subject",
        show=False,
    )
    assert isinstance(fig, plt.Figure)


def test_plot_metric_violins_no_paired(multi_subject_data):
    fig = plot_metric_violins(
        multi_subject_data,
        metric_cols=["R_f0", "peak_attenuation_db"],
        group_col="method",
        subject_col="subject",
        show_paired=False,
        show=False,
    )
    assert isinstance(fig, plt.Figure)


def test_plot_metric_violins_baseline_and_refs(multi_subject_data):
    fig = plot_metric_violins(
        multi_subject_data,
        metric_cols=["R_f0"],
        group_col="method",
        subject_col="subject",
        baseline_group="M0",
        reference_lines={"R_f0": [(1.0, {"color": "red", "ls": "--"})]},
        show=False,
    )
    assert isinstance(fig, plt.Figure)


def test_plot_metric_violins_no_data():
    data = {"method": [], "subject": [], "score": []}
    fig = plot_metric_violins(
        data,
        metric_cols=["score"],
        group_col="method",
        subject_col="subject",
        show=False,
    )
    assert isinstance(fig, plt.Figure)


def test_plot_null_distribution_basic():
    """Tests for null distribution plots."""
    rng = np.random.default_rng(0)
    null = rng.normal(0, 1, 1000)
    fig, p = plot_null_distribution(null, observed=1.5, show=False)
    assert isinstance(fig, plt.Figure)
    assert 0 < p < 1


def test_plot_forest_basic(multi_subject_data):
    """Tests for forest plots."""
    fig = plot_forest(
        multi_subject_data,
        metric_col="R_f0",
        group_col="method",
        subject_col="subject",
        show=False,
    )
    assert isinstance(fig, plt.Figure)


def test_plot_forest_ci_col(multi_subject_data):
    """Test plot_forest with explicit CI column."""
    data = dict(multi_subject_data)
    data["err"] = np.ones(len(data["R_f0"])) * 0.1
    fig = plot_forest(
        data,
        metric_col="R_f0",
        ci_col="err",
        group_col="method",
        subject_col="subject",
        show=False,
    )
    assert isinstance(fig, plt.Figure)


def test_plot_metric_bars_label_derivation():
    """Test metric label derivation when not provided."""
    data = {"group": ["A"], "my_metric_name": [1.0]}
    fig = plot_metric_bars(data, metric_cols=["my_metric_name"], show=False)
    assert isinstance(fig, plt.Figure)


def test_plot_subject_trajectories_single_group():
    """Test trajectories with only one group (edge case)."""
    data = {"subject": ["s1", "s2"], "group": ["A", "A"], "score": [1.0, 1.1]}
    fig = plot_metric_comparison(
        data, metric_col="score", group_col="group", subject_col="subject", show=False
    )
    assert isinstance(fig, plt.Figure)


def test_plot_metric_bars_extras(single_subject_data):
    """plot_metric_bars."""
    data = dict(single_subject_data)
    # Test with custom group colors and labels
    fig = plot_metric_bars(
        data,
        metric_cols=["R_f0"],
        group_col="method",
        group_colors={"M1": "red"},
        group_labels={"M1": "Method 1"},
        title="Custom Title",
        show=False,
    )
    assert isinstance(fig, plt.Figure)


def test_plot_tradeoff_scatter_refs(single_subject_data):
    """Cover reference lines in tradeoff scatter."""
    fig = plot_tradeoff_scatter(
        single_subject_data,
        x_col="below_noise_distortion_db",
        y_col="peak_attenuation_db",
        group_col="method",
        reference_x=0.0,
        reference_y=10.0,
        show=False,
    )
    assert isinstance(fig, plt.Figure)


def test_plot_forest_extended(multi_subject_data):
    fig = plot_forest(
        multi_subject_data,
        metric_col="R_f0",
        se_col="peak_attenuation_db",  # Fake SE for coverage
        group_col="method",
        subject_col="subject",
        baseline_group="M0",
        show=False,
    )
    assert isinstance(fig, plt.Figure)


def test_plot_metric_violins_single_metric(multi_subject_data):
    """Test plot_metric_violins with a single metric name."""
    fig = plot_metric_violins(
        multi_subject_data,
        metric_cols=["R_f0"],  # Test as list of 1
        group_col="method",
        subject_col="subject",
        show=False,
    )
    assert isinstance(fig, plt.Figure)


def test_plot_metric_bars_nan_mean():
    """Test plot_metric_bars with all NaN group."""
    data = {"group": ["A", "B"], "metric": [np.nan, 1.0]}
    fig = plot_metric_bars(
        data, metric_cols=["metric"], group_order=["A", "B"], show=False
    )
    assert isinstance(fig, plt.Figure)


def test_plot_forest_sd():
    """Test SD in plot_forest for subjects with missing data."""
    data = {"subject": ["s1", "s2"], "group": ["A", "A"], "metric": [1.0, np.nan]}
    fig = plot_forest(
        data, metric_col="metric", group_col="group", subject_col="subject", show=False
    )
    assert isinstance(fig, plt.Figure)


def test_plot_harmonic_attenuation_missing_series(freqs, gm_psd, cleaned_psds):
    """Test plot_harmonic_attenuation with missing series in data."""
    fig = plot_harmonic_attenuation(
        freqs,
        gm_psd,
        cleaned_psds,
        harmonics_hz=[50],
        series_order=["M1", "Missing"],
        show=False,
    )
    assert isinstance(fig, plt.Figure)
