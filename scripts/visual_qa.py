#!/usr/bin/env python
"""Visual QA — Exercise every mne_denoise.viz function on real data.

Loads the Runabout oddball dataset (sub-01), runs DSS + ZapLine,
then calls every public plot function with ``show=False`` and saves
PNG files to ``data/runabout/derivatives/mne-denoise/visual_qa/``.

Run with:
    python scripts/visual_qa.py
"""

from __future__ import annotations

import pathlib
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

# ── project imports ───────────────────────────────────────────────────
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from mne_denoise import viz
from mne_denoise.dss import DSS, AverageBias
from mne_denoise.zapline import ZapLine

warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("WARNING")

# ── paths ─────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data" / "runabout"
OUT_DIR = DATA_ROOT / "derivatives" / "mne-denoise" / "visual_qa"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Sub-directories for organized output
SIGSPEC_DIR = OUT_DIR / "signals_spectra"
SIGSPEC_DIR.mkdir(exist_ok=True)
COMPS_DIR = OUT_DIR / "components"
COMPS_DIR.mkdir(exist_ok=True)
SUMMARY_DSS_DIR = OUT_DIR / "summary_dss_input"
SUMMARY_DSS_DIR.mkdir(exist_ok=True)
SUMMARY_ZAP_DIR = OUT_DIR / "summary_zapline_input"
SUMMARY_ZAP_DIR.mkdir(exist_ok=True)
STATSSPEC_DIR = OUT_DIR / "stats_spectra"
STATSSPEC_DIR.mkdir(exist_ok=True)
THEME_DIR = OUT_DIR / "theme"
THEME_DIR.mkdir(exist_ok=True)

# ── config ────────────────────────────────────────────────────────────
SUB = "sub-01"
LINE_FREQ = 50.0
RESAMPLE_FREQ = 250
HP_FREQ = 0.5
LP_FREQ = 40.0
DSS_N_COMPONENTS = 5
DSS_N_KEEP = 3

passed, failed = [], []


def _try(name: str, func, *args, **kwargs):
    """Run func, track pass/fail, close figures."""
    print(f"  {name} ... ", end="", flush=True)
    try:
        result = func(*args, **kwargs)
        passed.append(name)
        print("✓")
        return result
    except Exception as exc:
        failed.append((name, str(exc)))
        print(f"FAILED: {exc}")
        return None
    finally:
        plt.close("all")


def _geometric_mean_psd_from_psd(psd):
    """Return geometric-mean PSD across channels."""
    psd = np.asarray(psd, dtype=float)
    return np.exp(np.mean(np.log(np.maximum(psd, 1e-30)), axis=0))


# ══════════════════════════════════════════════════════════════════════
# 1 ── Load & preprocess one subject
# ══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Visual QA — loading & preprocessing data")
print("=" * 60)

eeg_dir = DATA_ROOT / SUB / "eeg"
vhdr = sorted(eeg_dir.glob("*.vhdr"))[0]
raw = mne.io.read_raw_brainvision(str(vhdr), preload=True, verbose=False)

# Clean channel names
mapping = {}
for ch in raw.ch_names:
    cleaned = ch.encode("latin-1", errors="replace").decode("utf-8", errors="replace")
    if cleaned != ch:
        mapping[ch] = cleaned
if mapping:
    raw.rename_channels(mapping)

raw.resample(RESAMPLE_FREQ, verbose=False)
sfreq = raw.info["sfreq"]
raw.pick_types(eeg=True, verbose=False)

if "FCz" not in raw.ch_names:
    raw.add_reference_channels("FCz")
raw.set_montage("standard_1020", on_missing="warn")

mastoids = [ch for ch in ["TP9", "TP10"] if ch in raw.ch_names]
if len(mastoids) == 2:
    raw.set_eeg_reference(mastoids, verbose=False)
    raw.drop_channels(mastoids)
else:
    raw.set_eeg_reference("average", verbose=False)

raw.filter(l_freq=HP_FREQ, h_freq=None, verbose=False)

# ── ZapLine (before bandpass so we can visualize line noise removal) ──
print("  Running ZapLine ...")
raw_before_zap = raw.copy()
zap = ZapLine(sfreq=sfreq, line_freq=LINE_FREQ, n_remove="auto", n_harmonics=3)
raw = zap.fit_transform(raw)
print(f"  ZapLine removed {getattr(zap, 'n_removed_', '?')} components")

# Low-pass for ERP analysis
raw.filter(l_freq=None, h_freq=LP_FREQ, verbose=False)

# ── Epoch ──
print("  Epoching ...")
trig_dir = DATA_ROOT / "derivatives" / "trigger_corrected" / SUB / "eeg"
evt_file = sorted(trig_dir.glob("*_events.tsv"))[0]
tc_df = pd.read_csv(evt_file, sep="\t")
tc_df = tc_df[tc_df["trial_type"] != "empty"].reset_index(drop=True)
tc_df["sample_rs"] = np.round(tc_df["onset"].values * (sfreq / 500.0)).astype(int)

unique_types = sorted(tc_df["trial_type"].unique())
event_id = {tt: i + 1 for i, tt in enumerate(unique_types)}
events_arr = np.column_stack(
    [
        tc_df["sample_rs"].values,
        np.zeros(len(tc_df), dtype=int),
        np.array([event_id[tt] for tt in tc_df["trial_type"]]),
    ]
)
valid = (events_arr[:, 0] >= 0) & (events_arr[:, 0] < raw.n_times - 1)
events_arr = events_arr[valid]

epochs = mne.Epochs(
    raw,
    events_arr,
    event_id=event_id,
    tmin=-0.2,
    tmax=0.8,
    baseline=(-0.2, 0),
    preload=True,
    verbose=False,
)
epochs.drop_bad(reject={"eeg": 125e-6}, verbose=False)
print(f"  {len(epochs)} epochs after rejection")

# ── DSS on epochs ──
print("  Fitting DSS (AverageBias) ...")
n_ep = len(epochs)
epochs_train = epochs[np.arange(0, n_ep, 2)]
epochs_test = epochs[np.arange(1, n_ep, 2)]

bias = AverageBias(axis="epochs")
dss = DSS(bias=bias, n_components=DSS_N_COMPONENTS, return_type="sources")
dss.fit(epochs_train)

sources_train = dss.transform(epochs_train)
sources_kept = sources_train.copy()
sources_kept[:, DSS_N_KEEP:, :] = 0
recon = dss.inverse_transform(sources_kept)
if recon.ndim == 2:
    recon = recon[np.newaxis, ...]

epochs_denoised = mne.EpochsArray(
    recon,
    epochs_train.info,
    tmin=epochs_train.tmin,
    verbose=False,
)
print(f"  DSS eigenvalues: {dss.eigenvalues_[:5].round(5)}")

evoked_orig = epochs_train.average()
evoked_denoised = epochs_denoised.average()
info = epochs.info
times = epochs_train.times

# Data arrays for zapline viz
data_before_zap = raw_before_zap.get_data()
data_after_zap = raw.get_data()


# ══════════════════════════════════════════════════════════════════════
# 2 ── Theme tests
# ══════════════════════════════════════════════════════════════════════
print("\n── Theme ──")


def test_use_theme():
    """Test use_theme context manager."""
    with viz.use_theme():
        fig, ax = plt.subplots()
        ax.plot([0, 1, 2], [0, 1, 4], label="test")
        ax.set_title("use_theme() context manager")
        ax.legend()
        fig.savefig(THEME_DIR / "use_theme.png", dpi=150, bbox_inches="tight")
    return fig


_try("use_theme", test_use_theme)


def test_themed_figure():
    """Test themed_figure helper."""
    fig, ax = viz.themed_figure()
    ax.plot(np.linspace(0, 10, 50), np.sin(np.linspace(0, 10, 50)))
    ax.set_title("themed_figure()")
    fig.savefig(THEME_DIR / "themed_figure.png", dpi=150, bbox_inches="tight")
    return fig


_try("themed_figure", test_themed_figure)


def test_get_color():
    """Verify get_color returns correct colors for all keys."""
    from mne_denoise.viz import get_color

    for key in ["blue", "orange", "green", "dss", "zapline", "clean"]:
        c = get_color(key)
        assert c is not None, f"get_color({key!r}) returned None"
    return True


_try("get_color", test_get_color)


# ══════════════════════════════════════════════════════════════════════
# 3 ── Signal/Spectral primitives
# ══════════════════════════════════════════════════════════════════════
print("\n── Signals + Spectra ──")

_try(
    "plot_psd_comparison",
    viz.plot_psd_comparison,
    epochs_train,
    epochs_denoised,
    fmax=45,
    show=False,
    fname=str(SIGSPEC_DIR / "psd_comparison.png"),
)

_try(
    "plot_evoked_gfp_comparison",
    viz.plot_evoked_gfp_comparison,
    evoked_orig,
    evoked_denoised,
    times=evoked_orig.times,
    show=False,
    fname=str(SIGSPEC_DIR / "evoked_comparison.png"),
)

_try(
    "plot_channel_time_course_comparison",
    viz.plot_channel_time_course_comparison,
    epochs_train,
    epochs_denoised,
    picks=[0, 1],
    times=times,
    show=False,
    fname=str(SIGSPEC_DIR / "time_course_comparison.png"),
)

_try(
    "plot_power_ratio_map",
    viz.plot_power_ratio_map,
    epochs_train,
    epochs_denoised,
    info=info,
    show=False,
    fname=str(SIGSPEC_DIR / "power_map.png"),
)

_try(
    "plot_spectrogram_comparison",
    viz.plot_spectrogram_comparison,
    epochs_train,
    epochs_denoised,
    picks=[0, 1],
    times=times,
    fmin=1,
    fmax=40,
    show=False,
    fname=str(SIGSPEC_DIR / "spectrogram_comparison.png"),
)

_try(
    "plot_denoising_summary",
    viz.plot_denoising_summary,
    epochs_train,
    epochs_denoised,
    info=info,
    times=times,
    show=False,
    fname=str(SIGSPEC_DIR / "denoising_summary.png"),
)

_try(
    "plot_signal_overlay",
    viz.plot_signal_overlay,
    epochs_train,
    epochs_denoised,
    times=times,
    pick=0,
    show=False,
    fname=str(SIGSPEC_DIR / "overlay_comparison.png"),
)


# ── spectral PSD comparison (needs component data + sfreq) ──
def _spectral_psd():
    # Use DSS sources as "components"
    sources = dss.transform(epochs_train)  # (n_epochs, n_comp, n_times)
    avg_sources = sources.mean(axis=0)  # (n_comp, n_times)
    return viz.plot_component_psd_comparison(
        epochs_train,
        avg_sources,
        component_indices=list(range(min(3, avg_sources.shape[0]))),
        sfreq=sfreq,
        fmin=1,
        fmax=40,
        show=False,
        fname=str(SIGSPEC_DIR / "spectral_psd_comparison.png"),
    )


_try("plot_component_psd_comparison", _spectral_psd)


# ══════════════════════════════════════════════════════════════════════
# 4 ── Components module
# ══════════════════════════════════════════════════════════════════════
print("\n── Components ──")

_try(
    "plot_component_score_curve",
    viz.plot_component_score_curve,
    dss,
    show=False,
    fname=str(COMPS_DIR / "score_curve.png"),
)

_try(
    "plot_component_patterns",
    viz.plot_component_patterns,
    dss,
    info=info,
    n_components=5,
    show=False,
    fname=str(COMPS_DIR / "spatial_patterns.png"),
)

_try(
    "plot_component_summary",
    viz.plot_component_summary,
    dss,
    data=epochs_train,
    info=info,
    n_components=3,
    show=False,
    fname=str(COMPS_DIR / "component_summary.png"),
)

_try(
    "plot_component_epochs_image",
    viz.plot_component_epochs_image,
    dss,
    data=epochs_train,
    n_components=3,
    show=False,
    fname=str(COMPS_DIR / "component_image.png"),
)

_try(
    "plot_component_time_series",
    viz.plot_component_time_series,
    dss,
    data=epochs_train,
    n_components=3,
    show=False,
    fname=str(COMPS_DIR / "component_time_series.png"),
)


def _narrowband_scan():
    freqs = np.arange(1, 41)
    evals = np.random.default_rng(42).exponential(0.1, size=len(freqs))
    evals[9] = 0.8  # spike at 10 Hz
    return viz.plot_narrowband_score_scan(
        freqs,
        evals,
        peak_freq=10.0,
        show=False,
        fname=str(COMPS_DIR / "narrowband_scan.png"),
    )


_try("plot_narrowband_score_scan", _narrowband_scan)


def _tf_mask():
    times = np.linspace(-0.2, 0.8, 100)
    freqs = np.arange(1, 41)
    mask = np.zeros((len(freqs), len(times)))
    mask[8:12, 30:70] = 1.0  # 9–12 Hz, 100–500 ms
    return viz.plot_time_frequency_mask(
        mask,
        times,
        freqs,
        title="Example TF Mask",
        show=False,
        fname=str(COMPS_DIR / "tf_mask.png"),
    )


_try("plot_time_frequency_mask", _tf_mask)


def _component_spectrogram():
    sources = dss.transform(epochs_train)
    comp0 = sources[:, 0, :]  # (n_epochs, n_times)
    return viz.plot_component_spectrogram(
        comp0,
        sfreq=sfreq,
        title="DSS Component 1 Spectrogram",
        show=False,
        fname=str(COMPS_DIR / "component_spectrogram.png"),
    )


_try("plot_component_spectrogram", _component_spectrogram)


# ══════════════════════════════════════════════════════════════════════
# 5 ── Summary (DSS input)
# ══════════════════════════════════════════════════════════════════════
print("\n── Summary (DSS-style input) ──")

_try(
    "plot_component_cleaning_summary [dss]",
    viz.plot_component_cleaning_summary,
    scores=getattr(dss, "eigenvalues_", None),
    selected_count=getattr(dss, "n_selected_", 0),
    patterns=getattr(dss, "patterns_", None),
    removed=epochs_train.average().get_data() - epochs_denoised.average().get_data(),
    sources=getattr(dss, "sources_", None),
    sfreq=sfreq,
    info=info,
    title="Component Cleaning Summary (DSS, sub-01)",
    show=False,
    fname=str(SUMMARY_DSS_DIR / "component_cleaning_summary.png"),
)


# ══════════════════════════════════════════════════════════════════════
# 6 ── Summary (ZapLine input)
# ══════════════════════════════════════════════════════════════════════
print("\n── Summary (ZapLine-style input) ──")

_try(
    "plot_psd_comparison [zapline arrays]",
    viz.plot_psd_comparison,
    data_before_zap,
    data_after_zap,
    sfreq=sfreq,
    line_freq=LINE_FREQ,
    fmax=100,
    show=False,
    fname=str(SUMMARY_ZAP_DIR / "psd_comparison.png"),
)

_try(
    "plot_component_score_curve [zapline]",
    viz.plot_component_score_curve,
    zap,
    show=False,
    fname=str(SUMMARY_ZAP_DIR / "component_scores.png"),
)

_try(
    "plot_component_patterns [zapline]",
    viz.plot_component_patterns,
    zap,
    show=False,
    fname=str(SUMMARY_ZAP_DIR / "spatial_patterns.png"),
)

_try(
    "plot_component_cleaning_summary [zapline]",
    viz.plot_component_cleaning_summary,
    scores=getattr(zap, "scores_", None),
    selected_count=getattr(zap, "n_removed_", 0),
    patterns=getattr(zap, "patterns_", None),
    removed=data_before_zap - data_after_zap,
    sources=getattr(zap, "sources_", None),
    sfreq=sfreq,
    info=raw_before_zap.info,
    line_freq=LINE_FREQ,
    title="Component Cleaning Summary (ZapLine, sub-01)",
    show=False,
    fname=str(SUMMARY_ZAP_DIR / "component_cleaning_summary.png"),
)


# ══════════════════════════════════════════════════════════════════════
# 7 ── Stats + Spectral dashboards
# ══════════════════════════════════════════════════════════════════════
print("\n── Stats + Spectra ──")

# Load pre-computed group metrics from disk
# NOTE: benchmark viz was designed for line-noise metrics (R_f0, etc.)
DERIV = DATA_ROOT / "derivatives" / "mne-denoise"
ln_metrics = DERIV / "group" / "line_noise" / "metrics_all.tsv"
if ln_metrics.exists():
    df_bench = pd.read_csv(ln_metrics, sep="\t")
else:
    # Synthesize minimal test dataframe with line-noise columns
    rows = []
    rng = np.random.default_rng(42)
    for sub in ["sub-01", "sub-02", "sub-03"]:
        for mtag in ["M0", "M1", "M2"]:
            rows.append(
                {
                    "subject": sub,
                    "method": mtag,
                    "R_f0": rng.normal(0.5, 0.2),
                    "peak_attenuation_db": rng.normal(15, 5),
                    "below_noise_distortion_db": rng.normal(0.1, 0.05),
                    "overclean_proportion": rng.normal(0.05, 0.02),
                    "underclean_proportion": rng.normal(0.05, 0.02),
                }
            )
    df_bench = pd.DataFrame(rows)

METHOD_ORDER = sorted(df_bench["method"].unique())
STATS_DATA = {col: df_bench[col].to_numpy() for col in df_bench.columns}
METHOD_COLORS = {
    "M0": "#404040",
    "M1": "#2ca02c",
    "M2": "#d62728",
    "C0": "#404040",
    "C1": "#2ca02c",
    "C2": "#d62728",
}
METHOD_LABELS = {
    "M0": "Baseline",
    "M1": "ZapLine",
    "M2": "DSS-ZapLine",
    "C0": "Baseline",
    "C1": "Paper (ICA)",
    "C2": "DSS (AverageBias)",
}

_try(
    "plot_metric_bars",
    viz.plot_metric_bars,
    STATS_DATA,
    group_col="method",
    metric_cols=[
        "R_f0",
        "peak_attenuation_db",
        "below_noise_distortion_db",
        "overclean_proportion",
        "underclean_proportion",
    ],
    metric_labels=[
        "R(f0) - 1",
        "Peak Attenuation (dB)",
        "Below-Noise Distortion (dB) - 0",
        "Overclean Fraction",
        "Underclean Fraction",
    ],
    lower_better=[True, False, True, True, True],
    group_order=METHOD_ORDER,
    group_colors=METHOD_COLORS,
    show=False,
    fname=str(STATSSPEC_DIR / "metric_bars.png"),
)

_try(
    "plot_tradeoff_scatter",
    viz.plot_tradeoff_scatter,
    STATS_DATA,
    group_col="method",
    x_col="below_noise_distortion_db",
    y_col="peak_attenuation_db",
    group_order=METHOD_ORDER,
    group_colors=METHOD_COLORS,
    group_labels=METHOD_LABELS,
    x_label="Below-Noise Distortion (dB) - closer to 0 is better",
    y_label="Peak Attenuation (dB) - higher is better",
    reference_x=0.0,
    reference_y=10.0,
    show=False,
    fname=str(STATSSPEC_DIR / "tradeoff_scatter.png"),
)

_try(
    "plot_metric_comparison",
    viz.plot_metric_comparison,
    STATS_DATA,
    group_col="method",
    metric_col="R_f0",
    metric_label="R(f0) - Noise-Surround Ratio",
    group_order=METHOD_ORDER,
    group_colors=METHOD_COLORS,
    group_labels=METHOD_LABELS,
    title="Residual Line Noise — R(f₀)",
    reference_value=1.0,
    reference_label="Ideal (R=1)",
    show=False,
    fname=str(STATSSPEC_DIR / "metric_comparison.png"),
)

_try(
    "plot_metric_slopes",
    viz.plot_metric_slopes,
    STATS_DATA,
    group_col="method",
    metric_specs=[
        ("peak_attenuation_db", "Peak Attenuation (dB)"),
        ("below_noise_distortion_db", "Below-Noise Distortion (dB)"),
        ("R_f0", "R(f0)"),
    ],
    group_order=METHOD_ORDER,
    group_colors=METHOD_COLORS,
    show=False,
    fname=str(STATSSPEC_DIR / "metric_slopes.png"),
)

_try(
    "plot_metric_tradeoff_summary",
    viz.plot_metric_tradeoff_summary,
    STATS_DATA,
    group_col="method",
    subject_col="subject",
    x_col="below_noise_distortion_db",
    y_col="peak_attenuation_db",
    metric_col="R_f0",
    group_order=METHOD_ORDER,
    group_colors=METHOD_COLORS,
    group_labels=METHOD_LABELS,
    show=False,
    fname=str(STATSSPEC_DIR / "tradeoff_and_r.png"),
)


# ── PSD-based benchmark plots (need raw freq data) ──
def _make_psd_data():
    """Compute PSD arrays for benchmark gallery/overlay/harmonic funcs."""
    from scipy.signal import welch

    nperseg = min(512, data_before_zap.shape[-1])
    freqs_b, psd_b = welch(data_before_zap, fs=sfreq, nperseg=nperseg, axis=-1)
    gm_before = _geometric_mean_psd_from_psd(psd_b)

    cleaned_psds = {}
    for tag, d in [("C0", data_before_zap), ("C2", data_after_zap)]:
        freqs_a, psd_a = welch(d, fs=sfreq, nperseg=nperseg, axis=-1)
        gm = _geometric_mean_psd_from_psd(psd_a)
        cleaned_psds[tag] = (freqs_a, gm)

    return freqs_b, gm_before, cleaned_psds


freqs_b, gm_before, cleaned_psds = _make_psd_data()
nyquist = sfreq / 2.0
harmonics_hz = [LINE_FREQ * (h + 1) for h in range(3) if LINE_FREQ * (h + 1) < nyquist]

_try(
    "plot_psd_zoom_comparison",
    viz.plot_psd_zoom_comparison,
    freqs_b,
    gm_before,
    cleaned_psds["C2"][0],
    cleaned_psds["C2"][1],
    series_name="C2",
    title=SUB,
    zoom_freqs=harmonics_hz,
    series_colors=METHOD_COLORS,
    series_labels=METHOD_LABELS,
    fmax=sfreq / 2,
    show=False,
    fname=str(STATSSPEC_DIR / "qc_psd.png"),
)

_try(
    "plot_psd_gallery",
    viz.plot_psd_gallery,
    freqs_b,
    gm_before,
    cleaned_psds,
    zoom_freqs=harmonics_hz,
    fmax=sfreq / 2,
    title=SUB,
    series_order=list(cleaned_psds.keys()),
    series_colors=METHOD_COLORS,
    series_labels=METHOD_LABELS,
    show=False,
    fname=str(STATSSPEC_DIR / "psd_gallery.png"),
)

_try(
    "plot_psd_overlay",
    viz.plot_psd_overlay,
    freqs_b,
    gm_before,
    cleaned_psds,
    focus_freq=LINE_FREQ,
    fmax=sfreq / 2,
    title=SUB,
    series_order=list(cleaned_psds.keys()),
    series_colors=METHOD_COLORS,
    series_labels=METHOD_LABELS,
    show=False,
    fname=str(STATSSPEC_DIR / "subject_psd_overlay.png"),
)

_try(
    "plot_harmonic_attenuation",
    viz.plot_harmonic_attenuation,
    freqs_b,
    gm_before,
    cleaned_psds,
    harmonics_hz=harmonics_hz,
    subject=SUB,
    series_order=list(cleaned_psds.keys()),
    series_colors=METHOD_COLORS,
    series_labels=METHOD_LABELS,
    show=False,
    fname=str(STATSSPEC_DIR / "harmonic_attenuation.png"),
)


# ══════════════════════════════════════════════════════════════════════
# 8 ── Summary
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Visual QA Summary")
print("=" * 60)
print(f"  Passed: {len(passed)}/{len(passed) + len(failed)}")
for name in passed:
    print(f"    ✓ {name}")
if failed:
    print(f"\n  Failed: {len(failed)}")
    for name, err in failed:
        print(f"    ✗ {name}: {err}")
print(f"\n  Output: {OUT_DIR}")
print("Done.")
