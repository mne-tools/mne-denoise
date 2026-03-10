#!/usr/bin/env python
"""
Runabout ds003620 — Line-Noise Removal Benchmark (all 44 subjects).

Headless script version of runabout_line_noise_benchmark.ipynb.
Processes every subject, saves all per-subject artefacts *and* group-level
DataFrames, PSDs, and publication figures so that downstream visualisation
can be changed without re-running on the cluster.

Usage
-----
  # Single subject (SLURM array element or local test):
  python run_line_noise_benchmark.py --subject sub-01

  # All subjects sequentially (local or single SLURM job):
  python run_line_noise_benchmark.py --all

  # Slurm array mode — picks subject from $SLURM_ARRAY_TASK_ID:
  python run_line_noise_benchmark.py --slurm-array

  # Group aggregation only (after all per-subject jobs finished):
  python run_line_noise_benchmark.py --group-only

Output contract (per subject)
-----------------------------
  {DERIV_ROOT}/{sub}/line_noise/{method}/
      cleaned_raw.fif
      metrics.tsv
      condition_metrics.tsv
      qc_psd.png
      diagnostics.png
      model.json
  {DERIV_ROOT}/{sub}/line_noise/
      psd_comparison.png

Output contract (group)
-----------------------
  {DERIV_ROOT}/group/line_noise/
      all_metrics.tsv          <-- master DataFrame
      whole_recording.tsv      <-- df_all
      per_condition.tsv        <-- df_cond
      group_psds_{sub}.npz     <-- baseline + cleaned PSDs per subject
      psd_gallery.png
      metric_bars.png
      tradeoff_and_r.png
      paired_metrics.png
      harmonic_attenuation.png
      condition_r_f0_bars.png
      condition_metrics_grouped.png
      condition_psd_comparison.png
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

# ── Headless matplotlib BEFORE any other import that touches it ──────────────
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy import signal

# ── Ensure mne-denoise is importable ────────────────────────────────────────
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from mne_denoise.qa import (
    compute_all_qa_metrics,
)
from mne_denoise.viz import (
    plot_component_cleaning_summary,
    plot_harmonic_attenuation,
    plot_metric_bars,
    plot_metric_slopes,
    plot_metric_tradeoff_summary,
    plot_psd_gallery,
    plot_psd_overlay,
    plot_psd_zoom_comparison,
    set_theme,
)
from mne_denoise.viz.theme import COLORS, FONTS, style_axes, themed_legend
from mne_denoise.zapline import ZapLine

warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("WARNING")

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("line_noise")

# ═══════════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════════
DATASET_ID = "ds003620"
LINE_FREQ = 50.0
LINE_HARMONICS = 3
TASK = "oddball"
RESAMPLE_FREQ = 250
HIGHPASS_FREQ = 0.5
ORIG_SFREQ = 500.0
PSD_FMAX = 125.0
RANDOM_STATE = 42

CONDITIONS = ["lab", "oval", "campus"]
CONDITION_LABELS = {
    "lab": "Lab (seated)",
    "oval": "Oval (walking)",
    "campus": "Campus (walking)",
}
CONDITION_COLORS = {
    "lab": "#0072B2",
    "oval": "#E69F00",
    "campus": "#D55E00",
}

# Method setup (exclude M3-notch, keep M0/M1/M2)
METHOD_LABELS = {
    "M0": "Original",
    "M1": "ZapLine",
    "M2": "ZapLine adaptive",
}
METHOD_COLORS = {
    "M0": COLORS["before"],
    "M1": COLORS["primary"],
    "M2": COLORS["secondary"],
}
METHOD_ORDER = ["M0", "M1", "M2"]

# 44 subjects total
ALL_SUBJECTS = [f"sub-{i:02d}" for i in range(1, 45)]


def _resolve_paths():
    """Auto-detect DATA_ROOT / DERIV_ROOT for Narval vs local."""
    if "CC_CLUSTER" in os.environ:
        # Narval / Alliance cluster
        data_dir = Path(
            os.environ.get(
                "DATA_DIR",
                str(Path.home() / "scratch" / "mnedenoise" / "data"),
            )
        )
        data_root = data_dir / DATASET_ID
        deriv_root = data_root / "derivatives" / "mne-denoise"
    else:
        # Local
        data_root = _REPO / "data" / "runabout"
        deriv_root = data_root / "derivatives" / "mne-denoise"
    return data_root, deriv_root


DATA_ROOT, DERIV_ROOT = _resolve_paths()


# ═══════════════════════════════════════════════════════════════════════════════
#  Preprocessing
# ═══════════════════════════════════════════════════════════════════════════════


def load_and_preprocess(sub, ds_path):
    """Load raw BrainVision data and apply shared preprocessing."""
    eeg_dir = ds_path / sub / "eeg"
    vhdr_files = sorted(eeg_dir.glob("*.vhdr"))
    if not vhdr_files:
        raise FileNotFoundError(f"No .vhdr file found for {sub} in {eeg_dir}")
    raw = mne.io.read_raw_brainvision(str(vhdr_files[0]), preload=True, verbose=False)

    # Fix encoding issues in channel names
    mapping = {}
    for ch in raw.ch_names:
        cleaned = ch.encode("latin-1", errors="replace").decode(
            "utf-8", errors="replace"
        )
        if cleaned != ch:
            mapping[ch] = cleaned
    if mapping:
        raw.rename_channels(mapping)

    # Resample
    raw.resample(RESAMPLE_FREQ, verbose=False)

    # Pick EEG + add back online reference (FCz)
    raw.pick_types(eeg=True, verbose=False)
    if "FCz" not in raw.ch_names:
        raw.add_reference_channels("FCz")
    raw.set_montage("standard_1020", on_missing="warn")

    # Re-reference to linked mastoids
    mastoids = [ch for ch in ["TP9", "TP10"] if ch in raw.ch_names]
    if len(mastoids) == 2:
        raw.set_eeg_reference(mastoids, verbose=False)
        raw.drop_channels(mastoids)
    else:
        raw.set_eeg_reference("average", verbose=False)

    # High-pass filter
    raw.filter(
        l_freq=HIGHPASS_FREQ,
        h_freq=None,
        method="fir",
        fir_design="firwin",
        verbose=False,
    )
    return raw


# ═══════════════════════════════════════════════════════════════════════════════
#  Condition Block Parsing
# ═══════════════════════════════════════════════════════════════════════════════


def parse_condition_blocks(
    sub,
    ds_path=None,
    sfreq_orig=ORIG_SFREQ,
    sfreq_new=RESAMPLE_FREQ,
    gap_thresh=5.0,
    pad=1.0,
):
    """Identify contiguous condition blocks from trigger-corrected events."""
    if ds_path is None:
        ds_path = DATA_ROOT
    trig_path = (
        ds_path
        / "derivatives"
        / "trigger_corrected"
        / sub
        / "eeg"
        / f"{sub}_task-{TASK}_desc-trig_events.tsv"
    )
    if not trig_path.exists():
        log.warning("%s: trigger_corrected events not found — skipping", sub)
        return {}

    df = pd.read_csv(trig_path, sep="\t")
    df = df[df["trial_type"] != "empty"].copy()

    def _get_env(tt):
        for env in CONDITIONS:
            if tt.endswith(f"_{env}"):
                return env
        return None

    df["env"] = df["trial_type"].apply(_get_env)
    df = df.dropna(subset=["env"])
    df["onset_s"] = df["onset"] / sfreq_orig

    blocks = {c: [] for c in CONDITIONS}
    for env in CONDITIONS:
        env_df = df[df["env"] == env].sort_values("onset_s")
        if env_df.empty:
            continue
        onsets = env_df["onset_s"].values
        block_start = onsets[0]
        prev = onsets[0]
        for t in onsets[1:]:
            if t - prev > gap_thresh:
                blocks[env].append((max(0, block_start - pad), prev + pad))
                block_start = t
            prev = t
        blocks[env].append((max(0, block_start - pad), prev + pad))

    for env in CONDITIONS:
        n_blocks = len(blocks[env])
        if n_blocks:
            total_s = sum(e - s for s, e in blocks[env])
            log.info("  %s %s: %d block(s), %.0f s total", sub, env, n_blocks, total_s)
    return blocks


def crop_to_condition(raw, tmin, tmax):
    """Crop a Raw object to [tmin, tmax], clamping to valid range."""
    t_start = max(tmin, raw.times[0])
    t_end = min(tmax, raw.times[-1])
    if t_end <= t_start:
        return None
    return raw.copy().crop(tmin=t_start, tmax=t_end)


# ═══════════════════════════════════════════════════════════════════════════════
#  Method Definitions
# ═══════════════════════════════════════════════════════════════════════════════


def method_m0(raw):
    """M0: No line-noise removal (baseline)."""
    return raw.copy(), {"method": "M0_baseline", "description": "No cleaning"}, None


def method_m1(raw):
    """M1: Standard ZapLine (automatic component selection)."""
    zap = ZapLine(
        sfreq=raw.info["sfreq"],
        line_freq=LINE_FREQ,
        n_remove="auto",
        n_harmonics=LINE_HARMONICS,
    )
    raw_clean = zap.fit_transform(raw.copy())
    info = {
        "method": "M1_zapline_standard",
        "n_remove": "auto",
        "n_harmonics": LINE_HARMONICS,
        "n_removed_": getattr(zap, "n_removed_", None),
    }
    return raw_clean, info, zap


def method_m2(raw):
    """M2: Adaptive ZapLine+ (segment-wise adaptive)."""
    zap = ZapLine(
        sfreq=raw.info["sfreq"],
        line_freq=LINE_FREQ,
        adaptive=True,
        adaptive_params={
            "n_remove_params": {
                "sigma": 3.0,
                "min_remove": 1,
                "max_prop": 0.2,
            },
        },
        n_harmonics=LINE_HARMONICS,
    )
    raw_clean = zap.fit_transform(raw.copy())
    info = {
        "method": "M2_zapline_adaptive",
        "adaptive": True,
        "n_harmonics": LINE_HARMONICS,
        "n_removed_": getattr(zap, "n_removed_", None),
    }
    return raw_clean, info, zap


METHODS = {
    "M0": method_m0,
    "M1": method_m1,
    "M2": method_m2,
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Output Saving
# ═══════════════════════════════════════════════════════════════════════════════


def _plot_method_diagnostics(
    estimator,
    data_before,
    data_after,
    *,
    sfreq,
    title,
    show,
):
    """Create a generic component-cleaning diagnostics dashboard."""
    scores = (
        getattr(estimator, "scores_", None)
        if hasattr(estimator, "scores_")
        else getattr(estimator, "eigenvalues_", None)
    )
    selected_count = getattr(estimator, "n_removed_", None)
    if selected_count is None:
        selected_count = getattr(estimator, "n_selected_", 0)
    patterns = getattr(estimator, "patterns_", None)
    removed = data_before - data_after
    sources = getattr(estimator, "sources_", None)

    nperseg = min(data_before.shape[-1], int(sfreq * 2))
    freqs, psd_before = signal.welch(data_before, fs=sfreq, nperseg=nperseg, axis=-1)
    _, psd_after = signal.welch(data_after, fs=sfreq, nperseg=nperseg, axis=-1)

    return plot_component_cleaning_summary(
        scores=scores,
        selected_count=selected_count,
        patterns=patterns,
        removed=removed,
        sources=sources,
        sfreq=sfreq,
        freqs=freqs,
        psd_before=psd_before,
        psd_after=psd_after,
        line_freq=LINE_FREQ,
        title=title,
        show=show,
    )


def _geometric_mean_psd_raw(raw, *, fmax=125.0):
    """Compute geometric-mean PSD across channels from a Raw object."""
    psd = raw.compute_psd(fmax=fmax, verbose=False)
    data = psd.get_data()
    gm = np.exp(np.mean(np.log(np.maximum(data, 1e-30)), axis=0))
    return psd.freqs, gm


def _save_model_json(model_info, path):
    """Write model metadata dict to JSON (numpy-safe)."""

    def _default(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Cannot serialize {type(o)}")

    with open(path, "w") as f:
        json.dump(model_info, f, indent=2, default=_default)


def save_outputs(
    sub,
    method_tag,
    raw_before,
    raw_after,
    metrics_dict,
    model_info,
    estimator=None,
    condition_metrics=None,
    deriv_root=None,
):
    """Save outputs + generate per-subject diagnostics for one method."""
    if deriv_root is None:
        deriv_root = DERIV_ROOT
    out_dir = deriv_root / sub / "line_noise" / method_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cleaned raw
    raw_after.save(out_dir / "cleaned_raw.fif", overwrite=True, verbose=False)

    # 2. Metrics TSV (whole-recording)
    flat = {k: v for k, v in metrics_dict.items() if not isinstance(v, (list, dict))}
    df = pd.DataFrame([flat])
    df.insert(0, "subject", sub)
    df.insert(1, "method", method_tag)
    df.to_csv(out_dir / "metrics.tsv", sep="\t", index=False)

    # 2b. Condition-level metrics TSV
    if condition_metrics:
        cond_rows = []
        for cond, cmetrics in condition_metrics.items():
            cflat = {
                k: v for k, v in cmetrics.items() if not isinstance(v, (list, dict))
            }
            cflat["subject"] = sub
            cflat["method"] = method_tag
            cflat["condition"] = cond
            cond_rows.append(cflat)
        if cond_rows:
            df_cond = pd.DataFrame(cond_rows)
            cols = ["subject", "method", "condition"] + [
                c
                for c in df_cond.columns
                if c not in ("subject", "method", "condition")
            ]
            df_cond = df_cond[cols]
            df_cond.to_csv(out_dir / "condition_metrics.tsv", sep="\t", index=False)

    # 3. QC PSD figure
    freqs_b, gm_b = _geometric_mean_psd_raw(raw_before)
    freqs_a, gm_a = _geometric_mean_psd_raw(raw_after)
    zoom_annotations = None
    if metrics_dict:
        atten = metrics_dict.get("attenuation_per_harmonic_db", [])
        r_vals = metrics_dict.get("R_per_harmonic", [])
        zoom_annotations = []
        for idx in range(min(3, len(metrics_dict.get("harmonics_hz", [])))):
            if idx < len(atten) and idx < len(r_vals):
                zoom_annotations.append(
                    f"atten={atten[idx]:.1f} dB, R={r_vals[idx]:.2f}"
                )

    fig_psd = plot_psd_zoom_comparison(
        freqs_b,
        gm_b,
        freqs_a,
        gm_a,
        series_name=method_tag,
        title=f"{sub} — {METHOD_LABELS.get(method_tag, method_tag)}",
        zoom_freqs=metrics_dict.get("harmonics_hz", [50.0, 100.0, 150.0])[:3]
        if metrics_dict
        else [50.0, 100.0, 150.0],
        zoom_annotations=zoom_annotations,
        series_colors=METHOD_COLORS,
        series_labels=METHOD_LABELS,
        fmax=PSD_FMAX,
        show=False,
    )
    fig_psd.savefig(out_dir / "qc_psd.png", dpi=300, bbox_inches="tight")
    plt.close(fig_psd)

    # 4. Method-specific diagnostics dashboard (skip for M0)
    if method_tag in {"M1", "M2"} and estimator is not None:
        try:
            fig_diag = _plot_method_diagnostics(
                estimator,
                raw_before.get_data(),
                raw_after.get_data(),
                sfreq=raw_before.info["sfreq"],
                title=f"{sub} — {method_tag}",
                show=False,
            )
            fig_diag.savefig(out_dir / "diagnostics.png", dpi=300, bbox_inches="tight")
            plt.close(fig_diag)
        except Exception as e:
            log.warning("  [diag] %s diagnostic plot failed: %s", method_tag, e)

    # 5. Model metadata
    _save_model_json(model_info, out_dir / "model.json")
    return out_dir


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-Subject Processing
# ═══════════════════════════════════════════════════════════════════════════════


def process_subject(sub, ds_path=None, deriv_root=None):
    """Run all line-noise methods on one subject, save to disk.

    Returns
    -------
    results : list of dict
        One dict per method × condition with subject, method, condition, and metrics.
    """
    if ds_path is None:
        ds_path = DATA_ROOT
    if deriv_root is None:
        deriv_root = DERIV_ROOT

    log.info("=" * 60)
    log.info("  Processing %s", sub)
    log.info("=" * 60)

    raw_base = load_and_preprocess(sub, ds_path)
    sfreq = raw_base.info["sfreq"]
    log.info(
        "  Loaded: %d ch, %.1f s at %.0f Hz",
        raw_base.info["nchan"],
        raw_base.n_times / sfreq,
        sfreq,
    )

    cond_blocks = parse_condition_blocks(sub, ds_path)
    results = []
    cleaned_psds = {}

    # Baseline PSD (save for group analysis)
    freqs_base, gm_base = _geometric_mean_psd_raw(raw_base)

    for method_tag in METHOD_ORDER:
        log.info("  -- %s: %s", method_tag, METHOD_LABELS[method_tag])
        try:
            raw_clean, model_info, estimator = METHODS[method_tag](raw_base)

            # Whole-recording QA metrics
            metrics = compute_all_qa_metrics(
                raw_base,
                raw_clean,
                line_freq=LINE_FREQ,
                n_harmonics=LINE_HARMONICS,
            )

            # Condition-level metrics
            condition_metrics = {}
            for cond in CONDITIONS:
                if cond not in cond_blocks or not cond_blocks[cond]:
                    continue
                pieces_before, pieces_after = [], []
                for tmin, tmax in cond_blocks[cond]:
                    seg_b = crop_to_condition(raw_base, tmin, tmax)
                    seg_a = crop_to_condition(raw_clean, tmin, tmax)
                    if seg_b is not None and seg_a is not None:
                        pieces_before.append(seg_b)
                        pieces_after.append(seg_a)
                if not pieces_before:
                    continue
                if len(pieces_before) == 1:
                    raw_cond_b, raw_cond_a = pieces_before[0], pieces_after[0]
                else:
                    raw_cond_b = mne.concatenate_raws(pieces_before)
                    raw_cond_a = mne.concatenate_raws(pieces_after)

                cmetrics = compute_all_qa_metrics(
                    raw_cond_b,
                    raw_cond_a,
                    line_freq=LINE_FREQ,
                    n_harmonics=LINE_HARMONICS,
                )
                condition_metrics[cond] = cmetrics
                crow = {"subject": sub, "method": method_tag, "condition": cond}
                crow.update(
                    {
                        k: v
                        for k, v in cmetrics.items()
                        if not isinstance(v, (list, dict))
                    }
                )
                results.append(crow)
                del raw_cond_b, raw_cond_a

            # Save outputs + diagnostics
            save_outputs(
                sub,
                method_tag,
                raw_base,
                raw_clean,
                metrics,
                model_info,
                estimator,
                condition_metrics=condition_metrics,
                deriv_root=deriv_root,
            )

            # Store PSD for combined plot
            freqs_c, gm_c = _geometric_mean_psd_raw(raw_clean)
            cleaned_psds[method_tag] = (freqs_c, gm_c)

            # Whole-recording row
            row = {"subject": sub, "method": method_tag, "condition": "all"}
            row.update(
                {k: v for k, v in metrics.items() if not isinstance(v, (list, dict))}
            )
            results.append(row)

            cond_summary = ", ".join(
                f"{c}={condition_metrics[c]['R_f0']:.2f}"
                for c in CONDITIONS
                if c in condition_metrics
            )
            log.info(
                "    atten=%.1f dB, R=%.2f  [%s]",
                metrics["peak_attenuation_db"],
                metrics["R_f0"],
                cond_summary,
            )

            del raw_clean, estimator

        except Exception as e:
            log.error("    FAILED: %s", e)
            import traceback

            traceback.print_exc()
            results.append(
                {
                    "subject": sub,
                    "method": method_tag,
                    "condition": "all",
                    "error": str(e),
                }
            )

    # ── Per-subject PSD comparison figure ────────────────────────────────
    sub_fig_dir = deriv_root / sub / "line_noise"
    plot_psd_overlay(
        freqs_base,
        gm_base,
        cleaned_psds,
        focus_freq=LINE_FREQ,
        fmax=PSD_FMAX,
        n_harmonics=LINE_HARMONICS,
        title=sub,
        series_order=METHOD_ORDER,
        series_colors=METHOD_COLORS,
        series_labels=METHOD_LABELS,
        fname=sub_fig_dir / "psd_comparison.png",
        show=False,
    )
    plt.close("all")

    # ── Save per-subject PSDs for group analysis ─────────────────────────
    group_dir = deriv_root / "group" / "line_noise"
    group_dir.mkdir(parents=True, exist_ok=True)
    psd_dict = {"freqs_base": freqs_base, "gm_base": gm_base}
    for mtag, (f, g) in cleaned_psds.items():
        psd_dict[f"freqs_{mtag}"] = f
        psd_dict[f"gm_{mtag}"] = g
    np.savez_compressed(group_dir / f"group_psds_{sub}.npz", **psd_dict)

    del raw_base
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  Group Aggregation & Visualization
# ═══════════════════════════════════════════════════════════════════════════════


def run_group(subjects, deriv_root=None):
    """Aggregate all per-subject metrics and produce group figures."""
    if deriv_root is None:
        deriv_root = DERIV_ROOT

    group_dir = deriv_root / "group" / "line_noise"
    group_dir.mkdir(parents=True, exist_ok=True)

    set_theme()

    # ── Re-read per-subject metrics TSVs ─────────────────────────────────
    frames = []
    for sub in subjects:
        for mtag in METHOD_ORDER:
            # Whole-recording
            tsv = deriv_root / sub / "line_noise" / mtag / "metrics.tsv"
            if tsv.exists():
                df = pd.read_csv(tsv, sep="\t")
                df["condition"] = "all"
                frames.append(df)
            # Per-condition
            ctsv = deriv_root / sub / "line_noise" / mtag / "condition_metrics.tsv"
            if ctsv.exists():
                frames.append(pd.read_csv(ctsv, sep="\t"))

    if not frames:
        log.error("No metrics found — nothing to aggregate")
        return

    df_metrics = pd.concat(frames, ignore_index=True)

    # Clean
    if "error" in df_metrics.columns:
        df_clean = df_metrics.dropna(subset=["R_f0"])
    else:
        df_clean = df_metrics.copy()

    df_all = df_clean[df_clean["condition"] == "all"].copy()
    df_cond = df_clean[df_clean["condition"] != "all"].copy()

    log.info(
        "Whole-recording rows: %d (%d subjects x %d methods)",
        len(df_all),
        df_all["subject"].nunique(),
        df_all["method"].nunique(),
    )
    log.info(
        "Per-condition rows: %d (%d conditions)",
        len(df_cond),
        df_cond["condition"].nunique(),
    )

    # ── Save DataFrames ──────────────────────────────────────────────────
    df_metrics.to_csv(group_dir / "all_metrics.tsv", sep="\t", index=False)
    df_all.to_csv(group_dir / "whole_recording.tsv", sep="\t", index=False)
    df_cond.to_csv(group_dir / "per_condition.tsv", sep="\t", index=False)
    log.info("Saved DataFrames to %s", group_dir)

    # ── QA Step 1: PSD Gallery ───────────────────────────────────────────
    sub_show = subjects[0]
    harmonics_show = [LINE_FREQ * h for h in range(1, LINE_HARMONICS + 1)]

    raw_base = load_and_preprocess(sub_show, DATA_ROOT)
    freqs_base, gm_base = _geometric_mean_psd_raw(raw_base)

    cleaned_psds = {}
    for mtag in METHOD_ORDER[1:]:
        raw_path = deriv_root / sub_show / "line_noise" / mtag / "cleaned_raw.fif"
        if raw_path.exists():
            raw_c = mne.io.read_raw_fif(str(raw_path), preload=True, verbose=False)
            cleaned_psds[mtag] = _geometric_mean_psd_raw(raw_c)
            del raw_c
    del raw_base

    plot_psd_gallery(
        freqs_base,
        gm_base,
        cleaned_psds,
        zoom_freqs=harmonics_show,
        fmax=PSD_FMAX,
        title=sub_show,
        series_order=METHOD_ORDER[1:],
        series_colors=METHOD_COLORS,
        series_labels=METHOD_LABELS,
        fname=group_dir / "psd_gallery.png",
        show=False,
    )
    plt.close("all")

    # ── QA Step 2: Metric Bars ───────────────────────────────────────────
    df_plot = df_all.dropna(subset=["peak_attenuation_db"])
    stats_data = {col: df_plot[col].to_numpy() for col in df_plot.columns}
    plot_metric_bars(
        stats_data,
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
        fname=group_dir / "metric_bars.png",
        show=False,
    )
    plt.close("all")

    # ── QA Step 2c: Trade-off + R ────────────────────────────────────────
    plot_metric_tradeoff_summary(
        stats_data,
        group_col="method",
        subject_col="subject",
        x_col="below_noise_distortion_db",
        y_col="peak_attenuation_db",
        metric_col="R_f0",
        group_order=METHOD_ORDER,
        group_colors=METHOD_COLORS,
        group_labels=METHOD_LABELS,
        fname=group_dir / "tradeoff_and_r.png",
        show=False,
    )
    plt.close("all")

    # ── QA Step 3: Paired Comparison ─────────────────────────────────────
    if df_plot["subject"].nunique() > 1:
        plot_metric_slopes(
            stats_data,
            group_col="method",
            metric_specs=[
                ("peak_attenuation_db", "Peak Attenuation (dB)"),
                ("below_noise_distortion_db", "Below-Noise Distortion (dB)"),
                ("R_f0", "R(f0)"),
            ],
            group_order=METHOD_ORDER,
            group_colors=METHOD_COLORS,
            fname=group_dir / "metric_slopes.png",
            show=False,
        )
        plt.close("all")

    # ── QA Step 3b: Harmonic Attenuation ─────────────────────────────────
    sub_h = subjects[0]
    raw_base_h = load_and_preprocess(sub_h, DATA_ROOT)
    freqs_base_h, gm_base_h = _geometric_mean_psd_raw(raw_base_h)
    harmonics_hz = [
        LINE_FREQ * h
        for h in range(1, LINE_HARMONICS + 1)
        if LINE_FREQ * h < freqs_base_h[-1]
    ]
    cleaned_psds_h = {}
    for mtag in METHOD_ORDER[1:]:
        raw_path = deriv_root / sub_h / "line_noise" / mtag / "cleaned_raw.fif"
        if raw_path.exists():
            raw_c = mne.io.read_raw_fif(str(raw_path), preload=True, verbose=False)
            cleaned_psds_h[mtag] = _geometric_mean_psd_raw(raw_c)
            del raw_c
    del raw_base_h

    plot_harmonic_attenuation(
        freqs_base_h,
        gm_base_h,
        cleaned_psds_h,
        harmonics_hz=harmonics_hz,
        subject=sub_h,
        series_order=METHOD_ORDER[1:],
        series_colors=METHOD_COLORS,
        series_labels=METHOD_LABELS,
        fname=group_dir / "harmonic_attenuation.png",
        show=False,
    )
    plt.close("all")

    # ── Condition × Method: R(f₀) bars ──────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    fig.suptitle(
        "R(f₀) by Condition and Method", fontsize=FONTS["suptitle"], fontweight="bold"
    )
    bar_width = 0.2
    x = np.arange(len(METHOD_ORDER))

    for ax_idx, cond in enumerate(CONDITIONS):
        ax = axes[ax_idx]
        cond_data = df_cond[df_cond["condition"] == cond]
        means, sems = [], []
        for mtag in METHOD_ORDER:
            vals = cond_data[cond_data["method"] == mtag]["R_f0"].dropna()
            means.append(vals.mean() if len(vals) > 0 else 0)
            sems.append(vals.sem() if len(vals) > 1 else 0)
        ax.bar(
            x,
            means,
            bar_width * 3,
            yerr=sems,
            capsize=4,
            color=[METHOD_COLORS[m] for m in METHOD_ORDER],
            edgecolor="white",
            linewidth=0.8,
        )
        ax.set_title(CONDITION_LABELS[cond], fontsize=FONTS["title"], fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(METHOD_ORDER, fontsize=FONTS["tick"])
        ax.axhline(
            1.0, color="k", linestyle="--", linewidth=0.8, alpha=0.5, label="Ideal R=1"
        )
        if ax_idx == 0:
            ax.set_ylabel("R(f₀)", fontsize=FONTS["label"])
        style_axes(ax)
    themed_legend(axes[0], loc="upper right")
    fig.tight_layout()
    fig.savefig(group_dir / "condition_r_f0_bars.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ── Condition × Method: All key metrics grouped ──────────────────────
    cond_metrics_to_show = [
        "R_f0",
        "peak_attenuation_db",
        "below_noise_distortion_db",
        "overclean_proportion",
    ]
    cond_metric_labels = {
        "R_f0": "R(f₀)",
        "peak_attenuation_db": "Peak Atten. (dB)",
        "below_noise_distortion_db": "Distortion (dB)",
        "overclean_proportion": "Overclean",
    }

    fig, axes = plt.subplots(
        1, len(cond_metrics_to_show), figsize=(16, 4.5), sharey=False
    )
    fig.suptitle(
        "Condition × Method — Key QA Metrics",
        fontsize=FONTS["suptitle"],
        fontweight="bold",
    )

    for ax_idx, metric in enumerate(cond_metrics_to_show):
        ax = axes[ax_idx]
        xc = np.arange(len(CONDITIONS))
        bw = 0.18
        for m_idx, mtag in enumerate(METHOD_ORDER):
            vals_per_cond, errs_per_cond = [], []
            for cond in CONDITIONS:
                v = df_cond[
                    (df_cond["method"] == mtag) & (df_cond["condition"] == cond)
                ][metric].dropna()
                vals_per_cond.append(v.mean() if len(v) > 0 else 0)
                errs_per_cond.append(v.sem() if len(v) > 1 else 0)
            offset = (m_idx - len(METHOD_ORDER) / 2 + 0.5) * bw
            ax.bar(
                xc + offset,
                vals_per_cond,
                bw,
                yerr=errs_per_cond,
                capsize=3,
                color=METHOD_COLORS[mtag],
                label=METHOD_LABELS[mtag] if ax_idx == 0 else "",
                edgecolor="white",
                linewidth=0.5,
            )
        ax.set_title(cond_metric_labels[metric], fontsize=FONTS["title"])
        ax.set_xticks(xc)
        ax.set_xticklabels(
            [CONDITION_LABELS[c] for c in CONDITIONS],
            fontsize=FONTS["tick"],
            rotation=15,
            ha="right",
        )
        if metric == "R_f0":
            ax.axhline(1.0, color="k", ls="--", lw=0.8, alpha=0.5)
        elif metric == "below_noise_distortion_db":
            ax.axhline(0.0, color="k", ls="--", lw=0.8, alpha=0.5)
        style_axes(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(METHOD_ORDER),
        bbox_to_anchor=(0.5, -0.02),
        fontsize=FONTS["legend"],
        frameon=False,
        handlelength=1.2,
        handletextpad=0.5,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(
        group_dir / "condition_metrics_grouped.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    # ── Condition-level PSD comparison (first subject) ───────────────────
    sub_psd = subjects[0]
    raw_base_psd = load_and_preprocess(sub_psd, DATA_ROOT)
    cond_blocks_psd = parse_condition_blocks(sub_psd)

    cleaned_raws_psd = {}
    for mtag in METHOD_ORDER[1:]:
        raw_path = deriv_root / sub_psd / "line_noise" / mtag / "cleaned_raw.fif"
        if raw_path.exists():
            cleaned_raws_psd[mtag] = mne.io.read_raw_fif(
                str(raw_path), preload=True, verbose=False
            )

    n_methods_active = len(METHOD_ORDER[1:])
    fig, axes = plt.subplots(
        len(CONDITIONS),
        n_methods_active + 1,
        figsize=(4 * (n_methods_active + 1), 3.5 * len(CONDITIONS)),
        sharex=True,
        sharey=True,
    )
    fig.suptitle(
        f"Condition-Level PSD — {sub_psd}",
        fontsize=FONTS["suptitle"],
        fontweight="bold",
    )
    fmax_zoom = min(LINE_FREQ * (LINE_HARMONICS + 0.5), PSD_FMAX)

    for row_idx, cond in enumerate(CONDITIONS):
        if cond not in cond_blocks_psd or not cond_blocks_psd[cond]:
            continue
        pieces = []
        for tmin, tmax in cond_blocks_psd[cond]:
            seg = crop_to_condition(raw_base_psd, tmin, tmax)
            if seg is not None:
                pieces.append(seg)
        if not pieces:
            continue
        raw_cond_base = mne.concatenate_raws(pieces) if len(pieces) > 1 else pieces[0]
        freqs_cb, gm_cb = _geometric_mean_psd_raw(raw_cond_base)

        ax = axes[row_idx, 0]
        ax.plot(
            freqs_cb,
            10 * np.log10(gm_cb + 1e-30),
            color=COLORS["before"],
            linewidth=1,
            label="Before",
        )
        for h in range(1, LINE_HARMONICS + 1):
            ax.axvline(LINE_FREQ * h, color=COLORS["accent"], alpha=0.3, linewidth=0.5)
        ax.set_title(f"{CONDITION_LABELS[cond]} — Before", fontsize=FONTS["title"])
        ax.set_xlim(0, fmax_zoom)
        if row_idx == len(CONDITIONS) - 1:
            ax.set_xlabel("Frequency (Hz)", fontsize=FONTS["label"])
        ax.set_ylabel("Power (dB)", fontsize=FONTS["label"])
        style_axes(ax)

        for m_idx, mtag in enumerate(METHOD_ORDER[1:]):
            ax = axes[row_idx, m_idx + 1]
            ax.plot(
                freqs_cb,
                10 * np.log10(gm_cb + 1e-30),
                color=COLORS["before"],
                linewidth=0.8,
                alpha=0.4,
                label="Before",
            )
            if mtag in cleaned_raws_psd:
                pieces_c = []
                for tmin, tmax in cond_blocks_psd[cond]:
                    seg = crop_to_condition(cleaned_raws_psd[mtag], tmin, tmax)
                    if seg is not None:
                        pieces_c.append(seg)
                if pieces_c:
                    raw_cond_clean = (
                        mne.concatenate_raws(pieces_c)
                        if len(pieces_c) > 1
                        else pieces_c[0]
                    )
                    freqs_cc, gm_cc = _geometric_mean_psd_raw(raw_cond_clean)
                    ax.plot(
                        freqs_cc,
                        10 * np.log10(gm_cc + 1e-30),
                        color=METHOD_COLORS[mtag],
                        linewidth=1.2,
                        label=METHOD_LABELS[mtag],
                    )
                    del raw_cond_clean
            for h in range(1, LINE_HARMONICS + 1):
                ax.axvline(
                    LINE_FREQ * h, color=COLORS["accent"], alpha=0.3, linewidth=0.5
                )
            ax.set_title(f"{CONDITION_LABELS[cond]} — {mtag}", fontsize=FONTS["title"])
            ax.set_xlim(0, fmax_zoom)
            if row_idx == len(CONDITIONS) - 1:
                ax.set_xlabel("Frequency (Hz)", fontsize=FONTS["label"])
            style_axes(ax)

        del raw_cond_base

    del raw_base_psd
    for r in cleaned_raws_psd.values():
        del r
    fig.tight_layout()
    fig.savefig(
        group_dir / "condition_psd_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    log.info("Group figures saved to %s", group_dir)
    plt.close("all")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Runabout ds003620 — Line-Noise Removal Benchmark"
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--subject", type=str, help="Process a single subject (e.g. sub-01)"
    )
    grp.add_argument(
        "--slurm-array",
        action="store_true",
        help="Pick subject from $SLURM_ARRAY_TASK_ID",
    )
    grp.add_argument(
        "--all", action="store_true", help="Process all subjects sequentially + group"
    )
    grp.add_argument(
        "--group-only",
        action="store_true",
        help="Group aggregation only (after per-subject jobs)",
    )
    parser.add_argument(
        "--data-root", type=str, default=None, help="Override DATA_ROOT path"
    )
    parser.add_argument(
        "--deriv-root", type=str, default=None, help="Override DERIV_ROOT path"
    )
    args = parser.parse_args()

    global DATA_ROOT, DERIV_ROOT
    if args.data_root:
        DATA_ROOT = Path(args.data_root)
        DERIV_ROOT = DATA_ROOT / "derivatives" / "mne-denoise"
    if args.deriv_root:
        DERIV_ROOT = Path(args.deriv_root)

    if args.group_only:
        # Discover which subjects have been processed
        processed = sorted(
            [
                d.name
                for d in DERIV_ROOT.iterdir()
                if d.is_dir()
                and d.name.startswith("sub-")
                and (d / "line_noise").exists()
            ]
        )
        log.info("Group aggregation for %d subjects", len(processed))
        run_group(processed, DERIV_ROOT)
        return

    if args.slurm_array:
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 1))
        sub = f"sub-{task_id:02d}"
        log.info("SLURM array task %d → %s", task_id, sub)
        process_subject(sub)
        return

    if args.subject:
        process_subject(args.subject)
        return

    # --all: process every subject then run group
    t0 = time.perf_counter()
    all_results = []
    for sub in ALL_SUBJECTS:
        sub_path = DATA_ROOT / sub
        if not sub_path.exists():
            log.warning("Skipping %s — not found at %s", sub, sub_path)
            continue
        sub_results = process_subject(sub)
        all_results.extend(sub_results)

    elapsed = time.perf_counter() - t0
    log.info("All subjects processed in %.1f s", elapsed)

    # Discover what's actually on disk
    processed = sorted(
        [
            d.name
            for d in DERIV_ROOT.iterdir()
            if d.is_dir() and d.name.startswith("sub-") and (d / "line_noise").exists()
        ]
    )
    run_group(processed, DERIV_ROOT)


if __name__ == "__main__":
    main()
