#!/usr/bin/env python
"""
Runabout ds003620 — ERP DSS Benchmark (all 44 subjects).

Headless script version of runabout_erp_dss_benchmark.ipynb.
Processes every subject, saves all per-subject artefacts *and* group-level
DataFrames, null-control arrays, and publication figures so that downstream
visualisation can be changed without re-running on the cluster.

Usage
-----
  # Single subject (SLURM array element or local test):
  python run_erp_dss_benchmark.py --subject sub-01

  # All subjects sequentially (local or single SLURM job):
  python run_erp_dss_benchmark.py --all

  # Slurm array mode — picks subject from $SLURM_ARRAY_TASK_ID:
  python run_erp_dss_benchmark.py --slurm-array

  # Group aggregation only (after all per-subject jobs finished):
  python run_erp_dss_benchmark.py --group-only

Output contract (per subject)
-----------------------------
  {DERIV_ROOT}/{sub}/erp_dss/
      {pipeline}/
          metrics.tsv
          model.json
      preproc_info.json

Output contract (group)
-----------------------
  {DERIV_ROOT}/group/erp_dss/
      all_erp_metrics.tsv          <-- master DataFrame
      null_results.npz             <-- null g/AUC distributions
      qa_level_a_components.png
      qa_level_b_signal.png
      condition_diff_waves.png
      condition_interaction.png
      endpoint_metrics.png
      summary_null_control.png
      paired_comparison.png
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
from mne.preprocessing import ICA
from scipy import stats
from scipy.signal import welch as sp_welch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ── Ensure mne-denoise is importable ────────────────────────────────────────
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from mne_denoise.dss import DSS, AverageBias
from mne_denoise.zapline import ZapLine

warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("WARNING")

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("erp_dss")


# ═══════════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════════
DATASET_ID = "ds003620"
LINE_FREQ = 50.0
RESAMPLE_FREQ = 250
HP_FREQ = 0.5
LP_FREQ = 40.0
ASR_CUTOFF = 10
ICA_THRESHOLD = 0.89
EPOCH_TMIN = -0.2
EPOCH_TMAX = 0.8
REJECT_UV = 125e-6
STIMTRAK_LAG_S = 0.200
ORIG_SFREQ = 500.0
TASK = "oddball"
RANDOM_STATE = 42
N_PERM = 500

# ERP windows
ERP_WINDOWS = {
    "N1": (0.080, 0.130),
    "MMN": (0.100, 0.250),
    "P300": (0.250, 0.500),
}
BASELINE_WIN = (-0.200, 0.000)
EVAL_WIN = ERP_WINDOWS["P300"]

# ROI channels
PRIMARY_CH = "Cz"
P300_CH = "Pz"

# DSS parameters
DSS_N_COMPONENTS = 5
DSS_N_KEEP = 3

# Pipeline display
PIPE_ORDER = ["C0", "C1", "C2"]
PIPE_LABELS = {
    "C0": "Baseline",
    "C1": "Paper (ICA)",
    "C2": "DSS (AverageBias)",
}
PIPE_COLORS = {
    "C0": "#404040",
    "C1": "#2ca02c",
    "C2": "#d62728",
}

# Conditions from the dataset
CONDITIONS = ["lab", "oval", "campus"]
COND_LABELS = {"lab": "Lab", "oval": "Oval Office", "campus": "Campus"}

# 44 subjects total
ALL_SUBJECTS = [f"sub-{i:02d}" for i in range(1, 45)]


def _resolve_paths():
    """Auto-detect DATA_ROOT / DERIV_ROOT for Narval vs local."""
    if "CC_CLUSTER" in os.environ:
        data_dir = Path(
            os.environ.get(
                "DATA_DIR", str(Path.home() / "scratch" / "mnedenoise" / "data")
            )
        )
        data_root = data_dir / DATASET_ID
        deriv_root = data_root / "derivatives" / "mne-denoise"
    else:
        data_root = _REPO / "data" / "runabout"
        deriv_root = data_root / "derivatives" / "mne-denoise"
    return data_root, deriv_root


DATA_ROOT, DERIV_ROOT = _resolve_paths()


# ═══════════════════════════════════════════════════════════════════════════════
#  Full Preprocessing Pipeline
# ═══════════════════════════════════════════════════════════════════════════════


def _parse_trial_type(tt):
    """Parse a trigger_corrected trial_type string into (tone, task, condition)."""
    tone = "deviant" if tt.startswith("t_") else "standard"
    task = "ignore" if "dontcount" in tt.lower() else "count"
    for cond in ("lab", "oval", "campus"):
        if f"_{cond}" in tt:
            return tone, task, cond
    return tone, task, "unknown"


def full_preprocess(sub, ds_path):
    """Full preprocessing pipeline for one subject.

    Returns
    -------
    epochs : mne.Epochs
    raw_clean : mne.io.Raw
    ica_obj : mne.preprocessing.ICA
    preproc_info : dict
    """
    # Load
    eeg_dir = ds_path / sub / "eeg"
    vhdr_files = sorted(eeg_dir.glob("*.vhdr"))
    if not vhdr_files:
        raise FileNotFoundError(f"No .vhdr file for {sub}")
    raw = mne.io.read_raw_brainvision(str(vhdr_files[0]), preload=True, verbose=False)

    # Fix encoding
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
    sfreq = raw.info["sfreq"]

    # Pick EEG + FCz
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
        l_freq=HP_FREQ, h_freq=None, method="fir", fir_design="firwin", verbose=False
    )

    # ZapLine+ for line noise
    zap = ZapLine(sfreq=sfreq, line_freq=LINE_FREQ, n_remove="auto", n_harmonics=3)
    raw = zap.fit_transform(raw)
    n_zap_removed = getattr(zap, "n_removed_", None)

    # Bad channel detection
    ch_data = raw.get_data()
    ch_var = np.var(ch_data, axis=1)
    median_var = np.median(ch_var)
    bad_chs = []
    for ci, ch_name in enumerate(raw.ch_names):
        if ch_var[ci] < median_var * 0.01 or ch_var[ci] > median_var * 50:
            bad_chs.append(ch_name)
    raw.info["bads"] = bad_chs

    # Low-pass filter
    raw.filter(
        l_freq=None, h_freq=LP_FREQ, method="fir", fir_design="firwin", verbose=False
    )

    # ICA
    n_good = raw.info["nchan"] - len(bad_chs)
    n_ica = min(20, n_good - 1)
    ica = ICA(
        n_components=n_ica,
        method="infomax",
        fit_params={"extended": True},
        random_state=RANDOM_STATE,
        max_iter=1000,
    )
    ica.fit(raw, picks="eeg", verbose=False)

    # Automatic component rejection
    eog_idx = []
    for proxy_ch in ["Fp1", "Fp2"]:
        if proxy_ch in raw.ch_names:
            idx, _ = ica.find_bads_eog(
                raw, ch_name=proxy_ch, threshold=3.0, verbose=False
            )
            eog_idx.extend(idx)
    eog_idx = list(set(eog_idx))
    try:
        ecg_idx, _ = ica.find_bads_ecg(
            raw, method="correlation", threshold=0.25, verbose=False
        )
    except Exception:
        ecg_idx = []
    ica.exclude = list(set(eog_idx + ecg_idx))

    raw_clean = ica.apply(raw.copy(), verbose=False)

    # Interpolate
    if bad_chs:
        raw_clean.interpolate_bads(verbose=False)

    # Epoching (trigger-corrected events)
    trig_dir = ds_path / "derivatives" / "trigger_corrected" / sub / "eeg"
    evt_files = sorted(trig_dir.glob("*_events.tsv"))
    if not evt_files:
        raise FileNotFoundError(f"No trigger_corrected events for {sub}.")

    tc_df = pd.read_csv(evt_files[0], sep="\t")
    tc_df = tc_df[tc_df["trial_type"] != "empty"].reset_index(drop=True)

    tc_df["sample_rs"] = np.round(tc_df["onset"].values * (sfreq / ORIG_SFREQ)).astype(
        int
    )

    parsed = [_parse_trial_type(tt) for tt in tc_df["trial_type"]]
    tc_df["tone"] = [p[0] for p in parsed]
    tc_df["task"] = [p[1] for p in parsed]
    tc_df["condition"] = [p[2] for p in parsed]

    unique_types = sorted(tc_df["trial_type"].unique())
    event_id = {tt: i + 1 for i, tt in enumerate(unique_types)}

    events_arr = np.column_stack(
        [
            tc_df["sample_rs"].values,
            np.zeros(len(tc_df), dtype=int),
            np.array([event_id[tt] for tt in tc_df["trial_type"]]),
        ]
    )

    max_sample = raw_clean.n_times - 1
    valid = (events_arr[:, 0] >= 0) & (events_arr[:, 0] < max_sample)
    events_arr = events_arr[valid]
    tc_df = tc_df.iloc[np.where(valid)[0]].reset_index(drop=True)

    epochs = mne.Epochs(
        raw_clean,
        events_arr,
        event_id=event_id,
        tmin=EPOCH_TMIN,
        tmax=EPOCH_TMAX,
        baseline=(EPOCH_TMIN, 0),
        preload=True,
        verbose=False,
    )

    n_before = len(epochs)
    epochs.drop_bad(reject={"eeg": REJECT_UV}, verbose=False)
    n_after = len(epochs)

    kept = epochs.selection
    meta_df = tc_df.iloc[kept][["tone", "task", "condition", "trial_type"]].reset_index(
        drop=True
    )
    epochs.metadata = meta_df

    preproc_info = {
        "n_channels": raw_clean.info["nchan"],
        "sfreq": sfreq,
        "bad_channels": bad_chs,
        "ica_excluded": [int(x) for x in ica.exclude],
        "n_ica_components": int(ica.n_components_),
        "zapline_removed": n_zap_removed,
        "n_epochs_before_reject": int(n_before),
        "n_epochs_after_reject": int(n_after),
        "reject_rate": float((1 - n_after / max(n_before, 1)) * 100),
        "n_deviant": int((meta_df["tone"] == "deviant").sum()),
        "n_standard": int((meta_df["tone"] == "standard").sum()),
    }

    return epochs, raw_clean, ica, preproc_info


# ═══════════════════════════════════════════════════════════════════════════════
#  Endpoint Metric Functions
# ═══════════════════════════════════════════════════════════════════════════════


def hedges_g(x, y):
    """Hedges' g (bias-corrected Cohen's d)."""
    n1, n2 = len(x), len(y)
    if n1 < 15 or n2 < 15:
        return np.nan
    s_pooled = np.sqrt(
        ((n1 - 1) * np.var(x, ddof=1) + (n2 - 1) * np.var(y, ddof=1)) / (n1 + n2 - 2)
    )
    if s_pooled < 1e-15:
        return np.nan
    d = (np.mean(x) - np.mean(y)) / s_pooled
    correction = 1 - 3 / (4 * (n1 + n2) - 9)
    return d * correction


def peak_latency_ms(evoked_data, times_s, win, mode="pos"):
    """Peak latency (ms) via Fractional Area Latency (50% AUC)."""
    mask = (times_s >= win[0]) & (times_s <= win[1])
    if not mask.any():
        return np.nan
    segment = evoked_data[mask]
    if mode == "pos":
        val = np.maximum(segment, 0)
    elif mode == "neg":
        val = np.maximum(-segment, 0)
    else:
        val = np.abs(segment)
    cum_area = np.cumsum(val)
    if cum_area[-1] == 0:
        return np.nan
    half_point = cum_area[-1] / 2.0
    idx_50 = np.searchsorted(cum_area, half_point)
    return times_s[mask][idx_50] * 1000


def morphology_corr(ev_pipe, ev_base, times_s, excl_win):
    """Pearson r between pipeline and baseline waveforms OUTSIDE eval window."""
    mask = (times_s < excl_win[0]) | (times_s > excl_win[1])
    if mask.sum() < 3:
        return np.nan
    r, _ = stats.pearsonr(ev_pipe[mask], ev_base[mask])
    return r


def morphology_nrmse(ev_pipe, ev_base, times_s, excl_win):
    """Normalized RMSE vs baseline OUTSIDE eval window."""
    mask = (times_s < excl_win[0]) | (times_s > excl_win[1])
    if mask.sum() < 3:
        return np.nan
    pipe_seg = ev_pipe[mask]
    base_seg = ev_base[mask]
    rmse = np.sqrt(np.mean((pipe_seg - base_seg) ** 2))
    iqr = np.percentile(base_seg, 75) - np.percentile(base_seg, 25)
    return rmse / iqr if iqr > 0 else np.nan


def split_half_reliability(epoch_data_1ch):
    """Spearman-Brown corrected split-half reliability."""
    ev_even = epoch_data_1ch[0::2].mean(axis=0)
    ev_odd = epoch_data_1ch[1::2].mean(axis=0)
    r, _ = stats.pearsonr(ev_even, ev_odd)
    sb_r = (2 * r) / (1 + abs(r))
    return sb_r


def single_trial_auc(
    epochs_data_1ch, times, time_window, dev_mask, std_mask, random_state=42
):
    """Decode deviant vs standard from single-trial mean amplitude."""
    t_mask = (times >= time_window[0]) & (times <= time_window[1])
    features = epochs_data_1ch[:, t_mask].mean(axis=1).reshape(-1, 1)

    labels = np.full(len(epochs_data_1ch), -1)
    labels[dev_mask] = 1
    labels[std_mask] = 0
    valid = labels >= 0
    X, y = features[valid], labels[valid].astype(int)

    if len(np.unique(y)) < 2 or X.shape[0] < 10:
        return np.nan

    clf = make_pipeline(
        StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    )
    n_min = min(np.bincount(y))
    cv = StratifiedKFold(
        n_splits=min(5, n_min), shuffle=True, random_state=random_state
    )
    try:
        scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
        return scores.mean()
    except Exception:
        return np.nan


def compute_all_erp_metrics(
    epochs_test,
    evoked_test,
    evoked_baseline,
    ch_pz_idx,
    ch_cz_idx,
    times,
    dev_mask,
    std_mask,
    eval_win=None,
):
    """Compute all 6 endpoint metrics for one pipeline."""
    if eval_win is None:
        eval_win = EVAL_WIN
    if hasattr(epochs_test, "get_data"):
        data = epochs_test.get_data()
    else:
        data = epochs_test

    pz_data = data[:, ch_pz_idx, :] * 1e6

    # 1. Hedges' g
    t_mask = (times >= eval_win[0]) & (times <= eval_win[1])
    amp_dev = (
        pz_data[dev_mask][:, t_mask].mean(axis=1) if dev_mask.any() else np.array([])
    )
    amp_std = (
        pz_data[std_mask][:, t_mask].mean(axis=1) if std_mask.any() else np.array([])
    )
    g = (
        hedges_g(amp_dev, amp_std)
        if len(amp_dev) >= 2 and len(amp_std) >= 2
        else np.nan
    )

    # 2. Peak latency
    ev_pz = evoked_test.data[ch_pz_idx] * 1e6
    plat = peak_latency_ms(ev_pz, evoked_test.times, eval_win, mode="pos")

    # 3. Morphology correlation
    ev_cz = evoked_test.data[ch_cz_idx] * 1e6
    base_cz = evoked_baseline.data[ch_cz_idx] * 1e6
    mr = morphology_corr(ev_cz, base_cz, evoked_test.times, eval_win)
    nrmse = morphology_nrmse(ev_cz, base_cz, evoked_test.times, eval_win)

    # 4. Split-half reliability
    shr = split_half_reliability(pz_data)

    # 5. Single-trial AUC
    auc = single_trial_auc(pz_data, times, eval_win, dev_mask, std_mask)

    return {
        "hedges_g": g,
        "peak_lat_ms": plat,
        "morph_r": mr,
        "morph_nrmse": nrmse,
        "split_half_r": shr,
        "auc": auc,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Pipeline Application Functions
# ═══════════════════════════════════════════════════════════════════════════════


def apply_c0_baseline(epochs_train, epochs_test, **kwargs):
    """C0: Baseline — no spatial filter."""
    return epochs_test.copy(), {"pipeline": "C0_baseline"}


def apply_c1_ica(epochs_train, epochs_test, ica_obj=None, **kwargs):
    """C1: Paper pipeline (ICA)."""
    if ica_obj is None:
        return epochs_test.copy(), {"pipeline": "C1_ica", "error": "no ICA object"}
    epochs_out = ica_obj.apply(epochs_test.copy(), verbose=False)
    info = {
        "pipeline": "C1_ica",
        "n_components": int(ica_obj.n_components_),
        "excluded": [int(x) for x in ica_obj.exclude],
    }
    return epochs_out, info


def apply_c2_dss(
    epochs_train,
    epochs_test,
    n_components=DSS_N_COMPONENTS,
    n_keep=DSS_N_KEEP,
    **kwargs,
):
    """C2: DSS (AverageBias)."""
    bias = AverageBias(axis="epochs")
    dss = DSS(bias=bias, n_components=n_components, return_type="sources")
    dss.fit(epochs_train)

    sources_test = dss.transform(epochs_test)
    sources_kept = sources_test.copy()
    sources_kept[:, n_keep:, :] = 0
    data_recon = dss.inverse_transform(sources_kept)

    if data_recon.ndim == 2:
        data_recon = data_recon[np.newaxis, ...]
    elif data_recon.ndim == 3 and data_recon.shape[0] == len(epochs_test.ch_names):
        data_recon = np.transpose(data_recon, (2, 0, 1))

    epochs_out = mne.EpochsArray(
        data_recon, epochs_test.info, tmin=epochs_test.tmin, verbose=False
    )
    if epochs_test.metadata is not None:
        epochs_out.metadata = epochs_test.metadata.copy()

    info = {
        "pipeline": "C2_dss",
        "n_components": n_components,
        "n_keep": n_keep,
        "eigenvalues": (
            dss.eigenvalues_.tolist() if hasattr(dss, "eigenvalues_") else None
        ),
    }
    return epochs_out, info


PIPELINES = {
    "C0": apply_c0_baseline,
    "C1": apply_c1_ica,
    "C2": apply_c2_dss,
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-Subject Processing
# ═══════════════════════════════════════════════════════════════════════════════


def _jsonify(obj):
    """Recursively convert numpy types for JSON serialisation."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, list):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    return obj


def process_subject(sub, ds_path=None, deriv_root=None):
    """Run the full ERP benchmark pipeline for one subject.

    Returns
    -------
    results : dict  {pipeline_tag: {metrics_dict, pipeline_info}}
    """
    if ds_path is None:
        ds_path = DATA_ROOT
    if deriv_root is None:
        deriv_root = DERIV_ROOT

    log.info("=" * 60)
    log.info("  %s — ERP DSS Benchmark", sub)
    log.info("=" * 60)

    # Full preprocessing
    epochs, raw_clean, ica_obj, preproc_info = full_preprocess(sub, ds_path)
    log.info(
        "  Preprocessing: %d epochs, reject rate = %.1f%%",
        preproc_info["n_epochs_after_reject"],
        preproc_info["reject_rate"],
    )

    # Train/test split (anti-circularity)
    n_ep = len(epochs)
    train_idx = np.arange(0, n_ep, 2)
    test_idx = np.arange(1, n_ep, 2)
    epochs_train = epochs[train_idx]
    epochs_test = epochs[test_idx]
    log.info("  Train/test split: %d / %d", len(train_idx), len(test_idx))

    # Condition masks for test set
    meta_test = epochs_test.metadata
    if meta_test is not None and "tone" in meta_test.columns:
        test_tone = meta_test["tone"].values
        test_task = meta_test["task"].values
        dev_count_mask = (test_tone == "deviant") & (test_task == "count")
        std_count_mask = (test_tone == "standard") & (test_task == "count")
    else:
        dev_count_mask = np.zeros(len(test_idx), dtype=bool)
        std_count_mask = np.ones(len(test_idx), dtype=bool)
    log.info(
        "  Conditions (test): %d dev+count, %d std+count",
        dev_count_mask.sum(),
        std_count_mask.sum(),
    )

    # Channel indices
    ch_pz = epochs_test.ch_names.index(P300_CH)
    ch_cz = epochs_test.ch_names.index(PRIMARY_CH)
    times = epochs_test.times

    # Apply all pipelines
    results = {}
    pipe_epochs = {}
    pipe_evokeds = {}

    for ptag in PIPE_ORDER:
        log.info("  -- %s: %s", ptag, PIPE_LABELS[ptag])
        try:
            ep_out, p_info = PIPELINES[ptag](epochs_train, epochs_test, ica_obj=ica_obj)
            pipe_epochs[ptag] = ep_out
            pipe_evokeds[ptag] = ep_out.average()
            results[ptag] = {"info": p_info}
        except Exception as e:
            log.error("    FAILED: %s", e)
            pipe_epochs[ptag] = epochs_test.copy()
            pipe_evokeds[ptag] = epochs_test.average()
            results[ptag] = {"info": {"error": str(e)}}

    # Compute metrics (all against C0 baseline)
    ev_baseline = pipe_evokeds["C0"]
    for ptag in PIPE_ORDER:
        ep = pipe_epochs[ptag]
        ev = pipe_evokeds[ptag]
        if ptag == "C0":
            m = compute_all_erp_metrics(
                ep, ev, ev, ch_pz, ch_cz, times, dev_count_mask, std_count_mask
            )
            m["morph_r"] = 1.0
        else:
            m = compute_all_erp_metrics(
                ep, ev, ev_baseline, ch_pz, ch_cz, times, dev_count_mask, std_count_mask
            )
        results[ptag]["metrics"] = m
        log.info(
            "  %s: g=%.3f, lat=%.0fms, shr=%.3f",
            ptag,
            m["hedges_g"],
            m["peak_lat_ms"],
            m["split_half_r"],
        )

    # Save to disk
    for ptag in PIPE_ORDER:
        out_dir = deriv_root / sub / "erp_dss" / ptag
        out_dir.mkdir(parents=True, exist_ok=True)

        # Metrics TSV
        m = results[ptag]["metrics"]
        df = pd.DataFrame([m])
        df.insert(0, "subject", sub)
        df.insert(1, "pipeline", ptag)
        df.to_csv(out_dir / "metrics.tsv", sep="\t", index=False)

        # Model JSON
        info = results[ptag]["info"]
        with open(out_dir / "model.json", "w") as f:
            json.dump(_jsonify(info), f, indent=2)

    # Save preprocessing info
    info_dir = deriv_root / sub / "erp_dss"
    with open(info_dir / "preproc_info.json", "w") as f:
        json.dump(_jsonify(preproc_info), f, indent=2)

    # Clean up
    del epochs, raw_clean, ica_obj, epochs_train, epochs_test
    del pipe_epochs, pipe_evokeds

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  Group Aggregation from Disk
# ═══════════════════════════════════════════════════════════════════════════════


def aggregate_erp_from_disk(deriv_root=None, subjects=None):
    """Read all ERP metrics.tsv files into a single DataFrame."""
    if deriv_root is None:
        deriv_root = DERIV_ROOT
    frames = []
    if subjects is None:
        subjects = sorted(
            [
                d.name
                for d in deriv_root.iterdir()
                if d.is_dir() and d.name.startswith("sub-")
            ]
        )
    for sub in subjects:
        erp_dir = deriv_root / sub / "erp_dss"
        if not erp_dir.exists():
            continue
        for pipe_dir in sorted(erp_dir.iterdir()):
            tsv = pipe_dir / "metrics.tsv"
            if tsv.exists():
                frames.append(pd.read_csv(tsv, sep="\t"))
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
#  Group Visualization
# ═══════════════════════════════════════════════════════════════════════════════


def run_group(subjects, deriv_root=None):
    """Aggregate all per-subject metrics and produce group figures."""
    if deriv_root is None:
        deriv_root = DERIV_ROOT

    erp_fig_dir = deriv_root / "group" / "erp_dss"
    erp_fig_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate
    df_erp = aggregate_erp_from_disk(deriv_root, subjects)
    if df_erp.empty:
        log.error("No ERP metrics found — nothing to aggregate")
        return
    log.info("Aggregated %d rows from disk", len(df_erp))

    # Save master DataFrame
    df_erp.to_csv(erp_fig_dir / "all_erp_metrics.tsv", sep="\t", index=False)
    log.info("Saved %s", erp_fig_dir / "all_erp_metrics.tsv")

    metric_keys = [
        "hedges_g",
        "peak_lat_ms",
        "morph_r",
        "morph_nrmse",
        "split_half_r",
        "auc",
    ]
    col_headers = [
        "Hedges' g",
        "Peak (ms)",
        "Morph r",
        "Morph NRMSE",
        "Split-half r",
        "AUC",
    ]

    # ── Re-compute visualization data for first subject ──────────────────
    sub_show = subjects[0]
    log.info("Generating figures using %s for per-subject plots", sub_show)

    epochs_show, _, ica_show_obj, _ = full_preprocess(sub_show, DATA_ROOT)
    n_ep = len(epochs_show)
    train_idx = np.arange(0, n_ep, 2)
    test_idx = np.arange(1, n_ep, 2)
    epochs_train_show = epochs_show[train_idx]
    epochs_test_show = epochs_show[test_idx]

    meta_test_show = epochs_test_show.metadata
    if meta_test_show is not None and "tone" in meta_test_show.columns:
        test_tone_show = meta_test_show["tone"].values
        test_task_show = meta_test_show["task"].values
        dev_mask_show = (test_tone_show == "deviant") & (test_task_show == "count")
        std_mask_show = (test_tone_show == "standard") & (test_task_show == "count")
    else:
        dev_mask_show = np.zeros(len(test_idx), dtype=bool)
        std_mask_show = np.ones(len(test_idx), dtype=bool)

    times_ms = epochs_test_show.times * 1000

    # Apply all pipelines to show subject
    pipe_epochs_show = {}
    pipe_evokeds_show = {}
    for ptag in PIPE_ORDER:
        ep_out, _ = PIPELINES[ptag](
            epochs_train_show, epochs_test_show, ica_obj=ica_show_obj
        )
        pipe_epochs_show[ptag] = ep_out
        pipe_evokeds_show[ptag] = ep_out.average()

    # ══════════════════════════════════════════════════════════════════════
    #  QA Level A: Component Diagnostics
    # ══════════════════════════════════════════════════════════════════════
    bias_show = AverageBias(axis="epochs")
    dss_show = DSS(bias=bias_show, n_components=DSS_N_COMPONENTS, return_type="sources")
    dss_show.fit(epochs_train_show)
    evals = dss_show.eigenvalues_
    n_ev = len(evals)

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, DSS_N_COMPONENTS + 1, hspace=0.45, wspace=0.35)

    # Row 1: Eigenvalue spectrum
    ax_ev = fig.add_subplot(gs[0, :])
    colors_bar = [PIPE_COLORS["C2"]] * DSS_N_KEEP + ["#cccccc"] * (n_ev - DSS_N_KEEP)
    ax_ev.bar(range(n_ev), evals, color=colors_bar, edgecolor="k", linewidth=0.5)
    ax_ev.axvline(
        DSS_N_KEEP - 0.5,
        color="k",
        ls="--",
        lw=1.5,
        label=f"Keep threshold (top {DSS_N_KEEP})",
    )
    ax_ev.set_xlabel("Component index")
    ax_ev.set_ylabel("Eigenvalue (bias score)")
    ax_ev.set_title("DSS Eigenvalue Spectrum — AverageBias")
    ax_ev.legend()
    if n_ev >= 2:
        ratio = evals[0] / evals[1]
        ax_ev.annotate(
            f"\u03bb\u2081/\u03bb\u2082 = {ratio:.2f}\u00d7",
            xy=(0, evals[0]),
            xytext=(3, evals[0] * 0.9),
            fontsize=10,
            arrowprops={"arrowstyle": "->", "color": "red"},
        )

    # Row 2: Topographies
    for i in range(min(DSS_N_COMPONENTS, dss_show.patterns_.shape[1])):
        ax_t = fig.add_subplot(gs[1, i])
        mne.viz.plot_topomap(
            dss_show.patterns_[:, i], epochs_show.info, axes=ax_t, show=False
        )
        ax_t.set_title(f"DSS {i + 1}\n\u03bb={evals[i]:.4f}", fontsize=9)
    ax_lab = fig.add_subplot(gs[1, DSS_N_COMPONENTS])
    ax_lab.axis("off")
    ax_lab.text(
        0.1, 0.5, "Spatial\nPatterns", fontsize=11, va="center", fontweight="bold"
    )

    # Row 3: Time courses
    sources_train = dss_show.transform(epochs_train_show)
    times_tc = epochs_train_show.times * 1000
    for i in range(min(DSS_N_COMPONENTS, sources_train.shape[1])):
        ax_tc = fig.add_subplot(gs[2, i])
        src_mean = sources_train[:, i, :].mean(axis=0)
        src_sem = sources_train[:, i, :].std(axis=0) / np.sqrt(sources_train.shape[0])
        ax_tc.plot(times_tc, src_mean, color=PIPE_COLORS["C2"], lw=1.5)
        ax_tc.fill_between(
            times_tc,
            src_mean - src_sem,
            src_mean + src_sem,
            color=PIPE_COLORS["C2"],
            alpha=0.2,
        )
        ax_tc.axvline(0, color="gray", ls="--", alpha=0.5)
        ax_tc.axhline(0, color="gray", alpha=0.3)
        ax_tc.set_xlabel("Time (ms)")
        if i == 0:
            ax_tc.set_ylabel("Amplitude (a.u.)")
        ax_tc.set_title(f"DSS {i + 1}", fontsize=9)
    ax_lab2 = fig.add_subplot(gs[2, DSS_N_COMPONENTS])
    ax_lab2.axis("off")
    ax_lab2.text(
        0.1,
        0.5,
        "Component\nTime Courses\n(avg \u00b1 SEM)",
        fontsize=11,
        va="center",
        fontweight="bold",
    )

    fig.suptitle(
        f"QA Level A — DSS Component Diagnostics ({sub_show})",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(
        erp_fig_dir / "qa_level_a_components.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    # Save DSS eigenvalues/patterns for later analysis
    np.savez_compressed(
        erp_fig_dir / f"dss_components_{sub_show}.npz",
        eigenvalues=dss_show.eigenvalues_,
        patterns=dss_show.patterns_,
        filters=dss_show.filters_,
    )

    # ══════════════════════════════════════════════════════════════════════
    #  QA Level B: Signal Diagnostics
    # ══════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    _nperseg = min(256, epochs_test_show.get_data().shape[-1])

    # Row 1: PSD
    for col, (ch_name, ch_label) in enumerate(
        [(PRIMARY_CH, PRIMARY_CH), (P300_CH, P300_CH)]
    ):
        ax = axes[0, col]
        for ptag in PIPE_ORDER:
            ep = pipe_epochs_show[ptag]
            _ci = ep.ch_names.index(ch_name)
            mean_data = ep.get_data().mean(axis=0)
            freqs, psd = sp_welch(mean_data[_ci], fs=RESAMPLE_FREQ, nperseg=_nperseg)
            ax.semilogy(
                freqs,
                psd,
                color=PIPE_COLORS[ptag],
                lw=1.5,
                alpha=0.8,
                label=PIPE_LABELS[ptag],
            )
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (V\u00b2/Hz)")
        ax.set_title(f"Epoch-Averaged PSD at {ch_label}")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 45)

    # Row 2: Evoked overlay
    for col, ch_name in enumerate([PRIMARY_CH, P300_CH]):
        ax = axes[1, col]
        for ptag in PIPE_ORDER:
            ev = pipe_evokeds_show[ptag]
            _ci = ev.ch_names.index(ch_name)
            ax.plot(
                ev.times * 1000,
                ev.data[_ci] * 1e6,
                color=PIPE_COLORS[ptag],
                lw=1.8,
                alpha=0.85,
                label=PIPE_LABELS[ptag],
            )
        ax.axvline(0, color="gray", ls="--", alpha=0.5)
        ax.axhline(0, color="gray", alpha=0.3)
        for wname, (t0, t1) in ERP_WINDOWS.items():
            c = {"N1": "blue", "MMN": "purple", "P300": "green"}[wname]
            ax.axvspan(t0 * 1000, t1 * 1000, alpha=0.06, color=c)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude (\u00b5V)")
        ax.set_title(f"Evoked Overlay at {ch_name} (test set)")
        ax.legend(fontsize=8)

    # Row 3: Difference waves
    for col, ch_name in enumerate([PRIMARY_CH, P300_CH]):
        ax = axes[2, col]
        for ptag in PIPE_ORDER:
            ep = pipe_epochs_show[ptag]
            data = ep.get_data()
            _ci = ep.ch_names.index(ch_name)
            if dev_mask_show.sum() < 3 or std_mask_show.sum() < 3:
                ax.text(
                    0.5, 0.5, "Insufficient trials", transform=ax.transAxes, ha="center"
                )
                continue
            dev_epochs = data[dev_mask_show, _ci, :] * 1e6
            std_epochs = data[std_mask_show, _ci, :] * 1e6
            diff_wave = dev_epochs.mean(axis=0) - std_epochs.mean(axis=0)
            diff_var = dev_epochs.var(axis=0) / len(dev_epochs) + std_epochs.var(
                axis=0
            ) / len(std_epochs)
            diff_se = np.sqrt(diff_var)
            ax.plot(
                ep.times * 1000,
                diff_wave,
                color=PIPE_COLORS[ptag],
                lw=1.8,
                alpha=0.85,
                label=PIPE_LABELS[ptag],
            )
            ax.fill_between(
                ep.times * 1000,
                diff_wave - diff_se,
                diff_wave + diff_se,
                color=PIPE_COLORS[ptag],
                alpha=0.15,
                lw=0,
            )
        ax.axvline(0, color="gray", ls="--", alpha=0.5)
        ax.axhline(0, color="gray", alpha=0.3)
        for wname, (t0, t1) in ERP_WINDOWS.items():
            c = {"N1": "blue", "MMN": "purple", "P300": "green"}[wname]
            ax.axvspan(t0 * 1000, t1 * 1000, alpha=0.06, color=c)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude (\u00b5V)")
        ax.set_title(f"Difference Wave at {ch_name} (test set)")
        ax.legend(fontsize=8)

    fig.suptitle(
        f"QA Level B — Signal Diagnostics ({sub_show}, test set only)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(
        erp_fig_dir / "qa_level_b_signal.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════
    #  QA Level B+: Condition × Pipeline Interaction
    # ══════════════════════════════════════════════════════════════════════
    assert meta_test_show is not None and "condition" in meta_test_show.columns

    cond_pipe_diff = {}
    cond_pipe_diff_se = {}
    cond_pipe_g = {}
    cond_pipe_n = {}

    ch_idx_pz = pipe_epochs_show["C0"].ch_names.index(P300_CH)
    t0_samp = np.searchsorted(epochs_test_show.times, EVAL_WIN[0])
    t1_samp = np.searchsorted(epochs_test_show.times, EVAL_WIN[1])

    for cond in CONDITIONS:
        cond_mask = meta_test_show["condition"].values == cond
        for ptag in PIPE_ORDER:
            ep = pipe_epochs_show[ptag]
            data = ep.get_data()[:, ch_idx_pz, :] * 1e6
            dev_sel = dev_mask_show & cond_mask
            std_sel = std_mask_show & cond_mask
            cond_pipe_n[(cond, ptag)] = (int(dev_sel.sum()), int(std_sel.sum()))

            if dev_sel.sum() < 2 or std_sel.sum() < 2:
                cond_pipe_diff[(cond, ptag)] = np.full(ep.times.shape, np.nan)
                cond_pipe_diff_se[(cond, ptag)] = np.full(ep.times.shape, np.nan)
                cond_pipe_g[(cond, ptag)] = np.nan
                continue

            ev_dev = data[dev_sel].mean(axis=0)
            ev_std = data[std_sel].mean(axis=0)
            cond_pipe_diff[(cond, ptag)] = ev_dev - ev_std
            diff_var = (
                data[dev_sel].var(axis=0) / dev_sel.sum()
                + data[std_sel].var(axis=0) / std_sel.sum()
            )
            cond_pipe_diff_se[(cond, ptag)] = np.sqrt(diff_var)

            dev_win = data[dev_sel][:, t0_samp:t1_samp].mean(axis=1)
            std_win = data[std_sel][:, t0_samp:t1_samp].mean(axis=1)
            n_d, n_s = len(dev_win), len(std_win)
            pooled = np.sqrt(
                (
                    (n_d - 1) * dev_win.std(ddof=1) ** 2
                    + (n_s - 1) * std_win.std(ddof=1) ** 2
                )
                / (n_d + n_s - 2)
            )
            if pooled > 0:
                d_ = (dev_win.mean() - std_win.mean()) / pooled
                cf = 1 - 3 / (4 * (n_d + n_s - 2) - 1)
                cond_pipe_g[(cond, ptag)] = d_ * cf
            else:
                cond_pipe_g[(cond, ptag)] = 0.0

    # Figure: Condition difference waves
    fig1, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for i, cond in enumerate(CONDITIONS):
        ax = axes[i]
        for ptag in PIPE_ORDER:
            diff = cond_pipe_diff[(cond, ptag)]
            diff_se = cond_pipe_diff_se[(cond, ptag)]
            ax.plot(
                times_ms,
                diff,
                color=PIPE_COLORS[ptag],
                lw=1.8,
                alpha=0.85,
                label=PIPE_LABELS[ptag],
            )
            ax.fill_between(
                times_ms,
                diff - diff_se,
                diff + diff_se,
                color=PIPE_COLORS[ptag],
                alpha=0.15,
                lw=0,
            )
        ax.axvline(0, color="gray", ls="--", alpha=0.5)
        ax.axhline(0, color="gray", alpha=0.3)
        for wname, (t0, t1) in ERP_WINDOWS.items():
            c = {"N1": "blue", "MMN": "purple", "P300": "green"}[wname]
            ax.axvspan(t0 * 1000, t1 * 1000, alpha=0.06, color=c)
        ax.set_xlabel("Time (ms)")
        if i == 0:
            ax.set_ylabel("Amplitude (\u00b5V)")
        ax.set_title(COND_LABELS[cond])
        ax.legend(fontsize=8)
    fig1.suptitle(
        f"Condition \u00d7 Pipeline: Difference Waves at {P300_CH} ({sub_show})",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    fig1.savefig(
        erp_fig_dir / "condition_diff_waves.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig1)

    # Figure: Condition interaction
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(CONDITIONS))
    bar_w = 0.22
    for j, ptag in enumerate(PIPE_ORDER):
        vals = [cond_pipe_g.get((c, ptag), 0) for c in CONDITIONS]
        ax1.bar(
            x + j * bar_w,
            vals,
            bar_w,
            color=PIPE_COLORS[ptag],
            label=PIPE_LABELS[ptag],
            alpha=0.85,
            edgecolor="white",
            lw=0.5,
        )
    ax1.set_xticks(x + bar_w)
    ax1.set_xticklabels([COND_LABELS[c] for c in CONDITIONS])
    ax1.set_ylabel("Hedges' g")
    ax1.set_title("Effect Size by Condition \u00d7 Pipeline")
    ax1.legend(fontsize=9)
    ax1.axhline(0, color="gray", alpha=0.3)

    for ptag in PIPE_ORDER:
        vals = [cond_pipe_g.get((c, ptag), 0) for c in CONDITIONS]
        ax2.plot(
            range(len(CONDITIONS)),
            vals,
            "o-",
            color=PIPE_COLORS[ptag],
            lw=2,
            markersize=8,
            label=PIPE_LABELS[ptag],
        )
    ax2.set_xticks(range(len(CONDITIONS)))
    ax2.set_xticklabels([COND_LABELS[c] for c in CONDITIONS])
    ax2.set_ylabel("Hedges' g")
    ax2.set_title("Condition \u00d7 Pipeline Interaction")
    ax2.legend(fontsize=9)
    ax2.axhline(0, color="gray", alpha=0.3)

    fig2.suptitle(
        f"Condition \u00d7 Pipeline Effect-Size Interaction ({sub_show})",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    fig2.savefig(
        erp_fig_dir / "condition_interaction.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig2)

    # ══════════════════════════════════════════════════════════════════════
    #  QA Level C: Endpoint Metrics Bar Chart
    # ══════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5), constrained_layout=True)
    fig.suptitle(
        "QA Level C — Endpoint Metrics Summary", fontsize=13, fontweight="bold"
    )

    for ax, mk, ml in zip(axes, metric_keys, col_headers):
        means, sems = [], []
        for ptag in PIPE_ORDER:
            vals = df_erp.loc[df_erp["pipeline"] == ptag, mk].dropna()
            means.append(vals.mean() if len(vals) else 0)
            sems.append(vals.sem() if len(vals) > 1 else 0)
        xp = np.arange(len(PIPE_ORDER))
        colors = [PIPE_COLORS[p] for p in PIPE_ORDER]
        bars = ax.bar(
            xp,
            means,
            yerr=sems,
            color=colors,
            edgecolor="k",
            linewidth=0.5,
            capsize=3,
            width=0.6,
            zorder=3,
        )
        base_val = means[0]
        if not np.isnan(base_val):
            ax.axhline(base_val, color="grey", ls="--", lw=0.8, alpha=0.6)
        ax.set_xticks(xp)
        ax.set_xticklabels(
            [PIPE_LABELS[p] for p in PIPE_ORDER], fontsize=7, rotation=30, ha="right"
        )
        ax.set_ylabel(ml, fontsize=9)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        for b, v in zip(bars, means):
            if not np.isnan(v):
                fmt = f"{v:.2f}" if abs(v) < 100 else f"{v:.0f}"
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    b.get_height(),
                    fmt,
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    fig.savefig(
        erp_fig_dir / "endpoint_metrics.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════
    #  Anti-Circularity Null Control
    # ══════════════════════════════════════════════════════════════════════
    log.info("Running null control (%d permutations)...", N_PERM)
    rng = np.random.default_rng(RANDOM_STATE)
    null_pipelines = ["C2", "C0"]
    null_results = {p: {"g": [], "auc": []} for p in null_pipelines}

    for i_perm in range(N_PERM):
        for ptag in null_pipelines:
            ep = pipe_epochs_show[ptag]
            data = ep.get_data()
            idx_pz = ep.ch_names.index(P300_CH)
            times_ = ep.times
            t_mask = (times_ >= EVAL_WIN[0]) & (times_ <= EVAL_WIN[1])

            all_mask = dev_mask_show | std_mask_show
            n_dev = dev_mask_show.sum()
            shuffled_dev = np.zeros(len(data), dtype=bool)
            dev_idx = rng.choice(np.where(all_mask)[0], size=n_dev, replace=False)
            shuffled_dev[dev_idx] = True
            shuffled_std = all_mask & ~shuffled_dev

            amp_dev = data[shuffled_dev, idx_pz, :][:, t_mask].mean(axis=1) * 1e6
            amp_std = data[shuffled_std, idx_pz, :][:, t_mask].mean(axis=1) * 1e6
            null_results[ptag]["g"].append(hedges_g(amp_dev, amp_std))
            null_results[ptag]["auc"].append(
                single_trial_auc(
                    data[:, idx_pz, :] * 1e6,
                    times_,
                    EVAL_WIN,
                    shuffled_dev,
                    shuffled_std,
                    random_state=i_perm,
                )
            )

    for ptag in null_pipelines:
        null_results[ptag]["g"] = np.array(null_results[ptag]["g"])
        null_results[ptag]["auc"] = np.array(null_results[ptag]["auc"])

    # Save null results
    np.savez_compressed(
        erp_fig_dir / "null_results.npz",
        **{
            f"{ptag}_{mk}": null_results[ptag][mk]
            for ptag in null_pipelines
            for mk in ("g", "auc")
        },
    )
    log.info("Saved null_results.npz")

    # ══════════════════════════════════════════════════════════════════════
    #  Summary with Null Overlay
    # ══════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(
        1,
        len(metric_keys),
        figsize=(len(metric_keys) * 4, 4.5),
        constrained_layout=True,
    )
    fig.suptitle(
        "ERP Benchmark — Endpoint Summary with Null Control",
        fontsize=13,
        fontweight="bold",
    )

    for ax, mk, ml in zip(axes, metric_keys, col_headers):
        means = []
        for ptag in PIPE_ORDER:
            vals = df_erp.loc[df_erp["pipeline"] == ptag, mk].dropna()
            means.append(vals.mean() if len(vals) else np.nan)
        xp = np.arange(len(PIPE_ORDER))
        colors = [PIPE_COLORS[p] for p in PIPE_ORDER]
        bars = ax.bar(
            xp, means, color=colors, edgecolor="k", linewidth=0.5, width=0.6, zorder=3
        )
        if mk == "hedges_g" and "C2" in null_results:
            null_g = null_results["C2"]["g"]
            c2_x = PIPE_ORDER.index("C2")
            q025, q975 = np.percentile(null_g, [2.5, 97.5])
            ax.fill_between(
                [c2_x - 0.35, c2_x + 0.35],
                q025,
                q975,
                color="grey",
                alpha=0.25,
                zorder=1,
                label="null 95% CI",
            )
            ax.legend(fontsize=7)
        if mk == "auc":
            ax.axhline(0.5, color="k", ls=":", lw=0.7, alpha=0.5)
        ax.set_xticks(xp)
        ax.set_xticklabels(
            [PIPE_LABELS[p] for p in PIPE_ORDER], fontsize=7, rotation=30, ha="right"
        )
        ax.set_ylabel(ml, fontsize=9)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        for b, v in zip(bars, means):
            if not np.isnan(v):
                fmt = f"{v:.2f}" if abs(v) < 100 else f"{v:.0f}"
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    b.get_height(),
                    fmt,
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    fig.savefig(
        erp_fig_dir / "summary_null_control.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════
    #  Paired Subject-Level Comparison
    # ══════════════════════════════════════════════════════════════════════
    if df_erp["subject"].nunique() > 1:
        fig, axes = plt.subplots(1, len(metric_keys), figsize=(len(metric_keys) * 4, 5))
        for ax, mk, ml in zip(axes, metric_keys, col_headers):
            for sub_ in df_erp["subject"].unique():
                vals = []
                for ptag in PIPE_ORDER:
                    v = df_erp.loc[
                        (df_erp["subject"] == sub_) & (df_erp["pipeline"] == ptag), mk
                    ]
                    vals.append(v.iloc[0] if len(v) else np.nan)
                ax.plot(
                    range(len(PIPE_ORDER)),
                    vals,
                    "o-",
                    color="gray",
                    alpha=0.3,
                    markersize=3,
                )
            means_ = [
                df_erp.loc[df_erp["pipeline"] == p, mk].mean() for p in PIPE_ORDER
            ]
            ax.plot(
                range(len(PIPE_ORDER)),
                means_,
                "s-",
                color="k",
                markersize=8,
                lw=2,
                zorder=5,
                label="Group mean",
            )
            ax.set_xticks(range(len(PIPE_ORDER)))
            ax.set_xticklabels(
                [PIPE_LABELS[p] for p in PIPE_ORDER],
                fontsize=8,
                rotation=30,
                ha="right",
            )
            ax.set_ylabel(ml)
            ax.legend(fontsize=7)
            ax.grid(axis="y", alpha=0.3)

        plt.suptitle("Paired Subject-Level Comparison", fontsize=12, fontweight="bold")
        plt.tight_layout()
        fig.savefig(
            erp_fig_dir / "paired_comparison.png",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)

    # Clean up
    del epochs_show, epochs_train_show, epochs_test_show
    del pipe_epochs_show, pipe_evokeds_show
    plt.close("all")
    log.info("Group figures saved to %s", erp_fig_dir)


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Runabout ds003620 — ERP DSS Benchmark"
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
        processed = sorted(
            [
                d.name
                for d in DERIV_ROOT.iterdir()
                if d.is_dir() and d.name.startswith("sub-") and (d / "erp_dss").exists()
            ]
        )
        log.info("Group aggregation for %d subjects", len(processed))
        run_group(processed, DERIV_ROOT)
        return

    if args.slurm_array:
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 1))
        sub = f"sub-{task_id:02d}"
        log.info("SLURM array task %d -> %s", task_id, sub)
        process_subject(sub)
        return

    if args.subject:
        process_subject(args.subject)
        return

    # --all: process every subject then run group
    t0 = time.perf_counter()
    for sub in ALL_SUBJECTS:
        sub_path = DATA_ROOT / sub
        if not sub_path.exists():
            log.warning("Skipping %s — not found at %s", sub, sub_path)
            continue
        process_subject(sub)

    elapsed = time.perf_counter() - t0
    log.info("All subjects processed in %.1f s", elapsed)

    processed = sorted(
        [
            d.name
            for d in DERIV_ROOT.iterdir()
            if d.is_dir() and d.name.startswith("sub-") and (d / "erp_dss").exists()
        ]
    )
    run_group(processed, DERIV_ROOT)


if __name__ == "__main__":
    main()
