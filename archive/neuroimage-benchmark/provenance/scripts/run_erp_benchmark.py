#!/usr/bin/env python
"""ERP DSS Benchmark — standalone script (converted from notebook).

Run with:  python run_erp_benchmark.py
"""

import json
import sys
import time as _time
import warnings

import matplotlib

matplotlib.use("Agg")  # non-interactive — fast, no GUI
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

from mne_denoise.dss import DSS, AverageBias
from mne_denoise.zapline import ZapLine

warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("WARNING")

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "font.size": 10,
        "axes.titlesize": 11,
        "figure.facecolor": "white",
    }
)

# ══════════════════════════════════════════════════════════════════════════════
# 1 ── Configuration
# ══════════════════════════════════════════════════════════════════════════════
SMOKE = False
N_SUBJECTS = 5  # Cap number of subjects (set to 44 for full run)

DATASET_ID = "ds003620"
DATASET_VERSION = "1.1.1"
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

ERP_WINDOWS = {
    "N1": (0.080, 0.130),
    "MMN": (0.100, 0.250),
    "P300": (0.250, 0.500),
}
BASELINE_WIN = (-0.200, 0.000)
EVAL_WIN = ERP_WINDOWS["P300"]

PRIMARY_CH = "Cz"
P300_CH = "Pz"

DSS_N_COMPONENTS = 5
DSS_N_KEEP = 3

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

import pathlib

DATA_ROOT = pathlib.Path(r"d:\mne-denoise\condescending-liskov\data\runabout")
DERIV_ROOT = DATA_ROOT / "derivatives" / "mne-denoise"

RANDOM_STATE = 42
N_PERM = 500

print(f"Mode: {'SMOKE (1 subject)' if SMOKE else f'FULL ({N_SUBJECTS} subjects)'}")
print(f"Evaluation window: {EVAL_WIN[0] * 1000:.0f}–{EVAL_WIN[1] * 1000:.0f} ms (P300)")
print(f"MNE version: {mne.__version__}")
print(f"NumPy version: {np.__version__}")

# ══════════════════════════════════════════════════════════════════════════════
# 2 ── Data Discovery
# ══════════════════════════════════════════════════════════════════════════════
sub_dirs = sorted(
    d.name for d in DATA_ROOT.iterdir() if d.is_dir() and d.name.startswith("sub-")
)

if SMOKE:
    ALL_SUBJECTS = sub_dirs[:1]
else:
    ALL_SUBJECTS = sub_dirs[:N_SUBJECTS]

print(f"Found {len(sub_dirs)} subjects; using {len(ALL_SUBJECTS)}: {ALL_SUBJECTS}")

# ══════════════════════════════════════════════════════════════════════════════
# 3 ── Full Preprocessing Pipeline
# ══════════════════════════════════════════════════════════════════════════════
ORIG_SFREQ = 500.0


def _parse_trial_type(tt):
    tone = "deviant" if tt.startswith("t_") else "standard"
    task = "ignore" if "dontcount" in tt.lower() else "count"
    for cond in ("lab", "oval", "campus"):
        if f"_{cond}" in tt:
            return tone, task, cond
    return tone, task, "unknown"


def full_preprocess(sub, ds_path):
    eeg_dir = ds_path / sub / "eeg"
    vhdr_files = sorted(eeg_dir.glob("*.vhdr"))
    if not vhdr_files:
        raise FileNotFoundError(f"No .vhdr file for {sub}")
    raw = mne.io.read_raw_brainvision(str(vhdr_files[0]), preload=True, verbose=False)

    mapping = {}
    for ch in raw.ch_names:
        cleaned = ch.encode("latin-1", errors="replace").decode(
            "utf-8", errors="replace"
        )
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

    raw.filter(
        l_freq=HP_FREQ, h_freq=None, method="fir", fir_design="firwin", verbose=False
    )

    zap = ZapLine(sfreq=sfreq, line_freq=LINE_FREQ, n_remove="auto", n_harmonics=3)
    raw = zap.fit_transform(raw)
    n_zap_removed = getattr(zap, "n_removed_", None)

    ch_data = raw.get_data()
    ch_var = np.var(ch_data, axis=1)
    median_var = np.median(ch_var)
    bad_chs = []
    for ci, ch_name in enumerate(raw.ch_names):
        if ch_var[ci] < median_var * 0.01 or ch_var[ci] > median_var * 50:
            bad_chs.append(ch_name)
    raw.info["bads"] = bad_chs

    raw.filter(
        l_freq=None, h_freq=LP_FREQ, method="fir", fir_design="firwin", verbose=False
    )

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

    if bad_chs:
        raw_clean.interpolate_bads(verbose=False)

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


# Quick test
_test_ep, _test_raw, _test_ica, _test_info = full_preprocess(ALL_SUBJECTS[0], DATA_ROOT)
print(
    f"Test: {_test_info['n_epochs_after_reject']} epochs, "
    f"{_test_info['n_channels']} ch, "
    f"reject rate = {_test_info['reject_rate']:.1f}%"
)
print(f"  Deviant: {_test_info['n_deviant']},  Standard: {_test_info['n_standard']}")
del _test_ep, _test_raw, _test_ica, _test_info


# ══════════════════════════════════════════════════════════════════════════════
# 4 ── Endpoint Metric Functions
# ══════════════════════════════════════════════════════════════════════════════
def hedges_g(x, y):
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
    mask = (times_s < excl_win[0]) | (times_s > excl_win[1])
    if mask.sum() < 3:
        return np.nan
    r, _ = stats.pearsonr(ev_pipe[mask], ev_base[mask])
    return r


def morphology_nrmse(ev_pipe, ev_base, times_s, excl_win):
    mask = (times_s < excl_win[0]) | (times_s > excl_win[1])
    if mask.sum() < 3:
        return np.nan
    pipe_seg = ev_pipe[mask]
    base_seg = ev_base[mask]
    rmse = np.sqrt(np.mean((pipe_seg - base_seg) ** 2))
    iqr = np.percentile(base_seg, 75) - np.percentile(base_seg, 25)
    return rmse / iqr if iqr > 0 else np.nan


def split_half_reliability(epoch_data_1ch):
    ev_even = epoch_data_1ch[0::2].mean(axis=0)
    ev_odd = epoch_data_1ch[1::2].mean(axis=0)
    r, _ = stats.pearsonr(ev_even, ev_odd)
    sb_r = (2 * r) / (1 + abs(r))
    return sb_r


def single_trial_auc(
    epochs_data_1ch, times, time_window, dev_mask, std_mask, random_state=42
):
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
    eval_win=EVAL_WIN,
):
    if hasattr(epochs_test, "get_data"):
        data = epochs_test.get_data()
    else:
        data = epochs_test
    pz_data = data[:, ch_pz_idx, :] * 1e6
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
    ev_pz = evoked_test.data[ch_pz_idx] * 1e6
    plat = peak_latency_ms(ev_pz, evoked_test.times, eval_win, mode="pos")
    ev_cz = evoked_test.data[ch_cz_idx] * 1e6
    base_cz = evoked_baseline.data[ch_cz_idx] * 1e6
    mr = morphology_corr(ev_cz, base_cz, evoked_test.times, eval_win)
    nrmse = morphology_nrmse(ev_cz, base_cz, evoked_test.times, eval_win)
    shr = split_half_reliability(pz_data)
    auc = single_trial_auc(pz_data, times, eval_win, dev_mask, std_mask)
    return {
        "hedges_g": g,
        "peak_lat_ms": plat,
        "morph_r": mr,
        "morph_nrmse": nrmse,
        "split_half_r": shr,
        "auc": auc,
    }


print("✓ Metric functions defined")


# ══════════════════════════════════════════════════════════════════════════════
# 5 ── Pipeline Application Functions
# ══════════════════════════════════════════════════════════════════════════════
def apply_c0_baseline(epochs_train, epochs_test, **kwargs):
    return epochs_test.copy(), {"pipeline": "C0_baseline"}


def apply_c1_ica(epochs_train, epochs_test, ica_obj=None, **kwargs):
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
        data_recon,
        epochs_test.info,
        tmin=epochs_test.tmin,
        verbose=False,
    )
    if epochs_test.metadata is not None:
        epochs_out.metadata = epochs_test.metadata.copy()
    info = {
        "pipeline": "C2_dss",
        "n_components": n_components,
        "n_keep": n_keep,
        "eigenvalues": dss.eigenvalues_.tolist()
        if hasattr(dss, "eigenvalues_")
        else None,
    }
    return epochs_out, info


PIPELINES = {
    "C0": apply_c0_baseline,
    "C1": apply_c1_ica,
    "C2": apply_c2_dss,
}

print("✓ Pipeline functions defined")


# ══════════════════════════════════════════════════════════════════════════════
# 6 ── Per-Subject Processing
# ══════════════════════════════════════════════════════════════════════════════
def process_subject(sub, ds_path=None, deriv_root=DERIV_ROOT):
    if ds_path is None:
        ds_path = DATA_ROOT

    print(f"\n{'=' * 60}")
    print(f"  {sub} — ERP DSS Benchmark")
    print(f"{'=' * 60}")

    epochs, raw_clean, ica_obj, preproc_info = full_preprocess(sub, ds_path)
    print(
        f"  Preprocessing: {preproc_info['n_epochs_after_reject']} epochs, "
        f"reject rate = {preproc_info['reject_rate']:.1f}%"
    )

    n_ep = len(epochs)
    train_idx = np.arange(0, n_ep, 2)
    test_idx = np.arange(1, n_ep, 2)
    epochs_train = epochs[train_idx]
    epochs_test = epochs[test_idx]
    print(f"  Train/test split: {len(train_idx)} / {len(test_idx)}")

    meta_test = epochs_test.metadata
    if meta_test is not None and "tone" in meta_test.columns:
        test_tone = meta_test["tone"].values
        test_task = meta_test["task"].values
        dev_count_mask = (test_tone == "deviant") & (test_task == "count")
        std_count_mask = (test_tone == "standard") & (test_task == "count")
    else:
        dev_count_mask = np.zeros(len(test_idx), dtype=bool)
        std_count_mask = np.ones(len(test_idx), dtype=bool)
    print(
        f"  Conditions (test): {dev_count_mask.sum()} dev+count, "
        f"{std_count_mask.sum()} std+count"
    )

    ch_pz = epochs_test.ch_names.index(P300_CH)
    ch_cz = epochs_test.ch_names.index(PRIMARY_CH)
    times = epochs_test.times

    results = {}
    pipe_epochs = {}
    pipe_evokeds = {}

    for ptag in PIPE_ORDER:
        print(f"  ── {ptag}: {PIPE_LABELS[ptag]} ... ", end="", flush=True)
        try:
            ep_out, p_info = PIPELINES[ptag](
                epochs_train,
                epochs_test,
                ica_obj=ica_obj,
            )
            pipe_epochs[ptag] = ep_out
            pipe_evokeds[ptag] = ep_out.average()
            results[ptag] = {"info": p_info}
            print("✓")
        except Exception as e:
            print(f"FAILED: {e}")
            pipe_epochs[ptag] = epochs_test.copy()
            pipe_evokeds[ptag] = epochs_test.average()
            results[ptag] = {"info": {"error": str(e)}}

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
        print(
            f"  {ptag}: g={m['hedges_g']:.3f}, "
            f"lat={m['peak_lat_ms']:.0f}ms, "
            f"shr={m['split_half_r']:.3f}"
        )

    def _jsonify(obj):
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

    for ptag in PIPE_ORDER:
        out_dir = deriv_root / sub / "erp_dss" / ptag
        out_dir.mkdir(parents=True, exist_ok=True)
        m = results[ptag]["metrics"]
        df = pd.DataFrame([m])
        df.insert(0, "subject", sub)
        df.insert(1, "pipeline", ptag)
        df.to_csv(out_dir / "metrics.tsv", sep="\t", index=False)
        pipe_evokeds[ptag].save(
            out_dir / "evoked-ave.fif", overwrite=True, verbose=False
        )
        info = results[ptag]["info"]
        info_clean = _jsonify(info)
        with open(out_dir / "model.json", "w") as f:
            json.dump(info_clean, f, indent=2)

    info_dir = deriv_root / sub / "erp_dss"
    with open(info_dir / "preproc_info.json", "w") as f:
        json.dump(_jsonify(preproc_info), f, indent=2)

    bias_save = AverageBias(axis="epochs")
    dss_save = DSS(bias=bias_save, n_components=DSS_N_COMPONENTS, return_type="sources")
    dss_save.fit(epochs_train)
    dss_dir = deriv_root / sub / "erp_dss" / "dss_model"
    dss_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        dss_dir / "dss_model.npz",
        eigenvalues=dss_save.eigenvalues_,
        patterns=dss_save.patterns_,
        filters=dss_save.filters_,
    )
    print(f"  Saved DSS model → {dss_dir / 'dss_model.npz'}")

    del epochs, raw_clean, ica_obj, epochs_train, epochs_test
    del pipe_epochs, pipe_evokeds

    return results


print("✓ process_subject() defined")

# ══════════════════════════════════════════════════════════════════════════════
# 7 ── Main Processing Loop
# ══════════════════════════════════════════════════════════════════════════════
all_results = {}
t0 = _time.perf_counter()

for sub in ALL_SUBJECTS:
    all_results[sub] = process_subject(sub)

elapsed = _time.perf_counter() - t0
print(f"\n{'=' * 60}")
print(f"All subjects processed in {elapsed:.1f} s")
print(f"{'=' * 60}")


# ══════════════════════════════════════════════════════════════════════════════
# 8 ── Group Aggregation from Disk
# ══════════════════════════════════════════════════════════════════════════════
def aggregate_erp_from_disk(deriv_root=DERIV_ROOT, subjects=None):
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


df_erp = aggregate_erp_from_disk(subjects=ALL_SUBJECTS)
print(f"Aggregated {len(df_erp)} rows from disk")

ERP_FIG_DIR = DERIV_ROOT / "group" / "erp_dss"
ERP_FIG_DIR.mkdir(parents=True, exist_ok=True)
df_erp.to_csv(ERP_FIG_DIR / "metrics_all.tsv", sep="\t", index=False)
print(f"Saved: {ERP_FIG_DIR / 'metrics_all.tsv'}")
print(df_erp.to_string())

# ══════════════════════════════════════════════════════════════════════════════
# 9 ── QA Level A: Component Diagnostics
# ══════════════════════════════════════════════════════════════════════════════
print("\n── QA Level A: Component Diagnostics ──")
sub_show = ALL_SUBJECTS[0]
epochs_show, _, _, _ = full_preprocess(sub_show, DATA_ROOT)
n_ep = len(epochs_show)
train_idx = np.arange(0, n_ep, 2)
test_idx = np.arange(1, n_ep, 2)
epochs_train_show = epochs_show[train_idx]
epochs_test_show = epochs_show[test_idx]

bias_show = AverageBias(axis="epochs")
dss_show = DSS(bias=bias_show, n_components=DSS_N_COMPONENTS, return_type="sources")
dss_show.fit(epochs_train_show)

evals = dss_show.eigenvalues_
n_ev = len(evals)

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, DSS_N_COMPONENTS + 1, hspace=0.45, wspace=0.35)

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
ax_ev.set_title("DSS Eigenvalue Spectrum — AverageBias (trial reproducibility)")
ax_ev.legend()
if n_ev >= 2:
    ratio = evals[0] / evals[1]
    ax_ev.annotate(
        f"λ₁/λ₂ = {ratio:.2f}×",
        xy=(0, evals[0]),
        xytext=(3, evals[0] * 0.9),
        fontsize=10,
        arrowprops={"arrowstyle": "->", "color": "red"},
    )

for i in range(min(DSS_N_COMPONENTS, dss_show.patterns_.shape[1])):
    ax_t = fig.add_subplot(gs[1, i])
    mne.viz.plot_topomap(
        dss_show.patterns_[:, i], epochs_show.info, axes=ax_t, show=False
    )
    ax_t.set_title(f"DSS {i + 1}\nλ={evals[i]:.4f}", fontsize=9)

ax_lab = fig.add_subplot(gs[1, DSS_N_COMPONENTS])
ax_lab.axis("off")
ax_lab.text(0.1, 0.5, "Spatial\nPatterns", fontsize=11, va="center", fontweight="bold")

sources_train = dss_show.transform(epochs_train_show)
times_ms = epochs_train_show.times * 1000
for i in range(min(DSS_N_COMPONENTS, sources_train.shape[1])):
    ax_tc = fig.add_subplot(gs[2, i])
    src_mean = sources_train[:, i, :].mean(axis=0)
    src_sem = sources_train[:, i, :].std(axis=0) / np.sqrt(sources_train.shape[0])
    ax_tc.plot(times_ms, src_mean, color=PIPE_COLORS["C2"], lw=1.5)
    ax_tc.fill_between(
        times_ms,
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
    "Component\nTime Courses\n(avg ± SEM)",
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
    ERP_FIG_DIR / "qa_level_a_components.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)
print(f"Saved: {ERP_FIG_DIR / 'qa_level_a_components.png'}")
plt.close(fig)

print(f"Top eigenvalue: λ₁ = {evals[0]:.6f}")
if n_ev >= 2:
    print(f"λ₁/λ₂ = {evals[0] / evals[1]:.2f}×")
print(
    f"Cumulative (top-{DSS_N_KEEP}): "
    f"{evals[:DSS_N_KEEP].sum() / evals.sum() * 100:.1f}% of total bias"
)

# ══════════════════════════════════════════════════════════════════════════════
# 10 ── QA Level B: Signal Diagnostics
# ══════════════════════════════════════════════════════════════════════════════
print("\n── QA Level B: Signal Diagnostics ──")
pipe_epochs_show = {}
pipe_evokeds_show = {}
for ptag in PIPE_ORDER:
    _, _, ica_show, _ = full_preprocess(sub_show, DATA_ROOT)
    ep_out, _ = PIPELINES[ptag](epochs_train_show, epochs_test_show, ica_obj=ica_show)
    pipe_epochs_show[ptag] = ep_out
    pipe_evokeds_show[ptag] = ep_out.average()

meta_test_show = epochs_test_show.metadata
if meta_test_show is not None and "tone" in meta_test_show.columns:
    test_tone_show = meta_test_show["tone"].values
    test_task_show = meta_test_show["task"].values
    dev_mask_show = (test_tone_show == "deviant") & (test_task_show == "count")
    std_mask_show = (test_tone_show == "standard") & (test_task_show == "count")
else:
    dev_mask_show = np.zeros(len(epochs_test_show), dtype=bool)
    std_mask_show = np.ones(len(epochs_test_show), dtype=bool)

fig, axes = plt.subplots(3, 2, figsize=(16, 14))
times_ms = epochs_test_show.times * 1000

_nperseg = min(256, epochs_test_show.get_data().shape[-1])
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
    ax.set_ylabel("PSD (V²/Hz)")
    ax.set_title(f"Epoch-Averaged PSD at {ch_label}")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 45)

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
    for wname, (t0_, t1_) in ERP_WINDOWS.items():
        c = {"N1": "blue", "MMN": "purple", "P300": "green"}[wname]
        ax.axvspan(t0_ * 1000, t1_ * 1000, alpha=0.06, color=c)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title(f"Evoked Overlay at {ch_name} (test set)")
    ax.legend(fontsize=8)

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
        ev_dev = dev_epochs.mean(axis=0)
        ev_std = std_epochs.mean(axis=0)
        diff_wave = ev_dev - ev_std
        diff_var = (dev_epochs.var(axis=0) / len(dev_epochs)) + (
            std_epochs.var(axis=0) / len(std_epochs)
        )
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
    for wname, (t0_, t1_) in ERP_WINDOWS.items():
        c = {"N1": "blue", "MMN": "purple", "P300": "green"}[wname]
        ax.axvspan(t0_ * 1000, t1_ * 1000, alpha=0.06, color=c)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title(f"Difference Wave at {ch_name} (test set)")
    ax.legend(fontsize=8)

fig.suptitle(
    f"QA Level B — Signal Diagnostics ({sub_show}, test set only)",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout()
fig.savefig(
    ERP_FIG_DIR / "qa_level_b_signal.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)
print(f"Saved: {ERP_FIG_DIR / 'qa_level_b_signal.png'}")
plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# 11 ── QA Level B+: Condition × Pipeline Interaction
# ══════════════════════════════════════════════════════════════════════════════
print("\n── QA Level B+: Condition × Pipeline Interaction ──")
import seaborn as sns

CONDITIONS = ["lab", "oval", "campus"]
COND_LABELS = {"lab": "Lab", "oval": "Oval Office", "campus": "Campus"}

cond_pipe_diff = {}
cond_pipe_g = {}
cond_pipe_n = {}
cond_pipe_diff_se = {}

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
        diff_var = (data[dev_sel].var(axis=0) / len(dev_sel)) + (
            data[std_sel].var(axis=0) / len(std_sel)
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
            d = (dev_win.mean() - std_win.mean()) / pooled
            cf = 1 - 3 / (4 * (n_d + n_s - 2) - 1)
            cond_pipe_g[(cond, ptag)] = d * cf
        else:
            cond_pipe_g[(cond, ptag)] = 0.0

# Figure — Difference waves per condition
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
    for wname, (t0_, t1_) in ERP_WINDOWS.items():
        c = {"N1": "blue", "MMN": "purple", "P300": "green"}[wname]
        ax.axvspan(t0_ * 1000, t1_ * 1000, alpha=0.06, color=c)
    ax.set_xlabel("Time (ms)")
    if i == 0:
        ax.set_ylabel("Amplitude (µV)")
    ax.set_title(COND_LABELS[cond])
    ax.legend(fontsize=8)

fig1.suptitle(
    f"Condition × Pipeline: Difference Waves at {P300_CH} ({sub_show})",
    fontsize=13,
    fontweight="bold",
)
plt.tight_layout()
fig1.savefig(
    ERP_FIG_DIR / "condition_diff_waves.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)
print(f"Saved: {ERP_FIG_DIR / 'condition_diff_waves.png'}")
plt.close(fig1)

# Figure — Hedges' g violin + interaction
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

vdata = []
for cond in CONDITIONS:
    for ptag in PIPE_ORDER:
        g = cond_pipe_g.get((cond, ptag), np.nan)
        if not np.isnan(g):
            vdata.append(
                {
                    "Condition": COND_LABELS[cond],
                    "Pipeline": PIPE_LABELS[ptag],
                    "Hedges_g": g,
                    "ptag": ptag,
                }
            )
df_cond_g = pd.DataFrame(vdata)

if len(df_cond_g) > 0:
    cond_order = [COND_LABELS[c] for c in CONDITIONS]
    palette = {PIPE_LABELS[p]: PIPE_COLORS[p] for p in PIPE_ORDER}

    sns.stripplot(
        data=df_cond_g,
        x="Condition",
        y="Hedges_g",
        hue="Pipeline",
        order=cond_order,
        hue_order=[PIPE_LABELS[p] for p in PIPE_ORDER],
        palette=palette,
        dodge=True,
        size=9,
        alpha=0.85,
        ax=ax1,
        zorder=5,
    )
    ax1.axhline(0, color="gray", alpha=0.3)
    ax1.set_ylabel("Hedges' g")
    ax1.set_title("Effect Size by Condition × Pipeline")
    ax1.legend(fontsize=8)

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
ax2.set_title("Condition × Pipeline Interaction")
ax2.legend(fontsize=9)
ax2.axhline(0, color="gray", alpha=0.3)

fig2.suptitle(
    f"Condition × Pipeline Effect-Size Interaction ({sub_show})",
    fontsize=13,
    fontweight="bold",
)
plt.tight_layout()
fig2.savefig(
    ERP_FIG_DIR / "condition_interaction.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)
print(f"Saved: {ERP_FIG_DIR / 'condition_interaction.png'}")
plt.close(fig2)

print(
    f"\n{'Condition':<14} {'Pipeline':<20} {'Hedges g':>10}  {'n_dev':>6} {'n_std':>6}"
)
print("─" * 60)
for cond in CONDITIONS:
    for ptag in PIPE_ORDER:
        g = cond_pipe_g.get((cond, ptag), np.nan)
        nd, ns = cond_pipe_n.get((cond, ptag), (0, 0))
        print(
            f"{COND_LABELS[cond]:<14} {PIPE_LABELS[ptag]:<20} {g:>10.4f}"
            f"  {nd:>6d} {ns:>6d}"
        )

# ══════════════════════════════════════════════════════════════════════════════
# 12 ── QA Level C: Endpoint Metrics
# ══════════════════════════════════════════════════════════════════════════════
print("\n── QA Level C: Endpoint Metrics ──")
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

print(f"\n{'Pipeline':<12}  {'N':>4}  " + "  ".join(f"{h:>12}" for h in col_headers))
print("─" * 80)

for ptag in PIPE_ORDER:
    sub_df = df_erp[df_erp["pipeline"] == ptag]
    n = len(sub_df)
    row = f"{PIPE_LABELS[ptag]:<12}  {n:>4}  "
    for mk in metric_keys:
        vals = sub_df[mk].dropna()
        if len(vals) == 0:
            row += f"{'N/A':>12}  "
        elif len(vals) == 1:
            row += f"{vals.iloc[0]:>12.3f}  "
        else:
            row += f"{vals.mean():>6.3f}±{vals.std():>4.3f}  "
    print(row)
print("─" * 80)
print(f"N subjects = {df_erp['subject'].nunique()}")

# Violin + swarm plots
n_subjects = df_erp["subject"].nunique()
n_metrics = len(metric_keys)
fig, axes = plt.subplots(
    1, n_metrics, figsize=(4 * n_metrics, 5.5), constrained_layout=True
)
fig.suptitle(
    f"QA Level C — Endpoint Metrics (N = {n_subjects})", fontsize=13, fontweight="bold"
)

pipe_label_order = [PIPE_LABELS[p] for p in PIPE_ORDER]
palette = {PIPE_LABELS[p]: PIPE_COLORS[p] for p in PIPE_ORDER}

for ax, mk, ml in zip(axes, metric_keys, col_headers):
    vdata = []
    for ptag in PIPE_ORDER:
        for val in df_erp.loc[df_erp["pipeline"] == ptag, mk].dropna():
            vdata.append({"Pipeline": PIPE_LABELS[ptag], "value": val})
    if not vdata:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        continue
    df_v = pd.DataFrame(vdata)
    sns.violinplot(
        data=df_v,
        x="Pipeline",
        y="value",
        order=pipe_label_order,
        palette=palette,
        inner=None,
        linewidth=0.8,
        alpha=0.3,
        ax=ax,
        cut=0,
        density_norm="width",
    )
    sns.stripplot(
        data=df_v,
        x="Pipeline",
        y="value",
        order=pipe_label_order,
        palette=palette,
        size=3,
        alpha=0.7,
        jitter=0.12,
        ax=ax,
        zorder=5,
    )
    for sub in df_erp["subject"].unique():
        sub_vals = []
        for ptag in PIPE_ORDER:
            v = df_erp.loc[
                (df_erp["subject"] == sub) & (df_erp["pipeline"] == ptag), mk
            ]
            sub_vals.append(v.iloc[0] if len(v) else float("nan"))
        ax.plot(
            range(len(PIPE_ORDER)),
            sub_vals,
            "-",
            color="gray",
            alpha=0.1,
            lw=0.4,
            zorder=1,
        )
    if mk == "auc":
        ax.axhline(0.5, color="k", ls=":", lw=0.7, alpha=0.5, label="Chance")
    base_vals = df_erp.loc[df_erp["pipeline"] == "C0", mk].dropna()
    if len(base_vals) > 0:
        ax.axhline(base_vals.mean(), color="grey", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("")
    ax.set_ylabel(ml, fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=7, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3, zorder=0)

fig.savefig(
    ERP_FIG_DIR / "endpoint_metrics_violins.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)
print(f"Saved: {ERP_FIG_DIR / 'endpoint_metrics_violins.png'}")
plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# 13 ── Anti-Circularity Null Control
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Anti-Circularity Null Control ──")
rng = np.random.default_rng(RANDOM_STATE)

null_pipelines = ["C2", "C0"]
null_results = {p: {"g": [], "auc": []} for p in null_pipelines}

for i_perm in range(N_PERM):
    for ptag in null_pipelines:
        ep = pipe_epochs_show[ptag]
        data = ep.get_data()
        idx_pz = ep.ch_names.index(P300_CH)
        times = ep.times
        t_mask = (times >= EVAL_WIN[0]) & (times <= EVAL_WIN[1])

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
                times,
                EVAL_WIN,
                shuffled_dev,
                shuffled_std,
                random_state=i_perm,
            )
        )

for ptag in null_pipelines:
    null_results[ptag]["g"] = np.array(null_results[ptag]["g"])
    null_results[ptag]["auc"] = np.array(null_results[ptag]["auc"])

np.savez(
    ERP_FIG_DIR / "null_results.npz",
    C2_g=null_results["C2"]["g"],
    C2_auc=null_results["C2"]["auc"],
    C0_g=null_results["C0"]["g"],
    C0_auc=null_results["C0"]["auc"],
    subject=sub_show,
    n_perm=N_PERM,
)
print(f"Saved: {ERP_FIG_DIR / 'null_results.npz'}")

print("\nAnti-Circularity Null Control")
print("=" * 65)
print(
    f"{'Pipeline':<12}  {'Metric':<8}  {'Real':>8}  {'Null mean':>10}  "
    f"{'Null 95%':>14}  {'p':>6}"
)
print("─" * 65)

for ptag in null_pipelines:
    sub_row = df_erp[(df_erp["subject"] == sub_show) & (df_erp["pipeline"] == ptag)]
    real_g = sub_row["hedges_g"].iloc[0] if len(sub_row) else np.nan
    real_auc = sub_row["auc"].iloc[0] if len(sub_row) else np.nan

    null_g = null_results[ptag]["g"]
    p_g = (np.abs(null_g) >= np.abs(real_g)).mean()
    ci_g = np.percentile(null_g, [2.5, 97.5])
    print(
        f"{PIPE_LABELS[ptag]:<12}  {'g':<8}  {real_g:>8.3f}  "
        f"{null_g.mean():>10.3f}  [{ci_g[0]:+.3f},{ci_g[1]:+.3f}]  "
        f"{p_g:>6.3f}"
    )

    null_auc = null_results[ptag]["auc"]
    valid_auc = null_auc[~np.isnan(null_auc)]
    if len(valid_auc) > 0 and not np.isnan(real_auc):
        p_auc = (valid_auc >= real_auc).mean()
        ci_auc = np.percentile(valid_auc, [2.5, 97.5])
        print(
            f"{'':12}  {'AUC':<8}  {real_auc:>8.3f}  "
            f"{valid_auc.mean():>10.3f}  [{ci_auc[0]:.3f},{ci_auc[1]:.3f}]  "
            f"{p_auc:>6.3f}"
        )

# ══════════════════════════════════════════════════════════════════════════════
# 14 ── Figure 5 — Multipanel Storyboard
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Figure 5 — Multipanel Storyboard ──")
n_subjects = df_erp["subject"].nunique()
n_met = len(metric_keys)

fig = plt.figure(figsize=(n_met * 4 + 5, 5.5), constrained_layout=True)
gs = fig.add_gridspec(1, n_met + 1, width_ratios=[1] * n_met + [1.2])

pipe_label_order = [PIPE_LABELS[p] for p in PIPE_ORDER]
palette_v = {PIPE_LABELS[p]: PIPE_COLORS[p] for p in PIPE_ORDER}

for i, (mk, ml) in enumerate(zip(metric_keys, col_headers)):
    ax = fig.add_subplot(gs[0, i])
    vdata = []
    for ptag in PIPE_ORDER:
        for val in df_erp.loc[df_erp["pipeline"] == ptag, mk].dropna():
            vdata.append({"Pipeline": PIPE_LABELS[ptag], "value": val})
    if not vdata:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        continue
    df_v = pd.DataFrame(vdata)
    sns.violinplot(
        data=df_v,
        x="Pipeline",
        y="value",
        order=pipe_label_order,
        palette=palette_v,
        inner=None,
        linewidth=0.8,
        alpha=0.3,
        ax=ax,
        cut=0,
        density_norm="width",
    )
    sns.stripplot(
        data=df_v,
        x="Pipeline",
        y="value",
        order=pipe_label_order,
        palette=palette_v,
        size=3,
        alpha=0.7,
        jitter=0.12,
        ax=ax,
        zorder=5,
    )
    if mk == "hedges_g" and "C2" in null_results:
        null_g = null_results["C2"]["g"]
        c2_idx = PIPE_ORDER.index("C2")
        q025, q975 = np.percentile(null_g, [2.5, 97.5])
        ax.fill_between(
            [c2_idx - 0.35, c2_idx + 0.35],
            q025,
            q975,
            color="grey",
            alpha=0.25,
            zorder=1,
            label=f"Null 95% CI\n[{q025:+.2f}, {q975:+.2f}]",
        )
        ax.legend(fontsize=6, loc="lower right")
    if mk == "auc":
        ax.axhline(0.5, color="k", ls=":", lw=0.7, alpha=0.5)
    ax.set_xlabel("")
    ax.set_ylabel(ml, fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=6, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    if i == 0:
        ax.set_title(
            f"A  Violin + Dots (N={n_subjects})", fontweight="bold", fontsize=9
        )

ax_slope = fig.add_subplot(gs[0, n_met])
if n_subjects > 1:
    for sub in df_erp["subject"].unique():
        g_base = df_erp.loc[
            (df_erp["subject"] == sub) & (df_erp["pipeline"] == "C0"), "hedges_g"
        ]
        g_dss = df_erp.loc[
            (df_erp["subject"] == sub) & (df_erp["pipeline"] == "C2"), "hedges_g"
        ]
        if len(g_base) and len(g_dss):
            ax_slope.plot(
                [0, 1],
                [g_base.iloc[0], g_dss.iloc[0]],
                "o-",
                color=PIPE_COLORS["C2"],
                alpha=0.25,
                markersize=3,
                lw=0.7,
            )
    mean_base = df_erp.loc[df_erp["pipeline"] == "C0", "hedges_g"].mean()
    mean_dss = df_erp.loc[df_erp["pipeline"] == "C2", "hedges_g"].mean()
    ax_slope.plot(
        [0, 1],
        [mean_base, mean_dss],
        "s-",
        color=PIPE_COLORS["C2"],
        markersize=10,
        lw=3,
        label=f"Mean: {mean_base:.2f} → {mean_dss:.2f}",
        zorder=10,
    )
    if "C2" in null_results:
        null_g = null_results["C2"]["g"]
        q025, q975 = np.percentile(null_g, [2.5, 97.5])
        ax_slope.axhspan(
            q025,
            q975,
            xmin=0.6,
            xmax=1.0,
            color="grey",
            alpha=0.15,
            label="Null 95% CI",
        )
    ax_slope.axhline(0, color="gray", ls="--", alpha=0.3)
    ax_slope.set_xticks([0, 1])
    ax_slope.set_xticklabels(["Baseline", "DSS"], fontsize=9)
    ax_slope.set_ylabel("Hedges' g")
    ax_slope.set_title(
        "B  Paired Slopes\n(Baseline → DSS)", fontweight="bold", fontsize=9
    )
    ax_slope.legend(fontsize=7, loc="upper left")
    ax_slope.grid(axis="y", alpha=0.3)
else:
    ax_slope.text(
        0.5, 0.5, "≥ 2 subjects needed", transform=ax_slope.transAxes, ha="center"
    )

fig.suptitle(
    "Figure 5 — ERP Benchmark: Endpoint Summary with Null Control",
    fontsize=13,
    fontweight="bold",
)
fig.savefig(
    ERP_FIG_DIR / "fig5_multipanel_summary.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)
print(f"Saved: {ERP_FIG_DIR / 'fig5_multipanel_summary.png'}")
plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# 15 ── Paired Subject-Level Slope Plots
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Paired Subject-Level Slopes ──")
if df_erp["subject"].nunique() > 1:
    n_subjects = df_erp["subject"].nunique()
    fig, axes = plt.subplots(
        1, len(metric_keys), figsize=(len(metric_keys) * 4, 5), constrained_layout=True
    )

    for ax, mk, ml in zip(axes, metric_keys, col_headers):
        for sub in df_erp["subject"].unique():
            sub_vals = []
            for ptag in PIPE_ORDER:
                v = df_erp.loc[
                    (df_erp["subject"] == sub) & (df_erp["pipeline"] == ptag), mk
                ]
                sub_vals.append(v.iloc[0] if len(v) else np.nan)
            for j in range(len(PIPE_ORDER) - 1):
                ax.plot(
                    [j, j + 1],
                    [sub_vals[j], sub_vals[j + 1]],
                    color=PIPE_COLORS[PIPE_ORDER[j + 1]],
                    alpha=0.15,
                    lw=0.6,
                    zorder=1,
                )
            for j, ptag in enumerate(PIPE_ORDER):
                if not np.isnan(sub_vals[j]):
                    ax.scatter(
                        j,
                        sub_vals[j],
                        color=PIPE_COLORS[ptag],
                        s=8,
                        alpha=0.4,
                        zorder=2,
                    )
        means = [df_erp.loc[df_erp["pipeline"] == p, mk].mean() for p in PIPE_ORDER]
        ax.plot(
            range(len(PIPE_ORDER)),
            means,
            "s-",
            color="k",
            markersize=9,
            lw=2.5,
            zorder=5,
            label="Group mean",
        )
        for j, m in enumerate(means):
            if not np.isnan(m):
                fmt = f"{m:.2f}" if abs(m) < 100 else f"{m:.0f}"
                ax.annotate(
                    fmt,
                    (j, m),
                    textcoords="offset points",
                    xytext=(0, 10),
                    fontsize=7,
                    ha="center",
                    fontweight="bold",
                )
        if mk == "auc":
            ax.axhline(0.5, color="k", ls=":", lw=0.7, alpha=0.5)
        ax.set_xticks(range(len(PIPE_ORDER)))
        ax.set_xticklabels(
            [PIPE_LABELS[p] for p in PIPE_ORDER], fontsize=8, rotation=30, ha="right"
        )
        ax.set_ylabel(ml)
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)
        ax.annotate(
            f"N = {n_subjects}",
            xy=(0.02, 0.98),
            xycoords="axes fraction",
            fontsize=8,
            va="top",
        )

    plt.suptitle(
        "Paired Subject-Level Slopes — Within-Subject Trajectories",
        fontsize=12,
        fontweight="bold",
    )
    fig.savefig(
        ERP_FIG_DIR / "paired_slopes.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    print(f"Saved: {ERP_FIG_DIR / 'paired_slopes.png'}")
    plt.close(fig)
else:
    print("Only 1 subject — set SMOKE = False for paired comparisons.")

# ══════════════════════════════════════════════════════════════════════════════
# 16 ── Reproducibility
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Reproducibility ──")
print(f"Python: {sys.version}")
print(f"MNE: {mne.__version__}")
print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"Matplotlib: {matplotlib.__version__}")
print(f"Random state: {RANDOM_STATE}")
print(f"Subjects: {ALL_SUBJECTS}")
print(f"SMOKE mode: {SMOKE}")
print(f"DSS: n_components={DSS_N_COMPONENTS}, n_keep={DSS_N_KEEP}")
print(f"Null permutations: {N_PERM}")
print("\n✅ ERP benchmark complete.")
