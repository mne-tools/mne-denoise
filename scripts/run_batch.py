#!/usr/bin/env python
"""
Batch pipeline for ds003620 — Case 1 (line noise) & Case 2 / Part III (ERP).

Usage
-----
  # Single subject (local or inside a Slurm array task):
  python run_batch.py --subject sub-01

  # All subjects sequentially (local development):
  python run_batch.py --all

  # Slurm array mode — $SLURM_ARRAY_TASK_ID picks the subject:
  python run_batch.py --slurm-array

The script:
  1.  Downloads data (once, with sentinel check)
  2.  Runs the paper-faithful preprocessing pipeline (Stages 0-5)
  3.  Epochs and attaches condition metadata
  4.  Runs Case 1  — 4-method ZapLine/DSS line-noise benchmark (QA metrics)
  5.  Runs Case 2 / Part III — 4-pipeline ERP benchmark with train/test split
  6.  Saves per-subject JSON + NPZ to OUTPUT_DIR/<sub>/
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# ── Ensure mne-denoise importable ────────────────────────────────────────────
# When running from scripts/ inside the repo, add the repo root to sys.path
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import mne  # noqa: E402

# Silence MNE's verbose INFO messages
mne.set_log_level("WARNING")

# Third-party
import asrpy  # noqa: E402
from config import (  # noqa: E402
    ALL_SUBJECTS,
    ASR_CUTOFF,
    BASELINE,
    CASE1_N_HARMONICS,
    CORR_CRIT,
    DATASET_ROOT,
    DSS_N_COMPONENTS,
    DSS_N_KEEP,
    ERP_WINDOWS,
    EVENT_TYPE_MAP,
    EXCLUDE_CLASSES,
    FLATLINE_CRIT,
    GAP_THRESH,
    HP_FREQ,
    HP_TRANS,
    ICA_THRESHOLD,
    LINE_FREQ,
    LINENOISE_CRIT,
    LP_FREQ,
    LP_TRANS,
    MASTOID_CHS,
    MAX_BROKEN_FRAC,
    ORIG_SFREQ,
    OUTPUT_DIR,
    P300_CH,
    PIPE_ORDER,
    PRIMARY_CH,
    RANDOM_STATE,
    REJECT_THRESH,
    REJECT_TMAX,
    REJECT_TMIN,
    RESAMPLE_FREQ,
    STIMTRAK_LAG,
    TASK,
    TMAX,
    TMIN,
    TRIM_PAD,
    WIN_LEN_CORR,
    WIN_SIZE,
    WIN_STEP,
    XDAWN_N_COMPONENTS,
)
from mne.preprocessing import ICA, Xdawn  # noqa: E402
from mne_icalabel import label_components  # noqa: E402
from scipy import integrate, signal, stats  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

# mne-denoise imports
from mne_denoise.dss import DSS, AverageBias, LineNoiseBias  # noqa: E402
from mne_denoise.zapline import ZapLine  # noqa: E402

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("batch")


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION A – Data Acquisition
# ═══════════════════════════════════════════════════════════════════════════════


def check_subject_data(sub: str, data_dir: Path) -> None:
    """Verify that subject data has been downloaded.

    Data should be pre-downloaded using download_openneuro.py from a login node
    (compute nodes may not have internet access).
    """
    vhdr = list(data_dir.rglob(f"{sub}_task-{TASK}_eeg.vhdr"))
    if not vhdr:
        raise FileNotFoundError(
            f"{sub}: .vhdr not found under {data_dir}.\n"
            f"Run download_openneuro.py first, e.g.:\n"
            f"  python scripts/download_openneuro.py "
            f"--dataset ds003620 --task oddball --subjects {sub}\n"
        )
    trig_path = (
        data_dir
        / "derivatives"
        / "trigger_corrected"
        / sub
        / "eeg"
        / f"{sub}_task-{TASK}_desc-trig_events.tsv"
    )
    if not trig_path.exists():
        raise FileNotFoundError(
            f"{sub}: trigger_corrected events.tsv not found at {trig_path}.\n"
            f"Re-run download_openneuro.py without --skip-derivatives."
        )
    log.info(f"{sub}: data verified at {data_dir}")


def copy_to_tmpdir(sub: str, data_dir: Path, tmpdir: Path) -> Path:
    """Copy a subject's data to SLURM_TMPDIR for fast I/O. Returns new base."""
    local = tmpdir / "ds003620"
    if (local / sub / ".copy_done").exists():
        return local
    src = data_dir / sub
    dst = local / sub
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    # Also copy derivatives
    deriv_src = data_dir / "derivatives" / "trigger_corrected" / sub
    deriv_dst = local / "derivatives" / "trigger_corrected" / sub
    if deriv_src.exists():
        if deriv_dst.exists():
            shutil.rmtree(deriv_dst)
        shutil.copytree(deriv_src, deriv_dst)
    # Copy top-level BIDS files
    for f in ("dataset_description.json", "participants.tsv", "participants.json"):
        p = data_dir / f
        if p.exists() and not (local / f).exists():
            shutil.copy2(p, local / f)
    (local / sub / ".copy_done").write_text("ok")
    log.info(f"{sub}: copied to {local}")
    return local


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION B – QA Metric Functions
# ═══════════════════════════════════════════════════════════════════════════════

# ── Case 1 metrics (ZapLine-plus protocol) ────────────────────────────────────


def geometric_mean_psd(data, sfreq, nperseg=None):
    """Geometric-mean PSD across channels (mean of log-PSDs)."""
    if nperseg is None:
        nperseg = int(sfreq * 4)
    f, psd = signal.welch(data, fs=sfreq, nperseg=nperseg, axis=-1)
    log_psd = np.log10(psd + 1e-30)
    geo_mean = 10 ** log_psd.mean(axis=0)
    return f, geo_mean


def noise_surr_ratio(psd, freqs, f0, center_bw=0.05, side_bw=1.0):
    """R(f0) = mean(center) / mean(sidebands)."""
    center = (freqs >= f0 - center_bw) & (freqs <= f0 + center_bw)
    left = (freqs >= f0 - side_bw) & (freqs < f0 - center_bw)
    right = (freqs > f0 + center_bw) & (freqs <= f0 + side_bw)
    sides = left | right
    if not center.any() or not sides.any():
        return np.nan
    return psd[center].mean() / psd[sides].mean()


def peak_attenuation_db(psd_pre, psd_post, freqs, f0, bw=0.05):
    """Peak attenuation in dB at f0."""
    mask = (freqs >= f0 - bw) & (freqs <= f0 + bw)
    if not mask.any():
        return np.nan
    p_pre = psd_pre[mask].mean()
    p_post = psd_post[mask].mean()
    if p_post <= 0:
        return np.inf
    return 10 * np.log10(p_pre / p_post)


def below_noise_distortion(
    psd_pre, psd_post, freqs, f0, low_offset=11.0, high_offset=1.0
):
    """%ΔA below the noise frequency."""
    mask = (freqs >= f0 - low_offset) & (freqs <= f0 - high_offset)
    if not mask.any():
        return np.nan
    a_pre = integrate.trapezoid(psd_pre[mask], freqs[mask])
    a_post = integrate.trapezoid(psd_post[mask], freqs[mask])
    if a_pre <= 0:
        return np.nan
    return 100.0 * (a_post - a_pre) / a_pre


def overclean_proportion(psd_post, freqs, f0, low=-0.4, high=0.1, n_sigma=2.0):
    """Notch/overclean indicator."""
    region = (freqs >= f0 + low) & (freqs <= f0 + high)
    if not region.any():
        return np.nan
    region_psd = psd_post[region]
    center = (freqs >= f0 - 0.05) & (freqs <= f0 + 0.05)
    if center.any():
        center_power = psd_post[center].mean()
    else:
        center_power = region_psd.mean()
    sides = (freqs >= f0 - 1.0) & (freqs <= f0 + 1.0) & ~center
    if sides.any():
        side_power = psd_post[sides].mean()
        deviation = abs(center_power - side_power)
    else:
        deviation = np.std(region_psd)
        side_power = center_power
    threshold = side_power - n_sigma * deviation if sides.any() else center_power * 0.5
    return float((region_psd < threshold).mean())


def underclean_proportion(psd_post, psd_pre, freqs, f0, bw=0.05, threshold_frac=0.5):
    """Residual/underclean indicator."""
    center = (freqs >= f0 - bw) & (freqs <= f0 + bw)
    sides_mask = ((freqs >= f0 - 1.0) & (freqs < f0 - bw)) | (
        (freqs > f0 + bw) & (freqs <= f0 + 1.0)
    )
    if not center.any() or not sides_mask.any():
        return np.nan
    surround_level = psd_post[sides_mask].mean()
    threshold = surround_level * (1 + threshold_frac)
    return float((psd_post[center] > threshold).mean())


def compute_all_qa_metrics(
    data_before, data_after, sfreq, line_freq=50.0, n_harmonics=3
):
    """Compute all QA metrics (M0–M4) at each harmonic."""
    f, geo_pre = geometric_mean_psd(data_before, sfreq)
    _, geo_post = geometric_mean_psd(data_after, sfreq)
    results = {}
    for k in range(1, n_harmonics + 1):
        fk = line_freq * k
        if fk >= sfreq / 2:
            continue
        results[str(fk)] = {
            "R_pre": noise_surr_ratio(geo_pre, f, fk),
            "R_post": noise_surr_ratio(geo_post, f, fk),
            "attenuation_dB": peak_attenuation_db(geo_pre, geo_post, f, fk),
            "below_noise_pct": below_noise_distortion(geo_pre, geo_post, f, fk),
            "overclean_prop": overclean_proportion(geo_post, f, fk),
            "underclean_prop": underclean_proportion(geo_post, geo_pre, f, fk),
        }
    return results


# ── Case 2 / Part III metrics ────────────────────────────────────────────────


def hedges_g(x, y):
    """Hedges' g (bias-corrected Cohen's d)."""
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
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
    """Peak latency (ms) within a time window."""
    mask = (times_s >= win[0]) & (times_s <= win[1])
    if not mask.any():
        return np.nan
    segment = evoked_data[mask]
    if mode == "pos":
        peak_idx = np.argmax(segment)
    elif mode == "neg":
        peak_idx = np.argmin(segment)
    else:
        peak_idx = np.argmax(np.abs(segment))
    return float(times_s[mask][peak_idx] * 1000)


def morphology_corr(ev_pipe, ev_base, times_s, excl_win):
    """Pearson r outside the evaluation window."""
    mask = (times_s < excl_win[0]) | (times_s > excl_win[1])
    if mask.sum() < 3:
        return np.nan
    r, _ = stats.pearsonr(ev_pipe[mask], ev_base[mask])
    return float(r)


def split_half_reliability(epoch_data):
    """Spearman–Brown corrected split-half reliability."""
    ev_even = epoch_data[0::2].mean(axis=0)
    ev_odd = epoch_data[1::2].mean(axis=0)
    r, _ = stats.pearsonr(ev_even, ev_odd)
    sb_r = (2 * r) / (1 + abs(r))
    return float(sb_r)


def single_trial_auc(
    epochs_data_2d, times, time_window, dev_mask, std_mask, random_state=42
):
    """Decode deviant vs standard from single-trial mean amplitude."""
    t_mask = (times >= time_window[0]) & (times <= time_window[1])
    features = epochs_data_2d[:, t_mask].mean(axis=1).reshape(-1, 1)
    labels = np.full(len(epochs_data_2d), -1)
    labels[dev_mask] = 1
    labels[std_mask] = 0
    valid = labels >= 0
    X, y = features[valid], labels[valid].astype(int)
    if len(np.unique(y)) < 2 or X.shape[0] < 10:
        return np.nan
    clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
    n_min = min(np.bincount(y))
    cv = StratifiedKFold(
        n_splits=min(5, n_min), shuffle=True, random_state=random_state
    )
    try:
        scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
        return float(scores.mean())
    except Exception:
        return np.nan


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION C – Preprocessing Pipeline (Stages 0-5)
# ═══════════════════════════════════════════════════════════════════════════════


def load_raw(sub: str, data_dir: Path) -> mne.io.Raw:
    """Load BrainVision file for one subject."""
    vhdr_files = list(data_dir.resolve().rglob(f"{sub}_task-{TASK}_eeg.vhdr"))
    if not vhdr_files:
        raise FileNotFoundError(f"{sub}: .vhdr not found under {data_dir}")
    raw = mne.io.read_raw_brainvision(str(vhdr_files[0]), preload=True, verbose=False)
    raw.info["line_freq"] = LINE_FREQ
    return raw


def load_events_df(sub: str, data_dir: Path) -> pd.DataFrame:
    """Load trigger-corrected events.tsv and parse to DataFrame."""
    trig_path = (
        data_dir
        / "derivatives"
        / "trigger_corrected"
        / sub
        / "eeg"
        / f"{sub}_task-{TASK}_desc-trig_events.tsv"
    )
    if not trig_path.exists():
        raise FileNotFoundError(f"{sub}: trigger_corrected events.tsv not found")
    df = pd.read_csv(trig_path, sep="\t")
    df = df[df["trial_type"] != "empty"].copy()
    df = df.rename(columns={"onset": "onset_samp"})
    parsed = df["trial_type"].map(EVENT_TYPE_MAP)
    df["tone"] = parsed.apply(lambda x: x[0] if x else None)
    df["task"] = parsed.apply(lambda x: x[1] if x else None)
    df["env"] = parsed.apply(lambda x: x[2] if x else None)
    df["onset_s"] = df["onset_samp"] / ORIG_SFREQ
    return df.reset_index(drop=True)


def stage0_resample_trim_reref(raw: mne.io.Raw) -> tuple:
    """Stage 0: resample, trim, FCz reintro, re-ref. Returns (raw, orig_stim_onsets)."""
    raw.resample(RESAMPLE_FREQ, verbose=False)
    raw.pick_types(eeg=True, verbose=False)

    # Record pre-trim stimulus onsets
    event_times = []
    for ann in raw.annotations:
        desc = ann["description"]
        if desc.startswith("Stimulus/S") or desc.startswith("S"):
            event_times.append(ann["onset"])
    orig_stim_onsets = np.array(sorted(event_times))

    # Trim to task region
    if event_times:
        t_start = max(0, min(event_times) - TRIM_PAD)
        t_end = min(raw.times[-1], max(event_times) + TRIM_PAD)
        raw.crop(tmin=t_start, tmax=t_end)

        # Remove inter-block gaps
        ev_times2 = []
        for ann in raw.annotations:
            desc = ann["description"]
            if desc.startswith("Stimulus/S") or desc.startswith("S"):
                ev_times2.append(ann["onset"])
        ev_times2 = np.sort(ev_times2)
        if len(ev_times2) > 1:
            isi = np.diff(ev_times2)
            gap_idx = np.where(isi > GAP_THRESH)[0]
            if len(gap_idx):
                segments = []
                seg_start = raw.times[0]
                for gi in gap_idx:
                    seg_end = ev_times2[gi] + TRIM_PAD
                    segments.append((seg_start, seg_end))
                    seg_start = ev_times2[gi + 1] - TRIM_PAD
                segments.append((seg_start, raw.times[-1] + 1 / raw.info["sfreq"]))
                pieces = [
                    raw.copy().crop(
                        tmin=max(s, raw.times[0]),
                        tmax=min(e, raw.times[-1]),
                    )
                    for s, e in segments
                ]
                raw = mne.concatenate_raws(pieces)

    # Reintroduce FCz
    fcz_info = mne.create_info(["FCz"], raw.info["sfreq"], ch_types="eeg")
    fcz_raw = mne.io.RawArray(np.zeros((1, len(raw.times))), fcz_info, verbose=False)
    raw.add_channels([fcz_raw], force_update_info=True)
    raw.set_eeg_reference("average", verbose=False)

    # Re-ref to TP9/TP10 and drop
    mastoids_present = [ch for ch in MASTOID_CHS if ch in raw.ch_names]
    if len(mastoids_present) == 2:
        raw.set_eeg_reference(ref_channels=mastoids_present, verbose=False)
        raw.drop_channels(mastoids_present)

    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, on_missing="warn", verbose=False)
    return raw, orig_stim_onsets


def stage1_highpass(raw: mne.io.Raw) -> mne.io.Raw:
    """Stage 1: 0.5 Hz high-pass filter."""
    raw = raw.copy()
    raw.filter(
        l_freq=HP_FREQ,
        h_freq=None,
        method="fir",
        fir_window="hamming",
        l_trans_bandwidth=HP_TRANS,
        verbose=False,
    )
    return raw


def stage2_badch_asr(raw: mne.io.Raw) -> mne.io.Raw:
    """Stage 2: bad channel detection + ASR burst correction."""
    from scipy.signal import filtfilt, firwin

    raw = raw.copy()
    data = raw.get_data()
    n_ch = data.shape[0]

    # 2a. Flatline detection
    flatline_samps = int(FLATLINE_CRIT * raw.info["sfreq"])
    flat_bads = []
    for ch_idx, ch_name in enumerate(raw.ch_names):
        d = data[ch_idx]
        changes = np.diff(d) != 0
        run_lengths = np.diff(np.where(np.concatenate(([True], changes, [True])))[0])
        if run_lengths.max() >= flatline_samps:
            flat_bads.append(ch_name)

    # 2b. Channel correlation criterion
    win_samps = int(WIN_LEN_CORR * raw.info["sfreq"])
    step_samps = win_samps
    n_windows = max(1, (data.shape[1] - win_samps) // step_samps + 1)

    broken = np.zeros((n_ch, n_windows))
    for w in range(n_windows):
        s_idx = w * step_samps
        e_idx = min(s_idx + win_samps, data.shape[1])
        wnd = data[:, s_idx:e_idx].copy()
        wnd -= wnd.mean(axis=1, keepdims=True)
        C = np.corrcoef(wnd)
        for ch in range(n_ch):
            others = [j for j in range(n_ch) if j != ch]
            max_r = np.max(np.abs(C[ch, others]))
            broken[ch, w] = max_r < CORR_CRIT

    broken_frac = broken.mean(axis=1)
    corr_bads = [
        raw.ch_names[i] for i in range(n_ch) if broken_frac[i] > MAX_BROKEN_FRAC
    ]

    # 2c. Line noise criterion
    b = firwin(101, cutoff=45.0, fs=raw.info["sfreq"], pass_zero="lowpass")
    X_lp = filtfilt(b, [1.0], data, axis=1)
    X_hp = data - X_lp

    def mad(x, axis=1):
        med = np.median(x, axis=axis, keepdims=True)
        return np.median(np.abs(x - med), axis=axis)

    noisiness = mad(X_hp) / (mad(X_lp) + 1e-30)
    med_noise = np.median(noisiness)
    mad_noise = np.median(np.abs(noisiness - med_noise)) * 1.4826
    z_nsr = (
        (noisiness - med_noise) / mad_noise
        if mad_noise > 0
        else np.zeros_like(noisiness)
    )
    linenoise_bads = [raw.ch_names[i] for i in range(n_ch) if z_nsr[i] > LINENOISE_CRIT]

    all_bads = list(set(flat_bads + corr_bads + linenoise_bads))
    raw.info["bads"] = all_bads
    log.info(f"  Stage 2: bad channels = {all_bads}")

    # 2d. ASR
    good_names = [ch for ch in raw.ch_names if ch not in all_bads]
    if all_bads:
        r_good = raw.copy().pick(good_names)
    else:
        r_good = raw.copy()

    asr = asrpy.ASR(sfreq=r_good.info["sfreq"], cutoff=ASR_CUTOFF)
    asr.fit(r_good)
    r_good = asr.transform(r_good)

    if all_bads:
        r_bad = raw.copy().pick(all_bads)
        raw = r_good.add_channels([r_bad], force_update_info=True)
        raw.reorder_channels(
            [ch for ch in raw.ch_names if ch not in all_bads] + all_bads
        )
        raw.info["bads"] = all_bads
    else:
        raw = r_good

    return raw


def stage3_lowpass(raw: mne.io.Raw) -> mne.io.Raw:
    """Stage 3: 40 Hz low-pass filter."""
    raw = raw.copy()
    raw.filter(
        l_freq=None,
        h_freq=LP_FREQ,
        method="fir",
        fir_window="hamming",
        h_trans_bandwidth=LP_TRANS,
        verbose=False,
    )
    return raw


def stage4_ica(raw: mne.io.Raw) -> tuple:
    """Stage 4: ICA + ICLabel. Returns (cleaned_raw, ica_obj)."""
    ica = ICA(
        n_components=None,
        method="infomax",
        fit_params={"extended": True},
        random_state=RANDOM_STATE,
        max_iter=1000,
    )
    ica.fit(raw, verbose=False)

    ic_labs = label_components(raw, ica, method="iclabel")
    labels = ic_labs["labels"]
    y_pred_proba = ic_labs["y_pred_proba"]

    exclude_idx = [
        i
        for i, (label, prob) in enumerate(zip(labels, y_pred_proba))
        if label in EXCLUDE_CLASSES and prob >= ICA_THRESHOLD
    ]
    ica.exclude = exclude_idx
    log.info(f"  Stage 4: ICA {ica.n_components_} comp, excluded {exclude_idx}")
    raw_ica = ica.apply(raw.copy(), verbose=False)
    return raw_ica, ica


def stage5_interpolate(raw: mne.io.Raw) -> mne.io.Raw:
    """Stage 5: interpolate bad channels."""
    raw = raw.copy()
    raw.interpolate_bads(reset_bads=True, verbose=False)
    return raw


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION D – Epoching
# ═══════════════════════════════════════════════════════════════════════════════


def create_epochs(
    raw: mne.io.Raw,
    events_df: pd.DataFrame,
    orig_stim_onsets: np.ndarray,
) -> mne.Epochs:
    """Create epochs with condition metadata and moving-window rejection."""
    sfreq = raw.info["sfreq"]

    events, event_id = mne.events_from_annotations(raw, verbose=False)
    stim_event_id = {k: v for k, v in event_id.items() if "Segment" not in k}
    if not stim_event_id:
        raise ValueError("No stimulus events found")

    stim_codes = set(stim_event_id.values())
    mask = np.isin(events[:, 2], list(stim_codes))
    events_stim = events[mask]

    # StimTrak lag correction
    lag_samples = int(abs(STIMTRAK_LAG) * sfreq)
    events_shifted = events_stim.copy()
    events_shifted[:, 0] = np.maximum(events_stim[:, 0] - lag_samples, 0)

    ep = mne.Epochs(
        raw,
        events_shifted,
        event_id=stim_event_id,
        tmin=TMIN,
        tmax=TMAX,
        baseline=BASELINE,
        preload=True,
        verbose=False,
    )

    # Moving-window peak-peak amplitude rejection
    reject_tmin_idx = ep.time_as_index(REJECT_TMIN)[0]
    reject_tmax_idx = ep.time_as_index(REJECT_TMAX)[0]
    win_samp = int(WIN_SIZE * sfreq)
    step_samp = int(WIN_STEP * sfreq)
    data_check = ep.get_data()[:, :, reject_tmin_idx : reject_tmax_idx + 1]
    bad_epoch_mask = np.zeros(len(ep), dtype=bool)
    n_check = data_check.shape[2]
    for start in range(0, n_check - win_samp + 1, step_samp):
        window = data_check[:, :, start : start + win_samp]
        pp = window.max(axis=2) - window.min(axis=2)
        bad_epoch_mask |= (pp > REJECT_THRESH).any(axis=1)

    ep.drop(bad_epoch_mask, reason="peak-peak > 125 µV")
    n_stim = len(events_stim)
    n_kept = len(ep)
    log.info(f"  Epoching: {n_kept}/{n_stim} kept ({100 * n_kept / n_stim:.1f}%)")

    # Attach condition metadata
    trig_times = events_df["onset_samp"].values / ORIG_SFREQ
    kept_orig_idx = np.where(~bad_epoch_mask)[0]
    meta_rows = []
    for orig_idx in kept_orig_idx:
        if orig_stim_onsets is not None and orig_idx < len(orig_stim_onsets):
            evt_time = orig_stim_onsets[orig_idx]
        else:
            evt_time = events_stim[orig_idx, 0] / sfreq
        diffs = np.abs(trig_times - evt_time)
        best = np.argmin(diffs)
        if diffs[best] < 0.05:
            row = events_df.iloc[best]
            meta_rows.append(
                {
                    "orig_trial": int(orig_idx),
                    "tone": row["tone"],
                    "task": row["task"],
                    "env": row["env"],
                    "trial_type": row["trial_type"],
                    "condition": f"{row['task']}_{row['env']}",
                }
            )
        else:
            meta_rows.append(
                {
                    "orig_trial": int(orig_idx),
                    "tone": "unknown",
                    "task": "unknown",
                    "env": "unknown",
                    "trial_type": "unknown",
                    "condition": "unknown",
                }
            )

    ep.metadata = pd.DataFrame(meta_rows).reset_index(drop=True)
    n_matched = (ep.metadata["tone"] != "unknown").sum()
    log.info(f"  Metadata: {n_matched}/{n_kept} matched")
    return ep


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION E – Case 1: Line Noise Removal Benchmark
# ═══════════════════════════════════════════════════════════════════════════════


def run_case1(raw_hp: mne.io.Raw, sfreq: float) -> dict:
    """Run 4-method line noise benchmark. Returns results dict."""
    raw_eeg = raw_hp.copy().pick_types(eeg=True)
    data_orig = raw_eeg.get_data()
    results = {}
    base_period = int(sfreq / LINE_FREQ)

    # ── M1: Standard ZapLine ──────────────────────────────────────────────────
    log.info("  Case 1 M1: Standard ZapLine")
    try:
        zl_m1 = ZapLine(
            sfreq=sfreq, line_freq=LINE_FREQ, n_remove="auto", n_harmonics=3
        )
        raw_clean_m1 = zl_m1.fit_transform(raw_eeg)
        qa_m1 = compute_all_qa_metrics(
            data_orig,
            raw_clean_m1.get_data(),
            sfreq,
            LINE_FREQ,
            CASE1_N_HARMONICS,
        )
        results["M1"] = {
            "label": "Standard ZapLine",
            "n_removed": int(zl_m1.n_removed_),
            "qa_metrics": qa_m1,
        }
    except Exception as exc:
        log.warning(f"  M1 failed: {exc}")
        results["M1"] = {"label": "Standard ZapLine", "error": str(exc)}

    # ── M2: Adaptive ZapLine+ ─────────────────────────────────────────────────
    log.info("  Case 1 M2: Adaptive ZapLine+")
    try:
        zl_m2 = ZapLine(
            sfreq=sfreq,
            line_freq=LINE_FREQ,
            adaptive=True,
            adaptive_params={
                "n_remove_params": {
                    "sigma": 3.0,
                    "min_remove": 1,
                    "max_prop": 0.2,
                },
            },
        )
        raw_clean_m2 = zl_m2.fit_transform(raw_eeg)
        qa_m2 = compute_all_qa_metrics(
            data_orig,
            raw_clean_m2.get_data(),
            sfreq,
            LINE_FREQ,
            CASE1_N_HARMONICS,
        )
        n_rem_m2 = zl_m2.adaptive_results_.get("n_removed", 0)
        results["M2"] = {
            "label": "Adaptive ZapLine+",
            "n_removed": int(n_rem_m2) if np.isscalar(n_rem_m2) else n_rem_m2,
            "qa_metrics": qa_m2,
        }
    except Exception as exc:
        log.warning(f"  M2 failed: {exc}")
        results["M2"] = {"label": "Adaptive ZapLine+", "error": str(exc)}

    # ── M3 & M4: DSS + LineNoiseBias sweep ────────────────────────────────────
    bias_line = LineNoiseBias(freq=LINE_FREQ, sfreq=sfreq, method="fft")

    sweep_configs = {
        "M3a_combined_s3": {
            "smooth": None,
            "method": "combined",
            "threshold": 3.0,
            "n_comp": 10,
        },
        "M3b_maxgap": {
            "smooth": None,
            "method": "max_gap",
            "threshold": 1.2,
            "n_comp": 10,
        },
        "M3c_ratio_t1.5": {
            "smooth": None,
            "method": "ratio",
            "threshold": 1.5,
            "n_comp": 10,
        },
        "M3d_outlier_s2": {
            "smooth": None,
            "method": "outlier",
            "threshold": 2.0,
            "n_comp": 10,
        },
        "M4a_smooth1x": {
            "smooth": base_period,
            "method": "combined",
            "threshold": 3.0,
            "n_comp": 10,
        },
        "M4b_smooth2x": {
            "smooth": base_period * 2,
            "method": "combined",
            "threshold": 3.0,
            "n_comp": 10,
        },
        "M4c_smooth3x": {
            "smooth": base_period * 3,
            "method": "combined",
            "threshold": 3.0,
            "n_comp": 10,
        },
        "M4d_sm1x_maxgap": {
            "smooth": base_period,
            "method": "max_gap",
            "threshold": 1.2,
            "n_comp": 10,
        },
        "M4e_sm1x_ratio15": {
            "smooth": base_period,
            "method": "ratio",
            "threshold": 1.5,
            "n_comp": 10,
        },
        "M4f_sm1x_out2": {
            "smooth": base_period,
            "method": "outlier",
            "threshold": 2.0,
            "n_comp": 10,
        },
    }

    sweep_results = {}
    for tag, cfg in sweep_configs.items():
        try:
            dss = DSS(
                bias=bias_line,
                n_components=cfg["n_comp"],
                n_select="auto",
                selection_method=cfg["method"],
                selection_threshold=cfg["threshold"],
                smooth=cfg["smooth"],
                return_type="raw",
            )
            raw_clean = dss.fit_transform(raw_eeg)
            f_qa, geo_pre = geometric_mean_psd(data_orig, sfreq)
            _, geo_post = geometric_mean_psd(raw_clean.get_data(), sfreq)
            R_post = noise_surr_ratio(geo_post, f_qa, LINE_FREQ)
            below = below_noise_distortion(geo_pre, geo_post, f_qa, LINE_FREQ)
            sweep_results[tag] = {
                "R_post": R_post,
                "below_pct": below,
                "n_selected": int(dss.n_selected_),
                "eigenvalues": dss.eigenvalues_.tolist(),
                "raw_clean_data": raw_clean.get_data(),
            }
        except Exception as exc:
            log.warning(f"  {tag} failed: {exc}")

    # Pick best M3 and M4
    def pick_best(prefix):
        cands = {
            k: v
            for k, v in sweep_results.items()
            if k.startswith(prefix) and v["below_pct"] > -5.0
        }
        if not cands:
            cands = {k: v for k, v in sweep_results.items() if k.startswith(prefix)}
        if not cands:
            return None
        return min(cands, key=lambda k: cands[k]["R_post"])

    for prefix, label_base in [
        ("M3", "DSS+LineNoise (no smooth)"),
        ("M4", "DSS+LineNoise (smooth)"),
    ]:
        best_tag = pick_best(prefix)
        if best_tag:
            best = sweep_results[best_tag]
            qa = compute_all_qa_metrics(
                data_orig,
                best["raw_clean_data"],
                sfreq,
                LINE_FREQ,
                CASE1_N_HARMONICS,
            )
            results[prefix] = {
                "label": f"{label_base} [{best_tag}]",
                "n_removed": best["n_selected"],
                "qa_metrics": qa,
                "variant": best_tag,
                "eigenvalues": best["eigenvalues"],
            }
        else:
            results[prefix] = {"label": label_base, "error": "all variants failed"}

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION F – Case 2 / Part III: ERP Benchmark
# ═══════════════════════════════════════════════════════════════════════════════


def run_case2_benchmark(
    epochs_all: mne.Epochs,
    ica_obj: ICA,
) -> dict:
    """Run 4-pipeline ERP benchmark with train/test split. Returns results."""
    epochs_all.info["sfreq"]
    meta = epochs_all.metadata
    EVAL_WIN = ERP_WINDOWS["P300"]

    # ── Train / test split (odd / even) ───────────────────────────────────────
    n_ep = len(epochs_all)
    train_idx = np.arange(0, n_ep, 2)
    test_idx = np.arange(1, n_ep, 2)

    epochs_train = epochs_all[train_idx]
    epochs_test = epochs_all[test_idx]

    # Condition masks
    has_meta = meta is not None and "tone" in meta.columns
    if has_meta:
        cond_labels = meta["tone"].values
        task_labels = meta["task"].values
        test_tone = cond_labels[test_idx]
        test_task = task_labels[test_idx]
        test_dev_count_mask = (test_tone == "deviant") & (test_task == "count")
        test_std_count_mask = (test_tone == "standard") & (test_task == "count")
    else:
        test_dev_count_mask = np.zeros(len(test_idx), dtype=bool)
        test_std_count_mask = np.ones(len(test_idx), dtype=bool)

    pipe_epochs = {}
    pipe_evokeds = {}
    pipe_meta_out = {}

    # ── Pipeline 1: Baseline ──────────────────────────────────────────────────
    pipe_epochs["Baseline"] = epochs_test.copy()
    pipe_evokeds["Baseline"] = epochs_test.average()
    pipe_meta_out["Baseline"] = {"n_epochs": len(epochs_test), "components": "N/A"}

    # ── Pipeline 2: xDAWN ─────────────────────────────────────────────────────
    try:
        xdawn = Xdawn(n_components=XDAWN_N_COMPONENTS, reg="ledoit_wolf")
        xdawn.fit(epochs_train)
        xd_result = xdawn.apply(epochs_test.copy())
        if isinstance(xd_result, dict):
            merged = mne.concatenate_epochs(list(xd_result.values()))
            order = np.argsort(merged.events[:, 0])
            epochs_xdawn_test = mne.EpochsArray(
                merged.get_data()[order],
                merged.info,
                events=merged.events[order],
                tmin=merged.tmin,
                verbose=False,
            )
            if epochs_test.metadata is not None:
                epochs_xdawn_test.metadata = epochs_test.metadata.copy()
        else:
            epochs_xdawn_test = xd_result
        pipe_epochs["xDAWN"] = epochs_xdawn_test
        pipe_evokeds["xDAWN"] = epochs_xdawn_test.average()
        pipe_meta_out["xDAWN"] = {
            "n_epochs": len(epochs_xdawn_test),
            "components": XDAWN_N_COMPONENTS,
        }
    except Exception as exc:
        log.warning(f"  xDAWN failed: {exc}")
        pipe_epochs["xDAWN"] = epochs_test.copy()
        pipe_evokeds["xDAWN"] = epochs_test.average()
        pipe_meta_out["xDAWN"] = {"n_epochs": len(epochs_test), "components": "FAILED"}

    # ── Pipeline 3: ICA ───────────────────────────────────────────────────────
    epochs_ica_test = ica_obj.apply(epochs_test.copy(), verbose=False)
    pipe_epochs["ICA"] = epochs_ica_test
    pipe_evokeds["ICA"] = epochs_ica_test.average()
    pipe_meta_out["ICA"] = {
        "n_epochs": len(epochs_ica_test),
        "components": f"{len(ica_obj.exclude)} excluded of {ica_obj.n_components_}",
    }

    # ── Pipeline 4: DSS (AverageBias) ─────────────────────────────────────────
    bias_erp = AverageBias(axis="epochs")
    dss_erp = DSS(
        bias=bias_erp,
        n_components=DSS_N_COMPONENTS,
        return_type="sources",
    )
    dss_erp.fit(epochs_train)

    sources_test = dss_erp.transform(epochs_test)
    sources_kept = sources_test.copy()
    sources_kept[:, DSS_N_KEEP:, :] = 0
    data_dss_test = dss_erp.inverse_transform(sources_kept)
    if data_dss_test.ndim == 2:
        data_dss_test = data_dss_test[np.newaxis, ...]
    elif data_dss_test.ndim == 3 and data_dss_test.shape[0] == len(
        epochs_test.ch_names
    ):
        data_dss_test = np.transpose(data_dss_test, (2, 0, 1))

    epochs_dss_test = mne.EpochsArray(
        data_dss_test,
        epochs_test.info,
        tmin=epochs_test.tmin,
        verbose=False,
    )
    if epochs_test.metadata is not None:
        epochs_dss_test.metadata = epochs_test.metadata.copy()

    pipe_epochs["DSS"] = epochs_dss_test
    pipe_evokeds["DSS"] = epochs_dss_test.average()
    pipe_meta_out["DSS"] = {
        "n_epochs": len(epochs_dss_test),
        "components": f"{DSS_N_KEEP} kept of {DSS_N_COMPONENTS}",
        "eigenvalues": dss_erp.eigenvalues_.tolist(),
    }

    # ── Compute endpoint metrics ──────────────────────────────────────────────
    metrics = {}
    baseline_evoked_cz = None

    for name in PIPE_ORDER:
        ep = pipe_epochs[name]
        ev = pipe_evokeds[name]
        ch_names = ep.ch_names
        idx_cz = ch_names.index(PRIMARY_CH)
        idx_pz = ch_names.index(P300_CH)
        times = ep.times
        data = ep.get_data()
        dev_mask = test_dev_count_mask
        std_mask = test_std_count_mask

        # 1. Hedges' g
        t_mask = (times >= EVAL_WIN[0]) & (times <= EVAL_WIN[1])
        amp_dev = data[dev_mask, idx_pz, :][:, t_mask].mean(axis=1) * 1e6
        amp_std = data[std_mask, idx_pz, :][:, t_mask].mean(axis=1) * 1e6
        g = hedges_g(amp_dev, amp_std)

        # 2. Peak latency
        ev_data_pz = ev.data[idx_pz] * 1e6
        lat = peak_latency_ms(ev_data_pz, ev.times, EVAL_WIN, mode="pos")

        # 3. Morphology correlation
        ev_data_cz = ev.data[idx_cz] * 1e6
        if name == "Baseline":
            baseline_evoked_cz = ev_data_cz.copy()
            morph = 1.0
        else:
            morph = morphology_corr(ev_data_cz, baseline_evoked_cz, ev.times, EVAL_WIN)

        # 4. Split-half reliability
        shr = split_half_reliability(data[:, idx_pz, :] * 1e6)

        # 5. Single-trial AUC
        auc = single_trial_auc(
            data[:, idx_pz, :],
            times,
            EVAL_WIN,
            dev_mask,
            std_mask,
            random_state=RANDOM_STATE,
        )

        metrics[name] = {
            "hedges_g": _jsonable(g),
            "peak_lat_ms": _jsonable(lat),
            "morph_r": _jsonable(morph),
            "split_half_r": _jsonable(shr),
            "auc": _jsonable(auc),
            "n_epochs": int(len(ep)),
        }

    # ── Null-control permutation test (DSS + Baseline) ────────────────────────
    N_PERM = 500
    rng = np.random.default_rng(RANDOM_STATE)
    null_results = {}
    for name in ["DSS", "Baseline"]:
        null_g_list, null_auc_list = [], []
        ep = pipe_epochs[name]
        data = ep.get_data()
        idx_pz = ep.ch_names.index(P300_CH)
        times = ep.times
        t_mask = (times >= EVAL_WIN[0]) & (times <= EVAL_WIN[1])
        all_mask = test_dev_count_mask | test_std_count_mask
        n_dev = test_dev_count_mask.sum()

        for i_perm in range(N_PERM):
            shuffled = np.zeros(len(data), dtype=bool)
            dev_indices = rng.choice(np.where(all_mask)[0], size=n_dev, replace=False)
            shuffled[dev_indices] = True
            shuffled_std = all_mask & ~shuffled

            amp_dev_p = data[shuffled, idx_pz, :][:, t_mask].mean(axis=1) * 1e6
            amp_std_p = data[shuffled_std, idx_pz, :][:, t_mask].mean(axis=1) * 1e6
            null_g_list.append(hedges_g(amp_dev_p, amp_std_p))
            null_auc_list.append(
                single_trial_auc(
                    data[:, idx_pz, :],
                    times,
                    EVAL_WIN,
                    shuffled,
                    shuffled_std,
                    random_state=i_perm,
                )
            )

        null_g_arr = np.array(null_g_list)
        null_auc_arr = np.array(null_auc_list)
        real_g = metrics[name]["hedges_g"]
        real_auc = metrics[name]["auc"]

        p_g = float((np.abs(null_g_arr) >= abs(real_g)).mean())
        valid_auc = null_auc_arr[~np.isnan(null_auc_arr)]
        p_auc = (
            float((valid_auc >= real_auc).mean())
            if len(valid_auc) > 0 and not np.isnan(real_auc)
            else np.nan
        )
        null_results[name] = {
            "null_g_mean": _jsonable(np.nanmean(null_g_arr)),
            "null_g_ci95": [
                _jsonable(np.nanpercentile(null_g_arr, 2.5)),
                _jsonable(np.nanpercentile(null_g_arr, 97.5)),
            ],
            "p_g": _jsonable(p_g),
            "null_auc_mean": _jsonable(np.nanmean(valid_auc))
            if len(valid_auc)
            else None,
            "p_auc": _jsonable(p_auc),
        }

    return {
        "metrics": metrics,
        "null_control": null_results,
        "pipeline_meta": {
            k: {mk: mv for mk, mv in v.items() if mk != "eigenvalues"}
            for k, v in pipe_meta_out.items()
        },
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "dss_eigenvalues": pipe_meta_out.get("DSS", {}).get("eigenvalues"),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _jsonable(val):
    """Make a value JSON-serialisable."""
    if val is None:
        return None
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.floating | float):
        if np.isnan(val) or np.isinf(val):
            return None
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


def save_results(
    sub: str, case1: dict, case2: dict, preproc_info: dict, out_dir: Path
) -> None:
    """Save per-subject results as JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "subject": sub,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "preprocessing": preproc_info,
        "case1_line_noise": case1,
        "case2_erp_benchmark": case2,
    }
    out_path = out_dir / f"{sub}_results.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=_jsonable)
    log.info(f"  Results saved to {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION G – Main per-subject pipeline
# ═══════════════════════════════════════════════════════════════════════════════


def process_subject(sub: str) -> None:
    """Complete pipeline for one subject."""
    t0 = time.time()
    log.info(f"{'=' * 60}")
    log.info(f"  PROCESSING {sub}")
    log.info(f"{'=' * 60}")

    # ── Data acquisition ──────────────────────────────────────────────────────
    # Data must be pre-downloaded via download_openneuro.py.
    # On Narval, narval_job.sh copies data to $SLURM_TMPDIR and overrides
    # DATA_DIR before running this script, so DATASET_ROOT already points
    # to the fast local SSD copy.
    data_dir = DATASET_ROOT
    check_subject_data(sub, data_dir)

    # ── Load ──────────────────────────────────────────────────────────────────
    raw = load_raw(sub, data_dir)
    edf = load_events_df(sub, data_dir)
    log.info(
        f"  Loaded: {raw.info['nchan']} ch, "
        f"{raw.n_times / raw.info['sfreq']:.1f} s, "
        f"sfreq={raw.info['sfreq']} Hz"
    )

    # ── Preprocessing ─────────────────────────────────────────────────────────
    log.info("  Stage 0: resample, trim, re-ref")
    r0, orig_onsets = stage0_resample_trim_reref(raw)

    log.info("  Stage 1: high-pass filter")
    r1 = stage1_highpass(r0)

    log.info("  Stage 2: bad channels + ASR")
    r2 = stage2_badch_asr(r1)
    bad_chs = list(r2.info["bads"])

    log.info("  Stage 3: low-pass filter")
    r3 = stage3_lowpass(r2)

    log.info("  Stage 4: ICA + ICLabel")
    r4, ica_obj = stage4_ica(r3)
    ica_exclude = list(ica_obj.exclude)

    log.info("  Stage 5: interpolate bad channels")
    r5 = stage5_interpolate(r4)

    preproc_info = {
        "n_channels": len(r5.ch_names),
        "duration_s": float(r5.times[-1]),
        "sfreq": float(r5.info["sfreq"]),
        "bad_channels": bad_chs,
        "ica_excluded": ica_exclude,
        "ica_n_components": int(ica_obj.n_components_),
    }

    # ── Epoching ──────────────────────────────────────────────────────────────
    log.info("  Epoching")
    epochs = create_epochs(r5, edf, orig_onsets)
    preproc_info["n_epochs"] = len(epochs)

    # ── Case 1: Line Noise ────────────────────────────────────────────────────
    log.info("  Running Case 1: Line Noise Benchmark")
    case1_results = run_case1(r1, r1.info["sfreq"])

    # ── Case 2 / Part III: ERP Benchmark ──────────────────────────────────────
    log.info("  Running Case 2 / Part III: ERP Benchmark")
    case2_results = run_case2_benchmark(epochs, ica_obj)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = OUTPUT_DIR / sub
    save_results(sub, case1_results, case2_results, preproc_info, out_dir)

    elapsed = time.time() - t0
    log.info(f"  {sub} DONE in {elapsed / 60:.1f} min")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    """CLI entry point."""
    import os

    parser = argparse.ArgumentParser(
        description="Batch pipeline for ds003620 — Case 1 & Case 2"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--subject", type=str, help="Single subject ID (e.g. sub-01)")
    group.add_argument("--all", action="store_true", help="Process all 44 subjects")
    group.add_argument(
        "--slurm-array",
        action="store_true",
        help="Use $SLURM_ARRAY_TASK_ID (1-44) to pick subject",
    )
    args = parser.parse_args()

    if args.slurm_array:
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "1"))
        sub = f"sub-{task_id:02d}"
        log.info(f"Slurm array mode: TASK_ID={task_id} → {sub}")
        process_subject(sub)

    elif args.subject:
        process_subject(args.subject)

    elif args.all:
        for sub in ALL_SUBJECTS:
            try:
                process_subject(sub)
            except Exception:
                log.error(f"FAILED: {sub}")
                traceback.print_exc()
                continue


if __name__ == "__main__":
    main()
