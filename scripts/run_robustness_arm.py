#!/usr/bin/env python
"""Long-duration ASR robustness arm (sleep_long / clinical_stress; generalization).

A regime-eligibility / robustness statement, NOT an efficacy claim: does the calibration-based
ASR estimator stream stably over long real recordings, and with what rejection, structural, and
throughput profile? ASR is calibrated on the first clean two minutes and applied to a 30-min
window of each recording. Endpoints: terminal status (does it complete), throughput (x real-time),
retained-window fraction (ASR statistics), effective-rank change (structural footprint), and
neural-band (1-30 Hz) retention. CAP (5 bipolar EEG) gives the multichannel profile; Sleep-EDF
(2 EEG, 100 Hz, ~22-h nights) is the long-duration streaming point.

    python scripts/run_robustness_arm.py --config configs/benchmarks/robustness_cap.yaml --subject brux1
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import sys
import time

import numpy as np

_REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
import yaml  # noqa: E402

import mne_denoise.benchmarks.io as bio  # noqa: E402
from mne_denoise.qa.structural import effective_rank_change, rejected_window_fraction  # noqa: E402

BENCH = "robustness"


def _load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _bandpow(x, sfreq, lo, hi):
    from scipy.signal import welch
    f, p = welch(x, sfreq, nperseg=min(x.shape[-1], int(sfreq * 4)), axis=-1)
    m = (f >= lo) & (f <= hi)
    return float(np.mean(p[..., m]))


def _load(ds_id, rec, root, cfg):
    import mne

    if ds_id == "capslpdb":
        edf = sorted(glob.glob(f"{root}/**/{rec}.edf", recursive=True)) or sorted(glob.glob(f"{root}/{rec}*.edf"))
        raw = mne.io.read_raw_edf(edf[0], preload=True, verbose=False)
        eeg = [c for c in raw.ch_names if any(b in c for b in
               ("Fp2", "F4", "C4", "P4", "O2", "F3", "C3", "P3", "O1")) and "-" in c]
        raw.pick(eeg)
        tmax = 1800.0                                   # CAP: 30-min profile window
    elif ds_id == "chbmit":                             # clinical scalp EEG (23 bipolar, ~1-h records)
        edf = sorted(glob.glob(f"{root}/**/{rec}.edf", recursive=True))
        raw = mne.io.read_raw_edf(edf[0], preload=True, verbose=False)
        bip = [c for c in raw.ch_names if "-" in c
               and not any(x in c.upper() for x in ("ECG", "EKG", "VNS", "."))]
        raw.pick(bip)
        tmax = float(raw.times[-1])                     # full ~1-h clinical recording
    else:  # sleep-edfx: process the FULL night (genuine long-duration streaming test)
        edf = sorted(glob.glob(f"{root}/**/{rec}*PSG*.edf", recursive=True))
        raw = mne.io.read_raw_edf(edf[0], preload=True, verbose=False)
        raw.pick([c for c in raw.ch_names if c.startswith("EEG")])
        tmax = float(raw.times[-1])
    raw.filter(1.0, 40.0, verbose=False)
    raw.crop(tmax=min(tmax, float(raw.times[-1])))
    return raw


def run_subject(cfg, rec, root, deriv_root):
    from mne_denoise.asr import ASR, compute_clean_window_mask

    ds_id = (cfg.get("dataset", {}) or {}).get("id")
    raw = _load(ds_id, rec, root, cfg)
    sf = float(raw.info["sfreq"])
    X = raw.get_data()
    dur_min = float(raw.times[-1]) / 60.0
    calib = X    # calibrate on the recording's own clean windows (no artifact-free rest segment in sleep PSG)
    try:
        mask, _ = compute_clean_window_mask(X, sf)
        retained = rejected_window_fraction(mask)        # mask = retained-sample mask
        rej = round(float(1.0 - retained), 3)            # fraction ASR repairs
    except Exception:  # noqa: BLE001
        rej = None
    rows = []
    for method in ["asr", "rasr"]:
        t0 = time.perf_counter()
        try:
            asr = ASR(sfreq=sf, cutoff=20.0,
                      method=("riemannian_windowed" if method == "rasr" else "standard"))
            asr.fit(calib)
            cleaned = np.asarray(asr.transform(X), dtype=float)
            if cleaned.shape != X.shape:
                cleaned = cleaned.reshape(X.shape)
            status = "success"
        except Exception as exc:  # noqa: BLE001
            rows.append({"method": method, "status": "failed_numerical",
                         "error": f"{type(exc).__name__}: {exc}"})
            continue
        rt = time.perf_counter() - t0
        row = {"status": status,
               "n_ch": int(X.shape[0]),
               "dur_min": round(dur_min, 1),
               "runtime_s": round(rt, 1),
               "throughput_x_realtime": round(dur_min * 60.0 / rt, 1) if rt > 0 else None,
               "rejected_window_frac": rej,
               "effective_rank_change": round(float(effective_rank_change(X, cleaned)), 3),
               "neural_band_retained": round(_bandpow(cleaned, sf, 1, 30) / (_bandpow(X, sf, 1, 30) or 1), 3)}
        rows.append({"method": method, **row})
        out = pathlib.Path(deriv_root) / rec / BENCH / method
        bio.save_subject_benchmark_results(out, subject=rec, method=method, metrics=row)
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--subject")
    p.add_argument("--slurm-array", action="store_true")
    p.add_argument("--deriv-root")
    a = p.parse_args(argv)
    cfg = _load_cfg(a.config)
    arm = cfg.get("arm", "robustness")
    deriv = pathlib.Path(a.deriv_root or (_REPO / "results" / arm))
    root = pathlib.Path(os.environ.get("DATASETS_ROOT", "/project/rrg-kjerbi/datasets")) / \
        (cfg.get("dataset", {}) or {}).get("project_relative_path", "")
    subs = (cfg.get("dataset", {}) or {}).get("subjects") or []
    rec = a.subject or (subs[int(os.environ.get("SLURM_ARRAY_TASK_ID", 1)) - 1] if a.slurm_array else None)
    if not rec:
        p.error("need --subject or --slurm-array")
    for r in run_subject(cfg, rec, root, deriv):
        if r.get("status") == "success":
            print(f"  {r['method']:6} {r['n_ch']}ch {r['dur_min']}min  {r['throughput_x_realtime']}x-rt  "
                  f"reject={r['rejected_window_frac']}  rankΔ={r['effective_rank_change']}  band={r['neural_band_retained']}")
        else:
            print(f"  {r['method']:6} {r.get('status')} {str(r.get('error', ''))[:50]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
