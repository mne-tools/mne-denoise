#!/usr/bin/env python
"""Real cardiac-artifact benchmark runner (arm: cardiac_ds007554; generalization).

ds007554 records scalp EEG (.edf, 250 Hz) with a synchronized ECG physiological recording
(BIDS *_physio.tsv.gz, StartTime 0, aligned to the EEG). We resample the ECG onto the EEG
timeline, add it as a channel, and detect R-peaks with mne.preprocessing.find_ecg_events. The
native method is a cardiac DSS that builds a CycleAverageBias from the R-peak-locked cycles and
SUBTRACTS the recovered cardiac subspace (cleaned = EEG - DSS-cardiac-estimate); the comparator
is ECG-reference regression. Primary endpoint: QRS-locked residual amplitude
(coupling.event_locked_residual); preservation: neural-band power. Reported as generalization.

    python scripts/run_cardiac_arm.py --config configs/benchmarks/cardiac_ds007554.yaml --subject sub-001
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import pathlib
import sys

import numpy as np

_REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
import yaml  # noqa: E402

import mne_denoise.benchmarks.io as bio  # noqa: E402
from mne_denoise.benchmarks.preprocessing import apply_baseline  # noqa: E402
from mne_denoise.qa.coupling import event_locked_residual, regress_out  # noqa: E402

BENCH = "cardiac"


def _load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _bandpow(x, sfreq, lo, hi):
    from scipy.signal import welch
    f, p = welch(x, sfreq, nperseg=min(x.shape[-1], int(sfreq * 4)), axis=-1)
    m = (f >= lo) & (f <= hi)
    return float(np.mean(p[..., m]))


def _load_ds007554(root, subject, cfg):
    """EEG (.edf) + synchronized ECG physio added as a channel. Returns (raw_eeg_plus_ecg)."""
    import mne

    task = (cfg.get("dataset", {}) or {}).get("contaminated_task", "mentalarithmetic")
    edf = sorted(glob.glob(f"{root}/{subject}/**/*task-{task}_eeg.edf", recursive=True))
    if not edf:
        edf = sorted(glob.glob(f"{root}/{subject}/**/*_eeg.edf", recursive=True))
    if not edf:
        raise FileNotFoundError(f"no EEG .edf for {subject} under {root}")
    raw = mne.io.read_raw_edf(edf[0], preload=True, verbose=False)
    raw.pick("eeg")
    sf = float(raw.info["sfreq"])
    stem = pathlib.Path(edf[0]).name.replace("_eeg.edf", "")
    pe = sorted(glob.glob(f"{root}/{subject}/**/{stem}_recording-ECG_physio.tsv.gz", recursive=True)) or \
        sorted(glob.glob(f"{root}/{subject}/**/*{task}*ECG_physio.tsv.gz", recursive=True))
    if not pe:
        raise FileNotFoundError(f"no ECG physio for {subject} ({stem})")
    ecg = np.loadtxt(gzip.open(pe[0]))
    pj = pe[0].replace(".tsv.gz", ".json")
    esf = float(json.load(open(pj)).get("SamplingFrequency", sf)) if os.path.exists(pj) else sf
    n = raw.n_times
    ecg_rs = np.interp(np.arange(n) / sf, np.arange(len(ecg)) / esf, ecg)
    ecg_rs = (ecg_rs - ecg_rs.mean()) / (ecg_rs.std() or 1.0) * 1e-4   # z-score into a volt-ish scale
    er = mne.io.RawArray(ecg_rs[None, :], mne.create_info(["ECG"], sf, "ecg"), verbose=False)
    er.set_meas_date(raw.info["meas_date"])
    raw.add_channels([er], force_update_info=True)
    raw, _ = apply_baseline(raw, cfg.get("baseline_preprocessing"))
    return raw


def run_subject(cfg, subject, root, deriv_root, *, synthetic=False):
    import mne

    from mne_denoise.dss import DSS
    from mne_denoise.dss.denoisers import CycleAverageBias

    raw = _load_ds007554(root, subject, cfg)
    sf = float(raw.info["sfreq"])
    ecg_events, _, _ = mne.preprocessing.find_ecg_events(raw, ch_name="ECG", verbose=False)
    rp = (ecg_events[:, 0] - raw.first_samp).astype(int)
    rp = rp[(rp > 0) & (rp < raw.n_times)]
    eeg = raw.copy().pick("eeg")
    X = eeg.get_data()
    ecg = raw.copy().pick("ECG").get_data()
    tmin, tmax = -0.05, 0.10                         # QRS-locked window
    win = (int(-0.10 * sf), int(0.20 * sf))
    rows = []

    def _resid(data):
        return event_locked_residual(data, rp, sf, tmin=tmin, tmax=tmax)

    methods = ["none", "ecg_regression", "cardiac_dss"]
    for mid in methods:
        try:
            if mid == "none":
                cleaned = X
            elif mid == "ecg_regression":
                cleaned = regress_out(X, ecg)
            else:  # cardiac_dss: subtract the recovered cardiac subspace
                dss = DSS(bias=CycleAverageBias(event_samples=rp, window=win, sfreq=sf),
                          n_components=3, return_type="array")
                est = np.asarray(dss.fit_transform(eeg))
                est = est if est.shape == X.shape else est.reshape(X.shape)
                cleaned = X - est
        except Exception as exc:  # noqa: BLE001
            rows.append({"method": mid, "status": "failed_numerical", "error": f"{type(exc).__name__}: {exc}"})
            continue
        row = {"status": "success",
               "qrs_residual": float(_resid(cleaned)),
               "neural_band_power": _bandpow(cleaned, sf, 4, 30),     # preservation (theta-beta)
               "n_rpeaks": int(len(rp))}
        rows.append({"method": mid, **row})
        out = pathlib.Path(deriv_root) / subject / BENCH / mid
        bio.save_subject_benchmark_results(out, subject=subject, method=mid, metrics=row)
    return rows


def main(argv=None):
    import mne_denoise.benchmarks.datasets as D

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--subject")
    p.add_argument("--slurm-array", action="store_true")
    p.add_argument("--deriv-root")
    a = p.parse_args(argv)
    cfg = _load_cfg(a.config)
    arm = cfg.get("arm", "cardiac")
    deriv = pathlib.Path(a.deriv_root or (_REPO / "results" / arm))
    try:
        root = D.resolve_dataset_root(cfg["dataset"]["id"])
    except Exception:  # noqa: BLE001
        root = pathlib.Path(os.environ.get("DATASETS_ROOT", "/project/rrg-kjerbi/datasets")) / \
            (cfg.get("dataset", {}) or {}).get("project_relative_path", "")
    subs = (cfg.get("dataset", {}) or {}).get("subjects") or \
        sorted(p.name for p in pathlib.Path(root).glob("sub-*") if p.is_dir())
    subject = a.subject or (subs[int(os.environ.get("SLURM_ARRAY_TASK_ID", 1)) - 1] if a.slurm_array else None)
    if not subject:
        p.error("need --subject or --slurm-array")
    for r in run_subject(cfg, subject, pathlib.Path(root), deriv):
        if r.get("status") == "success":
            print(f"  {r['method']:16} QRS-resid={r['qrs_residual']:.3e}  neural={r['neural_band_power']:.3e}  rpeaks={r['n_rpeaks']}")
        else:
            print(f"  {r['method']:16} {r.get('status')} {str(r.get('error',''))[:50]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
