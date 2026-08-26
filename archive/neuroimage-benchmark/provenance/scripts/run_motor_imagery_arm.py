#!/usr/bin/env python
"""Motor-imagery external-preservation arm (eegmmidb; generalization).

Tests whether the package's denoisers PRESERVE the discriminative sensorimotor rhythm rather
than enhance an evoked response: left- vs right-fist motor imagery (eegmmidb runs R04/R08/R12,
annotations T1/T2) is decoded with CSP + LDA under 5-fold CV, on raw data and after ASR / DSS
denoising. A denoiser that preserves motor imagery leaves the cross-validated accuracy intact;
one that over-cleans the mu/beta ERD degrades it. Endpoint: CV accuracy (none vs asr vs dss).

    python scripts/run_motor_imagery_arm.py --config configs/benchmarks/mi_eegmmidb.yaml --subject S001
"""
from __future__ import annotations

import argparse
import glob
import os
import pathlib
import sys

import numpy as np

_REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
import yaml  # noqa: E402

import mne_denoise.benchmarks.io as bio  # noqa: E402

BENCH = "motor_imagery"
RUNS = (4, 8, 12)        # imagine opening/closing LEFT (T1) or RIGHT (T2) fist


def _load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _load_mi(root, subject):
    import mne

    raws = []
    for r in RUNS:
        f = sorted(glob.glob(f"{root}/{subject}/{subject}R{r:02d}.edf"))
        if not f:
            continue
        try:
            raws.append(mne.io.read_raw_edf(f[0], preload=True, verbose=False))
        except Exception:  # noqa: BLE001 - skip a truncated/corrupt download, use the valid runs
            continue
    if not raws:
        raise FileNotFoundError(f"no readable MI runs for {subject} under {root}")
    raw = mne.concatenate_raws(raws, verbose=False)
    raw.rename_channels({c: c.strip(".").upper() for c in raw.ch_names})
    raw.set_channel_types({c: "eeg" for c in raw.ch_names})
    return raw


def _epochs(raw):
    import mne

    r = raw.copy().filter(8.0, 30.0, verbose=False)          # mu + beta sensorimotor band
    events, eid = mne.events_from_annotations(r, verbose=False)
    keep = {k: v for k, v in eid.items() if k in ("T1", "T2")}
    epo = mne.Epochs(r, events, keep, tmin=0.5, tmax=3.5, baseline=None,
                     preload=True, reject=None, verbose=False)
    return epo


def _cv_acc(X, y):
    from mne.decoding import CSP
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline

    clf = Pipeline([("csp", CSP(n_components=6, reg="ledoit_wolf", log=True, norm_trace=False)),
                    ("lda", LinearDiscriminantAnalysis())])
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    return float(cross_val_score(clf, X, y, cv=cv).mean())


def run_subject(cfg, subject, root, deriv_root):
    raw = _load_mi(root, subject)
    sf = float(raw.info["sfreq"])
    ctx = {"sfreq": sf, "bandpass_band": [8.0, 30.0], "freq_band": [8.0, 30.0]}
    rows = []
    for method in ["none", "asr", "rasr"]:
        try:
            if method == "none":
                den = raw
            else:  # asr / rasr: the transient-artifact cleaners whose MI preservation we test
                from mne_denoise.asr import ASR
                X = raw.get_data()
                asr = ASR(sfreq=sf, cutoff=20.0,
                          method=("riemannian_windowed" if method == "rasr" else "standard"))
                asr.fit(X)
                den = raw.copy()
                den._data = np.asarray(asr.transform(X), dtype=float)
            epo = _epochs(den)
            y = epo.events[:, 2]
            if len(set(y)) < 2 or len(y) < 12:
                rows.append({"method": method, "status": "too_few_trials", "n_trials": int(len(y))}); continue
            acc = _cv_acc(epo.get_data(), y)
            row = {"status": "success", "cv_accuracy": round(acc, 4), "n_trials": int(len(y)),
                   "n_ch": int(len(epo.ch_names))}
            rows.append({"method": method, **row})
            out = pathlib.Path(deriv_root) / subject / BENCH / method
            bio.save_subject_benchmark_results(out, subject=subject, method=method, metrics=row)
        except Exception as exc:  # noqa: BLE001
            rows.append({"method": method, "status": "failed_numerical",
                         "error": f"{type(exc).__name__}: {exc}"})
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--subject")
    p.add_argument("--slurm-array", action="store_true")
    p.add_argument("--deriv-root")
    a = p.parse_args(argv)
    cfg = _load_cfg(a.config)
    arm = cfg.get("arm", "motor_imagery")
    deriv = pathlib.Path(a.deriv_root or (_REPO / "results" / arm))
    root = pathlib.Path(os.environ.get("DATASETS_ROOT", "/project/rrg-kjerbi/datasets")) / \
        (cfg.get("dataset", {}) or {}).get("project_relative_path", "")
    subs = (cfg.get("dataset", {}) or {}).get("subjects") or []
    subject = a.subject or (subs[int(os.environ.get("SLURM_ARRAY_TASK_ID", 1)) - 1] if a.slurm_array else None)
    if not subject:
        p.error("need --subject or --slurm-array")
    for r in run_subject(cfg, subject, root, deriv):
        if r.get("status") == "success":
            print(f"  {r['method']:5} acc={r['cv_accuracy']:.3f}  trials={r['n_trials']}")
        else:
            print(f"  {r['method']:5} {r.get('status')} {str(r.get('error', ''))[:50]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
