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


def _load_cap(root, subject, cfg):
    """CAP sleep PSG: EDF with a clean ECG channel (ECG1-ECG2) synchronized to bipolar EEG
    (Fp2-F4, F4-C4, ...). No cross-modal sync needed---ECG is a channel."""
    import glob

    import mne

    edf = sorted(glob.glob(f"{root}/**/{subject}.edf", recursive=True)) or \
        sorted(glob.glob(f"{root}/{subject}*.edf"))
    if not edf:
        raise FileNotFoundError(f"no CAP edf for {subject} under {root}")
    raw = mne.io.read_raw_edf(edf[0], preload=True, verbose=False)
    ecg = [c for c in raw.ch_names if "ECG" in c.upper() or "EKG" in c.upper()]
    eeg = [c for c in raw.ch_names if c not in ecg and any(b in c for b in
           ("Fp2", "F4", "C4", "P4", "O2", "F3", "C3", "P3", "O1", "A1", "A2")) and "-" in c]
    raw.rename_channels({ecg[0]: "ECG"})
    raw.set_channel_types({c: ("ecg" if c == "ECG" else "eeg") for c in eeg + ["ECG"]})
    raw.pick(eeg + ["ECG"])
    raw.crop(tmax=min(600.0, float(raw.times[-1])))   # 10 min is plenty of beats
    raw, _ = apply_baseline(raw, cfg.get("baseline_preprocessing"))
    return raw


def run_subject(cfg, subject, root, deriv_root, *, synthetic=False):
    import mne

    from mne_denoise.dss import DSS
    from mne_denoise.dss.denoisers import CycleAverageBias

    ds_id = (cfg.get("dataset", {}) or {}).get("id")
    raw = _load_cap(root, subject, cfg) if ds_id == "capslpdb" else _load_ds007554(root, subject, cfg)
    sf = float(raw.info["sfreq"])
    ecg_events, _, _ = mne.preprocessing.find_ecg_events(raw, ch_name="ECG", verbose=False)
    rp = (ecg_events[:, 0] - raw.first_samp).astype(int)
    rp = rp[(rp > int(0.15 * sf)) & (rp < raw.n_times - int(0.25 * sf))]   # keep edges clear of both windows
    eeg = raw.copy().pick("eeg")
    X = eeg.get_data()
    ecg = raw.copy().pick("ECG").get_data()
    tmin, tmax = -0.05, 0.10                         # QRS-locked window
    a, b = int(tmin * sf), int(tmax * sf)
    win = (int(-0.10 * sf), int(0.20 * sf))
    rows = []

    def _resid(data):                                # RMS of the QRS-locked AVERAGE (cardiac evoked, not broadband)
        seg = np.stack([data[:, p + a:p + b] for p in rp])
        return float(np.sqrt((seg.mean(0) ** 2).mean()))

    # config-driven: bespoke cardiac methods first, then any registered comparators named
    # in the config (e.g. wavelet_threshold) dispatched through the comparator registry
    cfg_methods = (list(cfg.get("methods_under_test", []) or [])
                   + list((cfg.get("comparators", {}) or {}).get("required", []) or []))
    methods = list(dict.fromkeys(["none", "ecg_regression", "cardiac_dss", "ssp_ecg", "ica_ecg"] + cfg_methods))
    ctx = {"sfreq": sf}
    for mid in methods:
        try:
            if mid == "none":
                cleaned = X
            elif mid == "ecg_regression":          # time-shifted ECG regression (captures the lagged QRS shape)
                L = int(0.12 * sf)
                R = np.vstack([np.roll(ecg[0], k) for k in range(-L, L + 1, max(1, L // 8))])
                cleaned = regress_out(X, R)
            elif mid == "cardiac_dss":             # subtract the recovered cardiac subspace
                dss = DSS(bias=CycleAverageBias(event_samples=rp, window=win, sfreq=sf),
                          n_components=3, return_type="array")
                est = np.asarray(dss.fit_transform(X), dtype=float)
                est = est if est.shape == X.shape else est.reshape(X.shape)
                cleaned = X - est
            elif mid == "ssp_ecg":                 # SSP-ECG: project out the QRS-locked spatial subspace
                seg = np.stack([X[:, p + win[0]:p + win[1]] for p in rp
                                if p + win[0] >= 0 and p + win[1] < X.shape[1]])
                U, _, _ = np.linalg.svd(seg.mean(0), full_matrices=False)
                P = U[:, :1]                       # top cardiac spatial projector (1 ECG SSP, the convention)
                cleaned = X - P @ (P.T @ X)
            elif mid == "ica_ecg":                 # ICA-ECG: drop components correlated with the ECG channel
                from sklearn.decomposition import FastICA
                ica = FastICA(n_components=min(20, X.shape[0] - 1), max_iter=400,
                              random_state=0, whiten="unit-variance")
                S = ica.fit_transform(X.T)         # (n_times, n_comp)
                cors = np.array([abs(np.corrcoef(S[:, i], ecg[0])[0, 1]) for i in range(S.shape[1])])
                bad = cors > 0.15
                if not bad.any() and cors.size and cors.max() > 0.1:
                    bad[int(cors.argmax())] = True  # else drop the single most ECG-correlated IC
                S[:, bad] = 0.0
                cleaned = ica.inverse_transform(S).T
            else:                                  # registered comparator (ssa, wavelet_threshold, ...)
                import mne_denoise.benchmarks.adapters  # noqa: F401  (populate registry)
                from mne_denoise.benchmarks import comparators

                cmp = comparators.get(mid)
                res = cmp.transform(eeg, cmp.fit(eeg, ctx), ctx)
                if res.status != "success":
                    rows.append({"method": mid, "status": res.status, "error": getattr(res, "error", None)})
                    continue
                cleaned = (res.cleaned.get_data() if hasattr(res.cleaned, "get_data")
                           else np.asarray(res.cleaned))
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
