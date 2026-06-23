#!/usr/bin/env python
"""Reference-coupled muscle/movement cleaning runner (arm: muscle_ds004505; Claim 4).

Reference-free (ASR/rASR/ICA) and reference-aware (regression/iCanClean/dual-layer)
methods are reported in SEPARATE tiers. Anti-circularity is enforced: reference-aware
methods FIT on the noise-layer references, but EMG contamination is SCORED on the
INDEPENDENT neck-EMG. Primary target = independent neck-EMG contamination
(HF power + coupling); preservation = neural-band (alpha/beta) power retention.

    python scripts/run_muscle_arm.py --config configs/benchmarks/muscle_ds004505.yaml --synthetic
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import yaml

_REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

import mne_denoise.benchmarks.adapters  # noqa: F401
from mne_denoise.benchmarks import comparators
from mne_denoise.benchmarks import datasets as D
from mne_denoise.benchmarks import io as bio
from mne_denoise.benchmarks import sweep as _sweep
from mne_denoise.benchmarks.preprocessing import apply_baseline
from mne_denoise.qa.coupling import hf_band_power, reference_coupling

BENCH = "muscle"
ALIAS = {"rasr_windowed": "rasr", "reference_regression": "regression",
         "dual_layer_regression": "regression", "ica_fixed": "ica_rank_matched",
         "asr_euclidean": "asr"}


def _load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _bandpow(data, sfreq, lo, hi):
    from scipy.integrate import trapezoid
    from scipy.signal import welch

    f, p = welch(data, sfreq, nperseg=min(1024, data.shape[-1]))
    m = (f >= lo) & (f <= hi)
    return float(np.mean(trapezoid(p[:, m], f[m], axis=1)))


def _synth_muscle(sfreq=250.0, n_eeg=16, n_noise=12, n_emg=4, secs=120, seed=0):
    """EEG = alpha+beta brain rhythms + broadband muscle (HF, spatially structured);
    noise-layer + neck-EMG references both capture the muscle. Cleaning should cut
    HF + coupling while preserving alpha/beta."""
    import mne

    rng = np.random.default_rng(seed)
    N = int(sfreq * secs)
    t = np.arange(N) / sfreq
    alpha = np.sin(2 * np.pi * 10 * t) * 1e-6
    beta = np.sin(2 * np.pi * 20 * t) * 0.6e-6
    a_pat = rng.standard_normal(n_eeg)[:, None]
    b_pat = rng.standard_normal(n_eeg)[:, None]
    muscle = rng.standard_normal(N)  # broadband HF source (whiten-ish)
    from scipy.signal import butter, sosfiltfilt

    sos = butter(4, [20, 100], btype="band", fs=sfreq, output="sos")
    muscle = sosfiltfilt(sos, muscle)
    m_pat = np.abs(rng.standard_normal(n_eeg))[:, None]
    eeg = a_pat * alpha[None, :] + b_pat * beta[None, :] + m_pat * muscle[None, :] * 6e-6
    eeg += rng.standard_normal((n_eeg, N)) * 3e-7
    noise = np.abs(rng.standard_normal(n_noise))[:, None] * muscle[None, :] * 6e-6 + rng.standard_normal((n_noise, N)) * 2e-7
    emg = np.abs(rng.standard_normal(n_emg))[:, None] * muscle[None, :] * 8e-6 + rng.standard_normal((n_emg, N)) * 2e-7
    names = [f"E{i}" for i in range(n_eeg)] + [f"N{i}" for i in range(n_noise)] + [f"M{i}" for i in range(n_emg)]
    types = ["eeg"] * n_eeg + ["misc"] * n_noise + ["emg"] * n_emg
    info = mne.create_info(names, sfreq, types)
    raw = mne.io.RawArray(np.vstack([eeg, noise, emg]), info, verbose=False)
    return raw, [f"N{i}" for i in range(n_noise)], [f"M{i}" for i in range(n_emg)]


def _methods(cfg):
    t = cfg.get("tiers", {}) or {}
    out = []
    for tier_name in ("reference_free", "reference_aware"):
        for mid in t.get(tier_name, []) or []:
            out.append((tier_name, mid))
    return out


def _load_ds004505(root, subject, cfg):
    """Load ds004505 table-tennis dual-layer EEG. EEG = scalp electrodes; noise_layer =
    paired 'N-' electrodes (fit reference); neck_emg = EMG channels (independent eval
    reference). Returns (raw, noise_layer_names, neck_emg_names)."""
    import csv as _csv
    import glob

    import mne

    cand = glob.glob(f"{root}/**/{subject}_task-TableTennis_eeg.set", recursive=True)
    if not cand:
        cand = glob.glob(f"{pathlib.Path(root).parent}/**/{subject}_task-TableTennis_eeg.set", recursive=True)
    if not cand:
        raise FileNotFoundError(f"ds004505 .set not found for {subject} under {root}")
    setf = pathlib.Path(cand[0])
    raw = mne.io.read_raw_eeglab(setf, preload=True, verbose=False)
    chf = setf.with_name(setf.name.replace("_eeg.set", "_channels.tsv"))
    bids_type = {}
    with open(chf) as f:
        for r in _csv.DictReader(f, delimiter="\t"):
            bids_type[r["name"]] = r["type"].upper()
    raw.set_channel_types({c: {"EEG": "eeg", "EMG": "emg", "MISC": "misc"}.get(bids_type.get(c, "MISC"), "misc")
                           for c in raw.ch_names})
    noise_layer = [c for c in raw.ch_names if c.startswith("N-")]
    neck_emg = [c for c in raw.ch_names if bids_type.get(c) == "EMG"]
    return raw, noise_layer, neck_emg


def _load_ds004784(root, subject, cfg):
    """Load ds004784 conductive-phantom EEG (task-All = injected brain + all artifacts).
    128 BioSemi scalp electrodes (A1-D32) are the EEG; EXG1-8 are external electrodes that
    pick up the artifacts (the reference-aware fit + scoring reference). The known clean
    brain (task-Brain) is a separate recording used only for the preservation discussion.
    Returns (raw, exg_names, exg_names)."""
    import glob

    import mne

    task = (cfg.get("dataset") or {}).get("contaminated_task", "All")
    cand = glob.glob(f"{root}/**/{subject}_task-{task}_eeg.set", recursive=True)
    if not cand:
        cand = glob.glob(f"{pathlib.Path(root).parent}/**/{subject}_task-{task}_eeg.set", recursive=True)
    if not cand:
        raise FileNotFoundError(f"ds004784 task-{task} .set not found for {subject} under {root}")
    raw = mne.io.read_raw_eeglab(pathlib.Path(cand[0]), preload=True, verbose=False)
    exg = [c for c in raw.ch_names if c.upper().startswith("EXG")]
    scalp = [c for c in raw.ch_names if len(c) >= 2 and c[0] in "ABCD" and c[1:].isdigit()]
    raw.set_channel_types({c: ("eeg" if c in scalp else "eog" if c in exg else "misc")
                           for c in raw.ch_names})
    return raw, exg, exg


def run_subject(cfg, subject, root, deriv_root, *, synthetic=False):
    if synthetic:
        raw, fit_refs, eval_refs = _synth_muscle()
    elif (cfg.get("dataset") or {}).get("id") == "phantom_ds004784":
        raw, fit_refs, eval_refs = _load_ds004784(root, subject, cfg)
    else:
        raw, fit_refs, eval_refs = _load_ds004505(root, subject, cfg)
    raw, _ = apply_baseline(raw, cfg.get("baseline_preprocessing"))
    sfreq = float(raw.info["sfreq"])
    emg_eval = raw.copy().pick(eval_refs).get_data()  # independent neck-EMG (scoring ref)
    tmax = float(raw.times[-1])
    sel = raw.copy().crop(tmax=tmax * 0.4)
    ev = raw.copy().crop(tmin=tmax * 0.4)
    emg_eval = emg_eval[:, emg_eval.shape[1] - ev.n_times:]
    ctx = {"sfreq": sfreq, "ref_channels": fit_refs}  # reference-aware FIT on noise-layer
    # pre-clean EEG contamination baseline
    eeg0 = ev.copy().pick("eeg").get_data()
    rows = []
    for tier, mid in _methods(cfg):
        rid = ALIAS.get(mid, mid)
        for suffix, mparams in _sweep.method_runs(cfg, rid):
            tag = f"{mid}__{suffix}" if suffix else mid
            try:
                cmp = comparators.get(rid, **mparams)
            except KeyError:
                rows.append({"method": mid, "tag": tag, "tier": tier, "status": "unavailable_dependency"}); continue
            try:
                state = cmp.fit(sel, ctx)
                res = cmp.transform(ev, state, ctx)
            except Exception as exc:  # noqa: BLE001
                rows.append({"method": mid, "tag": tag, "tier": tier, "status": "failed_numerical",
                             "error": f"{type(exc).__name__}: {exc}"}); continue
            if res.status != "success":
                rows.append({"method": mid, "tag": tag, "tier": tier, "status": res.status}); continue
            eeg = res.cleaned.copy().pick("eeg").get_data()
            row = {
                "status": "success", "tier": tier,
                "sweep_value": (suffix.split("-", 1)[1] if suffix else None),
                "hf_power": hf_band_power(eeg, sfreq, fmin=20.0, fmax=100.0),          # muscle contamination (lower better)
                "emg_coupling": reference_coupling(eeg, emg_eval),                      # independent neck-EMG (lower better)
                "alpha_power": _bandpow(eeg, sfreq, 8, 12),                             # preservation
                "beta_power": _bandpow(eeg, sfreq, 16, 24),                             # preservation
                "runtime_s": res.runtime_seconds,
            }
            rows.append({"method": mid, "tag": tag, **row})
            out_dir = pathlib.Path(deriv_root) / subject / BENCH / tag
            bio.save_subject_benchmark_results(out_dir, subject=subject, method=tag, metrics=row)
    # attach pre-clean reference values for context
    rows.insert(0, {"method": "_raw", "tier": "ref",
                    "hf_power": hf_band_power(eeg0, sfreq, fmin=20.0, fmax=100.0),
                    "emg_coupling": reference_coupling(eeg0, emg_eval),
                    "alpha_power": _bandpow(eeg0, sfreq, 8, 12),
                    "beta_power": _bandpow(eeg0, sfreq, 16, 24), "status": "reference"})
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--subject")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--group-only", action="store_true")
    p.add_argument("--deriv-root", default=None)
    a = p.parse_args(argv)
    cfg = _load_cfg(a.config)
    arm = cfg.get("arm", "muscle")
    deriv_root = pathlib.Path(a.deriv_root or (_REPO / "results" / arm))
    if a.group_only:
        bio.aggregate_benchmark_results(deriv_root)
        print(f"[group] {arm} -> {deriv_root}")
        return 0
    if a.synthetic:
        rows = run_subject(cfg, "sub-synth", None, deriv_root, synthetic=True)
        print(f"  {'method':22}{'tier':16}{'hf_power':>11}{'emg_coup':>9}{'alpha':>11}{'beta':>11}")
        for r in rows:
            if r.get("status") in ("success", "reference"):
                print(f"  {r['method']:22}{r.get('tier',''):16}{r['hf_power']:>11.2e}{r['emg_coupling']:>9.3f}{r['alpha_power']:>11.2e}{r['beta_power']:>11.2e}")
            else:
                print(f"  {r['method']:22}{r.get('tier',''):16}{r.get('status')}")
        return 0
    try:
        root = D.resolve_dataset_root(cfg["dataset"]["id"])
    except Exception:  # noqa: BLE001 - unregistered Tier-B id: fall back to the config path
        root = pathlib.Path(os.environ.get("DATASETS_ROOT", "/project/rrg-kjerbi/datasets")) / \
            (cfg.get("dataset", {}) or {}).get("project_relative_path", "")
    if a.subject:
        run_subject(cfg, a.subject, root, deriv_root)
        return 0
    p.error("one of --subject/--synthetic/--group-only required")


if __name__ == "__main__":
    raise SystemExit(main())
