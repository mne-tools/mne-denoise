#!/usr/bin/env python
"""Low-density / wearable robustness arm: sweep the channel count and measure how each
method degrades.

The wearable-EEG systematic review (Arpaia et al. 2025) reports that source-separation
methods (ICA / PCA / DSS) lose effectiveness as the channel count falls below ~8-32, while
ASR holds. This arm makes that empirical. Two modes:

* ``base_arm: ground_truth`` -- vary the simulation's ``n_channels`` (known sources at every
  density), giving an RRMSE degradation curve per method. The cleanest demonstration.
* ``base_arm: ocular`` / ``muscle`` -- subsample an existing arm's real recording to each
  channel count and re-score (uses ``benchmarks.subsample``). [real-data mode]

    python scripts/run_lowdensity_arm.py --config configs/benchmarks/lowdensity_ground_truth.yaml
"""
from __future__ import annotations

import argparse
import os
import pathlib
import statistics as st
import sys

import yaml

_REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))

from mne_denoise.benchmarks import io as bio  # noqa: E402

BENCH = "lowdensity"


def _load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def run_ground_truth(cfg, deriv_root, *, synthetic=False):
    """Sweep the simulation channel count; reuse the ground-truth replicate scorer."""
    import run_ground_truth_arm as G

    s = cfg.get("simulation", {}) or {}
    grid = cfg.get("channel_grid") or [32, 16, 8, 6, 4]
    regime = "forward" if "forward" in f"{s.get('regime', '')} {cfg.get('arm', '')}" else "generic"
    snr = float(s.get("snr_db", 0.0))
    nrep = 2 if synthetic else int(s.get("n_replicates", 10))
    n_brain = int(s.get("n_brain", 4))
    n_artifact = int(s.get("n_artifact", 4))
    methods = (list(cfg.get("methods_under_test", []) or [])
               + list((cfg.get("comparators", {}) or {}).get("required", []) or []))
    alias = {"iterativedss": "iterative_dss", "iterative_dss_tanh": "iterative_dss"}
    methods = list(dict.fromkeys(alias.get(m, m) for m in methods)) or \
        ["pca", "fastica", "iterative_dss", "oracle"]
    rows = []
    for n_ch in grid:
        for seed in range(nrep):
            rep = G.sim.simulate_replicate(regime=regime, n_channels=int(n_ch), n_brain=n_brain,
                                           n_artifact=n_artifact, n_train=2000, n_test=2000,
                                           snr_db=snr, seed=seed)
            for r in G.run_replicate(rep, methods):
                r["n_ch"] = int(n_ch)
                r["replicate"] = f"nch{n_ch}_seed{seed}"
                rows.append(r)
                if not synthetic:
                    out = pathlib.Path(deriv_root) / f"nch{n_ch}_seed{seed}" / BENCH / r["method"]
                    bio.save_subject_benchmark_results(out, subject=f"nch{n_ch}_seed{seed}",
                                                       method=r["method"], metrics=r)
    return rows, methods


def _print_curve(arm, base, methods, rows, metric="rrmse"):
    grid = sorted({r["n_ch"] for r in rows})
    print(f"arm={arm} base={base}  metric={metric}  channel_grid={grid}")
    for mth in methods:
        cells = []
        for n_ch in grid:
            vals = [r[metric] for r in rows
                    if r.get("method") == mth and r.get("n_ch") == n_ch
                    and r.get("status") == "success" and metric in r]
            cells.append(f"{n_ch}ch:{round(st.median(vals), 3) if vals else 'NA'}")
        print(f"  {mth:16} " + "  ".join(cells))


def _gate_heavy(cfg, n_ch, heavy_max=16, heavy=("eemd_cca", "emd", "eemd", "memd")):
    """Drop compute-prohibitive per-channel methods (EEMD-CCA, EMD) above ``heavy_max`` channels.
    EEMD-per-channel on full recordings does not scale to high density (the 64-ch case did not
    finish in 45 min), and these single-channel-origin methods are relevant at the low-density
    (wearable) end anyway."""
    if int(n_ch) <= heavy_max:
        return cfg
    import copy
    c = copy.deepcopy(cfg)
    c["tiers"] = {k: [m for m in (v or []) if m not in heavy]
                  for k, v in (cfg.get("tiers", {}) or {}).items()}
    c["methods_under_test"] = [m for m in (cfg.get("methods_under_test", []) or []) if m not in heavy]
    comp = c.get("comparators") or {}
    if comp.get("required"):
        comp["required"] = [m for m in comp["required"] if m not in heavy]
    return c


def run_real_data(cfg, deriv_root, *, synthetic=False):
    """Subsample a real arm's EEG channels to each density in ``channel_grid`` and re-score
    with that arm's methods + metrics (reuses run_{muscle,ocular}_arm.run_subject via the
    ``preloaded`` hook). Tests whether each method's primary endpoint degrades as channels fall."""
    import importlib
    import os

    from mne_denoise.benchmarks import subsample as ss
    from mne_denoise.benchmarks.preprocessing import apply_baseline

    base = cfg.get("base_arm")
    grid = cfg.get("channel_grid") or [64, 32, 16, 8, 4]
    ds = cfg.get("dataset", {}) or {}
    root = str(pathlib.Path(os.environ.get("DATASETS_ROOT", "/project/rrg-kjerbi/datasets"))
               / ds.get("project_relative_path", ""))
    subjects = ([os.environ["LD_SUBJECT"]] if os.environ.get("LD_SUBJECT")
                else list(ds.get("subjects") or ([] if not synthetic else ["synthetic"])))
    mod = importlib.import_module("run_muscle_arm" if base == "muscle" else "run_ocular_arm")
    _t = cfg.get("tiers", {}) or {}
    methods = list(dict.fromkeys(
        list(cfg.get("methods_under_test", []) or [])
        + list((cfg.get("comparators", {}) or {}).get("required", []) or [])
        + [m for tier in _t.values() for m in (tier or [])]))
    rows = []
    for subject in subjects:
        if base == "muscle":
            if synthetic:
                raw, fit_refs, eval_refs = mod._synth_muscle()
            else:
                raw, fit_refs, eval_refs = mod._load_ds004505(root, subject, cfg)
                raw, _ = apply_baseline(raw, cfg.get("baseline_preprocessing"))
            eeg_idx = [raw.ch_names.index(c) for c in raw.copy().pick("eeg").ch_names]
            for n_ch in grid:
                if n_ch > len(eeg_idx):
                    continue
                keep = ss.farthest_point_channels(raw.info, eeg_idx, int(n_ch))
                sub = raw.copy().pick([raw.ch_names[i] for i in keep] + list(fit_refs) + list(eval_refs))
                for r in mod.run_subject(_gate_heavy(cfg, n_ch), f"{subject}_nch{n_ch}", root, deriv_root,
                                         preloaded=(sub, fit_refs, eval_refs)):
                    r["n_ch"] = int(n_ch); r["base_subject"] = subject; rows.append(r)
        else:  # ocular
            if synthetic:
                epo, eog, win, picks = mod._synth_ocular()
            else:
                epo, eog, win, picks = mod._load_erp_core_ocular(root, subject, cfg)
            roi = [epo.ch_names[i] for i in (picks or [])]
            eeg_idx = [epo.ch_names.index(c) for c in epo.copy().pick("eeg").ch_names]
            for n_ch in grid:
                if n_ch > len(eeg_idx):
                    continue
                must = [epo.ch_names.index(c) for c in roi if c in epo.ch_names]
                keep = ss.farthest_point_channels(epo.info, eeg_idx, int(n_ch), must_include=must)
                sub = epo.copy().pick([epo.ch_names[i] for i in keep] + list(eog))
                sub_picks = [sub.ch_names.index(c) for c in roi if c in sub.ch_names] or None
                for r in mod.run_subject(_gate_heavy(cfg, n_ch), f"{subject}_nch{n_ch}", root, deriv_root,
                                         preloaded=(sub, eog, win, sub_picks)):
                    r["n_ch"] = int(n_ch); r["base_subject"] = subject; rows.append(r)
    return rows, methods


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--deriv-root")
    p.add_argument("--synthetic", action="store_true")
    a = p.parse_args(argv)
    cfg = _load_cfg(a.config)
    arm = cfg.get("arm", "lowdensity")
    deriv = pathlib.Path(a.deriv_root or (_REPO / "results" / arm))
    base = cfg.get("base_arm", "ground_truth")
    if base == "ground_truth":
        rows, methods = run_ground_truth(cfg, deriv, synthetic=a.synthetic)
    elif base in ("muscle", "ocular"):
        rows, methods = run_real_data(cfg, deriv, synthetic=a.synthetic)
    else:
        raise NotImplementedError(f"low-density mode for base_arm={base!r} not supported")
    metric = cfg.get("curve_metric") or {"muscle": "emg_coupling", "ocular": "blink_coupling"}.get(base, "rrmse")
    _print_curve(arm, base, methods, rows, metric=metric)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
