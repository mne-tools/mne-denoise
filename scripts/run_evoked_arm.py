#!/usr/bin/env python
"""Generic evoked-enhancement runner (arms: evoked_erp_core, evoked_ds000117).

Learns each method's spatial model on TRAIN trials and applies it to HELD-OUT trials
of two conditions (A vs B; e.g. face vs car / face vs scrambled), then scores the
held-out A-B effect at the component window/ROI (e.g. N170/M170) plus preservation
(amplitude, SME, split-half). Per-subject results go through ``benchmarks.io``;
the group A-B effect size (Hedges g across subjects) is computed at aggregation.

    python scripts/run_evoked_arm.py --config configs/benchmarks/evoked_erp_core.yaml --subject sub-001
    python scripts/run_evoked_arm.py --config ... --synthetic     # local smoke (no data)
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

import numpy as np
import yaml

_REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

import mne_denoise.benchmarks.adapters  # noqa: F401  (registers adapters)
from mne_denoise.benchmarks import comparators
from mne_denoise.benchmarks import datasets as D
from mne_denoise.benchmarks import io as bio
from mne_denoise.benchmarks import sweep as _sweep
from mne_denoise.qa import preservation as qp

BENCH = "evoked"


def _load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _synth_two_conditions(sfreq=256.0, nch=20, ntr=160, seed=0):
    """Two conditions sharing a posterior pattern; condition A has a larger N170."""
    import mne

    rng = np.random.default_rng(seed)
    nt = int(0.6 * sfreq)
    t = np.arange(nt) / sfreq - 0.1
    n170 = -np.exp(-((t - 0.155) ** 2) / (2 * 0.018 ** 2))  # negative deflection ~155 ms
    patt = rng.standard_normal(nch); patt /= np.linalg.norm(patt)
    info = mne.create_info([f"E{i}" for i in range(nch)], sfreq, "eeg")

    def make(amp):
        data = np.stack([
            patt[:, None] * (amp * n170)[None, :] * 4e-6 + rng.standard_normal((nch, nt)) * 3e-6
            for _ in range(ntr)
        ])
        ev = np.column_stack([np.arange(ntr) * nt, np.zeros(ntr, int), np.ones(ntr, int)])
        return mne.EpochsArray(data, info, ev, tmin=-0.1, verbose=False)

    return make(1.0), make(0.55)  # A larger N170 than B


def _trial_amps(epochs, tmin, tmax, picks=None):
    """Per-trial mean amplitude in [tmin, tmax] averaged over ROI picks."""
    x = epochs.get_data(copy=False)  # (n_trials, n_ch, n_times)
    times = epochs.times
    m = (times >= tmin) & (times <= tmax)
    if picks is None:
        picks = list(range(x.shape[1]))
    return x[:, picks, :][:, :, m].mean(axis=(1, 2))


def _roi_picks(epochs, cfg):
    roi = (cfg.get("metrics", {}) or {}).get("roi_channels")
    if roi:
        return [epochs.ch_names.index(c) for c in roi if c in epochs.ch_names] or None
    return None  # all channels (synthetic / default)


def run_subject(cfg, subject, root, deriv_root, *, synthetic=False):
    bp = cfg.get("baseline_preprocessing", {}) or {}
    steps = bp.get("steps", {}) or {}
    win = steps.get("n170_window_ms") or [110, 150]
    tmin, tmax = win[0] / 1000.0, win[1] / 1000.0
    if synthetic:
        epo_a, epo_b = _synth_two_conditions()
    else:
        raise NotImplementedError(
            "real evoked loading is wired per dataset via the config events; "
            "use --synthetic for the local smoke. (ERP-CORE/ds000117 loaders land with the pilot.)"
        )
    ctx = {"sfreq": float(epo_a.info["sfreq"])}
    picks = _roi_picks(epo_a, cfg)
    na, nb = len(epo_a), len(epo_b)
    tr_a, ev_a = epo_a[: na // 2], epo_a[na // 2:]
    tr_b, ev_b = epo_b[: nb // 2], epo_b[nb // 2:]
    train = mne_concat(tr_a, tr_b)  # filters learned on both conditions' train trials
    rows = []
    methods = list(cfg.get("methods_under_test", []) or []) + list(
        (cfg.get("comparators", {}) or {}).get("required", []) or []
    )
    methods = list(dict.fromkeys(m for m in methods if m != "event_destroyed_null"))
    for mid in methods:
        for suffix, mparams in _sweep.method_runs(cfg, mid):
            tag = f"{mid}__{suffix}" if suffix else mid
            try:
                cmp = comparators.get(mid, **mparams)
            except KeyError:
                rows.append({"method": mid, "tag": tag, "status": "unavailable_dependency"}); continue
            try:
                state = cmp.fit(train, ctx)
                ra = cmp.transform(ev_a, state, ctx)
                rb = cmp.transform(ev_b, state, ctx)
            except Exception as exc:  # noqa: BLE001
                rows.append({"method": mid, "tag": tag, "status": "failed_numerical",
                             "error": f"{type(exc).__name__}: {exc}"})
                continue
            if ra.status != "success" or rb.status != "success":
                rows.append({"method": mid, "tag": tag, "status": ra.status if ra.status != "success" else rb.status})
                continue
            amps_a = _trial_amps(ra.cleaned, tmin, tmax, picks)
            amps_b = _trial_amps(rb.cleaned, tmin, tmax, picks)
            diff = float(np.mean(amps_a) - np.mean(amps_b))
            sme_a = qp.analytic_sme(amps_a)
            try:
                sh = qp.split_half_reliability(ra.cleaned.get_data(copy=False)[:, picks or slice(None), :].mean(1))
            except Exception:  # noqa: BLE001
                sh = None
            row = {
                "status": "success",
                "sweep_value": (suffix.split("-", 1)[1] if suffix else None),
                "n170_amp_A": float(np.mean(amps_a)),
                "n170_amp_B": float(np.mean(amps_b)),
                "n170_diff": diff,                     # per-subject face-car contrast
                "n170_sme_A": float(sme_a),
                "split_half": (float(sh) if sh is not None else None),
                "runtime_s": ra.runtime_seconds,
            }
            rows.append({"method": mid, "tag": tag, **row})
            out_dir = pathlib.Path(deriv_root) / subject / BENCH / tag
            bio.save_subject_benchmark_results(out_dir, subject=subject, method=tag, metrics=row,
                                               model_info={"window_ms": win})
    return rows


def mne_concat(a, b):
    import mne

    return mne.concatenate_epochs([a, b], verbose=False)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--subject")
    p.add_argument("--slurm-array", action="store_true")
    p.add_argument("--all", action="store_true")
    p.add_argument("--group-only", action="store_true")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--deriv-root", default=None)
    a = p.parse_args(argv)
    cfg = _load_cfg(a.config)
    arm = cfg.get("arm", "evoked")
    deriv_root = pathlib.Path(a.deriv_root or (_REPO / "results" / arm))
    if a.group_only:
        g = bio.aggregate_benchmark_results(deriv_root)
        print(f"[group] {arm}: aggregated -> {deriv_root}")
        return 0
    if a.synthetic:
        rows = run_subject(cfg, "sub-synth", None, deriv_root, synthetic=True)
        for r in rows:
            if r.get("status") == "success":
                print(f"  {r['tag']:22} diff={r['n170_diff']:+.3e}  SME_A={r['n170_sme_A']:.3e}  split_half={r['split_half']}")
            else:
                print(f"  {r['tag']:22} {r.get('status')}")
        return 0
    root = D.resolve_dataset_root(cfg["dataset"]["id"])
    subject = a.subject or (f"sub-{int(os.environ.get('SLURM_ARRAY_TASK_ID', 1)):02d}" if a.slurm_array else None)
    if subject:
        run_subject(cfg, subject, root, deriv_root)
        return 0
    p.error("one of --subject/--slurm-array/--group-only/--synthetic required")


if __name__ == "__main__":
    raise SystemExit(main())
