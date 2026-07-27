#!/usr/bin/env python
"""Ocular-artifact runner (arm: ocular_erp_core).

Learns each method on TRAIN trials, applies to HELD-OUT trials, and scores blink
residual (coupling of cleaned EEG to the EOG channels) against N170 preservation
(mean amplitude at the posterior ROI/window). Reference-blind ICA/SSP and the
EOG-coupled EOG-DSS/regression are all held to the same data.

    python scripts/run_ocular_arm.py --config configs/benchmarks/ocular_erp_core.yaml --synthetic
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

import mne_denoise.benchmarks.adapters  # noqa: F401
from mne_denoise.benchmarks import comparators
from mne_denoise.benchmarks import datasets as D
from mne_denoise.benchmarks import io as bio
from mne_denoise.qa import preservation as qp
from mne_denoise.qa.coupling import reference_coupling

BENCH = "ocular"
ALIAS = {"eog_regression": "regression"}


def _load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _synth_ocular(sfreq=256.0, nch=20, ntr=200, seed=0):
    """N170 evoked in every trial (posterior) + blinks (frontal, EOG-coupled) in half."""
    import mne

    rng = np.random.default_rng(seed)
    nt = int(0.6 * sfreq)
    t = np.arange(nt) / sfreq - 0.1
    n170 = -np.exp(-((t - 0.155) ** 2) / (2 * 0.018 ** 2))
    blink = np.exp(-((t - 0.2) ** 2) / (2 * 0.05 ** 2))                 # slow frontal deflection
    post = np.abs(rng.standard_normal(nch)); post[: nch // 2] *= 0.2    # posterior-weighted
    post /= np.linalg.norm(post)
    front = np.zeros(nch); front[: nch // 4] = 1.0; front /= np.linalg.norm(front)
    eeg = np.zeros((ntr, nch, nt))
    veog = np.zeros((ntr, 1, nt))
    for k in range(ntr):
        x = post[:, None] * n170[None, :] * 3e-6 + rng.standard_normal((nch, nt)) * 2e-6
        if k % 2 == 0:                                                  # half the trials blink
            amp = 1.0 + 0.5 * rng.random()
            x += front[:, None] * blink[None, :] * amp * 2e-5
            veog[k, 0] = blink * amp * 2e-5 + rng.standard_normal(nt) * 1e-7
        else:
            veog[k, 0] = rng.standard_normal(nt) * 1e-7
        eeg[k] = x
    names = [f"E{i}" for i in range(nch)] + ["VEOG"]
    info = mne.create_info(names, sfreq, ["eeg"] * nch + ["eog"])
    data = np.concatenate([eeg, veog], axis=1)
    ev = np.column_stack([np.arange(ntr) * nt, np.zeros(ntr, int), np.ones(ntr, int)])
    epo = mne.EpochsArray(data, info, ev, tmin=-0.1, verbose=False)
    return epo, ["VEOG"], (0.13, 0.18), list(range(nch // 2, nch))      # epochs, eog, n170 window, posterior picks


def _methods(cfg):
    out = list(cfg.get("methods_under_test", []) or [])
    out += list((cfg.get("comparators", {}) or {}).get("required", []) or [])
    out = list(dict.fromkeys(out))
    return _restrict(out)


def _restrict(configured):
    """Narrow execution to a subset of the already-configured methods.

    ``mara`` is declared required in ocular_erp_core.yaml but has never run, because
    it lives in a GPL-isolated module behind MNE_DENOISE_ENABLE_GPL_MARA. Running it
    alone must not require editing the frozen config, which would change its
    config_hash and orphan the 360 attempts already hashed against it. This is
    execution-only scoping, so the subset is validated against the config rather
    than trusted: a name that is not already configured is an error, never a
    silently added method.
    """
    raw = (os.environ.get("ARM_ONLY_METHODS") or "").strip()
    if not raw:
        return configured
    want = [tok for tok in raw.replace(",", " ").split() if tok]
    unknown = [tok for tok in want if tok not in configured]
    if unknown:
        raise ValueError(
            f"ARM_ONLY_METHODS={raw!r} is not a subset of the configured methods "
            f"{configured}: {unknown}"
        )
    return [m for m in configured if m in want]


def _load_erp_core_ocular(root, subject, cfg):
    """Load ERP-CORE N170 for the ocular arm: all stimulus epochs (faces+cars, with their
    natural blinks) and the EOG channels retained (the ocular methods need them). Returns
    (epochs, eog_names, (tmin, tmax) N170 window, posterior picks)."""
    import csv as _csv
    import glob

    import mne

    from mne_denoise.benchmarks.preprocessing import apply_baseline

    cand = glob.glob(f"{root}/**/{subject}_ses-N170_task-N170_eeg.set", recursive=True)
    if not cand:
        cand = glob.glob(f"{pathlib.Path(root).parent}/**/{subject}_ses-N170_task-N170_eeg.set", recursive=True)
    if not cand:
        raise FileNotFoundError(f"N170 .set not found for {subject} under {root}")
    setf = pathlib.Path(cand[0])
    raw = mne.io.read_raw_eeglab(setf, preload=True, verbose=False)
    eog = [c for c in raw.ch_names if c.startswith(("HEOG", "VEOG"))]
    if eog:
        raw.set_channel_types({c: "eog" for c in eog})
    onsets = []
    with open(setf.with_name(setf.name.replace("_eeg.set", "_events.tsv"))) as f:
        for r in _csv.DictReader(f, delimiter="\t"):
            if r.get("trial_type") != "stimulus":
                continue
            try:
                v = int(float(r["value"])); on = float(r["onset"])
            except (ValueError, KeyError, TypeError):
                continue
            if 1 <= v <= 80:  # faces (1-40) vs cars (41-80): keep the contrast for preservation
                onsets.append((on, 1 if v <= 40 else 2))
    raw, _ = apply_baseline(raw, cfg.get("baseline_preprocessing"))
    sf = float(raw.info["sfreq"])
    ev = np.array([[int(round(o * sf)), 0, c] for o, c in onsets], dtype=int)
    epo = mne.Epochs(raw, ev, {"face": 1, "car": 2}, tmin=-0.2, tmax=0.8, baseline=(-0.2, 0.0),
                     preload=True, verbose=False)
    roi = (cfg.get("metrics", {}) or {}).get("roi_channels") or ["PO7", "PO8"]
    picks = [epo.ch_names.index(c) for c in roi if c in epo.ch_names] or None
    steps = (cfg.get("baseline_preprocessing", {}) or {}).get("steps", {}) or {}
    win = steps.get("n170_window_ms") or [110, 150]
    return epo, eog, (win[0] / 1000.0, win[1] / 1000.0), picks


def run_subject(cfg, subject, root, deriv_root, *, synthetic=False, preloaded=None):
    if preloaded is not None:
        epo, eog, (wlo, whi), picks = preloaded  # already baselined + channel-subsampled (low-density arm)
    elif synthetic:
        epo, eog, (wlo, whi), picks = _synth_ocular()
    elif cfg.get("dataset", {}).get("id") == "erp_core_n170":
        epo, eog, (wlo, whi), picks = _load_erp_core_ocular(root, subject, cfg)
    else:
        raise NotImplementedError(f"no ocular loader for dataset {cfg.get('dataset', {}).get('id')!r}")
    sfreq = float(epo.info["sfreq"])
    n = len(epo)
    tr, ev = epo[: n // 2], epo[n // 2:]
    ctx = {"sfreq": sfreq, "eog_channels": eog, "ref_channels": eog}
    eog_eval = ev.copy().pick(eog).get_data().transpose(1, 0, 2).reshape(len(eog), -1)
    rows = []
    for mid in _methods(cfg):
        rid = ALIAS.get(mid, mid)
        try:
            cmp = comparators.get(rid)
        except KeyError:
            rows.append({"method": mid, "status": "unavailable_dependency"}); continue
        try:
            state = cmp.fit(tr, ctx)
            res = cmp.transform(ev, state, ctx)
        except Exception as exc:  # noqa: BLE001
            rows.append({"method": mid, "status": "failed_numerical", "error": f"{type(exc).__name__}: {exc}"}); continue
        if res.status != "success":
            rows.append({"method": mid, "status": res.status}); continue
        eeg = res.cleaned.copy().pick("eeg")
        x = eeg.get_data()
        coup = reference_coupling(x.transpose(1, 0, 2).reshape(x.shape[1], -1), eog_eval)
        evoked = x.mean(0)                                              # (n_ch, n_times) held-out ERP
        n170 = qp.erp_mean_amplitude(evoked, epo.times, wlo, whi, picks=picks)
        try:                                                            # face-car N170 contrast: preserved face-selectivity
            af = eeg["face"].average().data
            ac = eeg["car"].average().data
            n170_diff = float(qp.erp_mean_amplitude(af, epo.times, wlo, whi, picks=picks)
                              - qp.erp_mean_amplitude(ac, epo.times, wlo, whi, picks=picks))
        except Exception:  # noqa: BLE001
            n170_diff = float("nan")
        row = {"status": "success", "blink_coupling": float(coup), "n170_amp": float(n170),
               "n170_diff": n170_diff, "runtime_s": res.runtime_seconds}
        rows.append({"method": mid, **row})
        out_dir = pathlib.Path(deriv_root) / subject / BENCH / mid
        bio.save_subject_benchmark_results(out_dir, subject=subject, method=mid, metrics=row)
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
    arm = cfg.get("arm", "ocular")
    deriv_root = pathlib.Path(a.deriv_root or (_REPO / "results" / arm))
    if a.group_only:
        bio.aggregate_benchmark_results(deriv_root)
        return 0
    if a.synthetic:
        rows = run_subject(cfg, "sub-synth", None, deriv_root, synthetic=True)
        print(f"  {'method':24}{'blink_coupling':>16}{'n170_amp':>12}")
        for r in rows:
            if r.get("status") == "success":
                print(f"  {r['method']:24}{r['blink_coupling']:>16.3f}{r['n170_amp']:>12.3e}")
            else:
                print(f"  {r['method']:24}{r.get('status'):>16}")
        return 0
    root = D.resolve_dataset_root(cfg["dataset"]["id"])
    if a.subject:
        run_subject(cfg, a.subject, root, deriv_root)
        return 0
    p.error("one of --subject/--synthetic/--group-only required")


if __name__ == "__main__":
    raise SystemExit(main())
