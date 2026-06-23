#!/usr/bin/env python
"""Generic line-noise benchmark runner (arms: line_injection, line_ds003620, line_ds000117).

Brings each subject to the paper-grounded baseline (configs' ``baseline_preprocessing``),
then runs every method-under-test + required comparator on IDENTICAL held-out data and
scores line attenuation R(f0) / peak attenuation plus off-target preservation. Per-subject
results go through ``benchmarks.io``; ``--group-only`` aggregates.

    python scripts/run_line_arm.py --config configs/benchmarks/line_noise_ds003620.yaml --subject sub-01
    python scripts/run_line_arm.py --config ... --slurm-array      # subject from $SLURM_ARRAY_TASK_ID
    python scripts/run_line_arm.py --config ... --all
    python scripts/run_line_arm.py --config ... --group-only
    python scripts/run_line_arm.py --config ... --synthetic         # local smoke (no staged data)
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

import mne_denoise.benchmarks.adapters  # noqa: F401  (registers method adapters)
from mne_denoise.benchmarks import comparators
from mne_denoise.benchmarks import datasets as D
from mne_denoise.benchmarks import io as bio
from mne_denoise.benchmarks import sweep as _sweep
from mne_denoise.benchmarks.preprocessing import apply_baseline, branch_context
from mne_denoise.qa import metrics as qm

BENCH = "line_noise"


def _load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _find_raw(root: pathlib.Path, subject: str):
    import mne

    sub = root / subject
    readers = [
        ("*.vhdr", mne.io.read_raw_brainvision),
        ("*_meg.fif", mne.io.read_raw_fif),
        ("*.edf", mne.io.read_raw_edf),
        ("*.set", mne.io.read_raw_eeglab),
        ("*.fif", mne.io.read_raw_fif),
    ]
    skip = {"derivatives", "sourcedata", ".git"}
    for pat, reader in readers:
        hits = [p for p in sub.rglob(pat) if not (skip & set(p.parts))]
        if hits:
            return reader(str(sorted(hits)[0]), preload=True, verbose=False)
    raise FileNotFoundError(f"no recording for {subject} under {root}")


def _load_ds000117_meg(root, subject):
    """ds000117 MEG line arm: load the RAW (non-SSS) run-01 FIF — line noise is intact
    only in the raw files (SSS/MaxFilter removes 50 Hz + harmonics + HPI). Magnetometers
    only (the baseline's mag_grad_separate; gradiometers are a separate sub-analysis)."""
    import glob

    import mne

    hits = sorted(glob.glob(f"{root}/{subject}/**/*task-facerecognition_run-*_meg.fif", recursive=True))
    hits = [h for h in hits if "proc-" not in pathlib.Path(h).name]  # raw only (no proc-sss)
    if not hits:
        raise FileNotFoundError(f"no raw MEG run for {subject} under {root}")
    raw = mne.io.read_raw_fif(hits[0], preload=True, verbose=False)
    raw.pick("mag")
    return raw


def _synth_subject(line_freq=50.0, sfreq=500.0, secs=60, nch=16, seed=0):
    import mne

    rng = np.random.default_rng(seed)
    N = int(sfreq * secs)
    t = np.arange(N) / sfreq
    brain = rng.standard_normal((nch, N)) * 1e-6
    amp = (0.3 + 0.7 * rng.random(nch))[:, None]
    line = np.sin(2 * np.pi * line_freq * t)[None, :] * amp * 4e-6
    line += 0.4 * np.sin(2 * np.pi * 2 * line_freq * t)[None, :] * amp * 4e-6
    info = mne.create_info([f"E{i}" for i in range(nch)], sfreq, "eeg")
    return mne.io.RawArray(brain + line, info, verbose=False)


def _split_select_eval(raw):
    """First 40% = selection split (param tuning); last 60% = held-out evaluation."""
    tmax = float(raw.times[-1])
    cut = tmax * 0.4
    return raw.copy().crop(tmax=cut), raw.copy().crop(tmin=cut)


def _methods(cfg) -> list[str]:
    comp = cfg.get("comparators", {}) or {}
    methods = list(cfg.get("methods_under_test", []) or [])
    methods += list(comp.get("required", []) or [])
    for g in (comp.get("required_comparator_group", {}) or {}).values():
        # the registered id for the non-spatial line comparator
        methods.append("non_spatial_line")
    return list(dict.fromkeys(methods))  # de-dup, keep order


def _scalars(d: dict) -> dict:
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = float(v)
    return out


def run_subject(cfg, subject, root, deriv_root, *, synthetic=False):
    ctx = branch_context(cfg)
    line_freq = ctx.get("line_freq", 50.0)
    ds_id = (cfg.get("dataset") or {}).get("id")
    if synthetic:
        raw = _synth_subject(line_freq=line_freq)
    elif ds_id == "ds000117":
        raw = _load_ds000117_meg(root, subject)
    else:
        raw = _find_raw(root, subject)
    raw, applied = apply_baseline(raw, cfg.get("baseline_preprocessing"))
    raw.pick("data")  # score on data channels
    ctx["sfreq"] = float(raw.info["sfreq"])
    sel, ev = _split_select_eval(raw)
    n_harm = 3
    rows = []
    for mid in _methods(cfg):
        for suffix, mparams in _sweep.method_runs(cfg, mid):
            tag = f"{mid}__{suffix}" if suffix else mid
            sweep_param = suffix.split("-", 1)[0] if suffix else None
            sweep_value = suffix.split("-", 1)[1] if suffix else None
            try:
                cmp = comparators.get(mid, **mparams)
            except KeyError:
                rows.append({"method": mid, "tag": tag, "status": "unavailable_dependency"})
                continue
            state = cmp.fit(sel, ctx)
            res = cmp.transform(ev, state, ctx)
            if res.status != "success":
                rows.append({"method": mid, "tag": tag, "status": res.status, "error": res.error})
                continue
            qa = qm.compute_all_qa_metrics(ev, res.cleaned, line_freq=line_freq, n_harmonics=n_harm)
            row = {
                "status": "success",
                "sweep_param": sweep_param,
                "sweep_value": sweep_value,
                "runtime_s": res.runtime_seconds,
                "n_removed": res.diagnostics.get("n_removed"),
                **_scalars(qa),
            }
            rows.append({"method": mid, "tag": tag, **row})
            out_dir = pathlib.Path(deriv_root) / subject / BENCH / tag
            bio.save_subject_benchmark_results(
                out_dir, subject=subject, method=tag, metrics=row,
                model_info={"baseline_applied": applied, "line_freq": line_freq,
                            "branch_point": (cfg.get("baseline_preprocessing") or {}).get("branch_point")},
            )
    return rows


def _all_subjects(root: pathlib.Path) -> list[str]:
    return sorted(p.name for p in root.glob("sub-*") if p.is_dir())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--subject")
    p.add_argument("--slurm-array", action="store_true")
    p.add_argument("--all", action="store_true")
    p.add_argument("--group-only", action="store_true")
    p.add_argument("--synthetic", action="store_true", help="local smoke; one synthetic subject")
    p.add_argument("--deriv-root", default=None)
    a = p.parse_args(argv)

    cfg = _load_cfg(a.config)
    arm = cfg.get("arm", "line")
    deriv_root = pathlib.Path(a.deriv_root or (_REPO / "results" / arm))

    if a.group_only:
        g = bio.aggregate_benchmark_results(deriv_root)
        print(f"[group] {arm}: aggregated {len(getattr(g, 'df_metrics', []))} rows -> {deriv_root}")
        return 0

    if a.synthetic:
        rows = run_subject(cfg, "sub-synth", None, deriv_root, synthetic=True)
        for r in rows:
            print(f"  {r['method']:18} {r.get('status')}  R_f0={r.get('R_f0', r.get('noise_surr_ratio'))}")
        return 0

    root = D.resolve_dataset_root(cfg["dataset"]["id"])
    if a.slurm_array:
        tid = int(os.environ.get("SLURM_ARRAY_TASK_ID", 1))
        subs = cfg["dataset"].get("subjects") or _all_subjects(root)
        subject = subs[tid - 1] if isinstance(subs, list) and subs and not str(subs[0]).startswith("pend") else f"sub-{tid:02d}"
        run_subject(cfg, subject, root, deriv_root)
        return 0
    if a.subject:
        run_subject(cfg, a.subject, root, deriv_root)
        return 0
    if a.all:
        for s in _all_subjects(root):
            run_subject(cfg, s, root, deriv_root)
        bio.aggregate_benchmark_results(deriv_root)
        return 0
    p.error("one of --subject/--slurm-array/--all/--group-only/--synthetic required")


if __name__ == "__main__":
    raise SystemExit(main())
