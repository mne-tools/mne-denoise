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


def _load_meg_masc(root, subject):
    """meg_masc KIT line arm: read the raw .con (ses-1, task-1), magnetometers only.
    A distinct MEG system (NYU KIT, ~157 axial-gradiometer 'mag' sensors) from the Elekta
    ds000117 arm; line noise empirically at 50 Hz. Cropped to 180 s (stationary line)."""
    import glob

    import mne

    hits = sorted(glob.glob(f"{root}/**/{subject}/ses-1/meg/{subject}_ses-1_task-1_meg.con", recursive=True))
    if not hits:
        hits = sorted(glob.glob(f"{root}/**/{subject}_ses-*_task-*_meg.con", recursive=True))
    if not hits:
        raise FileNotFoundError(f"no meg_masc .con for {subject} under {root}")
    raw = mne.io.read_raw_kit(hits[0], preload=False, verbose=False)
    raw.crop(tmax=min(180.0, float(raw.times[-1])))
    raw.pick("mag").load_data()
    return raw


def _load_lemon(root, subject):
    """LEMON resting EEG: the released BrainVision .vhdr/.vmrk reference a template filename
    (sub-010002) rather than the actual subject, so MNE can't resolve the data/marker files.
    Patch the headers into a temp dir with the correct names, then read."""
    import glob
    import re
    import shutil
    import tempfile

    import mne

    vhdr = sorted(glob.glob(f"{root}/{subject}/**/*.vhdr", recursive=True))
    if not vhdr:
        raise FileNotFoundError(f"no LEMON .vhdr for {subject} under {root}")
    vp = pathlib.Path(vhdr[0])
    base, d = vp.stem, vp.parent
    tmp = pathlib.Path(tempfile.mkdtemp())
    vt = vp.read_text()
    vt = re.sub(r"DataFile=.*", f"DataFile={base}.eeg", vt)
    vt = re.sub(r"MarkerFile=.*", f"MarkerFile={base}.vmrk", vt)
    (tmp / f"{base}.vhdr").write_text(vt)
    mt = (d / f"{base}.vmrk").read_text()
    mt = re.sub(r"DataFile=.*", f"DataFile={base}.eeg", mt)
    (tmp / f"{base}.vmrk").write_text(mt)
    shutil.copy(d / f"{base}.eeg", tmp / f"{base}.eeg")
    return mne.io.read_raw_brainvision(str(tmp / f"{base}.vhdr"), preload=True, verbose=False)


def _ds003620_env_segments(root, subject):
    """Map ds003620's single continuous recording to its 3 mobility environments.

    The recording is 6 sequential ~equal blocks = 3 environments (lab / oval / campus) x 2 task
    conditions (count / do-not-count), counterbalanced per subject (participants.tsv). The EEG
    has no block markers (only S 1 stimuli), so blocks are split by stimulus index into 6 equal
    segments and mapped to environments via the per-subject counterbalancing. Inter-block
    transitions (subjects physically relocate) are excluded by cropping between the first and last
    stimulus of each segment. Returns {env_name: [(tmin_s, tmax_s), ...]} (2 segments per env)."""
    import csv
    import glob

    num = int(subject.split("-")[1])
    evf = glob.glob(f"{root}/{subject}/**/*_events.tsv", recursive=True)
    jf = glob.glob(f"{root}/{subject}/**/*_eeg.json", recursive=True)
    sf = 500.0
    if jf:
        import json
        sf = float(json.load(open(jf[0])).get("SamplingFrequency", 500.0))
    ons = []
    with open(evf[0]) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            try:
                ons.append(float(r["onset"]))
            except (ValueError, KeyError):
                pass
    ons = np.sort(np.asarray(ons)) / sf  # samples -> seconds
    N = len(ons)
    bnd = [int(round(N * k / 6)) for k in range(7)]
    segs = [(float(ons[bnd[k]]), float(ons[min(bnd[k + 1] - 1, N - 1)])) for k in range(6)]
    # per-subject counterbalancing from participants.tsv (participant_id is "sub-<n>", no zero-pad)
    lvl = {"1": "lab", "2": "oval", "3": "campus"}
    prow = {}
    with open(f"{root}/participants.tsv") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r.get("participant_id") in (f"sub-{num}", subject):
                prow = r
                break
    cnt = [lvl.get(prow.get(f"enviro_{i}_count", ""), "lab") for i in (1, 2, 3)]
    dnc = [lvl.get(prow.get(f"enviro_{i}_d_count", ""), "lab") for i in (1, 2, 3)]
    order = (cnt + dnc) if str(prow.get("condition", "1")) == "1" else (dnc + cnt)
    env_segs = {}
    for seg, env in zip(segs, order):
        env_segs.setdefault(env, []).append(seg)
    return env_segs


def _concat_env(raw, segments):
    """Crop the raw to each (tmin, tmax) segment and concatenate (one mobility environment)."""
    import mne

    tmax_all = float(raw.times[-1])
    parts = [raw.copy().crop(tmin=max(0.0, t0), tmax=min(tmax_all, t1))
             for t0, t1 in segments if t1 > t0]
    return mne.concatenate_raws(parts) if len(parts) > 1 else parts[0]


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
    _lf = ctx.get("line_freq", 50.0)
    line_freq = _lf[0] if isinstance(_lf, (list, tuple)) else _lf  # scalar for synth + metric; ctx keeps the list for notch
    ds_id = (cfg.get("dataset") or {}).get("id")
    if synthetic:
        raw = _synth_subject(line_freq=line_freq)
    elif ds_id == "ds000117":
        raw = _load_ds000117_meg(root, subject)
    elif ds_id == "meg_masc":
        raw = _load_meg_masc(root, subject)
    elif ds_id == "lemon":
        raw = _load_lemon(root, subject)
    else:
        raw = _find_raw(root, subject)
    raw, applied = apply_baseline(raw, cfg.get("baseline_preprocessing"))
    raw.pick("data")  # score on data channels
    ctx["sfreq"] = float(raw.info["sfreq"])
    n_harm = 3
    rows = []
    env_seg = bool(cfg.get("env_segment")) and ds_id == "ds003620" and not synthetic
    envs = ["lab", "oval", "campus"] if env_seg else [None]
    env_map = _ds003620_env_segments(root, subject) if env_seg else None
    for env in envs:
        raw_e = raw if env is None else _concat_env(raw, env_map[env])
        sel, ev = _split_select_eval(raw_e)
        for mid in _methods(cfg):
            for suffix, mparams in _sweep.method_runs(cfg, mid):
                base = f"{mid}__{suffix}" if suffix else mid
                tag = f"{env}__{base}" if env else base
                sweep_param = suffix.split("-", 1)[0] if suffix else None
                sweep_value = suffix.split("-", 1)[1] if suffix else None
                try:
                    cmp = comparators.get(mid, **mparams)
                except KeyError:
                    rows.append({"method": mid, "tag": tag, "env": env, "status": "unavailable_dependency"})
                    continue
                state = cmp.fit(sel, ctx)
                res = cmp.transform(ev, state, ctx)
                if res.status != "success":
                    rows.append({"method": mid, "tag": tag, "env": env, "status": res.status, "error": res.error})
                    continue
                qa = qm.compute_all_qa_metrics(ev, res.cleaned, line_freq=line_freq, n_harmonics=n_harm)
                row = {
                    "status": "success",
                    "env": env,
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
                    model_info={"baseline_applied": applied, "line_freq": line_freq, "env": env,
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

    try:
        root = D.resolve_dataset_root(cfg["dataset"]["id"])
    except Exception:  # noqa: BLE001 - unregistered Tier-B id: fall back to the config path
        root = pathlib.Path(os.environ.get("DATASETS_ROOT", "/project/rrg-kjerbi/datasets")) / \
            (cfg.get("dataset", {}) or {}).get("project_relative_path", "")
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
