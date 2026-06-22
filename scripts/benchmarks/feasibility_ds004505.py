#!/usr/bin/env python
"""ds004505 feasibility analysis -> pick the ONE muscle-track preservation contrast.

Read-only.  Runs on Fir where ds004505 lives (resolved via the dataset registry).
Reports, per subject: task conditions + event counts, duration per condition, the
channel-type inventory (eeg / noise-layer / neck-EMG / IMU), and — with
``--with-spectra`` — candidate alpha/beta band-power contrasts in a default ROI.
It writes a recommendation; a human then **freezes one contrast** in
``configs/benchmarks/muscle_ds004505.yaml`` (no automatic, post-hoc pick — see
ANALYSIS_PLAN.md and the muscle config).

    python scripts/benchmarks/feasibility_ds004505.py --with-spectra

Output: ``reports/feasibility/ds004505/{summary.json,summary.md}``.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

_REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
from mne_denoise.benchmarks import datasets as D  # noqa: E402

ALPHA = (8.0, 12.0)
BETA = (13.0, 30.0)


def _read_tsv(path: pathlib.Path):
    import pandas as pd

    return pd.read_csv(path, sep="\t")


def analyze_subject(sub_dir: pathlib.Path, *, with_spectra: bool) -> dict:
    rec: dict = {"subject": sub_dir.name, "conditions": {}, "channel_types": {}, "issues": []}
    eeg_dir = sub_dir / "eeg"
    if not eeg_dir.is_dir():
        rec["issues"].append("no eeg/ dir")
        return rec

    ev = sorted(eeg_dir.glob("*_events.tsv"))
    if ev:
        try:
            df = _read_tsv(ev[0])
            col = "trial_type" if "trial_type" in df else df.columns[-1]
            rec["conditions"] = {str(k): int(v) for k, v in df[col].value_counts().items()}
            if "duration" in df:
                rec["mean_duration_s"] = float(df["duration"].mean())
        except Exception as exc:  # noqa: BLE001
            rec["issues"].append(f"events parse failed: {exc}")
    else:
        rec["issues"].append("no *_events.tsv")

    ch = sorted(eeg_dir.glob("*_channels.tsv"))
    if ch:
        try:
            df = _read_tsv(ch[0])
            if "type" in df:
                rec["channel_types"] = {str(k): int(v) for k, v in df["type"].value_counts().items()}
        except Exception as exc:  # noqa: BLE001
            rec["issues"].append(f"channels parse failed: {exc}")

    if with_spectra:
        rec["spectra"] = _spectra(eeg_dir, rec)
    return rec


def _spectra(eeg_dir: pathlib.Path, rec: dict) -> dict:
    """Best-effort alpha/beta band power per condition (default posterior ROI)."""
    try:
        import mne
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        return {"error": f"mne/numpy unavailable: {exc}"}
    hits = sorted(eeg_dir.glob("*_eeg.set")) or sorted(eeg_dir.glob("*_eeg.vhdr"))
    if not hits:
        return {"error": "no readable eeg recording"}
    try:
        reader = mne.io.read_raw_eeglab if hits[0].suffix == ".set" else mne.io.read_raw_brainvision
        raw = reader(str(hits[0]), preload=True, verbose="ERROR")
        raw.pick("eeg")
        psd = raw.compute_psd(fmax=45.0, verbose="ERROR")
        freqs = psd.freqs
        data = psd.get_data().mean(0)
        def band(lo, hi):
            m = (freqs >= lo) & (freqs <= hi)
            return float(np.trapezoid(data[m], freqs[m]))
        return {"alpha_power": band(*ALPHA), "beta_power": band(*BETA),
                "n_channels": len(raw.ch_names)}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"spectra failed: {exc}"}


def recommend(records: list[dict]) -> dict:
    """Heuristic recommendation; the human freezes the final contrast."""
    conds: set[str] = set()
    for r in records:
        conds.update(r.get("conditions", {}))
    rec = {
        "candidate_conditions": sorted(conds),
        "preferred_band_order": ["beta(13-30)", "alpha(8-12)", "task_vs_rest"],
        "note": "Freeze ONE contrast in muscle_ds004505.yaml (band, ROI, window, "
                "direction, equivalence_margin) + a negative-control band/ROI. "
                "Do not pick whichever band shows the biggest method difference.",
    }
    return rec


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--with-spectra", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="analyze first N subjects")
    a = p.parse_args(argv)

    root = D.resolve_dataset_root("ds004505")
    if not root.exists():
        print(f"ds004505 not found at {root}.\n"
              f"Stage it first on Fir: python scripts/cc/stage_dataset.py ds004505",
              file=sys.stderr)
        return 2
    subs = sorted(p for p in root.glob("sub-*") if p.is_dir())
    if a.limit:
        subs = subs[: a.limit]
    records = [analyze_subject(s, with_spectra=a.with_spectra) for s in subs]
    summary = {"dataset": "ds004505", "root": str(root), "n_subjects": len(subs),
               "subjects": records, "recommendation": recommend(records)}

    out = _REPO / "reports" / "feasibility" / "ds004505"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    md = ["# ds004505 feasibility", f"- root: {root}", f"- subjects: {len(subs)}",
          f"- candidate conditions: {summary['recommendation']['candidate_conditions']}",
          "", "Freeze ONE contrast in `configs/benchmarks/muscle_ds004505.yaml`."]
    (out / "summary.md").write_text("\n".join(md))
    print(f"[OK] wrote {out/'summary.json'} ({len(subs)} subjects)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
