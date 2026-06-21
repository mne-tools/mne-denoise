"""Tier-C ICLabel + dipolar-IC scoring for Chang 2020 reproduction.

For each dataset that already has a Stage 2 output folder under
``reports/paper_validation/standard_asr/<label>/``, this script:

1. Reloads the dataset the same way ``run_asr_paper_validation.py`` does.
2. Reproduces the same burst injection (``random_state=97``).
3. Builds two copies of the data: input EEG and ASR-cleaned EEG at ``--cutoff``.
4. Fits extended-infomax ICA on each copy (``n_components=0.99``,
   ``random_state=97``).
5. Runs ``mne_icalabel.iclabel_label_components`` to classify each IC into
   brain / muscle / eye / heart / line-noise / channel-noise / other.
6. Reports an IC's dipolar status as ``ICLabel brain probability >= 0.5`` --
   the modern reproducibility convention when MATLAB ``dipfit`` is not
   available. See the deviation report for why this differs from Chang 2020's
   exact dipfit residual-variance threshold.

Outputs:

- ``reports/paper_validation/standard_asr/<label>/tier_C_iclabel.json``
  + ``tier_C_iclabel.csv`` (per-class counts before vs after)
- ``reports/paper_validation/standard_asr/<label>/tier_C_dipolar.csv``
  (dipolar count before vs after)
- ``reports/paper_validation/standard_asr/tier_C_summary.json`` aggregating
  every dataset.
"""

# ruff: noqa: I001

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import mne
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mne_denoise.asr import ASR

from scripts.run_asr_paper_validation import (  # noqa: E402
    DatasetSpec,
    _inject_bursts,
    _load_dataset,
    _reader_for_path,
)

ICLABEL_CLASSES = (
    "brain",
    "muscle artifact",
    "eye blink",
    "heart beat",
    "line noise",
    "channel noise",
    "other",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=ROOT / "reports" / "paper_validation" / "standard_asr",
    )
    parser.add_argument("--cutoff", type=float, default=20.0)
    parser.add_argument(
        "--method",
        choices=("standard", "riemannian", "juggler_dbscan", "juggler_gev"),
        default="standard",
        help=(
            "ASR backend. Use 'riemannian' for the rASR sprint, "
            "'juggler_dbscan'/'juggler_gev' for the Juggler sprint."
        ),
    )
    parser.add_argument("--random-state", type=int, default=97)
    parser.add_argument("--ica-n-components", type=float, default=0.99)
    parser.add_argument(
        "--max-iter",
        type=int,
        default=500,
        help="ICA max_iter; infomax sometimes needs >200 to converge on EEG",
    )
    parser.add_argument(
        "--brain-prob-threshold",
        type=float,
        default=0.5,
        help="ICLabel brain-class threshold used as dipolar-IC proxy",
    )
    parser.add_argument(
        "--burst-count",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--burst-duration",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--burst-amplitude",
        type=float,
        default=12.0,
    )
    return parser.parse_args()


def _dataset_spec_from_summary(item: dict, config_ds: dict) -> DatasetSpec:
    path = Path(item["path"])
    return DatasetSpec(
        path=path,
        label=item["label"],
        reader=_reader_for_path(path),
        max_duration_s=float(config_ds.get("max_duration_s", 120.0)),
        resample_hz=float(config_ds.get("resample_hz", 250.0)),
        highpass_hz=float(config_ds.get("highpass_hz", 1.0)),
    )


def _ensure_montage_for_iclabel(raw: mne.io.BaseRaw) -> bool:
    """Set standard 10-20 montage so ICLabel's spatial features make sense."""
    info = raw.info
    has_positions = False
    montage = info.get_montage()
    if montage is not None:
        for ch in info["chs"]:
            loc = ch.get("loc")
            if (
                loc is not None
                and np.isfinite(loc[:3]).all()
                and np.linalg.norm(loc[:3]) > 0
            ):
                has_positions = True
                break
    if not has_positions:
        try:
            std = mne.channels.make_standard_montage("standard_1020")
            raw.set_montage(std, match_case=False, on_missing="ignore", verbose=False)
            for ch in raw.info["chs"]:
                loc = ch.get("loc")
                if (
                    loc is not None
                    and np.isfinite(loc[:3]).all()
                    and np.linalg.norm(loc[:3]) > 0
                ):
                    has_positions = True
                    break
        except Exception:  # noqa: BLE001
            pass
    return has_positions


def _set_average_reference(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """ICLabel was trained on common-average-referenced data."""
    raw_avg = raw.copy()
    raw_avg.set_eeg_reference("average", projection=False, verbose=False)
    return raw_avg


def _fit_ica(
    raw: mne.io.BaseRaw,
    *,
    n_components: float,
    random_state: int,
    max_iter: int,
) -> mne.preprocessing.ICA:
    # ICLabel-recommended: extended infomax, EEG only
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method="infomax",
        random_state=random_state,
        max_iter=max_iter,
        fit_params={"extended": True},
    )
    ica.fit(raw, picks=eeg_picks, verbose=False)
    return ica


def _classify_iclabel(raw: mne.io.BaseRaw, ica: mne.preprocessing.ICA) -> dict:
    """Run mne_icalabel and return per-IC class probabilities + counts.

    Uses the lower-level ``iclabel_label_components`` because the high-level
    ``label_components`` API only returns argmax probability per IC, not the
    full per-class probability matrix we need.
    """
    from mne_icalabel.iclabel import iclabel_label_components

    proba = np.asarray(
        iclabel_label_components(raw, ica, inplace=False)
    )  # (n_components, 7)
    if proba.ndim != 2 or proba.shape[1] != len(ICLABEL_CLASSES):
        raise RuntimeError(
            f"Unexpected ICLabel proba shape {proba.shape}; "
            f"expected (n_components, {len(ICLABEL_CLASSES)})"
        )
    argmax = np.argmax(proba, axis=1)
    labels = [ICLABEL_CLASSES[i] for i in argmax]
    counts = {cls: int(sum(1 for L in labels if cls == L)) for cls in ICLABEL_CLASSES}
    return {
        "proba": proba.tolist(),
        "labels": labels,
        "counts": counts,
        "brain_prob_per_ic": proba[:, 0].tolist(),
    }


def _dipolar_count(brain_probs: list[float], threshold: float) -> int:
    return int(sum(1 for p in brain_probs if p >= threshold))


def _process_dataset(
    summary_item: dict,
    summary: dict,
    out_dir: Path,
    args: argparse.Namespace,
) -> dict:
    cfg_ds = next(
        (
            d
            for d in summary.get("config", {}).get("datasets", [])
            if d.get("label") == summary_item["label"]
        ),
        {},
    )
    ds = _dataset_spec_from_summary(summary_item, cfg_ds)
    raw = _load_dataset(ds)

    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    rng = np.random.default_rng(args.random_state)
    inject_info = summary_item.get("injection", {})
    if inject_info.get("enabled", True):
        raw_in, _, _ = _inject_bursts(
            raw,
            eeg_picks=eeg_picks,
            rng=rng,
            n_bursts=args.burst_count,
            burst_duration=args.burst_duration,
            amplitude=args.burst_amplitude,
        )
    else:
        raw_in = raw.copy()

    # Apply ASR (standard / riemannian / juggler_* depending on --method)
    asr_kwargs = {
        "sfreq": raw_in.info["sfreq"],
        "cutoff": args.cutoff,
        "picks": "eeg",
        "max_mem_mb": 512,
        "copy": True,
        "random_state": args.random_state,
        "verbose": False,
    }
    if args.method.startswith("juggler_"):
        from mne_denoise.asr import JugglerASR

        strategy = args.method.split("_", 1)[1]  # "dbscan" or "gev"
        est = JugglerASR(**asr_kwargs, strategy=strategy)
    elif args.method == "riemannian":
        asr_kwargs.update(method="riemannian", experimental=True)
        est = ASR(**asr_kwargs)
    else:
        est = ASR(**asr_kwargs)
    raw_clean = est.fit_transform(raw_in)

    # ICLabel was trained on EEG bandpassed 1-100 Hz; apply LP at 100 Hz here.
    # Highpass at 1 Hz already happened in _load_dataset.
    for r in (raw_in, raw_clean):
        if r.info["lowpass"] is None or r.info["lowpass"] > 100.0:
            r.filter(l_freq=None, h_freq=100.0, picks="eeg", verbose=False)

    # Set montage + common-average reference for ICLabel
    has_montage_in = _ensure_montage_for_iclabel(raw_in)
    _ensure_montage_for_iclabel(raw_clean)
    if not has_montage_in:
        return {
            "label": summary_item["label"],
            "status": "skipped",
            "reason": "No usable EEG montage; ICLabel requires electrode positions.",
        }

    raw_in_avg = _set_average_reference(raw_in)
    raw_clean_avg = _set_average_reference(raw_clean)

    ica_before = _fit_ica(
        raw_in_avg,
        n_components=args.ica_n_components,
        random_state=args.random_state,
        max_iter=args.max_iter,
    )
    ica_after = _fit_ica(
        raw_clean_avg,
        n_components=args.ica_n_components,
        random_state=args.random_state,
        max_iter=args.max_iter,
    )

    before = _classify_iclabel(raw_in_avg, ica_before)
    after = _classify_iclabel(raw_clean_avg, ica_after)

    dipolar_before = _dipolar_count(
        before["brain_prob_per_ic"], args.brain_prob_threshold
    )
    dipolar_after = _dipolar_count(
        after["brain_prob_per_ic"], args.brain_prob_threshold
    )

    iclabel_payload = {
        "label": summary_item["label"],
        "method": args.method,
        "cutoff": float(args.cutoff),
        "brain_prob_threshold": float(args.brain_prob_threshold),
        "n_components_before": len(before["labels"]),
        "n_components_after": len(after["labels"]),
        "counts_before": before["counts"],
        "counts_after": after["counts"],
        "dipolar_before": dipolar_before,
        "dipolar_after": dipolar_after,
    }

    # Tag the output filename with --method when it's NOT the default 'standard',
    # so juggler_dbscan vs juggler_gev (etc.) don't clobber each other.
    suffix = "" if args.method == "standard" else f"_{args.method}"
    (out_dir / f"tier_C_iclabel{suffix}.json").write_text(
        json.dumps(iclabel_payload, indent=2), encoding="utf-8"
    )

    # tier_C_iclabel.csv — one row per class
    with (out_dir / f"tier_C_iclabel{suffix}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        w = csv.writer(fh)
        w.writerow(["class", "count_before", "count_after", "delta"])
        for cls in ICLABEL_CLASSES:
            b = before["counts"].get(cls, 0)
            a = after["counts"].get(cls, 0)
            w.writerow([cls, b, a, a - b])

    # tier_C_dipolar.csv — single-row summary
    with (out_dir / f"tier_C_dipolar{suffix}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "dataset",
                "cutoff",
                "brain_prob_threshold",
                "dipolar_before",
                "dipolar_after",
                "delta",
            ]
        )
        w.writerow(
            [
                summary_item["label"],
                args.cutoff,
                args.brain_prob_threshold,
                dipolar_before,
                dipolar_after,
                dipolar_after - dipolar_before,
            ]
        )

    return {**iclabel_payload, "status": "ok"}


def main() -> int:
    args = _parse_args()
    reports_dir = args.reports_dir.resolve()
    summary_path = reports_dir / "summary.json"
    if not summary_path.exists():
        raise SystemExit(
            f"summary.json not found at {summary_path}. "
            "Run scripts/run_asr_paper_validation.py first."
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    try:
        from mne_icalabel import label_components  # noqa: F401
    except ImportError:
        raise SystemExit(
            "mne_icalabel is not installed. Run: pip install mne-icalabel"
        ) from None

    overall = []
    for item in summary.get("results", []):
        out_dir = reports_dir / item["label"]
        if not out_dir.exists():
            print(f"[skip] {item['label']}: dataset output dir missing")
            continue
        try:
            rec = _process_dataset(item, summary, out_dir, args)
            overall.append(rec)
            if rec.get("status") == "skipped":
                print(f"[{item['label']}] skipped: {rec.get('reason')}")
                continue
            print(
                f"[{item['label']}] "
                f"brain {rec['counts_before']['brain']}->{rec['counts_after']['brain']} "
                f"eye {rec['counts_before']['eye blink']}->{rec['counts_after']['eye blink']} "
                f"muscle {rec['counts_before']['muscle artifact']}->{rec['counts_after']['muscle artifact']} "
                f"dipolar {rec['dipolar_before']}->{rec['dipolar_after']}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[{item['label']}] FAILED: {type(exc).__name__}: {exc}")
            overall.append(
                {"label": item["label"], "status": "error", "error": str(exc)}
            )

    (reports_dir / "tier_C_summary.json").write_text(
        json.dumps({"cutoff": args.cutoff, "records": overall}, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {reports_dir / 'tier_C_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
