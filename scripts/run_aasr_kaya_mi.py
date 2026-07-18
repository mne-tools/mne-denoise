#!/usr/bin/env python
"""Run the public Kaya motor-imagery replication from Tsai et al."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.io import loadmat
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mne_denoise.asr import AdaptiveASR  # noqa: E402
from mne_denoise.benchmarks.config import assert_submission_ready  # noqa: E402
from scripts.asr_paper_protocols import (  # noqa: E402
    tsai_demo_update_slices,
    tsai_fft_bandpass,
)

VARIANTS = ("init", "mw", "psp", "psw")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _git_state() -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return commit, dirty


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert_submission_ready(config, source=str(path))
    return config


def _event_onsets(marker: np.ndarray, code: int) -> np.ndarray:
    selected = np.asarray(marker).reshape(-1) == code
    return np.flatnonzero(selected & np.r_[True, ~selected[:-1]])


def _extract_trials(
    data: np.ndarray,
    marker: np.ndarray,
    *,
    samples_per_trial: int,
    codes: tuple[int, ...],
) -> tuple[list[np.ndarray], list[int], list[int]]:
    events = []
    for code in codes:
        events.extend((int(onset), code) for onset in _event_onsets(marker, code))
    events.sort()
    trials = []
    labels = []
    onsets = []
    for onset, code in events:
        stop = onset + samples_per_trial
        if stop > data.shape[1]:
            raise ValueError(f"trial at sample {onset} exceeds recording boundary")
        trials.append(data[:, onset:stop])
        labels.append(code)
        onsets.append(onset)
    return trials, labels, onsets


def _load_filtered_recording(
    path: Path,
    *,
    channels: tuple[str, ...],
    sfreq: float,
    low: float,
    high: float,
) -> tuple[np.ndarray, np.ndarray]:
    recording = loadmat(path, squeeze_me=True, struct_as_record=False).get("o")
    if recording is None:
        raise ValueError(f"MATLAB variable 'o' is absent: {path}")
    observed_sfreq = float(recording.sampFreq)
    if not np.isclose(observed_sfreq, sfreq):
        raise ValueError(f"unexpected sampling frequency in {path}: {observed_sfreq}")
    names = [str(name) for name in np.asarray(recording.chnames).reshape(-1)]
    missing = sorted(set(channels) - set(names))
    if missing:
        raise ValueError(f"missing channels in {path}: {missing}")
    picks = [names.index(name) for name in channels]
    data = np.asarray(recording.data, dtype=np.float64)[:, picks].T
    filtered = tsai_fft_bandpass(data, sfreq, low=low, high=high)
    return filtered, np.asarray(recording.marker).reshape(-1)


def prepare_cache(
    config_path: Path,
    dataset_root: Path,
    inventory_path: Path,
    output_dir: Path,
    *,
    locked: bool,
) -> Path:
    """Prepare immutable subject streams from the recovered public cohort."""
    config = _load_config(config_path)
    commit, dirty = _git_state()
    if locked and dirty:
        raise RuntimeError("refusing locked cache preparation from a dirty worktree")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    candidate = inventory["paper_candidate"]
    dataset = config["dataset"]
    if candidate["trial_count"] != int(dataset["public_trial_count"]):
        raise ValueError("inventory does not match the frozen public trial count")
    if candidate["subjects"] != list(dataset["public_subject_ids"]):
        raise ValueError("inventory subject identifiers do not match the frozen cohort")
    if candidate["paradigms"] != [dataset["included_paradigm"]]:
        raise ValueError("inventory paradigm does not match the frozen cohort")

    preprocessing = config["preprocessing"]
    channels = tuple(preprocessing["channels"])
    sfreq = float(preprocessing["source_sampling_frequency_hz"])
    samples_per_trial = int(round(sfreq * float(preprocessing["trial_duration_s"])))
    low, high = map(float, preprocessing["bandpass_hz"])
    codes = tuple(int(value) for value in dataset["included_marker_codes"])
    manifest = json.loads(
        (dataset_root / "figshare_manifest.json").read_text(encoding="utf-8")
    )
    source_metadata = {
        item["relative_path"]: {
            "size": int(item["size"]),
            "computed_md5": item.get("computed_md5"),
            "supplied_md5": item.get("supplied_md5"),
        }
        for item in manifest["files"]
    }

    by_subject: dict[str, list[str]] = defaultdict(list)
    for relative_path in candidate["relative_paths"]:
        name = Path(relative_path).name
        subject = name.split("-Subject", 1)[1][0]
        by_subject[subject].append(relative_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    subject_records = []
    for subject in dataset["public_subject_ids"]:
        trial_blocks: list[np.ndarray] = []
        labels: list[int] = []
        session_ids: list[int] = []
        source_onsets: list[int] = []
        paths = sorted(by_subject[subject], key=lambda value: Path(value).name)
        for session_index, relative_path in enumerate(paths):
            path = dataset_root / relative_path
            filtered, marker = _load_filtered_recording(
                path,
                channels=channels,
                sfreq=sfreq,
                low=low,
                high=high,
            )
            trials, trial_labels, onsets = _extract_trials(
                filtered,
                marker,
                samples_per_trial=samples_per_trial,
                codes=codes,
            )
            trial_blocks.extend(trials)
            labels.extend(trial_labels)
            session_ids.extend([session_index] * len(trials))
            source_onsets.extend(onsets)
            print(f"[{subject}] {path.name}: {len(trials)} hand trials")
        stream = np.concatenate(trial_blocks, axis=1)
        cache_path = output_dir / f"subject-{subject}.npz"
        temporary = cache_path.with_name(cache_path.stem + ".tmp.npz")
        np.savez(
            temporary,
            data=stream,
            labels=np.asarray(labels, dtype=np.uint8),
            session_ids=np.asarray(session_ids, dtype=np.uint8),
            source_onsets=np.asarray(source_onsets, dtype=np.int64),
            channels=np.asarray(channels),
            samples_per_trial=np.asarray(samples_per_trial),
        )
        os.replace(temporary, cache_path)
        subject_records.append(
            {
                "subject": subject,
                "trial_count": len(labels),
                "class_counts": {
                    str(code): int(np.sum(np.asarray(labels) == code)) for code in codes
                },
                "session_count": len(paths),
                "relative_cache_path": cache_path.relative_to(output_dir).as_posix(),
                "cache_sha256": _sha256(cache_path),
                "source_files": [
                    {"relative_path": value, **source_metadata[value]} for value in paths
                ],
            }
        )
        print(f"[{subject}] cached {len(labels)} trials at {cache_path}")

    observed_trials = sum(item["trial_count"] for item in subject_records)
    if observed_trials != int(dataset["public_trial_count"]):
        raise RuntimeError(f"prepared {observed_trials} trials, expected {dataset['public_trial_count']}")
    record = {
        "schema_version": 1,
        "generated_utc": _utc_now(),
        "repository_commit": commit,
        "dirty_worktree": dirty,
        "config_path": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "inventory_path": str(inventory_path.resolve()),
        "inventory_sha256": _sha256(inventory_path),
        "figshare_manifest_sha256": _sha256(dataset_root / "figshare_manifest.json"),
        "dataset_root": str(dataset_root.resolve()),
        "subject_count": len(subject_records),
        "trial_count": observed_trials,
        "samples_per_trial": samples_per_trial,
        "channels": list(channels),
        "subjects": subject_records,
    }
    record["manifest_hash"] = _json_hash(record)
    output = output_dir / "prepared_manifest.json"
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return output


def _run_variant(
    data: np.ndarray,
    *,
    sfreq: float,
    cutoff: float,
    variant: str,
    update_window_s: float,
) -> tuple[np.ndarray, AdaptiveASR, float]:
    update_samples = int(round(sfreq * update_window_s))
    slices = tsai_demo_update_slices(data.shape[1], sfreq, update_window_s)
    if not slices:
        raise ValueError("subject stream is shorter than two update intervals")
    start = time.perf_counter()
    common = {"sfreq": sfreq, "cutoff": cutoff, "blocksize": 10, "verbose": False}
    if variant == "init":
        model = AdaptiveASR(variant="psw", **common).fit(data[:, :update_samples])
    elif variant == "mw":
        model = AdaptiveASR(variant="mw", mw_mode="final_state", **common).fit(
            data[:, slices[-1]]
        )
    elif variant in ("psp", "psw"):
        model = AdaptiveASR(variant=variant, **common).fit(data[:, slices[0]])
        for update_slice in slices[1:]:
            model.partial_fit(data[:, update_slice])
    else:
        raise ValueError(f"unknown variant: {variant}")
    model.reset_process_state()
    cleaned = model.transform(data)
    return cleaned, model, time.perf_counter() - start


def _connectivity_features(
    stream: np.ndarray, n_trials: int, samples_per_trial: int
) -> np.ndarray:
    n_channels = stream.shape[0]
    trials = stream.reshape(n_channels, n_trials, samples_per_trial).transpose(1, 0, 2)
    centered = trials - trials.mean(axis=2, keepdims=True)
    norms = np.linalg.norm(centered, axis=2)
    numerators = np.einsum("tcs,tds->tcd", centered, centered, optimize=True)
    denominators = norms[:, :, None] * norms[:, None, :]
    correlations = np.divide(
        numerators,
        denominators,
        out=np.zeros_like(numerators),
        where=denominators > np.finfo(float).eps,
    )
    upper = np.triu_indices(n_channels, k=1)
    return correlations[:, upper[0], upper[1]]


def _diagnostic_summary(model: AdaptiveASR) -> dict[str, Any]:
    diagnostics = getattr(model, "diagnostics_", {})
    return {
        "rank": int(model.rank_),
        "fraction_reconstructed_windows": float(
            diagnostics.get("fraction_reconstructed_windows", 0.0)
        ),
        "fraction_reconstructed_samples": float(
            diagnostics.get("fraction_reconstructed_samples", 0.0)
        ),
        "max_components_reconstructed": int(
            diagnostics.get("max_components_reconstructed", 0)
        ),
    }


def run_cell(
    config_path: Path,
    prepared_manifest_path: Path,
    *,
    subject: str,
    variant: str,
    cutoff: float,
    output: Path,
    locked: bool,
) -> int:
    """Run one participant, variant, and cutoff cell with a terminal record."""
    started = _utc_now()
    commit, dirty = _git_state()
    base_record: dict[str, Any] = {
        "schema_version": 1,
        "started_utc": started,
        "repository_commit": commit,
        "dirty_worktree": dirty,
        "subject": subject,
        "variant": variant,
        "cutoff": cutoff,
    }
    exit_code = 0
    try:
        if locked and dirty:
            raise RuntimeError("refusing locked run from a dirty worktree")
        config = _load_config(config_path)
        if variant not in VARIANTS or variant not in config["methods"]:
            raise ValueError(f"variant is not frozen in the protocol: {variant}")
        if cutoff not in [float(value) for value in config["processing"]["cutoffs"]]:
            raise ValueError(f"cutoff is not frozen in the protocol: {cutoff}")
        manifest = json.loads(prepared_manifest_path.read_text(encoding="utf-8"))
        if locked and manifest["repository_commit"] != commit:
            raise RuntimeError("prepared cache and analysis commits differ")
        if manifest["config_sha256"] != _sha256(config_path):
            raise RuntimeError("prepared cache and analysis configurations differ")
        subject_record = next(
            item for item in manifest["subjects"] if item["subject"] == subject
        )
        cache_path = prepared_manifest_path.parent / subject_record["relative_cache_path"]
        cache_hash = _sha256(cache_path)
        if cache_hash != subject_record["cache_sha256"]:
            raise RuntimeError(f"cache checksum mismatch: {cache_path}")
        with np.load(cache_path, allow_pickle=False) as cache:
            data = np.asarray(cache["data"], dtype=np.float64)
            labels = np.asarray(cache["labels"], dtype=int)
            samples_per_trial = int(cache["samples_per_trial"])
        sfreq = float(config["preprocessing"]["source_sampling_frequency_hz"])
        cleaned, model, asr_time = _run_variant(
            data,
            sfreq=sfreq,
            cutoff=cutoff,
            variant=variant,
            update_window_s=float(config["processing"]["update_window_s"]),
        )
        feature_start = time.perf_counter()
        features = _connectivity_features(cleaned, len(labels), samples_per_trial)
        feature_time = time.perf_counter() - feature_start
        classification = config["classification"]
        cv = StratifiedKFold(
            n_splits=10,
            shuffle=bool(classification["shuffle"]),
            random_state=int(classification["random_seed"]),
        )
        classifier = SVC(kernel="linear", C=float(classification["svm_c"]))
        classifier_start = time.perf_counter()
        fold_scores = cross_val_score(classifier, features, labels, cv=cv, n_jobs=1)
        classifier_time = time.perf_counter() - classifier_start
        residual = cleaned - data
        record = {
            **base_record,
            "status": "passed",
            "completed_utc": _utc_now(),
            "config_sha256": _sha256(config_path),
            "prepared_manifest_hash": manifest["manifest_hash"],
            "cache_sha256": cache_hash,
            "n_trials": int(len(labels)),
            "class_counts": {
                str(code): int(np.sum(labels == code)) for code in np.unique(labels)
            },
            "feature_count": int(features.shape[1]),
            "accuracy": float(np.mean(fold_scores)),
            "fold_accuracies": [float(value) for value in fold_scores],
            "relative_rms_change": float(
                np.sqrt(np.mean(residual**2))
                / max(np.sqrt(np.mean(data**2)), np.finfo(float).tiny)
            ),
            "asr_wall_time_s": asr_time,
            "feature_wall_time_s": feature_time,
            "classifier_wall_time_s": classifier_time,
            **_diagnostic_summary(model),
        }
    except Exception as exc:  # noqa: BLE001
        exit_code = 1
        record = {
            **base_record,
            "status": "failed",
            "completed_utc": _utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    return exit_code


def _bootstrap_mean_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def merge_results(
    config_path: Path,
    prepared_manifest_path: Path,
    results_dir: Path,
    output_dir: Path,
) -> int:
    """Merge terminal cell records and verify the complete frozen factorial."""
    config = _load_config(config_path)
    manifest = json.loads(prepared_manifest_path.read_text(encoding="utf-8"))
    records = [json.loads(path.read_text(encoding="utf-8")) for path in results_dir.glob("*.json")]
    expected = {
        (subject["subject"], variant, float(cutoff))
        for subject in manifest["subjects"]
        for variant in VARIANTS
        for cutoff in config["processing"]["cutoffs"]
    }
    observed = {(row["subject"], row["variant"], float(row["cutoff"])) for row in records}
    missing = sorted(expected - observed)
    extras = sorted(observed - expected)
    duplicates = len(records) - len(observed)
    passed = [row for row in records if row["status"] == "passed"]
    failed = [row for row in records if row["status"] != "passed"]

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "per_subject_metrics.csv"
    fields = [
        "subject",
        "variant",
        "cutoff",
        "status",
        "accuracy",
        "relative_rms_change",
        "fraction_reconstructed_windows",
        "rank",
        "asr_wall_time_s",
        "error_type",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(records, key=lambda row: (row["variant"], row["cutoff"], row["subject"])))

    aggregate = []
    for variant in VARIANTS:
        for cutoff in map(float, config["processing"]["cutoffs"]):
            cell = [
                row
                for row in passed
                if row["variant"] == variant and float(row["cutoff"]) == cutoff
            ]
            values = np.asarray([row["accuracy"] for row in cell], dtype=float)
            if values.size:
                low, high = _bootstrap_mean_ci(
                    values,
                    int(config["processing"]["random_seed"])
                    + int(cutoff) * 10
                    + VARIANTS.index(variant),
                )
                aggregate.append(
                    {
                        "variant": variant,
                        "cutoff": cutoff,
                        "n": int(values.size),
                        "mean_accuracy": float(values.mean()),
                        "median_accuracy": float(np.median(values)),
                        "bootstrap_95_ci": [low, high],
                    }
                )
    report = {
        "schema_version": 1,
        "generated_utc": _utc_now(),
        "config_sha256": _sha256(config_path),
        "prepared_manifest_hash": manifest["manifest_hash"],
        "expected_cell_count": len(expected),
        "terminal_record_count": len(records),
        "passed_count": len(passed),
        "failed_count": len(failed),
        "missing_cells": missing,
        "extra_cells": extras,
        "duplicate_count": duplicates,
        "aggregate": aggregate,
    }
    (output_dir / "aggregate.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return int(bool(missing or extras or duplicates or failed))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--dataset-root", type=Path, required=True)
    prepare.add_argument("--inventory", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--locked", action="store_true")
    run = subparsers.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--prepared-manifest", type=Path, required=True)
    run.add_argument("--subject", required=True)
    run.add_argument("--variant", choices=VARIANTS, required=True)
    run.add_argument("--cutoff", type=float, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--locked", action="store_true")
    merge = subparsers.add_parser("merge")
    merge.add_argument("--config", type=Path, required=True)
    merge.add_argument("--prepared-manifest", type=Path, required=True)
    merge.add_argument("--results-dir", type=Path, required=True)
    merge.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Run the selected preparation, cell, or merge command."""
    args = _parse_args()
    if args.command == "prepare":
        output = prepare_cache(
            args.config,
            args.dataset_root,
            args.inventory,
            args.output_dir,
            locked=args.locked,
        )
        print(output)
        return 0
    if args.command == "run":
        return run_cell(
            args.config,
            args.prepared_manifest,
            subject=args.subject,
            variant=args.variant,
            cutoff=args.cutoff,
            output=args.output,
            locked=args.locked,
        )
    return merge_results(
        args.config,
        args.prepared_manifest,
        args.results_dir,
        args.output_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
