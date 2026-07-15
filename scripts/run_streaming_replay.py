#!/usr/bin/env python
"""Run frozen causal replay tests for ContinuousDSS adaptation and deployment."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

import numpy as np
import yaml

_REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from mne_denoise.benchmarks.config import assert_submission_ready
from mne_denoise.benchmarks.intended import (
    clean_correlation,
    locked_seed,
    relative_rmse,
)
from mne_denoise.benchmarks.provenance import AttemptRecorder, build_run_record
from mne_denoise.dss import DSS, BandpassBias
from mne_denoise.experimental import ContinuousDSS

_EPS = float(np.finfo(np.float64).eps)


def _replay_substrate(regime, sfreq, duration, block_size, seed):
    rng = np.random.default_rng(seed)
    n_channels = 16
    n_times = int(round(sfreq * duration))
    time = np.arange(n_times) / sfreq
    pattern_start = rng.normal(size=n_channels)
    pattern_start /= np.linalg.norm(pattern_start)
    pattern_end = rng.normal(size=n_channels)
    pattern_end -= pattern_start * (pattern_start @ pattern_end)
    pattern_end /= np.linalg.norm(pattern_end)
    if regime == "abrupt_target_change":
        alpha = (np.arange(n_times) >= n_times // 2).astype(float)
    elif regime == "gradual_spatial_drift":
        alpha = np.linspace(0.0, 1.0, n_times)
    else:
        alpha = np.zeros(n_times)
    patterns = pattern_start[:, None] * (1.0 - alpha) + pattern_end[:, None] * alpha
    patterns /= np.maximum(np.linalg.norm(patterns, axis=0, keepdims=True), _EPS)
    target_wave = np.sin(2 * np.pi * 10.0 * time) * (
        1.0 + 0.2 * np.sin(2 * np.pi * 0.1 * time)
    )
    target = patterns * target_wave
    background = 0.35 * rng.normal(size=(n_channels, n_times))
    observed = target + background
    if regime == "contaminated_initialization":
        initial = min(n_times, block_size * 8)
        observed[:, :initial] += 5.0 * rng.normal(size=(n_channels, initial))
    event_block = max(2, (n_times // block_size) // 2)
    return observed, target, patterns, event_block


def _offline(observed, sfreq):
    model = DSS(
        BandpassBias((8.0, 12.0), sfreq),
        n_components=1,
        normalize_input=False,
        return_type="raw",
        verbose=False,
    ).fit(observed)
    return np.asarray(model.transform(observed)), model


def _online(method, observed, sfreq, block_size, regime, event_block):
    adaptive = method == "adaptive_dss"
    model = ContinuousDSS(
        observed.shape[0],
        sfreq,
        bias="bandpass",
        freq_band=(8.0, 12.0),
        n_components=1,
        lambda_baseline=0.99 if adaptive else 0.995,
        lambda_biased=0.95 if adaptive else 0.99,
        solve_interval=2 if adaptive else 10,
        warmup_blocks=8,
        block_size=block_size,
        channel_names=[f"EEG{index:02d}" for index in range(observed.shape[0])],
        mode="enhance",
        experimental=True,
    ).fit()
    outputs = []
    filter_history = []
    failure_detected = False
    names = list(model.channel_names_)
    for block_index, start in enumerate(range(0, observed.shape[1], block_size)):
        block = observed[:, start : start + block_size]
        if regime == "missing_block" and block_index == event_block:
            missing = block.copy()
            missing[:] = np.nan
            model.process_block(missing, names)
            output = block.copy()
            failure_detected = model.failure_counts_["nonfinite_block"] > 0
        elif regime == "reordered_block" and block_index == event_block:
            try:
                model.process_block(block[::-1], names[::-1])
            except ValueError:
                failure_detected = True
            output = model.process_block(block, names)
        else:
            output = model.process_block(block, names)
        outputs.append(output)
        filter_history.append(None if model.filters_ is None else model.filters_[0].copy())
    return np.concatenate(outputs, axis=1), model, filter_history, failure_detected


def _angle(left, right):
    left = left / max(np.linalg.norm(left), _EPS)
    right = right / max(np.linalg.norm(right), _EPS)
    return float(np.degrees(np.arccos(np.clip(abs(left @ right), 0.0, 1.0))))


def _settling_time(
    history,
    reference,
    *,
    start_block,
    block_size,
    sfreq,
    threshold_degrees,
    consecutive_blocks,
):
    angles = [np.nan if value is None else _angle(value, reference) for value in history]
    for index in range(int(start_block), len(angles) - int(consecutive_blocks) + 1):
        window = np.asarray(angles[index : index + int(consecutive_blocks)])
        if np.all(np.isfinite(window)) and np.all(window <= float(threshold_degrees)):
            return float((index - int(start_block)) * block_size / sfreq)
    return None


def _boundary_discontinuity(output, block_size):
    starts = np.arange(block_size, output.shape[1], block_size, dtype=int)
    if not starts.size:
        return 0.0
    boundary = output[:, starts] - output[:, starts - 1]
    differences = np.diff(output, axis=1)
    keep = np.ones(differences.shape[1], dtype=bool)
    keep[starts - 1] = False
    denominator = np.sqrt(np.mean(differences[:, keep] ** 2)) if np.any(keep) else 0.0
    return float(np.sqrt(np.mean(boundary**2)) / max(denominator, _EPS))


def _metrics(
    output,
    target,
    method,
    model,
    offline_model,
    history,
    block_size,
    sfreq,
    failure_detected,
    event_block,
    regime,
    settling_angle,
    settling_consecutive,
):
    final_filter = offline_model.filters_[0] if method == "offline_dss" else model.filters_[0]
    diagnostics = {} if method == "offline_dss" else model.get_diagnostics()
    angles = [
        _angle(value, offline_model.filters_[0])
        for value in history
        if value is not None
    ]
    settling_start = event_block if regime in {
        "abrupt_target_change", "missing_block", "reordered_block", "gradual_spatial_drift"
    } else 0
    settling = (
        0.0
        if method == "offline_dss"
        else _settling_time(
            history,
            offline_model.filters_[0],
            start_block=settling_start,
            block_size=block_size,
            sfreq=sfreq,
            threshold_degrees=settling_angle,
            consecutive_blocks=settling_consecutive,
        )
    )
    return {
        "method": method,
        "status": "success",
        "subspace_angle_to_offline_reference": _angle(final_filter, offline_model.filters_[0]),
        "known_clean_waveform_relative_rmse": relative_rmse(output, target),
        "target_correlation": clean_correlation(output, target),
        "settling_time_s": settling,
        "boundary_discontinuity": _boundary_discontinuity(output, block_size),
        "algorithmic_latency_ms": float(block_size / sfreq * 1000.0),
        "end_to_end_latency_ms": float(diagnostics.get("mean_processing_time_s", 0.0) * 1000.0),
        "realtime_factor": float(diagnostics.get("real_time_factor", 0.0)),
        "failure_detected": bool(failure_detected),
        "n_solves": int(diagnostics.get("n_solves", 1)),
        "mean_online_subspace_angle": float(np.mean(angles)) if angles else 0.0,
    }


def _write_tsv(path, rows):
    fields = sorted({key for row in rows for key in row})
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with pathlib.Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    config_path = pathlib.Path(args.config).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert_submission_ready(cfg, source=str(config_path))
    spec = cfg["simulation"]
    regimes = list(spec["regimes"])
    block_sizes = list(spec["block_sizes"])
    repetitions = int(spec["replicates_per_cell"])
    duration = float(spec["duration_s"])
    if args.smoke:
        duration, repetitions = 12.0, 1
        block_sizes = [64]
    methods = [*cfg["methods_under_test"], *cfg["comparators"]["required"]]
    root = pathlib.Path(args.output_root).resolve()
    sfreq = float(spec["sfreq_hz"])
    rows = []
    for regime in regimes:
        for block_size in block_sizes:
            for replicate in range(repetitions):
                seed = locked_seed(spec["seeds"]["global"], cfg["arm"], regime, block_size, replicate)
                observed, target, _, event_block = _replay_substrate(regime, sfreq, duration, block_size, seed)
                offline_output, offline_model = _offline(observed, sfreq)
                unit_id = f"{regime}_block{block_size}_seed{replicate:03d}"
                for method in methods:
                    method_dir = root / unit_id / method
                    record = build_run_record(
                        arm=cfg["arm"], method=method, unit_id=unit_id,
                        config_path=config_path, dataset_manifest=args.dataset_manifest,
                        repo_root=_REPO, seed=seed,
                        information_tier="full_recording" if method == "offline_dss" else "causal_local",
                        allow_dirty=args.allow_dirty,
                    )
                    try:
                        with AttemptRecorder(method_dir, record):
                            if method == "offline_dss":
                                output, model, history, detected = offline_output, offline_model, [], False
                            else:
                                output, model, history, detected = _online(method, observed, sfreq, block_size, regime, event_block)
                            metrics = _metrics(
                                output, target, method, model, offline_model, history,
                                block_size, sfreq, detected, event_block, regime,
                                float(spec["settling_angle_degrees"]),
                                int(spec["settling_consecutive_blocks"]),
                            )
                            metrics.update({"unit_id": unit_id, "regime": regime, "block_size": block_size, "replicate": replicate, "seed": seed})
                            method_dir.mkdir(parents=True, exist_ok=True)
                            (method_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
                            (method_dir / "model.json").write_text(json.dumps({"class": type(model).__name__, "adaptive_state_changes_during_transform": method != "offline_dss"}, indent=2), encoding="utf-8")
                            rows.append(metrics)
                    except Exception as error:
                        rows.append({"method": method, "status": "failed", "unit_id": unit_id, "seed": seed, "error": f"{type(error).__name__}: {error}"})
    _write_tsv(root / "raw_metrics.tsv", rows)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(_REPO / "configs/benchmarks/streaming_replay.yaml"))
    parser.add_argument("--dataset-manifest", default=str(_REPO / "configs/manifests/synthetic_v1.json"))
    parser.add_argument("--output-root", default=str(_REPO / "results/streaming_replay"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)
    rows = run(args)
    success = sum(row.get("status") == "success" for row in rows)
    print(f"attempts={len(rows)} successes={success} output={args.output_root}")
    return 0 if success == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
