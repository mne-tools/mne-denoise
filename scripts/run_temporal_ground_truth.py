#!/usr/bin/env python
"""Run intended-regime temporal and time-frequency known-target validation."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

import numpy as np
import yaml
from scipy import signal

_REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from mne_denoise.benchmarks.config import assert_submission_ready
from mne_denoise.benchmarks.intended import (
    clean_correlation,
    clean_multichannel_eeg,
    locked_seed,
    relative_rmse,
)
from mne_denoise.benchmarks.provenance import AttemptRecorder, build_run_record
from mne_denoise.benchmarks.sharding import add_shard_arguments, args_select_unit
from mne_denoise.dss.denoisers import QuasiPeriodicDenoiser
from mne_denoise.emd import EMDDenoiser
from mne_denoise.ssa import SSA
from mne_denoise.wavelet import WaveletICA, wavelet_denoise_multichannel

_EPS = float(np.finfo(np.float64).eps)


def _substrate(artifact_type, ratio_db, sfreq, duration, seed, n_channels):
    rng = np.random.default_rng(locked_seed(seed, "artifact"))
    n_times = int(round(sfreq * duration))
    clean = clean_multichannel_eeg(n_channels=n_channels, n_times=n_times, sfreq=sfreq, seed=locked_seed(seed, "clean"))
    time = np.arange(n_times) / sfreq
    erp_events = np.arange(int(sfreq), n_times - int(sfreq), int(2 * sfreq))
    erp_pattern = rng.normal(size=n_channels)
    erp_pattern /= np.linalg.norm(erp_pattern)
    erp_template = np.zeros(n_times)
    half = max(2, int(0.2 * sfreq))
    pulse = signal.windows.gaussian(2 * half + 1, std=max(1, 0.04 * sfreq))
    for event in erp_events:
        erp_template[event - half : event + half + 1] += pulse
    clean += 0.4 * np.outer(erp_pattern, erp_template)
    artifact = np.zeros_like(clean)
    pattern = rng.normal(size=n_channels)
    pattern /= np.linalg.norm(pattern)
    if artifact_type == "quasiperiodic":
        waveform = signal.sawtooth(2 * np.pi * 1.2 * time, width=0.15)
        waveform *= 0.7 + 0.3 * np.sin(2 * np.pi * 0.03 * time)
        artifact = np.outer(pattern, waveform)
    elif artifact_type == "chirp":
        artifact = np.outer(pattern, signal.chirp(time, 3.0, time[-1], 45.0))
    elif artifact_type == "drift":
        artifact = np.outer(pattern, signal.sawtooth(2 * np.pi * 0.15 * time, width=0.8))
    elif artifact_type == "transient":
        for event in np.linspace(0.1, 0.9, 8) * n_times:
            start = int(event)
            stop = min(n_times, start + int(0.15 * sfreq))
            artifact[:, start:stop] += np.outer(pattern, signal.windows.tukey(stop - start))
    elif artifact_type == "broadband_muscle":
        mask = np.zeros(n_times, dtype=bool)
        for event in np.linspace(0.1, 0.85, 7) * n_times:
            start = int(event)
            stop = min(n_times, start + int(0.5 * sfreq))
            mask[start:stop] = True
        high = signal.butter(3, 25.0, fs=sfreq, btype="high", output="sos")
        noise = signal.sosfilt(high, rng.normal(size=clean.shape), axis=1)
        artifact[:, mask] = noise[:, mask]
    else:
        raise KeyError(artifact_type)
    artifact *= 10.0 ** (float(ratio_db) / 20.0) * np.sqrt(np.mean(clean**2)) / max(np.sqrt(np.mean(artifact**2)), _EPS)
    return clean, clean + artifact, artifact, erp_events


def _eemd_cca(data, params, smoke):
    from PyEMD import EEMD

    from mne_denoise.cca import AutoCCA

    outputs = []
    for channel in data:
        eemd = EEMD(
            trials=1 if smoke else int(params["ensemble_trials"]),
            parallel=False,
        )
        eemd.noise_seed(0)
        imfs = eemd.eemd(channel, max_imf=int(params["maximum_imfs"]))
        if imfs.ndim < 2 or imfs.shape[0] < 2:
            outputs.append(channel)
        else:
            cleaned = AutoCCA(rho_threshold=float(params["canonical_correlation_threshold"])).fit_transform(imfs)
            outputs.append(cleaned.sum(axis=0))
    return np.asarray(outputs)


def _clean(method, data, sfreq, cfg, smoke):
    params = cfg["method_parameters"]
    if method == "none":
        return data.copy(), None
    if method == "ssa":
        p = params["ssa"]
        model = SSA(sfreq, window_length=int(p["window_length_samples"]), drop_freq_max=float(p["drop_frequency_max_hz"]), n_check=int(p["checked_eigentriples"]))
        return model.fit_transform(data), model
    if method == "quasiperiodic_denoiser":
        p = params["quasiperiodic_denoiser"]
        model = QuasiPeriodicDenoiser(peak_distance=int(p["minimum_peak_distance_samples"]), peak_height_percentile=float(p["peak_height_percentile"]))
        estimate = np.stack([model.denoise(channel) for channel in data])
        return data - estimate, model
    if method == "wavelet_threshold":
        return wavelet_denoise_multichannel(data, params["wavelet_threshold"]["wavelet"]), None
    if method == "wica":
        p = params["wica"]
        model = WaveletICA(wavelet=p["wavelet"], max_iter=int(p["max_iterations"])).fit(data)
        return model.transform(data), model
    if method in {"emd", "eemd"}:
        p = params[method]
        model = EMDDenoiser(sfreq, method=method, freq_cutoff=float(p["frequency_cutoff_hz"]), max_imf=int(p["maximum_imfs"]), trials=1 if smoke else int(p.get("ensemble_trials", 8)))
        return model.fit_transform(data), model
    if method == "eemd_cca":
        return _eemd_cca(data, params[method], smoke), None
    raise KeyError(method)


def _score(output, clean, artifact, erp_events, sfreq, method):
    residual = output - clean
    half = max(1, int(0.05 * sfreq))
    clean_erp = np.mean([clean[:, event - half : event + half + 1].mean(axis=1) for event in erp_events], axis=0)
    output_erp = np.mean([output[:, event - half : event + half + 1].mean(axis=1) for event in erp_events], axis=0)
    return {
        "method": method,
        "status": "success",
        "known_artifact_residual_energy": float(np.sum(residual**2) / max(np.sum(artifact**2), _EPS)),
        "known_clean_waveform_relative_rmse": relative_rmse(output, clean),
        "clean_correlation": clean_correlation(output, clean),
        "spectral_distortion": relative_rmse(np.abs(np.fft.rfft(output, axis=1)), np.abs(np.fft.rfft(clean, axis=1))),
        "erp_amplitude_error": relative_rmse(output_erp, clean_erp),
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
    artifacts = list(spec["artifact_types"])
    ratios = list(spec["artifact_to_signal_db"])
    repetitions = int(spec["replicates_per_cell"])
    duration = float(spec["duration_s"])
    n_channels = int(spec["n_channels"])
    if args.smoke:
        ratios, repetitions, duration, n_channels = [0], 1, 3.0, 2
    methods = [*cfg["methods_under_test"], *cfg["comparators"]["required"]]
    sfreq = float(spec["sfreq_hz"])
    root = pathlib.Path(args.output_root).resolve()
    rows = []
    for artifact_type in artifacts:
        for ratio in ratios:
            for replicate in range(repetitions):
                seed = locked_seed(spec["seeds"]["global"], cfg["arm"], artifact_type, ratio, replicate)
                clean, contaminated, artifact, events = _substrate(
                    artifact_type,
                    ratio,
                    sfreq,
                    duration,
                    seed,
                    n_channels,
                )
                unit_id = f"{artifact_type}_db{ratio}_seed{replicate:03d}"
                if not args_select_unit(args, unit_id):
                    continue
                for method in methods:
                    method_dir = root / unit_id / method
                    record = build_run_record(arm=cfg["arm"], method=method, unit_id=unit_id, config_path=config_path, dataset_manifest=args.dataset_manifest, repo_root=_REPO, seed=seed, information_tier="recording_local", allow_dirty=args.allow_dirty)
                    try:
                        with AttemptRecorder(method_dir, record) as active:
                            intended = artifact_type in np.atleast_1d(
                                cfg["intended_regime"].get(method, [])
                            ).tolist()
                            if (
                                cfg["execution_policy"][
                                    "run_only_declared_intended_regimes"
                                ]
                                and method != "none"
                                and not intended
                            ):
                                active.status = cfg["execution_policy"][
                                    "off_regime_status"
                                ]
                                model = None
                                metrics = {
                                    "method": method,
                                    "status": active.status,
                                }
                            else:
                                try:
                                    output, model = _clean(
                                        method,
                                        contaminated,
                                        sfreq,
                                        cfg,
                                        args.smoke,
                                    )
                                    metrics = _score(
                                        output,
                                        clean,
                                        artifact,
                                        events,
                                        sfreq,
                                        method,
                                    )
                                except (ImportError, ModuleNotFoundError) as error:
                                    active.status = "unavailable_dependency"
                                    model = None
                                    metrics = {
                                        "method": method,
                                        "status": "unavailable_dependency",
                                        "error": f"{type(error).__name__}: {error}",
                                    }
                            metrics.update({"unit_id": unit_id, "artifact_type": artifact_type, "artifact_to_signal_db": ratio, "replicate": replicate, "seed": seed, "intended_regime_match": intended})
                            method_dir.mkdir(parents=True, exist_ok=True)
                            (method_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
                            (method_dir / "model.json").write_text(json.dumps({"class": type(model).__name__ if model is not None else None}, indent=2), encoding="utf-8")
                            rows.append(metrics)
                    except Exception as error:
                        rows.append({"method": method, "status": "failed", "unit_id": unit_id, "seed": seed, "error": f"{type(error).__name__}: {error}"})
    _write_tsv(root / "raw_metrics.tsv", rows)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(_REPO / "configs/benchmarks/temporal_ground_truth.yaml"))
    parser.add_argument("--dataset-manifest", default=str(_REPO / "configs/manifests/synthetic_v1.json"))
    parser.add_argument("--output-root", default=str(_REPO / "results/temporal_ground_truth"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    add_shard_arguments(parser)
    args = parser.parse_args(argv)
    rows = run(args)
    accepted = sum(
        row.get("status")
        in {
            "success",
            "unavailable_dependency",
            "skipped_outside_intended_regime",
        }
        for row in rows
    )
    print(f"attempts={len(rows)} accepted={accepted} output={args.output_root}")
    return 0 if accepted == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
