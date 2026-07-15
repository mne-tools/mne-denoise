#!/usr/bin/env python
"""Run known-source sensor/forward-model validation on MNE sample geometry."""

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
    clean_multichannel_eeg,
    locked_seed,
    relative_rmse,
)
from mne_denoise.benchmarks.provenance import AttemptRecorder, build_run_record
from mne_denoise.benchmarks.sharding import add_shard_arguments, args_select_unit
from mne_denoise.mwf import MWF
from mne_denoise.sound import SOUND
from mne_denoise.sspsir import SSPSIR

_EPS = float(np.finfo(np.float64).eps)


def _load_leadfield(modality, max_channels, n_sources, seed, mismatch_mm):
    import mne

    sample = pathlib.Path(mne.datasets.sample.data_path(download=False))
    path = sample / "MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif"
    if not path.is_file():
        raise FileNotFoundError(f"MNE sample forward not found: {path}")
    forward = mne.read_forward_solution(path, verbose=False)
    forward = mne.convert_forward_solution(
        forward, surf_ori=True, force_fixed=True, use_cps=True, verbose=False
    )
    forward = mne.pick_types_forward(
        forward,
        meg=modality == "meg",
        eeg=modality == "eeg",
        exclude=[],
    )
    gain = np.asarray(forward["sol"]["data"], dtype=float)
    if gain.shape[0] > max_channels:
        indices = np.linspace(0, gain.shape[0] - 1, max_channels, dtype=int)
        gain = gain[indices]
    coordinates = np.vstack(
        [space["rr"][space["vertno"]] for space in forward["src"]]
    )
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(gain.shape[1], n_sources, replace=False))
    true = gain[:, selected]
    if mismatch_mm <= 0:
        fitted = true.copy()
    else:
        fitted_columns = []
        target = mismatch_mm / 1000.0
        for index in selected:
            distances = np.linalg.norm(coordinates - coordinates[index], axis=1)
            distances[index] = np.inf
            neighbor = int(np.argmin(np.abs(distances - target)))
            fitted_columns.append(gain[:, neighbor])
        fitted = np.column_stack(fitted_columns)
    true -= true.mean(axis=0, keepdims=True)
    fitted -= fitted.mean(axis=0, keepdims=True)
    return true, fitted


def _substrate(modality, artifact_type, severity, cfg, seed, mismatch):
    spec = cfg["simulation"]
    sfreq = float(spec["sfreq_hz"])
    n_times = int(round(float(spec["duration_s"]) * sfreq))
    n_sources = int(spec["cortical_sources_per_replicate"])
    true_leadfield, fitted_leadfield = _load_leadfield(
        modality,
        int(spec["max_channels_per_modality"]),
        n_sources,
        locked_seed(seed, "forward"),
        float(spec["source_displacement_mm"][mismatch]),
    )
    sources = clean_multichannel_eeg(
        n_channels=n_sources,
        n_times=n_times,
        sfreq=sfreq,
        seed=locked_seed(seed, "sources"),
    )
    clean = true_leadfield @ sources
    clean /= max(np.sqrt(np.mean(clean**2)), _EPS)
    sources /= max(np.sqrt(np.mean((true_leadfield @ sources) ** 2)), _EPS)
    rng = np.random.default_rng(locked_seed(seed, "artifact"))
    artifact = np.zeros_like(clean)
    mask = np.ones(n_times, dtype=bool)
    if artifact_type == "sensor_local":
        active = rng.choice(clean.shape[0], max(1, clean.shape[0] // 10), replace=False)
        artifact[active] = rng.normal(size=(len(active), n_times))
    elif artifact_type == "spatially_correlated_environmental":
        patterns = rng.normal(size=(clean.shape[0], 3))
        # Make the environmental field explicitly forward-inconsistent.
        projector = true_leadfield @ np.linalg.pinv(true_leadfield)
        patterns = (np.eye(clean.shape[0]) - projector) @ patterns
        artifact = patterns @ rng.normal(size=(3, n_times))
    elif artifact_type == "masked_broadband":
        mask = np.zeros(n_times, dtype=bool)
        for start in np.linspace(0.1, 0.8, 6) * n_times:
            start = int(start)
            stop = min(n_times, start + int(0.5 * sfreq))
            mask[start:stop] = True
        artifact[:, mask] = rng.normal(size=(clean.shape[0], int(mask.sum())))
    elif artifact_type == "tms_pulse":
        mask = np.zeros(n_times, dtype=bool)
        pattern = rng.normal(size=(clean.shape[0], 2))
        for start in np.linspace(0.1, 0.85, 8) * n_times:
            start = int(start)
            stop = min(n_times, start + int(0.08 * sfreq))
            waveform = np.exp(-np.arange(stop - start) / max(1.0, 0.01 * sfreq))
            artifact[:, start:stop] += pattern @ np.vstack([waveform, -np.gradient(waveform)])
            mask[start:stop] = True
    else:
        raise KeyError(artifact_type)
    artifact *= (10.0 ** (float(severity) / 20.0)) * np.sqrt(
        np.mean(clean[:, mask] ** 2)
    ) / max(np.sqrt(np.mean(artifact[:, mask] ** 2)), _EPS)
    return {
        "sfreq": sfreq,
        "sources": sources,
        "clean": clean,
        "artifact": artifact,
        "contaminated": clean + artifact,
        "mask": mask,
        "true_leadfield": true_leadfield,
        "fitted_leadfield": fitted_leadfield,
    }


def _sns(data, n_neighbors=10):
    covariance = data @ data.T / data.shape[1]
    operator = np.zeros((data.shape[0], data.shape[0]))
    for index in range(data.shape[0]):
        candidates = np.delete(np.arange(data.shape[0]), index)
        correlations = np.abs(covariance[index, candidates]) / np.maximum(
            np.sqrt(covariance[index, index] * np.diag(covariance)[candidates]), _EPS
        )
        neighbors = candidates[np.argsort(correlations)[-min(n_neighbors, len(candidates)) :]]
        coefficients = np.linalg.solve(
            covariance[np.ix_(neighbors, neighbors)]
            + 1e-8 * np.trace(covariance) / data.shape[0] * np.eye(len(neighbors)),
            covariance[neighbors, index],
        )
        operator[index, neighbors] = coefficients
    return operator @ data, operator


def _clean(method, substrate):
    data = substrate["contaminated"]
    model = None
    if method == "none":
        output = data.copy()
    elif method == "pca_rank_matched":
        left, _, _ = np.linalg.svd(data - data.mean(axis=1, keepdims=True), full_matrices=False)
        basis = left[:, : substrate["sources"].shape[0]]
        output = basis @ basis.T @ data
        model = {"basis": basis}
    elif method == "sns":
        output, operator = _sns(data)
        model = {"operator": operator}
    elif method == "sound":
        model = SOUND(
            forward={"sol": {"data": substrate["fitted_leadfield"]}},
            random_state=0,
        ).fit(data)
        output = np.asarray(model.transform(data))
    elif method == "mwf":
        model = MWF(sfreq=substrate["sfreq"])
        output = model.fit_transform(data, mask=substrate["mask"])
    elif method == "ssp_sir":
        model = SSPSIR(
            n_components=2,
            high_pass=min(80.0, substrate["sfreq"] / 2 - 1),
            forward={"sol": {"data": substrate["fitted_leadfield"]}},
            sfreq=substrate["sfreq"],
        ).fit(data)
        output = np.asarray(model.transform(data))
    elif method == "ssp":
        artifact_data = data[:, substrate["mask"]]
        covariance = artifact_data @ artifact_data.T / max(artifact_data.shape[1], 1)
        values, vectors = np.linalg.eigh(covariance)
        basis = vectors[:, np.argsort(values)[-2:]]
        output = (np.eye(data.shape[0]) - basis @ basis.T) @ data
        model = {"basis": basis, "mask_source": "known_artifact_intervals"}
    else:
        raise KeyError(method)
    return output, model


def _score(output, substrate, method):
    clean = substrate["clean"]
    artifact = substrate["artifact"]
    mask = substrate["mask"]
    source_estimate = np.linalg.pinv(substrate["true_leadfield"]) @ output
    residual = output[:, mask] - clean[:, mask]
    return {
        "method": method,
        "status": "success",
        "known_artifact_residual_energy": float(
            np.sum(residual**2) / max(np.sum(artifact[:, mask] ** 2), _EPS)
        ),
        "known_cortical_source_relative_rmse": relative_rmse(
            source_estimate, substrate["sources"]
        ),
        "sensor_rmse": relative_rmse(output, clean),
        "source_correlation": clean_correlation(source_estimate, substrate["sources"]),
        "topographic_correlation": clean_correlation(output, clean),
        "spectral_distortion": relative_rmse(
            np.abs(np.fft.rfft(output, axis=1)),
            np.abs(np.fft.rfft(clean, axis=1)),
        ),
        "rank_before": int(np.linalg.matrix_rank(substrate["contaminated"])),
        "rank_after": int(np.linalg.matrix_rank(output)),
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
    modalities = list(spec["modalities"])
    artifacts = list(spec["artifact_types"])
    severities = list(spec["severity_db"])
    mismatches = list(spec["leadfield_mismatch"])
    repetitions = int(spec["replicates_per_primary_cell"])
    if args.smoke:
        modalities, severities, mismatches, repetitions = ["eeg"], [0], ["none"], 1
        spec["duration_s"] = 4
        spec["max_channels_per_modality"] = 16
        spec["cortical_sources_per_replicate"] = 6
    methods = [
        *cfg["methods_under_test"],
        *cfg["comparators"]["required"],
        *cfg["comparators"]["secondary"],
    ]
    root = pathlib.Path(args.output_root).resolve()
    rows = []
    for modality in modalities:
        for artifact in artifacts:
            for severity in severities:
                for mismatch in mismatches:
                    for replicate in range(repetitions):
                        seed = locked_seed(spec["seeds"]["global"], cfg["arm"], modality, artifact, severity, mismatch, replicate)
                        substrate = _substrate(modality, artifact, severity, cfg, seed, mismatch)
                        unit_id = f"{modality}_{artifact}_db{severity}_{mismatch}_seed{replicate:03d}"
                        if not args_select_unit(args, unit_id):
                            continue
                        for method in methods:
                            method_dir = root / unit_id / method
                            tier = (
                                "artifact_mask_aware"
                                if method in {"mwf", "ssp"}
                                else "forward_aware"
                                if method in {"sound", "ssp_sir"}
                                else "blind"
                            )
                            record = build_run_record(
                                arm=cfg["arm"], method=method, unit_id=unit_id,
                                config_path=config_path, dataset_manifest=args.dataset_manifest,
                                repo_root=_REPO, seed=seed, information_tier=tier,
                                allow_dirty=args.allow_dirty,
                            )
                            try:
                                with AttemptRecorder(method_dir, record) as active:
                                    output, model = _clean(method, substrate)
                                    metrics = _score(output, substrate, method)
                                    intended = cfg["intended_regime"].get(method)
                                    intended_match = (
                                        artifact in intended
                                        if isinstance(intended, list)
                                        else artifact == intended
                                    )
                                    metrics.update({
                                        "unit_id": unit_id, "modality": modality,
                                        "artifact_type": artifact, "severity_db": severity,
                                        "leadfield_mismatch": mismatch, "replicate": replicate,
                                        "seed": seed,
                                        "intended_regime_match": intended_match,
                                        "information_tier": tier,
                                    })
                                    active.effective_rank_before = metrics["rank_before"]
                                    active.effective_rank_after = metrics["rank_after"]
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
    parser.add_argument("--config", default=str(_REPO / "configs/benchmarks/sensor_forward_ground_truth.yaml"))
    parser.add_argument("--dataset-manifest", default=str(_REPO / "configs/manifests/synthetic_v1.json"))
    parser.add_argument("--output-root", default=str(_REPO / "results/sensor_forward_ground_truth"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    add_shard_arguments(parser)
    args = parser.parse_args(argv)
    rows = run(args)
    success = sum(row.get("status") == "success" for row in rows)
    print(f"attempts={len(rows)} successes={success} output={args.output_root}")
    return 0 if success == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
