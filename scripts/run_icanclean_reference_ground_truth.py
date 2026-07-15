#!/usr/bin/env python
"""Run the frozen reference-quality and neural-leakage simulation."""

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
    locked_seed,
    reference_mixture,
    relative_rmse,
)
from mne_denoise.benchmarks.provenance import AttemptRecorder, build_run_record
from mne_denoise.benchmarks.sharding import add_shard_arguments, args_select_unit
from mne_denoise.cca import AutoCCA
from mne_denoise.dss import DSS, ReferenceBias
from mne_denoise.icanclean import ICanClean
from mne_denoise.mwf import MWF
from mne_denoise.qa.coupling import regress_out

_EPS = float(np.finfo(np.float64).eps)


def _pseudo_reference(data: np.ndarray, sfreq: float, n_components: int, highpass: float):
    sos = signal.butter(3, highpass, fs=sfreq, btype="high", output="sos")
    high = signal.sosfiltfilt(sos, data, axis=1)
    left, _, _ = np.linalg.svd(high, full_matrices=False)
    return left[:, : min(n_components, left.shape[1])].T @ high


def _icanclean(primary, reference, sfreq, params):
    combined = np.vstack([primary, reference])
    n_primary = primary.shape[0]
    model = ICanClean(
        sfreq=sfreq,
        primary_channels=list(range(n_primary)),
        ref_channels=list(range(n_primary, combined.shape[0])),
        mode=params["mode"],
        clean_with=params["clean_with"],
        segment_len=float(params["segment_len_s"]),
        overlap=float(params["overlap"]),
        threshold=float(params["r_squared_threshold"]),
        max_reject_fraction=float(params["max_reject_fraction"]),
        verbose=False,
    )
    output = np.asarray(model.fit_transform(combined))[:n_primary]
    return output, model, int(np.sum(model.n_removed_))


def _clean(method, mixture, sfreq, cfg):
    data = mixture.contaminated
    reference = mixture.reference_fit
    params = cfg["method_parameters"]
    model = None
    removed = None
    if method == "none":
        output = data.copy()
    elif method == "oracle":
        output = mixture.clean.copy()
    elif method == "regression":
        output = regress_out(data, reference)
    elif method == "tspca":
        shifts = int(params["tspca"]["n_shifts_each_direction"])
        lagged = np.vstack([np.roll(reference, shift, axis=1) for shift in range(-shifts, shifts + 1)])
        output = regress_out(data, lagged)
    elif method == "icanclean_dual_layer":
        output, model, removed = _icanclean(data, reference, sfreq, params["icanclean"])
    elif method == "icanclean_pseudoreference":
        pseudo = _pseudo_reference(
            data,
            sfreq,
            int(params["pseudoreference"]["pca_components"]),
            float(params["pseudoreference"]["highpass_hz"]),
        )
        output, model, removed = _icanclean(data, pseudo, sfreq, params["icanclean"])
    elif method == "reference_bias_dss":
        count = min(
            reference.shape[0],
            int(params["reference_bias_dss"]["max_components"]),
        )
        model = DSS(
            bias=ReferenceBias(
                reference, ridge=float(params["reference_bias_dss"]["ridge"])
            ),
            n_components=count,
            normalize_input=False,
            return_type="raw",
            verbose=False,
        ).fit(data)
        artifact_estimate = np.asarray(model.transform(data))
        output = data - (artifact_estimate - data.mean(axis=1, keepdims=True))
        removed = count
    elif method == "autocca":
        model = AutoCCA(
            rho_threshold=float(params["autocca"]["rho_threshold"])
        ).fit(data)
        output = model.transform(data)
        removed = data.shape[0] - int(model.n_kept_)
    elif method == "mwf":
        envelope = np.mean(reference**2, axis=0)
        mask = envelope > np.quantile(
            envelope, float(params["mwf"]["reference_envelope_quantile"])
        )
        model = MWF(sfreq=sfreq, reg=float(params["mwf"]["regularization"]))
        output = model.fit_transform(data, mask=mask)
    else:
        raise KeyError(method)
    return np.asarray(output), model, removed


def _band_power(data, sfreq, low=8.0, high=30.0):
    frequencies, psd = signal.welch(data, sfreq, nperseg=min(512, data.shape[1]), axis=1)
    keep = (frequencies >= low) & (frequencies <= high)
    return float(np.trapezoid(psd[:, keep], frequencies[keep], axis=1).mean())


def _max_reference_correlation(data, reference):
    data = data - data.mean(axis=1, keepdims=True)
    reference = reference - reference.mean(axis=1, keepdims=True)
    data /= np.maximum(np.linalg.norm(data, axis=1, keepdims=True), _EPS)
    reference /= np.maximum(np.linalg.norm(reference, axis=1, keepdims=True), _EPS)
    return float(np.max(np.abs(data @ reference.T)))


def _score(output, mixture, method, removed, sfreq):
    uncleaned_error = np.sqrt(np.mean((mixture.contaminated - mixture.clean) ** 2))
    cleaning_error = np.sqrt(np.mean((output - mixture.clean) ** 2))
    return {
        "method": method,
        "status": "success",
        "ground_truth_data_quality_score": float(
            100.0 * (1.0 - cleaning_error / max(uncleaned_error, _EPS))
        ),
        "known_clean_waveform_relative_rmse": relative_rmse(output, mixture.clean),
        "clean_correlation": clean_correlation(output, mixture.clean),
        "neural_band_retention": _band_power(output, sfreq)
        / max(_band_power(mixture.clean, sfreq), _EPS),
        "disjoint_reference_coupling": _max_reference_correlation(
            output, mixture.reference_score
        ),
        "rank_before": int(np.linalg.matrix_rank(mixture.contaminated)),
        "rank_after": int(np.linalg.matrix_rank(output)),
        "removed_components": removed,
    }


def _write_tsv(path: pathlib.Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(".tsv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run(args):
    config_path = pathlib.Path(args.config).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert_submission_ready(cfg, source=str(config_path))
    simulation = cfg["simulation"]
    sfreq = float(simulation["sfreq_hz"])
    duration = float(simulation["duration_s"])
    n_channels = int(simulation["n_primary_channels"])
    repetitions = int(simulation["replicates_per_cell"])
    snrs = list(simulation["reference_snr_db"])
    counts = list(simulation["reference_count"])
    leakage = list(simulation["neural_leakage_fraction"])
    coupling = list(simulation["coupling"])
    if args.smoke:
        duration, n_channels, repetitions = 8.0, 16, 1
        snrs, counts = [0], [4]
        leakage = [leakage[0], leakage[-1]]
    methods = list(
        dict.fromkeys(
            [
                *cfg["tiers"]["reference_free"],
                *cfg["tiers"]["reference_aware"],
                *cfg["comparators"]["required"],
            ]
        )
    )
    root = pathlib.Path(args.output_root).resolve()
    global_seed = int(simulation["seeds"]["global"])
    rows = []
    for snr in snrs:
        for count in counts:
            for leak in leakage:
                for drift in coupling:
                    for replicate in range(repetitions):
                        seed = locked_seed(
                            global_seed, cfg["arm"], snr, count, leak, drift, replicate
                        )
                        mixture = reference_mixture(
                            reference_snr_db=float(snr),
                            reference_count=int(count),
                            neural_leakage_fraction=float(leak),
                            coupling=drift,
                            n_channels=n_channels,
                            n_times=int(round(duration * sfreq)),
                            sfreq=sfreq,
                            seed=seed,
                        )
                        unit_id = f"snr{snr}_r{count}_leak{leak}_{drift}_seed{replicate:03d}"
                        if not args_select_unit(args, unit_id):
                            continue
                        for method in methods:
                            tier = (
                                "oracle"
                                if method == "oracle"
                                else "reference_aware"
                                if method in cfg["tiers"]["reference_aware"]
                                else "reference_free"
                            )
                            method_dir = root / unit_id / method
                            record = build_run_record(
                                arm=cfg["arm"],
                                method=method,
                                unit_id=unit_id,
                                config_path=config_path,
                                dataset_manifest=args.dataset_manifest,
                                repo_root=_REPO,
                                seed=seed,
                                information_tier=tier,
                                allow_dirty=args.allow_dirty,
                            )
                            try:
                                with AttemptRecorder(method_dir, record) as active:
                                    output, model, removed = _clean(method, mixture, sfreq, cfg)
                                    metrics = _score(output, mixture, method, removed, sfreq)
                                    metrics.update(
                                        {
                                            "unit_id": unit_id,
                                            "reference_snr_db": snr,
                                            "reference_count": count,
                                            "neural_leakage_fraction": leak,
                                            "coupling": drift,
                                            "replicate": replicate,
                                            "seed": seed,
                                            "information_tier": tier,
                                        }
                                    )
                                    active.effective_rank_before = metrics["rank_before"]
                                    active.effective_rank_after = metrics["rank_after"]
                                    (method_dir / "metrics.json").write_text(
                                        json.dumps(metrics, indent=2, sort_keys=True),
                                        encoding="utf-8",
                                    )
                                    state = {
                                        "class": None if model is None else type(model).__name__,
                                        "parameters": cfg["method_parameters"].get(method, {}),
                                    }
                                    (method_dir / "model.json").write_text(
                                        json.dumps(state, indent=2, sort_keys=True, default=str),
                                        encoding="utf-8",
                                    )
                                    rows.append(metrics)
                            except Exception as error:
                                rows.append(
                                    {
                                        "method": method,
                                        "status": "failed",
                                        "unit_id": unit_id,
                                        "seed": seed,
                                        "error": f"{type(error).__name__}: {error}",
                                    }
                                )
    _write_tsv(root / "raw_metrics.tsv", rows)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=str(_REPO / "configs/benchmarks/icanclean_reference_ground_truth.yaml")
    )
    parser.add_argument(
        "--dataset-manifest", default=str(_REPO / "configs/manifests/synthetic_v1.json")
    )
    parser.add_argument(
        "--output-root", default=str(_REPO / "results/icanclean_reference_ground_truth")
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    add_shard_arguments(parser)
    args = parser.parse_args(argv)
    rows = run(args)
    successes = sum(row.get("status") == "success" for row in rows)
    print(f"attempts={len(rows)} successes={successes} output={args.output_root}")
    return 0 if successes == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
