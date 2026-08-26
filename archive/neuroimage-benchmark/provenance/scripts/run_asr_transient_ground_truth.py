#!/usr/bin/env python
"""Run the frozen paired known-truth ASR-family benchmark."""

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

from mne_denoise.asr import ASR, AdaptiveASR, GuidedASR, JugglerASR
from mne_denoise.benchmarks.config import assert_submission_ready
from mne_denoise.benchmarks.intended import (
    clean_correlation,
    locked_seed,
    relative_rmse,
    transient_mixture,
)
from mne_denoise.benchmarks.provenance import (
    AttemptRecorder,
    build_run_record,
)
from mne_denoise.benchmarks.sharding import add_shard_arguments, args_select_unit
from mne_denoise.dss.denoisers import BandpassBias
from mne_denoise.mwf import MWF
from mne_denoise.wavelet import wavelet_denoise_multichannel


def _method(method: str, artifact_type: str, sfreq: float, cutoff: float):
    common = {
        "sfreq": sfreq,
        "cutoff": cutoff,
        "calibration": "manual",
        "filter_kind": "none",
        "picks": None,
        "verbose": False,
    }
    if method == "asr":
        return ASR(method="standard", **common)
    if method == "rasr":
        return ASR(method="riemannian_windowed", **common)
    if method == "adaptive_asr":
        adaptive_common = {
            key: value
            for key, value in common.items()
            if key not in {"calibration", "filter_kind"}
        }
        return AdaptiveASR(variant="psw", **adaptive_common)
    if method == "juggler_asr":
        juggler_common = {
            key: value
            for key, value in common.items()
            if key not in {"calibration", "filter_kind"}
        }
        return JugglerASR(strategy="gev", selection_filter_kind="none", **juggler_common)
    if method == "guided_asr":
        bands = {
            "blink": (0.7, 4.0),
            "emg_burst": (30.0, min(90.0, sfreq * 0.45)),
            "electrode_pop": (0.7, 8.0),
            "low_rank_covariance_burst": (0.7, min(45.0, sfreq * 0.45)),
        }
        return GuidedASR(
            method="riemannian_windowed",
            experimental=True,
            reconstruction="soft",
            artifact_biases=[BandpassBias(bands[artifact_type], sfreq)],
            preserve_biases=[BandpassBias((8.0, 12.0), sfreq)],
            **{key: value for key, value in common.items() if key != "experimental"},
        )
    if method == "none":
        return None
    raise KeyError(method)


def _f1(truth: np.ndarray, prediction: np.ndarray | None) -> float | None:
    if prediction is None or prediction.shape != truth.shape:
        return None
    tp = int(np.sum(truth & prediction))
    fp = int(np.sum(~truth & prediction))
    fn = int(np.sum(truth & ~prediction))
    return float(2 * tp / max(2 * tp + fp + fn, 1))


def _score(model, mixture, method: str, sfreq: float) -> tuple[np.ndarray, dict]:
    if method == "mwf":
        model = MWF(sfreq=sfreq, reg=1e-6)
        cleaned = np.asarray(model.fit_transform(mixture.contaminated, mask=mixture.artifact_mask))
        diagnostics = {"mask_source": "known_injected_intervals"}
    elif method == "wavelet_threshold":
        model = None
        cleaned = wavelet_denoise_multichannel(mixture.contaminated, wavelet="sym5")
        diagnostics = {"wavelet": "sym5"}
    elif model is None:
        cleaned = mixture.contaminated.copy()
        diagnostics = {}
    else:
        model.fit(mixture.calibration)
        cleaned = np.asarray(model.transform(mixture.contaminated))
        diagnostics = model.get_diagnostics() if hasattr(model, "get_diagnostics") else {}
    mask = mixture.artifact_mask
    artifact_error = cleaned[:, mask] - mixture.clean[:, mask]
    injected = mixture.contaminated[:, mask] - mixture.clean[:, mask]
    residual_ratio = float(
        np.sqrt(np.mean(artifact_error**2))
        / max(np.sqrt(np.mean(injected**2)), np.finfo(float).eps)
    )
    clean_error = relative_rmse(cleaned[:, ~mask], mixture.clean[:, ~mask])
    sample_mask = diagnostics.get("sample_mask")
    if isinstance(sample_mask, np.ndarray) and sample_mask.ndim > 1:
        sample_mask = np.any(sample_mask, axis=0)
    metrics = {
        "method": method,
        "status": "success",
        "contaminated_interval_residual_energy_ratio": residual_ratio,
        "clean_interval_relative_rmse": clean_error,
        "whole_signal_rrmse": relative_rmse(cleaned, mixture.clean),
        "clean_correlation": clean_correlation(cleaned, mixture.clean),
        "contaminated_window_f1": _f1(mask, sample_mask),
        "rank_before": int(np.linalg.matrix_rank(mixture.contaminated)),
        "rank_after": int(np.linalg.matrix_rank(cleaned)),
        "fraction_samples_flagged": float(np.mean(sample_mask))
        if isinstance(sample_mask, np.ndarray)
        else None,
    }
    return cleaned, metrics


def _sensitivity_cells(cfg: dict) -> list[dict]:
    defaults = dict(cfg["sensitivity_design"]["fixed_defaults"])
    cells = [{**defaults, "factor": "baseline", "level": "baseline"}]
    mapping = {
        "cutoff": "cutoff",
        "calibration_contamination_fraction": "calibration_contamination_fraction",
        "channel_count": "channel_count",
        "burst_duration_s": "burst_duration_s",
    }
    for factor, key in mapping.items():
        for value in cfg["sensitivity"][key]:
            if value == defaults[factor]:
                continue
            cells.append({**defaults, factor: value, "factor": factor, "level": value})
    return cells


def _model_parameters(model) -> dict:
    if model is None:
        return {}
    if hasattr(model, "get_params"):
        return model.get_params()
    return {
        key: value
        for key, value in vars(model).items()
        if not key.endswith("_") and isinstance(value, (str, int, float, bool, type(None)))
    }


def _write_tsv(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(".tsv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run(args) -> list[dict]:
    config_path = pathlib.Path(args.config).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert_submission_ready(cfg, source=str(config_path))
    simulation = cfg["simulation"]
    sfreq = float(simulation["sfreq_hz"])
    n_channels = int(simulation["n_channels"])
    duration_s = float(simulation["duration_s"])
    calibration_s = float(simulation["clean_calibration_s"])
    repetitions = int(simulation["core_replicates_per_cell"])
    artifact_types = list(simulation["artifact_types"])
    severities = list(simulation["artifact_to_signal_db"])
    if args.campaign == "sensitivity":
        cells = _sensitivity_cells(cfg)
        repetitions = int(simulation["sensitivity_replicates_per_cell"])
    else:
        defaults = cfg["sensitivity_design"]["fixed_defaults"]
        cells = [{**defaults, "factor": "core", "level": "frozen_default"}]
    if args.sensitivity_index is not None:
        cells = [cells[int(args.sensitivity_index)]]
    if args.smoke:
        duration_s, calibration_s, repetitions = 8.0, 3.0, 1
        severities = severities[:1]
        cells = [{**cells[0], "channel_count": 16}]
    if args.artifact_type:
        artifact_types = [args.artifact_type]
    if args.severity is not None:
        severities = [args.severity]
    if args.replicate is not None:
        replicate_indices = [args.replicate]
    else:
        replicate_indices = list(range(repetitions))
    methods = [
        *cfg["methods_under_test"],
        *cfg["comparators"]["required"],
        *cfg["comparators"]["secondary"],
    ]
    if args.methods:
        requested = args.methods.split(",")
        methods = [method for method in methods if method in requested]
    output_root = pathlib.Path(args.output_root).resolve()
    global_seed = int(simulation["seeds"]["global"])
    rows: list[dict] = []

    for cell_index, cell in enumerate(cells):
        for artifact_type in artifact_types:
            for severity in severities:
                for replicate in replicate_indices:
                    seed = locked_seed(
                        global_seed, cfg["arm"], artifact_type, severity, replicate
                    )
                    mixture = transient_mixture(
                        artifact_type=artifact_type,
                        artifact_to_signal_db=float(severity),
                        n_channels=int(cell.get("channel_count", n_channels)),
                        sfreq=sfreq,
                        duration_s=duration_s,
                        calibration_s=calibration_s,
                        seed=seed,
                        burst_duration_s=float(cell["burst_duration_s"]),
                        calibration_contamination_fraction=float(
                            cell["calibration_contamination_fraction"]
                        ),
                    )
                    factor_label = str(cell["factor"]).replace("_", "-")
                    unit_id = (
                        f"{args.campaign}_{factor_label}-{cell['level']}_"
                        f"{artifact_type}_db{severity}_seed{replicate:03d}"
                    )
                    if not args_select_unit(args, unit_id):
                        continue
                    for method in methods:
                        method_dir = output_root / unit_id / method
                        tier = (
                            "oracle_mask_aware"
                            if method == "mwf"
                            else "target_aware_local"
                            if method == "guided_asr"
                            else "blind_local"
                            if method == "wavelet_threshold"
                            else "blind"
                        )
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
                                model = (
                                    None
                                    if method in {"mwf", "wavelet_threshold"}
                                    else _method(
                                        method, artifact_type, sfreq, float(cell["cutoff"])
                                    )
                                )
                                _, metrics = _score(model, mixture, method, sfreq)
                                metrics.update(
                                    {
                                        "unit_id": unit_id,
                                        "artifact_type": artifact_type,
                                        "artifact_to_signal_db": severity,
                                        "replicate": replicate,
                                        "seed": seed,
                                        "campaign": args.campaign,
                                        "sensitivity_cell_index": cell_index,
                                        "sensitivity_factor": cell["factor"],
                                        "sensitivity_level": cell["level"],
                                        "cutoff": cell["cutoff"],
                                        "calibration_contamination_fraction": cell[
                                            "calibration_contamination_fraction"
                                        ],
                                        "channel_count": cell["channel_count"],
                                        "burst_duration_s": cell["burst_duration_s"],
                                        "information_tier": tier,
                                    }
                                )
                                active.effective_rank_before = metrics["rank_before"]
                                active.effective_rank_after = metrics["rank_after"]
                                (method_dir / "metrics.json").write_text(
                                    json.dumps(metrics, indent=2, sort_keys=True),
                                    encoding="utf-8",
                                )
                                model_state = {
                                    "class": None if model is None else type(model).__name__,
                                    "parameters": _model_parameters(model),
                                    "diagnostic_scalars": {
                                        key: value
                                        for key, value in metrics.items()
                                        if isinstance(
                                            value, (str, int, float, bool, type(None))
                                        )
                                    },
                                }
                                (method_dir / "model.json").write_text(
                                    json.dumps(
                                        model_state, indent=2, sort_keys=True, default=str
                                    ),
                                    encoding="utf-8",
                                )
                                rows.append(metrics)
                        except (ImportError, ModuleNotFoundError) as error:
                            # Missing required method code is an execution failure,
                            # not a scientific failure rate.  The Slurm wrapper
                            # must fail closed so the environment can be repaired.
                            rows.append(
                                {
                                    "method": method,
                                    "status": "unavailable_dependency",
                                    "unit_id": unit_id,
                                    "artifact_type": artifact_type,
                                    "artifact_to_signal_db": severity,
                                    "replicate": replicate,
                                    "seed": seed,
                                    "error": f"{type(error).__name__}: {error}",
                                }
                            )
                        except Exception as error:  # terminal record contains traceback
                            rows.append(
                                {
                                    "method": method,
                                    "status": "failed",
                                    "unit_id": unit_id,
                                    "artifact_type": artifact_type,
                                    "artifact_to_signal_db": severity,
                                    "replicate": replicate,
                                    "seed": seed,
                                    "error": f"{type(error).__name__}: {error}",
                                }
                            )
    _write_tsv(output_root / "raw_metrics.tsv", rows)
    return rows


def _runner_exit_status(rows) -> int:
    """Return success when every scheduled attempt produced a terminal result.

    Method-level numerical or calibration failures are benchmark outcomes and
    must remain in the failure denominator. Missing dependencies and unknown
    statuses instead indicate that the frozen execution environment is invalid.
    """
    if not rows:
        return 1
    accepted = {"success", "failed"}
    return int(any(row.get("status") not in accepted for row in rows))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=str(_REPO / "configs/benchmarks/asr_transient_ground_truth.yaml")
    )
    parser.add_argument(
        "--dataset-manifest", default=str(_REPO / "configs/manifests/synthetic_v1.json")
    )
    parser.add_argument("--output-root", default=str(_REPO / "results/asr_transient_ground_truth"))
    parser.add_argument("--campaign", choices=("core", "sensitivity"), default="core")
    parser.add_argument("--sensitivity-index", type=int)
    parser.add_argument("--artifact-type")
    parser.add_argument("--severity", type=float)
    parser.add_argument("--replicate", type=int)
    parser.add_argument("--methods", help="Comma-separated method subset")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    add_shard_arguments(parser)
    args = parser.parse_args(argv)
    rows = run(args)
    successes = sum(row.get("status") == "success" for row in rows)
    failures = sum(row.get("status") == "failed" for row in rows)
    print(
        f"attempts={len(rows)} successes={successes} failures={failures} "
        f"output={args.output_root}"
    )
    return _runner_exit_status(rows)


if __name__ == "__main__":
    raise SystemExit(main())
