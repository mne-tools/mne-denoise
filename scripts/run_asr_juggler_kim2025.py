#!/usr/bin/env python
"""Run the procedural Kim et al. (2025) Juggler ASR simulation."""

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

from mne_denoise.asr import ASR, JugglerASR
from mne_denoise.benchmarks.config import assert_submission_ready
from mne_denoise.benchmarks.intended import locked_seed, relative_rmse
from mne_denoise.benchmarks.provenance import AttemptRecorder, build_run_record
from mne_denoise.benchmarks.sharding import add_shard_arguments, args_select_unit

_EPS = np.finfo(float).eps


def _pink_noise(rng: np.random.Generator, n_times: int) -> np.ndarray:
    """Generate one standardized real pink-noise trace."""
    frequencies = np.fft.rfftfreq(n_times)
    spectrum = np.fft.rfft(rng.standard_normal(n_times))
    scale = np.zeros_like(frequencies)
    scale[1:] = 1.0 / np.sqrt(frequencies[1:])
    trace = np.fft.irfft(spectrum * scale, n=n_times)
    trace -= np.mean(trace)
    return trace / max(float(np.std(trace)), _EPS)


def _event_starts(start: int, stop: int, interval_s: float, sfreq: float) -> np.ndarray:
    step = max(1, int(round(interval_s * sfreq)))
    return np.arange(start, stop, step, dtype=int)


def _unit_peak(values: np.ndarray) -> np.ndarray:
    return values / max(float(np.max(np.abs(values))), _EPS)


def _artifact_template(kind: str, duration_s: float, sfreq: float) -> np.ndarray:
    n_samples = max(3, int(round(duration_s * sfreq)))
    if kind == "biphasic_gaussian_derivative":
        x = np.linspace(-3.5, 3.5, n_samples)
        return _unit_peak(-x * np.exp(-0.5 * x**2))
    if kind == "raised_cosine":
        return np.sin(np.linspace(0.0, np.pi, n_samples)) ** 2
    if kind == "low_frequency_half_sine":
        return np.sin(np.linspace(0.0, np.pi, n_samples))
    raise ValueError(f"unknown artifact waveform: {kind}")


def _inject_events(
    destination: np.ndarray,
    artifact: np.ndarray,
    mask: np.ndarray,
    *,
    starts: np.ndarray,
    channel_weights: list[float],
    template: np.ndarray,
    peak_uv: float,
    rng: np.random.Generator,
) -> None:
    n_times = destination.shape[1]
    for channel, weight in enumerate(channel_weights):
        for start in starts:
            stop = min(int(start) + template.size, n_times)
            if stop <= start:
                continue
            length = stop - int(start)
            # The paper states that artifact channels were independent. Small
            # deterministic shape and amplitude perturbations implement that
            # property while retaining the reported spatial weights.
            perturbation = signal.savgol_filter(
                rng.standard_normal(length),
                window_length=min(length if length % 2 else length - 1, 7),
                polyorder=min(2, max(0, length - 2)),
                mode="interp",
            )
            waveform = template[:length] + 0.05 * perturbation
            waveform = _unit_peak(waveform) * float(peak_uv) * 1e-6 * float(weight)
            destination[channel, start:stop] += waveform
            artifact[channel, start:stop] += waveform
            mask[start:stop] = True


def simulate_kim2025(
    cfg: dict,
    *,
    motion_1_interval_s: float,
    seed: int,
    smoke: bool = False,
) -> dict[str, np.ndarray | float]:
    """Construct the reported pink-noise, alpha, and three-artifact mixture."""
    simulation = cfg["simulation"]
    n_channels = int(simulation["n_channels"])
    n_times = int(simulation["n_times"])
    sfreq = float(simulation["sfreq_hz"])
    if smoke:
        # Keep the reported 100-channel topology: reducing the montage while
        # retaining 6-10 contaminated channels changes the intended sparse-
        # artifact regime into a dense one and makes the smoke test misleading.
        n_times = min(n_times, 4096)
    rng = np.random.default_rng(seed)

    common = _pink_noise(rng, n_times)
    independent = np.vstack([_pink_noise(rng, n_times) for _ in range(n_channels)])
    rho = float(simulation["cross_channel_correlation"])
    background = np.sqrt(rho) * common + np.sqrt(1.0 - rho) * independent
    background *= float(simulation["background_scale_uv"]) * 1e-6
    time = np.arange(n_times) / sfreq
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(n_channels, 1))
    alpha = float(simulation["alpha_peak_amplitude_uv"]) * 1e-6 * np.sin(
        2.0 * np.pi * float(simulation["alpha_frequency_hz"]) * time + phases
    )
    clean = background + alpha
    mixture = clean.copy()
    artifact = np.zeros_like(clean)
    masks = {
        "motion_1": np.zeros(n_times, dtype=bool),
        "blink": np.zeros(n_times, dtype=bool),
        "motion_2": np.zeros(n_times, dtype=bool),
    }

    support = simulation["motion_1_support_fraction"]
    starts = _event_starts(
        int(round(float(support[0]) * n_times)),
        int(round(float(support[1]) * n_times)),
        motion_1_interval_s,
        sfreq,
    )
    specification = simulation["waveform_operationalization"]["motion_1"]
    _inject_events(
        mixture,
        artifact,
        masks["motion_1"],
        starts=starts,
        channel_weights=list(simulation["motion_1_channel_weights"]),
        template=_artifact_template(
            specification["kind"], float(specification["duration_s"]), sfreq
        ),
        peak_uv=float(specification["peak_uv"]),
        rng=rng,
    )

    support = simulation["blink_support_fraction"]
    starts = _event_starts(
        int(round(float(support[0]) * n_times)),
        int(round(float(support[1]) * n_times)),
        float(simulation["blink_interval_s"]),
        sfreq,
    )
    specification = simulation["waveform_operationalization"]["blink"]
    _inject_events(
        mixture,
        artifact,
        masks["blink"],
        starts=starts,
        channel_weights=list(simulation["blink_channel_weights"]),
        template=_artifact_template(
            specification["kind"], float(specification["duration_s"]), sfreq
        ),
        peak_uv=float(specification["peak_uv"]),
        rng=rng,
    )

    support = simulation["motion_2_support_fraction"]
    starts = _event_starts(
        int(round(float(support[0]) * n_times)),
        int(round(float(support[1]) * n_times)),
        float(simulation["motion_2_interval_s"]),
        sfreq,
    )
    specification = simulation["waveform_operationalization"]["motion_2"]
    _inject_events(
        mixture,
        artifact,
        masks["motion_2"],
        starts=starts,
        channel_weights=list(simulation["motion_2_channel_weights"]),
        template=_artifact_template(
            specification["kind"], float(specification["duration_s"]), sfreq
        ),
        peak_uv=float(specification["peak_uv"]),
        rng=rng,
    )
    masks["any"] = masks["motion_1"] | masks["blink"] | masks["motion_2"]
    return {
        "clean": clean,
        "mixture": mixture,
        "artifact": artifact,
        "sfreq": sfreq,
        **masks,
    }


def _method(name: str, *, sfreq: float, cutoff: float, cfg: dict):
    common = cfg["common_parameters"]
    parameters = {
        "sfreq": sfreq,
        "cutoff": cutoff,
        "blocksize": int(common["blocksize"]),
        "filter_kind": str(common["filter_kind"]),
        "window_length": float(common["window_length_s"]),
        "window_criterion": float(
            common["post_asr_window_rejection"]["retained_bad_channel_fraction"]
        ),
        "window_criterion_tolerances": tuple(
            common["post_asr_window_rejection"]["z_tolerances"]
        ),
        "picks": None,
        "verbose": False,
    }
    if name == "asr_original_5pct":
        return ASR(
            method="standard",
            calibration="auto",
            ref_max_bad_channels=float(
                cfg["methods"][name]["ref_max_bad_channels"]
            ),
            ref_tolerances=tuple(
                common["post_asr_window_rejection"]["z_tolerances"]
            ),
            **parameters,
        )
    if name in {"juggler_dbscan", "juggler_gev"}:
        strategy = name.removeprefix("juggler_")
        extra = cfg["methods"][name]
        return JugglerASR(
            strategy=strategy,
            selection_filter_kind="asr",
            dbscan_top_k=int(extra.get("dbscan_top_k", 5)),
            dbscan_eps=extra.get("dbscan_eps", "paper"),
            dbscan_min_samples=extra.get("dbscan_min_samples", "paper"),
            **parameters,
        )
    raise KeyError(name)


def _band_power(data: np.ndarray, sfreq: float, low: float, high: float) -> float:
    frequencies, psd = signal.welch(
        data, fs=sfreq, nperseg=min(data.shape[1], int(round(4 * sfreq))), axis=1
    )
    keep = (frequencies >= low) & (frequencies <= high)
    return float(np.trapezoid(psd[:, keep], frequencies[keep], axis=1).mean())


def _score(model, simulation: dict[str, np.ndarray | float]) -> dict:
    mixture = np.asarray(simulation["mixture"])
    clean = np.asarray(simulation["clean"])
    motion_1 = np.asarray(simulation["motion_1"], dtype=bool)
    clean_mask = ~np.asarray(simulation["any"], dtype=bool)
    sfreq = float(simulation["sfreq"])
    cleaned = model.fit_transform(mixture)
    residual = cleaned - clean
    injected = mixture - clean
    calibration = getattr(model, "calibration_info_", {}) or {}
    reference_fraction = calibration.get("reference_selected_fraction")
    if reference_fraction is None:
        reference_fraction = calibration.get("calibration_clean_sample_fraction")
    return {
        "status": "success",
        "motion_1_residual_energy_ratio": float(
            np.sum(residual[:, motion_1] ** 2)
            / max(float(np.sum(injected[:, motion_1] ** 2)), _EPS)
        ),
        "clean_interval_relative_rmse": relative_rmse(
            cleaned[:, clean_mask], clean[:, clean_mask]
        ),
        "whole_signal_relative_rmse": relative_rmse(cleaned, clean),
        "alpha_10hz_power_retention": _band_power(cleaned, sfreq, 9.5, 10.5)
        / max(_band_power(clean, sfreq, 9.5, 10.5), _EPS),
        "reference_selected_fraction": reference_fraction,
        "post_rejection_retained_fraction": float(
            np.mean(model.get_rejection_mask())
        ),
        "fraction_reconstructed_windows": float(
            model.diagnostics_.get("fraction_reconstructed_windows", 0.0)
        ),
        "rank_before": int(np.linalg.matrix_rank(mixture @ mixture.T)),
        "rank_after": int(np.linalg.matrix_rank(cleaned @ cleaned.T)),
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
    intervals = list(cfg["simulation"]["motion_1_intervals_s"])
    cutoffs = list(cfg["common_parameters"]["cutoffs"])
    methods = list(cfg["methods"])
    replicates = int(cfg["simulation"]["replicates_per_interval"])
    if args.smoke:
        intervals, cutoffs, replicates = intervals[:1], [5], 1
    if args.methods:
        requested = args.methods.split(",")
        methods = [method for method in methods if method in requested]
    output_root = pathlib.Path(args.output_root).resolve()
    rows = []
    for interval in intervals:
        for cutoff in cutoffs:
            for replicate in range(replicates):
                seed = locked_seed(
                    int(cfg["simulation"]["seeds"]["global"]),
                    cfg["arm"],
                    interval,
                    replicate,
                )
                unit_id = f"motion1_interval-{interval:.2f}_k-{cutoff}_seed-{replicate:03d}"
                if not args_select_unit(args, unit_id):
                    continue
                simulation = simulate_kim2025(
                    cfg,
                    motion_1_interval_s=float(interval),
                    seed=seed,
                    smoke=args.smoke,
                )
                for method_name in methods:
                    method_dir = output_root / unit_id / method_name
                    record = build_run_record(
                        arm=cfg["arm"],
                        method=method_name,
                        unit_id=unit_id,
                        config_path=config_path,
                        dataset_manifest=args.dataset_manifest,
                        repo_root=_REPO,
                        seed=seed,
                        information_tier="blind_recording_adaptive",
                        allow_dirty=args.allow_dirty,
                    )
                    try:
                        with AttemptRecorder(method_dir, record) as active:
                            model = _method(
                                method_name,
                                sfreq=float(simulation["sfreq"]),
                                cutoff=float(cutoff),
                                cfg=cfg,
                            )
                            metrics = _score(model, simulation)
                            metrics.update(
                                {
                                    "unit_id": unit_id,
                                    "method": method_name,
                                    "motion_1_interval_s": interval,
                                    "cutoff": cutoff,
                                    "replicate": replicate,
                                    "seed": seed,
                                    "evidence_tier": cfg["evidence"]["tier"],
                                }
                            )
                            active.effective_rank_before = metrics["rank_before"]
                            active.effective_rank_after = metrics["rank_after"]
                            (method_dir / "metrics.json").write_text(
                                json.dumps(metrics, indent=2, sort_keys=True),
                                encoding="utf-8",
                            )
                            (method_dir / "model.json").write_text(
                                json.dumps(
                                    {
                                        "class": type(model).__name__,
                                        "parameters": model.get_params(deep=False),
                                        "calibration_info": {
                                            key: value
                                            for key, value in model.calibration_info_.items()
                                            if np.isscalar(value) or value is None
                                        },
                                    },
                                    indent=2,
                                    sort_keys=True,
                                    default=str,
                                ),
                                encoding="utf-8",
                            )
                            rows.append(metrics)
                    except Exception as error:
                        rows.append(
                            {
                                "unit_id": unit_id,
                                "method": method_name,
                                "motion_1_interval_s": interval,
                                "cutoff": cutoff,
                                "replicate": replicate,
                                "seed": seed,
                                "status": "failed",
                                "error": f"{type(error).__name__}: {error}",
                            }
                        )
    _write_tsv(output_root / "raw_metrics.tsv", rows)
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(_REPO / "configs/benchmarks/asr_juggler_kim2025.yaml"),
    )
    parser.add_argument(
        "--dataset-manifest",
        default=str(_REPO / "configs/manifests/synthetic_v1.json"),
    )
    parser.add_argument(
        "--output-root", default=str(_REPO / "results/asr_juggler_kim2025")
    )
    parser.add_argument("--methods", help="Comma-separated subset")
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
    return int(not rows or any(row.get("status") not in {"success", "failed"} for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
