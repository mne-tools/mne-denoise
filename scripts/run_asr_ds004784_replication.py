#!/usr/bin/env python
"""Reproduce published ds004784 ASR cells and run the locked ASR family."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import sys
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np
import yaml
from scipy import integrate, interpolate, signal

_REPO = pathlib.Path(__file__).resolve().parents[1]
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_SCRIPT_DIR))

from run_ds004784_phantom import (
    _corrected_data_quality_score,
    _effective_rank,
    _fit_clean_forward,
    _load_ground_truth,
)

from mne_denoise.asr import ASR, AdaptiveASR, GuidedASR, JugglerASR
from mne_denoise.benchmarks.config import assert_submission_ready
from mne_denoise.benchmarks.intended import clean_correlation, relative_rmse
from mne_denoise.benchmarks.provenance import (
    AttemptRecorder,
    build_run_record,
    sha256_file,
)
from mne_denoise.dss.denoisers import BandpassBias

_EPS = float(np.finfo(np.float64).eps)
_SYNC_DESCRIPTION = "65471"


@dataclass(frozen=True)
class Cell:
    campaign: str
    repeat: int
    condition: str
    method: str
    calibration_source: str
    cutoff: float | None
    expected_dqs: float | None = None

    @property
    def unit_id(self) -> str:
        cutoff = "na" if self.cutoff is None else f"{self.cutoff:g}".replace(".", "p")
        return (
            f"{self.campaign}_repeat{self.repeat}_{self.condition}_"
            f"{self.method}_{self.calibration_source}_k{cutoff}"
        )


def _read_config(path: pathlib.Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert_submission_ready(config, source=str(path))
    return config


def _validate_manifest(
    path: pathlib.Path, *, repeat: int, conditions: list[str]
) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_prefix = f"ds004784-repeat{repeat}"
    if not str(payload.get("dataset_id", "")).startswith(expected_prefix):
        raise RuntimeError(
            f"manifest identifies {payload.get('dataset_id')!r}, expected {expected_prefix!r}"
        )
    paths = {item["path"] for item in payload.get("files", [])}
    required = {
        f"derivatives/Data/Imported/NMM10_{condition}_{repeat}.{extension}"
        for condition in conditions
        for extension in ("set", "fdt")
    }
    required.add("stimuli/GTdata_croppedToRisingEdge.mat")
    missing = required - paths
    if missing:
        raise RuntimeError(f"manifest is missing ds004784 inputs: {sorted(missing)}")
    return payload


def _validate_reference_manifest(path: pathlib.Path, *, conditions: list[str]) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    paths = {item["path"] for item in payload.get("files", [])}
    required = {
        f"derivatives/Data/Preprocessed/ASR/prep_ASR_NMM10_{condition}_1.{extension}"
        for condition in conditions
        for extension in ("set", "fdt")
        if not (condition == "Clean" and extension == "fdt")
    }
    missing = required - paths
    if missing:
        raise RuntimeError(
            f"reference manifest is missing distributed ASR outputs: {sorted(missing)}"
        )
    return payload


def exact_cutoffs(config: dict) -> list[float]:
    values: list[float] = []
    for segment in config["published_protocol"]["exact_cutoff_grid"]["segments"]:
        start = Fraction(str(segment["start"]))
        stop = Fraction(str(segment["stop"]))
        step = Fraction(str(segment["step"]))
        count = int((stop - start) / step) + 1
        values.extend(float(start + index * step) for index in range(count))
    expected = int(config["published_protocol"]["exact_cutoff_grid"]["expected_count"])
    if len(values) != expected or len(set(values)) != expected:
        raise RuntimeError(
            f"published cutoff grid produced {len(values)} values, expected {expected}"
        )
    return values


def published_reference_cells(config: dict) -> list[Cell]:
    repeat = int(config["dataset"]["published_repeat"])
    cells: list[Cell] = []
    for condition in config["dataset"]["conditions"]:
        cells.append(
            Cell(
                campaign="published_reference",
                repeat=repeat,
                condition=condition,
                method="none",
                calibration_source="not_applicable",
                cutoff=None,
                expected_dqs=float(
                    config["published_protocol"]["expected_raw_dqs"][condition]
                ),
            )
        )
        expected = config["published_protocol"]["expected_best_cells"][condition]
        for key, calibration_source in (
            ("external_clean", "external_clean"),
            ("target_selection", "target_selection"),
        ):
            cells.append(
                Cell(
                    campaign="published_reference",
                    repeat=repeat,
                    condition=condition,
                    method="asr_standard",
                    calibration_source=calibration_source,
                    cutoff=float(expected[key]["cutoff"]),
                    expected_dqs=float(expected[key]["corrected_dqs"]),
                )
            )
    return cells


def family_cells(config: dict) -> list[Cell]:
    specification = config["family_replication"]
    repeat = int(specification["repeat"])
    cells: list[Cell] = []
    for condition in config["dataset"]["conditions"]:
        cells.append(
            Cell(
                campaign="family_replication",
                repeat=repeat,
                condition=condition,
                method="none",
                calibration_source="not_applicable",
                cutoff=None,
            )
        )
        for method, calibration_sources in specification["methods"].items():
            if method == "none":
                continue
            for calibration_source in calibration_sources:
                for cutoff in specification["cutoff_sweep"]:
                    cells.append(
                        Cell(
                            campaign="family_replication",
                            repeat=repeat,
                            condition=condition,
                            method=method,
                            calibration_source=calibration_source,
                            cutoff=float(cutoff),
                        )
                    )
    return cells


def campaign_cells(config: dict, campaign: str) -> list[Cell]:
    if campaign == "published_reference":
        return published_reference_cells(config)
    if campaign == "family_replication":
        return family_cells(config)
    raise ValueError(f"unknown campaign {campaign!r}")


def _scalp_indices(ch_names: list[str]) -> list[int]:
    indices = [
        index
        for index, name in enumerate(ch_names)
        if len(name) >= 2 and name[0] in "ABCD" and name[1:].isdigit()
    ]
    if len(indices) != 128:
        raise RuntimeError(
            f"expected 128 ds004784 scalp channels, found {len(indices)}"
        )
    return indices


def _load_scalp_recording(path: pathlib.Path) -> dict[str, Any]:
    import mne

    raw = mne.io.read_raw_eeglab(path, preload=True, verbose=False)
    sfreq = float(raw.info["sfreq"])
    sync = [
        float(onset)
        for onset, description in zip(
            raw.annotations.onset, raw.annotations.description
        )
        if str(description).startswith(_SYNC_DESCRIPTION)
    ]
    if len(sync) != 2 or sync[1] <= sync[0]:
        raise RuntimeError(
            f"expected two ordered synchronization events in {path}, found {sync}"
        )
    start = int(round(sync[0] * sfreq))
    stop = int(round(sync[1] * sfreq)) + 1
    data = np.asarray(raw.get_data(picks=_scalp_indices(raw.ch_names)))
    if start < 0 or stop > data.shape[1]:
        raise RuntimeError(
            f"synchronization interval [{start}, {stop}) lies outside {path}"
        )
    return {
        "data": data,
        "sfreq": sfreq,
        "sync_onsets_s": sync,
        "sync_samples": (start, stop),
    }


def _synchronized_data(data: np.ndarray, sync_samples: tuple[int, int]) -> np.ndarray:
    start, stop = sync_samples
    if start < 0 or stop <= start or stop > data.shape[1]:
        raise ValueError(
            f"invalid synchronization interval [{start}, {stop}) for {data.shape[1]} samples"
        )
    return data[:, start:stop]


def _align(data: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    gt_times = np.arange(ground_truth.shape[1], dtype=float) / 512.0
    source_times = np.linspace(0.0, gt_times[-1], data.shape[1])
    warper = interpolate.interp1d(
        source_times,
        data,
        axis=1,
        kind="linear",
        assume_sorted=True,
        bounds_error=False,
        fill_value="extrapolate",
    )
    return np.asarray(warper(gt_times))


def _band_power(
    data: np.ndarray, sfreq: float, low: float = 8.0, high: float = 30.0
) -> float:
    frequencies, psd = signal.welch(
        data, sfreq, nperseg=min(2048, data.shape[1]), axis=1
    )
    keep = (frequencies >= low) & (frequencies <= high)
    return float(integrate.trapezoid(psd[:, keep], frequencies[keep], axis=1).mean())


def _estimator(config: dict, cell: Cell, sfreq: float):
    if cell.method == "none":
        return None
    common = {
        "sfreq": sfreq,
        "cutoff": float(cell.cutoff),
        "picks": None,
        "max_mem_mb": 2048,
        "verbose": False,
    }
    reference_tolerances = tuple(
        float(value)
        for value in config["published_protocol"]["method"][
            "reference_window_tolerances"
        ]
    )
    if cell.method == "asr_standard":
        calibration = (
            "manual" if cell.calibration_source == "external_clean" else "auto"
        )
        return ASR(
            method="standard",
            calibration=calibration,
            filter_kind="asr",
            ref_tolerances=reference_tolerances,
            **common,
        )
    if cell.method == "rasr_windowed":
        calibration = (
            "manual" if cell.calibration_source == "external_clean" else "auto"
        )
        return ASR(
            method="riemannian_windowed",
            calibration=calibration,
            filter_kind="asr",
            ref_tolerances=reference_tolerances,
            **common,
        )
    if cell.method == "rasr_legacy":
        calibration = (
            "manual" if cell.calibration_source == "external_clean" else "auto"
        )
        return ASR(
            method="riemannian",
            experimental=True,
            calibration=calibration,
            filter_kind="asr",
            ref_tolerances=reference_tolerances,
            **common,
        )
    if cell.method in {"adaptive_psp", "adaptive_psw"}:
        return AdaptiveASR(variant=cell.method.removeprefix("adaptive_"), **common)
    if cell.method in {"adaptive_mw_final_state", "adaptive_mw_sliding"}:
        mode = cell.method.removeprefix("adaptive_mw_")
        return AdaptiveASR(
            variant="mw",
            mw_mode=mode,
            mw_window_length=float(config["family_replication"]["mw_window_length_s"]),
            **common,
        )
    if cell.method in {"juggler_dbscan", "juggler_gev"}:
        return JugglerASR(
            strategy=cell.method.removeprefix("juggler_"),
            filter_kind="asr",
            selection_filter_kind="asr",
            **common,
        )
    if cell.method == "guided_asr":
        artifact_biases = [
            BandpassBias(tuple(band), sfreq)
            for band in config["family_replication"]["guided_artifact_bands_hz"][
                cell.condition
            ]
        ]
        preserve_biases = [
            BandpassBias(tuple(band), sfreq)
            for band in config["family_replication"]["guided_preserve_bands_hz"]
        ]
        calibration = (
            "manual" if cell.calibration_source == "external_clean" else "auto"
        )
        return GuidedASR(
            method="riemannian_windowed",
            calibration=calibration,
            filter_kind="asr",
            ref_tolerances=reference_tolerances,
            reconstruction="soft",
            artifact_biases=artifact_biases,
            preserve_biases=preserve_biases,
            **common,
        )
    raise KeyError(cell.method)


def _adaptive_updates(
    model: AdaptiveASR, data: np.ndarray, sfreq: float, chunk_s: float
) -> int:
    chunk_samples = max(int(model.blocksize), int(round(chunk_s * sfreq)))
    updates = 0
    for start in range(0, data.shape[1], chunk_samples):
        chunk = data[:, start : start + chunk_samples]
        if chunk.shape[1] < int(model.blocksize):
            continue
        model.partial_fit(chunk)
        updates += 1
    model.reset_process_state()
    return updates


def _clean(
    config: dict,
    cell: Cell,
    target: np.ndarray,
    clean_calibration: np.ndarray,
    sfreq: float,
) -> tuple[np.ndarray, Any, dict]:
    model = _estimator(config, cell, sfreq)
    if model is None:
        return target.copy(), None, {}
    if cell.method in {"adaptive_mw_final_state", "adaptive_mw_sliding"}:
        cleaned, diagnostics = model.fit_transform(target, return_diagnostics=True)
        return np.asarray(cleaned), model, diagnostics
    if cell.method in {"adaptive_psp", "adaptive_psw"}:
        initial_s = float(config["family_replication"]["adaptive_initial_window_s"])
        chunk_s = float(config["family_replication"]["adaptive_update_chunk_s"])
        if cell.calibration_source == "external_clean_adaptive":
            model.fit(clean_calibration)
            updates = _adaptive_updates(model, target, sfreq, chunk_s)
        elif cell.calibration_source == "target_initial_adaptive":
            initial = max(int(model.blocksize), int(round(initial_s * sfreq)))
            model.fit(target[:, :initial])
            updates = _adaptive_updates(model, target[:, initial:], sfreq, chunk_s)
        else:
            raise ValueError(
                f"invalid adaptive calibration source {cell.calibration_source!r}"
            )
        cleaned, diagnostics = model.transform(target, return_diagnostics=True)
        diagnostics = dict(diagnostics)
        diagnostics["benchmark_adaptive_update_count"] = updates
        return np.asarray(cleaned), model, diagnostics
    if cell.calibration_source == "external_clean":
        model.fit(clean_calibration)
    elif cell.calibration_source in {"target_selection", "recording_local"}:
        model.fit(target)
    else:
        raise ValueError(f"invalid calibration source {cell.calibration_source!r}")
    cleaned, diagnostics = model.transform(target, return_diagnostics=True)
    return np.asarray(cleaned), model, diagnostics


def _sample_mask(diagnostics: dict, n_times: int) -> np.ndarray | None:
    value = diagnostics.get("sample_mask")
    if not isinstance(value, np.ndarray):
        return None
    if value.ndim > 1:
        value = np.any(value, axis=0)
    if value.shape != (n_times,):
        return None
    return value.astype(bool)


def _calibration_metrics(model: Any) -> dict[str, int | float | str | None]:
    if model is None:
        return {
            "calibration_reference_samples": None,
            "calibration_candidate_samples": None,
            "calibration_reference_fraction": None,
            "calibration_reference_mask_sha256": None,
            "calibration_clean_windows": None,
            "calibration_candidate_windows": None,
            "calibration_clean_window_fraction": None,
        }
    info = getattr(model, "calibration_info_", {})
    if not isinstance(info, dict):
        info = {}

    sample_mask = info.get("reference_sample_mask")
    if not isinstance(sample_mask, np.ndarray):
        sample_mask = info.get("clean_sample_mask")
    selected_samples = info.get("reference_selected_samples")
    candidate_samples = info.get("reference_candidate_samples")
    if isinstance(sample_mask, np.ndarray):
        sample_mask = np.asarray(sample_mask, dtype=bool).ravel()
        selected_samples = int(np.sum(sample_mask))
        candidate_samples = int(sample_mask.size)
    elif selected_samples is None:
        selected_samples = info.get("calibration_samples")

    window_mask = info.get("clean_window_mask")
    selected_windows = info.get("n_clean_windows")
    candidate_windows = info.get("n_calibration_windows")
    if isinstance(window_mask, np.ndarray):
        selected_windows = int(np.sum(window_mask))
        candidate_windows = int(window_mask.size)

    selected_samples = None if selected_samples is None else int(selected_samples)
    candidate_samples = None if candidate_samples is None else int(candidate_samples)
    selected_windows = None if selected_windows is None else int(selected_windows)
    candidate_windows = None if candidate_windows is None else int(candidate_windows)
    return {
        "calibration_reference_samples": selected_samples,
        "calibration_candidate_samples": candidate_samples,
        "calibration_reference_fraction": (
            None
            if selected_samples is None or not candidate_samples
            else float(selected_samples / candidate_samples)
        ),
        "calibration_reference_mask_sha256": (
            None
            if not isinstance(sample_mask, np.ndarray)
            else hashlib.sha256(
                np.packbits(sample_mask, bitorder="little").tobytes()
            ).hexdigest()
        ),
        "calibration_clean_windows": selected_windows,
        "calibration_candidate_windows": candidate_windows,
        "calibration_clean_window_fraction": (
            None
            if selected_windows is None or not candidate_windows
            else float(selected_windows / candidate_windows)
        ),
    }


def _write_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    temporary.replace(path)


def run_cell(args) -> dict:
    config_path = pathlib.Path(args.config).resolve()
    config = _read_config(config_path)
    cells = campaign_cells(config, args.campaign)
    if args.index < 0 or args.index >= len(cells):
        raise IndexError(f"cell index {args.index} outside 0..{len(cells) - 1}")
    cell = cells[args.index]
    conditions = list(config["dataset"]["conditions"])
    manifest_path = pathlib.Path(args.dataset_manifest).resolve()
    manifest = _validate_manifest(
        manifest_path, repeat=cell.repeat, conditions=conditions
    )
    reference_manifest_path = None
    reference_manifest = None
    if cell.campaign == "published_reference":
        if not args.reference_manifest:
            raise ValueError("published_reference requires --reference-manifest")
        reference_manifest_path = pathlib.Path(args.reference_manifest).resolve()
        reference_manifest = _validate_reference_manifest(
            reference_manifest_path, conditions=conditions
        )

    root = pathlib.Path(args.dataset_root).resolve()
    target_path = root / config["dataset"]["filename_template"].format(
        condition=cell.condition, repeat=cell.repeat
    )
    clean_path = root / config["dataset"]["filename_template"].format(
        condition="Clean", repeat=cell.repeat
    )
    target_recording = _load_scalp_recording(target_path)
    clean_recording = _load_scalp_recording(clean_path)
    sfreq = float(target_recording["sfreq"])
    if not np.isclose(sfreq, clean_recording["sfreq"]):
        raise RuntimeError("target and clean calibration sampling frequencies differ")

    ground_truth = _load_ground_truth(root / config["dataset"]["ground_truth_path"])
    target = target_recording["data"]
    clean_calibration = clean_recording["data"]
    raw_aligned = _align(
        _synchronized_data(target, target_recording["sync_samples"]), ground_truth
    )
    clean_aligned = _align(
        _synchronized_data(clean_calibration, clean_recording["sync_samples"]),
        ground_truth,
    )
    fit_split = int(round(0.4 * ground_truth.shape[1]))
    clean_target = _fit_clean_forward(clean_aligned, ground_truth, fit_split)
    brain = ground_truth[:10]

    output_dir = pathlib.Path(args.output_root).resolve() / cell.campaign / cell.unit_id
    information_tier = {
        "external_clean": "external_clean_calibration",
        "target_selection": "recording_local_clean_window_selection",
        "external_clean_adaptive": "external_clean_initialization_and_target_updates",
        "target_initial_adaptive": "target_initialization_and_target_updates",
        "recording_local": "recording_local_adaptive_or_reference_selection",
        "not_applicable": "no_cleaning_control",
    }[cell.calibration_source]
    record = build_run_record(
        arm=config["arm"],
        method=cell.method,
        unit_id=cell.unit_id,
        config_path=config_path,
        dataset_manifest=manifest_path,
        repo_root=_REPO,
        seed=None,
        information_tier=information_tier,
        allow_dirty=args.allow_dirty,
    )
    with AttemptRecorder(output_dir, record) as active:
        cleaned, model, diagnostics = _clean(
            config, cell, target, clean_calibration, sfreq
        )
        aligned = _align(
            _synchronized_data(cleaned, target_recording["sync_samples"]),
            ground_truth,
        )
        dqs, dqs_uncorrected, correction = _corrected_data_quality_score(
            aligned, raw_aligned, brain
        )
        raw_dqs, _, _ = _corrected_data_quality_score(raw_aligned, raw_aligned, brain)
        mask = _sample_mask(diagnostics, target.shape[1])
        rank_before = _effective_rank(raw_aligned)
        rank_after = _effective_rank(aligned)
        metrics = {
            "arm": config["arm"],
            "campaign": cell.campaign,
            "cell_index": int(args.index),
            "unit_id": cell.unit_id,
            "technical_repeat": cell.repeat,
            "condition": cell.condition,
            "method": cell.method,
            "calibration_source": cell.calibration_source,
            "information_tier": information_tier,
            "cutoff": cell.cutoff,
            "status": "success",
            "ground_truth_data_quality_score": dqs,
            "ground_truth_data_quality_score_uncorrected": dqs_uncorrected,
            "ground_truth_preservation_correction": correction,
            "raw_ground_truth_data_quality_score": raw_dqs,
            "data_quality_score_change": dqs - raw_dqs,
            "expected_published_data_quality_score": cell.expected_dqs,
            "published_data_quality_score_error": (
                None if cell.expected_dqs is None else dqs - cell.expected_dqs
            ),
            "known_clean_waveform_relative_rmse": relative_rmse(aligned, clean_target),
            "known_clean_waveform_correlation": clean_correlation(
                aligned, clean_target
            ),
            "neural_band_retention_vs_raw": _band_power(cleaned, sfreq)
            / max(_band_power(target, sfreq), _EPS),
            "rank_before": rank_before,
            "rank_after": rank_after,
            "fraction_samples_flagged": (
                None if mask is None else float(np.mean(mask))
            ),
            "dataset_manifest_content_sha256": manifest["content_sha256"],
            "reference_manifest_content_sha256": (
                None
                if reference_manifest is None
                else reference_manifest["content_sha256"]
            ),
            "preprocessing": "released_1Hz_highpass_no_rereference_no_channel_rejection_512Hz",
            "score_definition": "Downey_Ferris_corrected_DQS_full_synchronized_recording",
        }
        metrics.update(_calibration_metrics(model))
        if (
            cell.campaign == "published_reference"
            and cell.method == "asr_standard"
            and cell.calibration_source == "external_clean"
        ):
            reference_path = root / config["dataset"][
                "distributed_asr_template"
            ].format(condition=cell.condition)
            distributed = _load_scalp_recording(reference_path)["data"]
            sample_count = min(cleaned.shape[1], distributed.shape[1])
            left = cleaned[:, :sample_count]
            right = distributed[:, :sample_count]
            metrics.update(
                {
                    "distributed_reference_relative_rmse": relative_rmse(left, right),
                    "distributed_reference_correlation": clean_correlation(left, right),
                    "distributed_reference_samples": sample_count,
                    "distributed_reference_manifest_sha256": sha256_file(
                        reference_manifest_path
                    ),
                }
            )
            np.save(output_dir / "cleaned.npy", cleaned)
        active.effective_rank_before = rank_before
        active.effective_rank_after = rank_after
        _write_json(output_dir / "metrics.json", metrics)
        _write_json(
            output_dir / "model.json",
            {
                "class": None if model is None else type(model).__name__,
                "parameters": {} if model is None else model.get_params(deep=False),
                "diagnostic_keys": sorted(diagnostics),
                "fit_scope": cell.calibration_source,
            },
        )
    return metrics


def _write_tsv(path: pathlib.Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def merge_campaign(args) -> dict:
    config_path = pathlib.Path(args.config).resolve()
    config = _read_config(config_path)
    expected_cells = campaign_cells(config, args.campaign)
    root = pathlib.Path(args.output_root).resolve() / args.campaign
    rows: list[dict] = []
    terminals: list[dict] = []
    missing: list[str] = []
    for cell in expected_cells:
        cell_dir = root / cell.unit_id
        terminal_path = cell_dir / "terminal_status.json"
        if not terminal_path.is_file():
            missing.append(cell.unit_id)
            continue
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        terminals.append(terminal)
        metrics_path = cell_dir / "metrics.json"
        if metrics_path.is_file():
            row = json.loads(metrics_path.read_text(encoding="utf-8"))
        else:
            row = {
                "campaign": cell.campaign,
                "unit_id": cell.unit_id,
                "technical_repeat": cell.repeat,
                "condition": cell.condition,
                "method": cell.method,
                "calibration_source": cell.calibration_source,
                "cutoff": cell.cutoff,
                "status": terminal.get("status", "failed"),
                "error_type": terminal.get("error_type"),
                "error_message": terminal.get("error_message"),
            }
        row.update(
            {
                "git_commit": terminal.get("git_commit"),
                "git_dirty": terminal.get("git_dirty"),
                "config_hash": terminal.get("config_hash"),
                "dataset_manifest_hash": terminal.get("dataset_manifest_hash"),
                "environment_hash": terminal.get("environment_hash"),
                "runtime_seconds": terminal.get("runtime_seconds"),
                "peak_memory_mb": terminal.get("peak_memory_mb"),
                "slurm_job_id": terminal.get("slurm_job_id"),
                "slurm_array_task_id": terminal.get("slurm_array_task_id"),
            }
        )
        rows.append(row)

    signatures = {
        key: sorted({str(record.get(key)) for record in terminals})
        for key in (
            "git_commit",
            "git_dirty",
            "config_hash",
            "dataset_manifest_hash",
            "environment_hash",
        )
    }
    invalid_signatures = {
        key: values
        for key, values in signatures.items()
        if len(values) != 1 or (key == "git_dirty" and values != ["False"])
    }
    observed_ids = [row["unit_id"] for row in rows]
    duplicates = sorted(
        {unit_id for unit_id in observed_ids if observed_ids.count(unit_id) > 1}
    )
    successes = sum(row.get("status") == "success" for row in rows)
    failures = len(rows) - successes
    summary = {
        "campaign": args.campaign,
        "expected_cells": len(expected_cells),
        "observed_terminal_cells": len(terminals),
        "successful_cells": successes,
        "failed_cells": failures,
        "missing_cells": missing,
        "duplicate_cells": duplicates,
        "provenance_signatures": signatures,
        "invalid_provenance_signatures": invalid_signatures,
        "config_sha256": sha256_file(config_path),
        "raw_metrics_sha256": None,
    }
    _write_tsv(root / "raw_metrics.tsv", rows)
    summary["raw_metrics_sha256"] = sha256_file(root / "raw_metrics.tsv")
    if args.campaign == "published_reference":
        errors = [
            abs(float(row["published_data_quality_score_error"]))
            for row in rows
            if row.get("published_data_quality_score_error") is not None
        ]
        correlations = [
            float(row["distributed_reference_correlation"])
            for row in rows
            if row.get("distributed_reference_correlation") is not None
        ]
        summary.update(
            {
                "max_absolute_published_dqs_error": max(errors, default=None),
                "min_distributed_reference_correlation": min(
                    correlations, default=None
                ),
            }
        )
    _write_json(root / "merge_summary.json", summary)
    if (
        missing or duplicates or failures or invalid_signatures
    ) and not args.allow_incomplete:
        raise RuntimeError(
            "campaign merge is incomplete or provenance-invalid; see merge_summary.json"
        )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "merge"):
        sub = subparsers.add_parser(name)
        sub.add_argument(
            "--config",
            default=str(_REPO / "configs/benchmarks/asr_ds004784_replication.yaml"),
        )
        sub.add_argument(
            "--campaign",
            choices=("published_reference", "family_replication"),
            required=True,
        )
        sub.add_argument("--output-root", required=True)
    run_parser = subparsers.choices["run"]
    run_parser.add_argument("--dataset-root", required=True)
    run_parser.add_argument("--dataset-manifest", required=True)
    run_parser.add_argument("--reference-manifest")
    run_parser.add_argument("--index", type=int, required=True)
    run_parser.add_argument("--allow-dirty", action="store_true")
    merge_parser = subparsers.choices["merge"]
    merge_parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "run":
        result = run_cell(args)
    else:
        result = merge_campaign(args)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return result


if __name__ == "__main__":
    main()
