#!/usr/bin/env python
"""Run the frozen six-condition ds004784 phantom technical replication.

Repeat 1 is development evidence.  Repeat 2 is the locked technical
replication and cannot be run from a dirty package worktree.  Cleaning is
performed in the recording time base; outputs are then linearly aligned to
the distributed ground-truth sources using the two synchronization events.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from fractions import Fraction

import numpy as np
import yaml
from scipy import interpolate, io, signal

_REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from mne_denoise.benchmarks.config import assert_submission_ready
from mne_denoise.benchmarks.intended import clean_correlation, relative_rmse
from mne_denoise.benchmarks.provenance import AttemptRecorder, build_run_record
from mne_denoise.cca import AutoCCA
from mne_denoise.dss import DSS, ReferenceBias
from mne_denoise.icanclean import ICanClean
from mne_denoise.mwf import MWF

_EPS = float(np.finfo(np.float64).eps)
_SYNC_DESCRIPTION = "65471"


def _read_manifest(path: pathlib.Path, *, repeat: int) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = f"ds004784-repeat{repeat}"
    if not str(payload.get("dataset_id", "")).startswith(expected):
        raise RuntimeError(
            f"manifest {path} identifies {payload.get('dataset_id')!r}, expected {expected!r}"
        )
    paths = {item["path"] for item in payload.get("files", [])}
    expected_pairs = {
        f"derivatives/Data/Imported/NMM10_{condition}_{repeat}.{extension}"
        for condition in ("Clean", "Eyes", "Jaw", "Motion", "Neck", "All")
        for extension in ("set", "fdt")
    }
    missing = expected_pairs - paths
    if missing:
        raise RuntimeError(f"manifest is missing locked phantom inputs: {sorted(missing)}")
    if "stimuli/GTdata_croppedToRisingEdge.mat" not in paths:
        raise RuntimeError("manifest does not include the distributed ground-truth source matrix")
    return payload


def _channel_groups(ch_names: list[str]) -> tuple[list[int], list[int], list[int]]:
    scalp, noise, external = [], [], []
    for index, name in enumerate(ch_names):
        if name.startswith("N-"):
            noise.append(index)
        elif name.upper().startswith("EXG"):
            external.append(index)
        elif len(name) >= 2 and name[0] in "ABCD" and name[1:].isdigit():
            scalp.append(index)
    if (len(scalp), len(noise), len(external)) != (128, 128, 8):
        raise RuntimeError(
            "unexpected ds004784 channel layout: "
            f"scalp={len(scalp)}, noise={len(noise)}, external={len(external)}"
        )
    return scalp, noise, external


def _resample(data: np.ndarray, source_sfreq: float, target_sfreq: float) -> np.ndarray:
    ratio = Fraction(target_sfreq / source_sfreq).limit_denominator(10000)
    return signal.resample_poly(data, ratio.numerator, ratio.denominator, axis=1)


def _load_recording(path: pathlib.Path, *, resample_hz: float) -> dict:
    import mne

    raw = mne.io.read_raw_eeglab(path, preload=True, verbose=False)
    sfreq = float(raw.info["sfreq"])
    sync = [
        float(onset)
        for onset, description in zip(raw.annotations.onset, raw.annotations.description)
        if str(description).startswith(_SYNC_DESCRIPTION)
    ]
    if len(sync) != 2 or sync[1] <= sync[0]:
        raise RuntimeError(f"expected two ordered synchronization events in {path}, found {sync}")
    start = int(round(sync[0] * sfreq))
    stop = int(round(sync[1] * sfreq)) + 1
    data = raw.get_data()[:, start:stop]
    scalp, noise, external = _channel_groups(raw.ch_names)

    # Match the released analysis: average-reference each physical sensor layer
    # separately so that the dual-layer reference cannot alter the scalp average.
    groups = []
    for picks in (scalp, noise, external):
        selected = np.asarray(data[picks], dtype=np.float64)
        groups.append(selected - selected.mean(axis=0, keepdims=True))
    primary, reference, score_reference = groups
    if not np.isclose(sfreq, resample_hz):
        primary = _resample(primary, sfreq, resample_hz)
        reference = _resample(reference, sfreq, resample_hz)
        score_reference = _resample(score_reference, sfreq, resample_hz)
    return {
        "primary": primary,
        "reference": reference,
        "score_reference": score_reference,
        "sfreq": float(resample_hz),
        "duration_s": float(sync[1] - sync[0]),
        "sync_onsets_s": sync,
    }


def _load_ground_truth(path: pathlib.Path) -> np.ndarray:
    payload = io.loadmat(path, simplify_cells=True)
    if "GTdata" not in payload:
        raise RuntimeError(f"{path} has no GTdata variable")
    ground_truth = np.asarray(payload["GTdata"], dtype=np.float64)
    if ground_truth.ndim != 2 or ground_truth.shape[1] < 20:
        raise RuntimeError(f"unexpected GTdata shape {ground_truth.shape}")
    return ground_truth[:, :20].T


def _align_to_ground_truth(
    data: np.ndarray,
    *,
    raw_start_fraction: float,
    ground_truth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    gt_times = np.arange(ground_truth.shape[1], dtype=float) / 512.0
    start_time = raw_start_fraction * gt_times[-1]
    source_times = np.linspace(start_time, gt_times[-1], data.shape[1])
    target = gt_times[gt_times >= start_time]
    warper = interpolate.interp1d(
        source_times,
        data,
        axis=1,
        kind="linear",
        assume_sorted=True,
        bounds_error=False,
        fill_value="extrapolate",
    )
    return np.asarray(warper(target)), target


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


def _reference_projection(reference_fit, reference_score, n_components):
    mean = reference_fit.mean(axis=1, keepdims=True)
    centered = reference_fit - mean
    left, _, _ = np.linalg.svd(centered, full_matrices=False)
    basis = left[:, : min(int(n_components), left.shape[1])]
    return basis.T @ centered, basis.T @ (reference_score - mean)


def _lagged(reference: np.ndarray, shifts: int) -> np.ndarray:
    blocks = []
    for shift in range(-shifts, shifts + 1):
        block = np.zeros_like(reference)
        if shift < 0:
            block[:, :shift] = reference[:, -shift:]
        elif shift > 0:
            block[:, shift:] = reference[:, :-shift]
        else:
            block = reference.copy()
        blocks.append(block)
    return np.vstack(blocks)


def _regression_clean(primary_fit, primary_score, reference_fit, reference_score, ridge=1e-8):
    covariance = reference_fit @ reference_fit.T
    weights = (
        primary_fit
        @ reference_fit.T
        @ np.linalg.pinv(covariance + float(ridge) * np.eye(covariance.shape[0]))
    )
    return primary_score - weights @ reference_score


def _clean_real(method, primary_fit, primary_score, reference_fit, reference_score, sfreq, cfg):
    params = cfg["method_parameters"]
    n_ref_pca = int(cfg["analysis_policy"]["global_reference_pca_components"])
    model, removed = None, None
    if method == "none":
        output = primary_score.copy()
    elif method == "icanclean_dual_layer":
        output, model, removed = _icanclean(primary_score, reference_score, sfreq, params["icanclean"])
    elif method == "icanclean_pseudoreference":
        pseudo = _pseudo_reference(
            primary_score,
            sfreq,
            int(params["pseudoreference"]["pca_components"]),
            float(params["pseudoreference"]["highpass_hz"]),
        )
        output, model, removed = _icanclean(primary_score, pseudo, sfreq, params["icanclean"])
    elif method in {"regression", "tspca"}:
        ref_fit, ref_score = _reference_projection(reference_fit, reference_score, n_ref_pca)
        if method == "tspca":
            shifts = int(params["tspca"]["n_shifts_each_direction"])
            ref_fit, ref_score = _lagged(ref_fit, shifts), _lagged(ref_score, shifts)
        output = _regression_clean(primary_fit, primary_score, ref_fit, ref_score)
    elif method == "reference_bias_dss":
        count = min(
            int(params["reference_bias_dss"]["max_components"]), reference_fit.shape[0]
        )
        model = DSS(
            bias=ReferenceBias(
                reference_fit, ridge=float(params["reference_bias_dss"]["ridge"])
            ),
            n_components=count,
            normalize_input=False,
            return_type="raw",
            verbose=False,
        ).fit(primary_fit)
        artifact_estimate = np.asarray(model.transform(primary_score))
        output = primary_score - (artifact_estimate - primary_score.mean(axis=1, keepdims=True))
        removed = count
    elif method == "autocca":
        model = AutoCCA(rho_threshold=float(params["autocca"]["rho_threshold"])).fit(
            primary_fit
        )
        output = np.asarray(model.transform(primary_score))
        removed = primary_fit.shape[0] - int(model.n_kept_)
    elif method == "mwf":
        projected_fit, projected_score = _reference_projection(
            reference_fit, reference_score, n_ref_pca
        )
        del projected_fit
        envelope = np.mean(projected_score**2, axis=0)
        mask = envelope > np.quantile(
            envelope, float(params["mwf"]["reference_envelope_quantile"])
        )
        model = MWF(sfreq=sfreq, reg=float(params["mwf"]["regularization"]))
        output = np.asarray(model.fit_transform(primary_score, mask=mask))
    else:
        raise KeyError(method)
    return np.asarray(output), model, removed


def _row_correlation(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left - left.mean(axis=1, keepdims=True)
    right = right - right.mean(axis=1, keepdims=True)
    left = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), _EPS)
    right = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), _EPS)
    return left @ right.T


def _best_possible_ground_truth_correlation(data: np.ndarray, brain: np.ndarray) -> np.ndarray:
    gram = data @ data.T
    unmixing = brain @ data.T @ np.linalg.pinv(gram)
    reconstructed = unmixing @ data
    return np.diag(_row_correlation(reconstructed, brain))


def _corrected_data_quality_score(
    cleaned: np.ndarray, raw: np.ndarray, brain: np.ndarray
) -> tuple[float, float, float]:
    correlations = _row_correlation(cleaned, brain)
    uncorrected = float(100.0 * np.mean(np.sum(correlations**2, axis=1)))
    post_best = _best_possible_ground_truth_correlation(cleaned, brain)
    raw_best = _best_possible_ground_truth_correlation(raw, brain)
    ratios = (post_best**2) / np.maximum(raw_best**2, _EPS)
    correction = float(np.min(ratios))
    return uncorrected * correction, uncorrected, correction


def _band_power(data, sfreq=512.0, low=8.0, high=30.0):
    frequencies, psd = signal.welch(
        data, sfreq, nperseg=min(1024, data.shape[1]), axis=1
    )
    keep = (frequencies >= low) & (frequencies <= high)
    return float(np.trapezoid(psd[:, keep], frequencies[keep], axis=1).mean())


def _effective_rank(data: np.ndarray) -> int:
    covariance = data @ data.T / max(data.shape[1] - 1, 1)
    return int(np.linalg.matrix_rank(covariance))


def _fit_clean_forward(clean_primary: np.ndarray, ground_truth: np.ndarray, split: int):
    brain = ground_truth[:10]
    design = np.vstack([brain[:, :split], np.ones((1, split))])
    coefficients = clean_primary[:, :split] @ np.linalg.pinv(design)
    return coefficients @ np.vstack([brain, np.ones((1, brain.shape[1]))])


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
    repeat = int(args.repeat)
    locked_repeat = int(cfg["dataset"]["locked_repeat"])
    if repeat == locked_repeat and args.allow_dirty:
        raise RuntimeError("the locked repeat cannot be run with --allow-dirty")
    manifest_path = pathlib.Path(args.dataset_manifest).resolve()
    manifest = _read_manifest(manifest_path, repeat=repeat)
    del manifest

    dataset_root = pathlib.Path(args.dataset_root).resolve()
    imported = dataset_root / "derivatives" / "Data" / "Imported"
    gt_path = dataset_root / cfg["dataset"]["ground_truth_path"]
    ground_truth = _load_ground_truth(gt_path)
    conditions = list(cfg["dataset"]["conditions"])
    if args.conditions:
        requested = args.conditions.split(",")
        conditions = [condition for condition in conditions if condition in requested]
    methods = list(
        dict.fromkeys(
            [
                *cfg["tiers"]["reference_free"],
                *cfg["tiers"]["reference_aware"],
                *cfg["comparators"]["required"],
            ]
        )
    )
    if args.methods:
        requested = args.methods.split(",")
        methods = [method for method in methods if method in requested]
    if args.smoke:
        if not args.conditions:
            conditions = [condition for condition in conditions if condition in {"Clean", "All"}]
        if not args.methods:
            methods = [
                method
                for method in methods
                if method in {"none", "icanclean_dual_layer", "oracle"}
            ]

    resample_hz = float(cfg["analysis_policy"]["resample_hz"])
    recordings = {}
    for condition in {"Clean", *conditions}:
        path = imported / cfg["dataset"]["filename_template"].format(
            condition=condition, repeat=repeat
        )
        recordings[condition] = _load_recording(path, resample_hz=resample_hz)
        if args.smoke:
            n_samples = min(
                recordings[condition]["primary"].shape[1], int(round(30 * resample_hz))
            )
            for key in ("primary", "reference", "score_reference"):
                recordings[condition][key] = recordings[condition][key][:, :n_samples]

    if args.smoke:
        usable = int(round(30 * 512))
        ground_truth = ground_truth[:, :usable]
    clean_aligned, _ = _align_to_ground_truth(
        recordings["Clean"]["primary"], raw_start_fraction=0.0, ground_truth=ground_truth
    )
    fit_fraction = float(cfg["analysis_policy"]["phantom_fit_fraction"])
    gt_split = int(round(fit_fraction * ground_truth.shape[1]))
    clean_target = _fit_clean_forward(clean_aligned, ground_truth, gt_split)[:, gt_split:]
    gt_brain_score = ground_truth[:10, gt_split:]

    root = pathlib.Path(args.output_root).resolve()
    rows = []
    for condition in conditions:
        recording = recordings[condition]
        split = int(round(fit_fraction * recording["primary"].shape[1]))
        primary_fit = recording["primary"][:, :split]
        primary_score = recording["primary"][:, split:]
        reference_fit = recording["reference"][:, :split]
        reference_score = recording["reference"][:, split:]
        raw_aligned, _ = _align_to_ground_truth(
            primary_score, raw_start_fraction=fit_fraction, ground_truth=ground_truth
        )
        external_aligned, _ = _align_to_ground_truth(
            recording["score_reference"][:, split:],
            raw_start_fraction=fit_fraction,
            ground_truth=ground_truth,
        )
        sample_count = min(
            raw_aligned.shape[1], clean_target.shape[1], gt_brain_score.shape[1]
        )
        raw_aligned = raw_aligned[:, :sample_count]
        target = clean_target[:, :sample_count]
        brain = gt_brain_score[:, :sample_count]
        external = external_aligned[:, :sample_count]

        for method in methods:
            tier = (
                "oracle"
                if method == "oracle"
                else "reference_aware"
                if method in cfg["tiers"]["reference_aware"]
                else "reference_free"
            )
            unit_id = f"repeat{repeat}_{condition}"
            method_dir = root / unit_id / method
            record = build_run_record(
                arm="icanclean_ds004784_phantom",
                method=method,
                unit_id=unit_id,
                config_path=config_path,
                dataset_manifest=manifest_path,
                repo_root=_REPO,
                seed=None,
                information_tier=tier,
                allow_dirty=args.allow_dirty,
            )
            try:
                with AttemptRecorder(method_dir, record) as active:
                    if method == "oracle":
                        aligned, model, removed = target.copy(), None, 0
                    else:
                        output, model, removed = _clean_real(
                            method,
                            primary_fit,
                            primary_score,
                            reference_fit,
                            reference_score,
                            resample_hz,
                            cfg,
                        )
                        aligned, _ = _align_to_ground_truth(
                            output, raw_start_fraction=fit_fraction, ground_truth=ground_truth
                        )
                        aligned = aligned[:, :sample_count]
                    dqs, dqs_raw, correction = _corrected_data_quality_score(
                        aligned, raw_aligned, brain
                    )
                    rank_before = _effective_rank(raw_aligned)
                    rank_after = _effective_rank(aligned)
                    metrics = {
                        "arm": "icanclean_ds004784_phantom",
                        "unit_id": unit_id,
                        "condition": condition,
                        "technical_repeat": repeat,
                        "evidence_role": (
                            "locked_technical_replication"
                            if repeat == locked_repeat
                            else "development_parameter_selection"
                        ),
                        "method": method,
                        "information_tier": tier,
                        "status": "success",
                        "ground_truth_data_quality_score": dqs,
                        "ground_truth_data_quality_score_uncorrected": dqs_raw,
                        "ground_truth_preservation_correction": correction,
                        "known_clean_waveform_relative_rmse": relative_rmse(aligned, target),
                        "clean_correlation": clean_correlation(aligned, target),
                        "neural_band_retention": _band_power(aligned)
                        / max(_band_power(target), _EPS),
                        "disjoint_reference_coupling": float(
                            np.max(np.abs(_row_correlation(aligned, external)))
                        ),
                        "rank_before": rank_before,
                        "rank_after": rank_after,
                        "removed_components": removed,
                        "scored_samples_at_512_hz": sample_count,
                        "manifest_content_sha256": json.loads(
                            manifest_path.read_text(encoding="utf-8")
                        )["content_sha256"],
                        "score_definition": "Downey_Ferris_corrected_DQS_on_heldout_60_percent",
                    }
                    active.effective_rank_before = rank_before
                    active.effective_rank_after = rank_after
                    (method_dir / "metrics.json").write_text(
                        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
                    )
                    (method_dir / "model.json").write_text(
                        json.dumps(
                            {
                                "class": None if model is None else type(model).__name__,
                                "removed_components": removed,
                                "fit_scope": cfg["analysis_policy"]["fit_transform_scope"],
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
                        "arm": "icanclean_ds004784_phantom",
                        "unit_id": unit_id,
                        "condition": condition,
                        "technical_repeat": repeat,
                        "method": method,
                        "status": "failed",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
    _write_tsv(root / "raw_metrics.tsv", rows)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(_REPO / "configs/benchmarks/icanclean_reference_ground_truth.yaml"),
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--repeat", choices=("1", "2"), default="2")
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument(
        "--output-root", default=str(_REPO / "results/icanclean_ds004784_phantom")
    )
    parser.add_argument("--conditions", help="Comma-separated subset")
    parser.add_argument("--methods", help="Comma-separated subset")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)
    rows = run(args)
    successes = sum(row.get("status") == "success" for row in rows)
    print(f"attempts={len(rows)} successes={successes} output={args.output_root}")
    return 0 if successes == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
