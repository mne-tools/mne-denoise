"""Paper-driven ASR validation.

Reproduces the headline metrics from:
- Chang et al. 2020 (IEEE TBME 67:1114-1121): cutoff sweep, % data modified,
  % variance reduced.
- Blum et al. 2019 (Front. Hum. Neurosci. 13:141): computation time per variant,
  eye-blink amplitude reduction at Fp1/Fp2.
- Kim et al. 2025 (J. Neurosci. Methods 420:110465): PSD 1/f-bump parameter,
  reference-data fraction selected per variant.

This script does NOT cover the ICLabel / dipolar-IC metrics from those papers
(Tier B/C). Tier A metrics are designed to run end-to-end without optional
deps beyond mne, numpy, scipy, matplotlib.
"""

# ruff: noqa: I001

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy import optimize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mne_denoise.asr import ASR, AdaptiveASR, JugglerASR

DEFAULT_CUTOFFS = (1.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0, 50.0, 70.0, 100.0, 120.0)

VARIANTS = (
    "standard",
    "standard_lowmem",
    "riemannian",
    "adaptive_psw",
    "adaptive_psp",
    "juggler_dbscan",
    "juggler_gev",
)

FRONTAL_CHANNELS = ("Fp1", "Fp2", "AFp1", "AFp2", "EOG1", "EOG2", "VEOG")


@dataclass
class DatasetSpec:
    """Specification of one validation dataset."""

    path: Path
    label: str
    reader: str  # 'brainvision', 'edf', 'set', 'fif'
    max_duration_s: float
    resample_hz: float
    highpass_hz: float


def main() -> int:
    args = _parse_args()
    out_root = args.output_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    datasets = _collect_datasets(args)
    if not datasets:
        raise SystemExit(
            "No datasets to validate. Use --data-file to add explicit files."
        )

    summary: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "mne_version": mne.__version__,
        "numpy_version": np.__version__,
        "config": {
            "cutoff_sweep_variants": list(args.sweep_variants),
            "cutoffs": list(args.cutoffs),
            "cross_dataset_cutoff": float(args.cross_cutoff),
            "datasets": [
                {
                    "label": ds.label,
                    "path": str(ds.path),
                    "max_duration_s": ds.max_duration_s,
                    "resample_hz": ds.resample_hz,
                    "highpass_hz": ds.highpass_hz,
                }
                for ds in datasets
            ],
        },
        "results": [],
    }

    for ds in datasets:
        raw = _load_dataset(ds)
        ds_dir = out_root / ds.label
        ds_dir.mkdir(parents=True, exist_ok=True)
        ds_result = _validate_dataset(
            raw,
            ds,
            cutoffs=args.cutoffs,
            sweep_variants=args.sweep_variants,
            cross_cutoff=args.cross_cutoff,
            random_state=args.random_state,
            inject_bursts=not args.no_inject,
            burst_count=args.burst_count,
            burst_duration=args.burst_duration,
            burst_amplitude=args.burst_amplitude,
            out_dir=ds_dir,
        )
        summary["results"].append(ds_result)
        del raw
        gc.collect()

    _write_summary(summary, out_root)
    print(f"\nWrote summary to {out_root / 'summary.json'}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-file",
        action="append",
        type=Path,
        default=[],
        help="Additional EEG file to validate",
    )
    parser.add_argument(
        "--cutoffs",
        type=float,
        nargs="+",
        default=list(DEFAULT_CUTOFFS),
        help="Cutoff values for the sweep (Chang 2020 style)",
    )
    parser.add_argument(
        "--sweep-variants",
        nargs="+",
        default=["standard", "adaptive_psw", "juggler_gev"],
        choices=VARIANTS,
        help="Variants used in the full cutoff sweep (others run only at cross-cutoff)",
    )
    parser.add_argument(
        "--cross-cutoff",
        type=float,
        default=20.0,
        help="Cutoff used for the cross-dataset comparison (Chang 2020 recommended 20-30)",
    )
    parser.add_argument("--max-duration", type=float, default=120.0)
    parser.add_argument("--resample", type=float, default=250.0)
    parser.add_argument("--highpass", type=float, default=1.0)
    parser.add_argument("--burst-count", type=int, default=8)
    parser.add_argument("--burst-duration", type=float, default=0.5)
    parser.add_argument("--burst-amplitude", type=float, default=12.0)
    parser.add_argument("--no-inject", action="store_true")
    parser.add_argument("--random-state", type=int, default=97)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "reports" / "paper_validation",
    )
    return parser.parse_args()


def _collect_datasets(args: argparse.Namespace) -> list[DatasetSpec]:
    explicit = [_dataset_for_path(p, args) for p in args.data_file]
    if explicit:
        return explicit

    cache_root = ROOT / ".cache" / "asr_datasets"
    candidates = [
        (
            "mobile_bci_erp",
            cache_root
            / "mobile_bci"
            / "sub-01"
            / "ses-02"
            / "eeg"
            / "sub-01_ses-02_task-ERP_eeg.vhdr",
        ),
        (
            "mobile_bci_ssvep",
            cache_root
            / "mobile_bci"
            / "sub-01"
            / "ses-02"
            / "eeg"
            / "sub-01_ses-02_task-SSVEP_eeg.vhdr",
        ),
        (
            "ds003775_rest_ec",
            cache_root
            / "ds003775"
            / "sub-001"
            / "ses-t1"
            / "eeg"
            / "sub-001_ses-t1_task-resteyesc_eeg.edf",
        ),
    ]
    datasets = []
    for label, path in candidates:
        if not path.exists():
            continue
        datasets.append(
            DatasetSpec(
                path=path,
                label=label,
                reader=_reader_for_path(path),
                max_duration_s=args.max_duration,
                resample_hz=args.resample,
                highpass_hz=args.highpass,
            )
        )
    return datasets


def _dataset_for_path(path: Path, args: argparse.Namespace) -> DatasetSpec:
    label = path.stem
    return DatasetSpec(
        path=path,
        label=label,
        reader=_reader_for_path(path),
        max_duration_s=args.max_duration,
        resample_hz=args.resample,
        highpass_hz=args.highpass,
    )


def _reader_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".vhdr": "brainvision",
        ".edf": "edf",
        ".set": "set",
        ".fif": "fif",
    }.get(suffix, "")


def _load_dataset(ds: DatasetSpec) -> mne.io.BaseRaw:
    if ds.reader == "brainvision":
        raw = mne.io.read_raw_brainvision(ds.path, preload=False, verbose=False)
    elif ds.reader == "edf":
        raw = mne.io.read_raw_edf(ds.path, preload=False, verbose=False)
    elif ds.reader == "set":
        raw = mne.io.read_raw_eeglab(ds.path, preload=False, verbose=False)
    elif ds.reader == "fif":
        raw = mne.io.read_raw_fif(ds.path, preload=False, verbose=False)
    else:
        raise ValueError(f"Unsupported file: {ds.path}")
    raw.load_data(verbose=False)
    if ds.max_duration_s > 0:
        tmax = min(ds.max_duration_s, raw.n_times / raw.info["sfreq"])
        raw.crop(tmin=0.0, tmax=tmax, include_tmax=False)
    if ds.resample_hz > 0 and raw.info["sfreq"] > ds.resample_hz:
        raw.resample(ds.resample_hz, verbose=False)
    picks = mne.pick_types(raw.info, eeg=True, meg=False, exclude=[])
    if len(picks) == 0:
        raise RuntimeError(f"No EEG channels in {ds.path}")
    if ds.highpass_hz > 0:
        raw.filter(ds.highpass_hz, None, picks=picks, verbose=False)
    # Per Kothe/Mullen, drop obviously bad channels before ASR.
    # Anything with std > 5× the channel-wise median std is flagged.
    eeg_data = raw.get_data(picks=picks)
    stds = np.std(eeg_data, axis=1)
    median_std = float(np.median(stds))
    threshold = max(median_std * 5.0, 500e-6)  # 500 µV cap also catches drift
    bad_idx = np.where(stds > threshold)[0]
    bad_names = [raw.ch_names[picks[i]] for i in bad_idx]
    if bad_names:
        raw.info["bads"] = list(set(raw.info["bads"]) | set(bad_names))
        print(
            f"[{ds.label}] Pre-ASR bad channels: {bad_names} "
            f"(median std {median_std:.2e} V, threshold {threshold:.2e} V)"
        )
    return raw


def _validate_dataset(
    raw: mne.io.BaseRaw,
    ds: DatasetSpec,
    *,
    cutoffs: list[float],
    sweep_variants: list[str],
    cross_cutoff: float,
    random_state: int,
    inject_bursts: bool,
    burst_count: int,
    burst_duration: float,
    burst_amplitude: float,
    out_dir: Path,
) -> dict[str, Any]:
    eeg_picks = mne.pick_types(raw.info, eeg=True, meg=False, exclude=[])
    rng = np.random.default_rng(random_state)

    if inject_bursts:
        input_raw, burst_mask, injection_info = _inject_bursts(
            raw,
            eeg_picks=eeg_picks,
            rng=rng,
            n_bursts=burst_count,
            burst_duration=burst_duration,
            amplitude=burst_amplitude,
        )
    else:
        input_raw = raw.copy()
        burst_mask = np.zeros(raw.n_times, dtype=bool)
        injection_info = {"enabled": False}

    psd_before = _compute_psd(input_raw, eeg_picks)
    frontal_picks = _resolve_frontal_picks(raw)

    sweep_rows: list[dict[str, Any]] = []
    cross_rows: list[dict[str, Any]] = []

    # 1. Cutoff sweep for selected variants
    for variant in sweep_variants:
        for cutoff in cutoffs:
            row = _run_variant(
                variant,
                cutoff,
                input_raw,
                eeg_picks,
                frontal_picks,
                psd_before,
                burst_mask,
                random_state,
            )
            row["dataset"] = ds.label
            row["mode"] = "cutoff_sweep"
            sweep_rows.append(row)
            print(
                f"[{ds.label}] sweep {variant:18s} k={cutoff:>6.1f}  "
                f"status={row['status']:9s} t={row.get('wall_time_s', 0):.1f}s  "
                f"pct_mod={row.get('pct_data_modified', 0) * 100:5.1f}%  "
                f"pct_var={row.get('pct_variance_reduced', 0) * 100:5.1f}%"
            )

    # 2. Cross-variant comparison at cross-cutoff (variants not already swept)
    cross_variants = [v for v in VARIANTS if v not in sweep_variants]
    for variant in cross_variants:
        row = _run_variant(
            variant,
            cross_cutoff,
            input_raw,
            eeg_picks,
            frontal_picks,
            psd_before,
            burst_mask,
            random_state,
        )
        row["dataset"] = ds.label
        row["mode"] = "cross_variant"
        cross_rows.append(row)
        print(
            f"[{ds.label}] cross {variant:18s} k={cross_cutoff:>6.1f}  "
            f"status={row['status']:9s} t={row.get('wall_time_s', 0):.1f}s  "
            f"pct_mod={row.get('pct_data_modified', 0) * 100:5.1f}%  "
            f"pct_var={row.get('pct_variance_reduced', 0) * 100:5.1f}%"
        )

    rows = sweep_rows + cross_rows

    # Save raw rows
    json_path = out_dir / "rows.json"
    json_path.write_text(json.dumps(_json_safe(rows), indent=2), encoding="utf-8")

    csv_path = out_dir / "rows.csv"
    _write_csv(rows, csv_path)

    # Generate plots
    _plot_cutoff_sweep(sweep_rows, out_dir, ds.label)
    _plot_computation_time(rows, out_dir, ds.label, cross_cutoff)
    _plot_reference_fraction(rows, out_dir, ds.label)
    _plot_psd_overlay(rows, psd_before, out_dir, ds.label, cross_cutoff)
    _plot_blink_reduction(rows, out_dir, ds.label, cross_cutoff)

    return {
        "label": ds.label,
        "path": str(ds.path),
        "n_eeg": int(len(eeg_picks)),
        "sfreq": float(input_raw.info["sfreq"]),
        "duration_s": float(input_raw.n_times / input_raw.info["sfreq"]),
        "injection": injection_info,
        "frontal_channels": list(frontal_picks["names"]),
        "n_rows": len(rows),
        "rows_path": str(json_path),
        "csv_path": str(csv_path),
        "plot_paths": sorted(str(p) for p in out_dir.glob("*.png")),
    }


def _run_variant(
    variant: str,
    cutoff: float,
    input_raw: mne.io.BaseRaw,
    eeg_picks: np.ndarray,
    frontal_picks: dict[str, Any],
    psd_before: dict[str, Any],
    burst_mask: np.ndarray,
    random_state: int,
) -> dict[str, Any]:
    sfreq = input_raw.info["sfreq"]
    common: dict[str, Any] = {
        "sfreq": sfreq,
        "cutoff": cutoff,
        "picks": "eeg",
        "max_mem_mb": 512,
        "copy": True,
        "random_state": random_state,
        "verbose": False,
    }
    try:
        if variant == "standard":
            est = ASR(**common)
        elif variant == "standard_lowmem":
            est = ASR(**{**common, "max_mem_mb": 8})
        elif variant == "riemannian":
            est = ASR(**common, method="riemannian", experimental=True)
        elif variant == "adaptive_psw":
            est = AdaptiveASR(**{**common, "max_mem_mb": 8}, variant="psw")
        elif variant == "adaptive_psp":
            est = AdaptiveASR(**{**common, "max_mem_mb": 8}, variant="psp")
        elif variant == "juggler_dbscan":
            est = JugglerASR(**{**common, "max_mem_mb": 8}, strategy="dbscan")
        elif variant == "juggler_gev":
            est = JugglerASR(**{**common, "max_mem_mb": 8}, strategy="gev")
        else:
            raise ValueError(f"Unknown variant {variant}")

        t0 = time.perf_counter()
        cleaned = est.fit_transform(input_raw)
        dt = time.perf_counter() - t0
    except Exception as exc:  # noqa: BLE001
        return {
            "variant": variant,
            "cutoff": cutoff,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(limit=4),
        }

    before = input_raw.get_data(picks=eeg_picks)
    after = cleaned.get_data(picks=eeg_picks)
    diag = est.diagnostics_

    # Chang 2020 Fig 2a: fraction of WINDOWS in which any component was
    # rejected. Sample-mask fraction is always ~100% for standard ASR because
    # its sliding cosine blend touches every sample.
    pct_data_modified = float(diag.get("fraction_reconstructed_windows", 0.0))
    pct_data_modified_samples = float(
        np.mean(diag.get("sample_mask", np.zeros(before.shape[1])))
    )
    var_before = float(np.var(before))
    var_after = float(np.var(after))
    pct_variance_reduced = (var_before - var_after) / max(
        var_before, np.finfo(float).eps
    )
    # Variance reduction restricted to non-burst (background) samples — closer
    # to Chang 2020's "background variance change" because injected bursts
    # dominate the global variance.
    if np.any(burst_mask) and np.any(~burst_mask):
        bg = ~burst_mask
        var_bg_before = float(np.var(before[:, bg]))
        var_bg_after = float(np.var(after[:, bg]))
        pct_variance_reduced_bg = (var_bg_before - var_bg_after) / max(
            var_bg_before, np.finfo(float).eps
        )
    else:
        pct_variance_reduced_bg = pct_variance_reduced

    # Burst-region attenuation (sanity)
    burst_attenuation = (
        _burst_attenuation_db(input_raw, cleaned, eeg_picks, burst_mask)
        if np.any(burst_mask)
        else None
    )

    # PSD 1/f-bump fit (Kim 2025)
    psd_after = _compute_psd(cleaned, eeg_picks)
    bump_before = _fit_psd_bump_parameter(psd_before)
    bump_after = _fit_psd_bump_parameter(psd_after)

    # Eye-blink reduction (Blum 2019) on frontal channels
    blink = _blink_amplitude_change(
        input_raw,
        cleaned,
        frontal_picks,
    )

    # Reference data fraction (Kim 2025 Fig 8)
    calibration_info = getattr(est, "calibration_info_", {}) or {}
    ref_fraction = _reference_fraction(calibration_info)

    # Mean components reconstructed per window (Chang 2020 Fig 4)
    mean_components_reconstructed = (
        float(np.mean(diag.get("n_components_reconstructed", [0.0])))
        if "n_components_reconstructed" in diag
        else 0.0
    )

    return {
        "variant": variant,
        "cutoff": cutoff,
        "status": "passed",
        "wall_time_s": dt,
        "pct_data_modified": pct_data_modified,
        "pct_data_modified_samples": pct_data_modified_samples,
        "pct_variance_reduced": float(pct_variance_reduced),
        "pct_variance_reduced_bg": float(pct_variance_reduced_bg),
        "variance_before": var_before,
        "variance_after": var_after,
        "burst_attenuation_db": burst_attenuation,
        "psd_bump_before": bump_before,
        "psd_bump_after": bump_after,
        "psd_bump_reduction": (
            bump_before["bump_height"] - bump_after["bump_height"]
            if bump_before and bump_after
            else None
        ),
        "blink_before_uv_pp": blink.get("before_uv_pp"),
        "blink_after_uv_pp": blink.get("after_uv_pp"),
        "blink_reduction_pct": blink.get("reduction_pct"),
        "reference_fraction": ref_fraction,
        "mean_components_reconstructed": mean_components_reconstructed,
        "memory_mode_proc": diag.get("memory_mode", "unknown"),
        "memory_mode_cal": calibration_info.get("memory_mode", "unknown"),
        "psd_freqs": psd_before["freqs"].tolist(),
        "psd_before_uv2_hz": psd_before["mean_db"].tolist(),
        "psd_after_uv2_hz": psd_after["mean_db"].tolist(),
    }


def _inject_bursts(
    raw: mne.io.BaseRaw,
    *,
    eeg_picks: np.ndarray,
    rng: np.random.Generator,
    n_bursts: int,
    burst_duration: float,
    amplitude: float,
) -> tuple[mne.io.BaseRaw, np.ndarray, dict[str, Any]]:
    corrupt = raw.copy()
    sfreq = float(raw.info["sfreq"])
    burst_len = max(1, int(round(burst_duration * sfreq)))
    guard = max(burst_len, int(round(2.0 * sfreq)))
    n_times = raw.n_times
    if n_times <= 2 * guard + burst_len:
        starts = np.array([max(0, (n_times - burst_len) // 2)], dtype=int)
    else:
        starts = np.linspace(guard, n_times - guard - burst_len, n_bursts).astype(int)
    mask = np.zeros(n_times, dtype=bool)
    eeg = raw.get_data(picks=eeg_picks)
    channel_scale = float(max(np.median(np.std(eeg, axis=1)), np.finfo(float).eps))
    data = corrupt._data
    for start in starts:
        stop = min(start + burst_len, n_times)
        actual_len = stop - start
        spatial = rng.standard_normal(len(eeg_picks))
        spatial /= max(np.linalg.norm(spatial), np.finfo(float).eps)
        temporal = rng.standard_normal(actual_len)
        temporal += 0.75 * np.sin(np.linspace(0, 6 * np.pi, actual_len))
        data[eeg_picks, start:stop] += (
            amplitude * channel_scale * np.outer(spatial, temporal)
        )
        mask[start:stop] = True
    return (
        corrupt,
        mask,
        {
            "enabled": True,
            "n_bursts": int(len(starts)),
            "burst_duration_s": burst_duration,
            "burst_samples": int(np.sum(mask)),
            "burst_fraction": float(np.mean(mask)),
            "amplitude_scale": amplitude,
        },
    )


def _burst_attenuation_db(
    input_raw: mne.io.BaseRaw,
    cleaned: mne.io.BaseRaw,
    eeg_picks: np.ndarray,
    burst_mask: np.ndarray,
) -> float:
    before = input_raw.get_data(picks=eeg_picks)
    after = cleaned.get_data(picks=eeg_picks)
    rms_before = np.sqrt(np.mean(before[:, burst_mask] ** 2))
    rms_after = np.sqrt(np.mean(after[:, burst_mask] ** 2))
    return float(20.0 * np.log10((rms_before + 1e-30) / (rms_after + 1e-30)))


def _resolve_frontal_picks(raw: mne.io.BaseRaw) -> dict[str, Any]:
    chs = raw.ch_names
    selected = [c for c in chs if c in FRONTAL_CHANNELS]
    if not selected:
        # Fallback: take the two most frontal-named channels
        candidates = [c for c in chs if c.lower().startswith(("fp", "af"))]
        selected = candidates[:2]
    if not selected:
        selected = chs[:2]
    indices = np.asarray(
        mne.pick_channels(chs, include=selected, ordered=True), dtype=int
    )
    return {"names": selected, "indices": indices}


def _blink_amplitude_change(
    input_raw: mne.io.BaseRaw,
    cleaned: mne.io.BaseRaw,
    frontal_picks: dict[str, Any],
) -> dict[str, Any]:
    if frontal_picks["indices"].size == 0:
        return {}
    before = input_raw.get_data(picks=frontal_picks["indices"])
    after = cleaned.get_data(picks=frontal_picks["indices"])
    # Peak-to-peak in microvolts (data is in V)
    before_pp = float(np.percentile(np.ptp(before, axis=1), 95) * 1e6)
    after_pp = float(np.percentile(np.ptp(after, axis=1), 95) * 1e6)
    reduction = (before_pp - after_pp) / max(before_pp, np.finfo(float).eps) * 100.0
    return {
        "before_uv_pp": before_pp,
        "after_uv_pp": after_pp,
        "reduction_pct": float(reduction),
    }


def _compute_psd(raw: mne.io.BaseRaw, eeg_picks: np.ndarray) -> dict[str, Any]:
    sfreq = raw.info["sfreq"]
    fmax = min(80.0, sfreq / 2 - 1)
    psd = raw.compute_psd(
        picks=eeg_picks,
        fmin=1.0,
        fmax=fmax,
        verbose=False,
    )
    freqs = np.asarray(psd.freqs)
    data = np.asarray(psd.get_data())  # (n_channels, n_freqs)
    mean_uv2 = np.mean(data, axis=0) * 1e12  # V^2/Hz to uV^2/Hz
    mean_db = 10.0 * np.log10(mean_uv2 + 1e-30)
    return {"freqs": freqs, "mean_uv2": mean_uv2, "mean_db": mean_db}


def _fit_psd_bump_parameter(psd: dict[str, Any]) -> dict[str, Any] | None:
    """Kim 2025 1/f + Gaussian bump fit. Returns bump_height and 1/f params."""
    freqs = psd["freqs"]
    mean_uv2 = psd["mean_uv2"]
    mask = (freqs >= 3.0) & (freqs <= 55.0)
    f = freqs[mask]
    y = mean_uv2[mask]
    if f.size < 10:
        return None

    def model(
        x: np.ndarray, p1: float, p2: float, p3: float, p4: float, p5: float
    ) -> np.ndarray:
        return p1 / np.power(x, p2) + p3 * np.exp(-((x - p4) ** 2) / (2 * p5**2))

    p0 = [float(y[0]) * f[0], 1.0, float(y[-1]), 25.0, 10.0]
    bounds = (
        [0.0, 0.0, 0.0, 5.0, 0.1],
        [np.inf, 5.0, np.inf, 80.0, 30.0],
    )
    try:
        popt, _ = optimize.curve_fit(model, f, y, p0=p0, bounds=bounds, maxfev=8000)
    except Exception:  # noqa: BLE001
        return None
    y_pred = model(f, *popt)
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, np.finfo(float).eps)
    return {
        "scale": float(popt[0]),
        "exponent": float(popt[1]),
        "bump_height": float(popt[2]),
        "bump_center": float(popt[3]),
        "bump_width": float(popt[4]),
        "r2": float(r2),
    }


def _reference_fraction(calibration_info: dict[str, Any]) -> float | None:
    if not calibration_info:
        return None
    # Juggler-specific (point-by-point reference selection)
    n_sel = calibration_info.get("reference_selected_samples")
    n_cand = calibration_info.get("reference_candidate_samples")
    if n_sel is not None and n_cand is not None and n_cand > 0:
        return float(n_sel) / float(n_cand)
    # Standard/adaptive ASR: window-based clean fraction
    n_clean = calibration_info.get("n_clean_windows")
    n_tot = calibration_info.get("n_calibration_windows")
    if n_clean is not None and n_tot is not None and n_tot > 0:
        return float(n_clean) / float(n_tot)
    # Older naming
    if "calibration_clean_window_fraction" in calibration_info:
        return float(calibration_info["calibration_clean_window_fraction"])
    if "calibration_clean_sample_fraction" in calibration_info:
        return float(calibration_info["calibration_clean_sample_fraction"])
    return None


def _plot_cutoff_sweep(
    sweep_rows: list[dict[str, Any]],
    out_dir: Path,
    label: str,
) -> None:
    if not sweep_rows:
        return
    variants = sorted({r["variant"] for r in sweep_rows})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for variant in variants:
        rows = sorted(
            [r for r in sweep_rows if r["variant"] == variant],
            key=lambda r: r["cutoff"],
        )
        passed = [r for r in rows if r["status"] == "passed"]
        if not passed:
            continue
        ks = [r["cutoff"] for r in passed]
        modified = [r["pct_data_modified"] * 100 for r in passed]
        var_reduced = [
            r.get("pct_variance_reduced_bg", r["pct_variance_reduced"]) * 100
            for r in passed
        ]
        ax1.plot(ks, modified, marker="o", label=variant)
        ax2.plot(ks, var_reduced, marker="o", label=variant)
    for ax, ylab, ttl in (
        (
            ax1,
            "% windows with rejection",
            "Chang 2020 Fig 2a (fraction of windows where ASR rejected a PC)",
        ),
        (
            ax2,
            "% background-variance reduced",
            "Chang 2020 Fig 2a (non-burst variance reduction)",
        ),
    ):
        ax.set_xlabel("Cutoff k (SDs)")
        ax.set_ylabel(ylab)
        ax.set_xscale("log")
        ax.set_title(ttl)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle(f"{label} — cutoff sweep")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_chang2020_cutoff_sweep.png", dpi=140)
    plt.close(fig)


def _plot_computation_time(
    rows: list[dict[str, Any]],
    out_dir: Path,
    label: str,
    cross_cutoff: float,
) -> None:
    use = [
        r
        for r in rows
        if r["status"] == "passed" and abs(r["cutoff"] - cross_cutoff) < 1e-9
    ]
    if not use:
        return
    use = sorted(use, key=lambda r: r["wall_time_s"])
    fig, ax = plt.subplots(figsize=(8, 4))
    variants = [r["variant"] for r in use]
    times = [r["wall_time_s"] for r in use]
    ax.barh(variants, times, color="steelblue")
    for variant, t in zip(variants, times):
        ax.text(t * 1.01, variant, f"{t:.1f}s", va="center", fontsize=9)
    ax.set_xlabel("Wall time per fit_transform (s)")
    ax.set_title(f"{label} — Blum 2019 Fig 5 equivalent (k={cross_cutoff})")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_blum2019_computation_time.png", dpi=140)
    plt.close(fig)


def _plot_reference_fraction(
    rows: list[dict[str, Any]],
    out_dir: Path,
    label: str,
) -> None:
    # One value per (variant, cutoff)
    use = [
        r
        for r in rows
        if r["status"] == "passed" and r.get("reference_fraction") is not None
    ]
    if not use:
        return
    variants = sorted({r["variant"] for r in use})
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for variant in variants:
        sub = sorted(
            [r for r in use if r["variant"] == variant], key=lambda r: r["cutoff"]
        )
        ks = [r["cutoff"] for r in sub]
        fracs = [r["reference_fraction"] * 100 for r in sub]
        ax.plot(ks, fracs, marker="o", label=variant)
    ax.set_xscale("log")
    ax.set_xlabel("Cutoff k (SDs)")
    ax.set_ylabel("% reference samples / windows used")
    ax.set_title(f"{label} — Kim 2025 Fig 8 equivalent")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_kim2025_reference_fraction.png", dpi=140)
    plt.close(fig)


def _plot_psd_overlay(
    rows: list[dict[str, Any]],
    psd_before: dict[str, Any],
    out_dir: Path,
    label: str,
    cross_cutoff: float,
) -> None:
    use = [
        r
        for r in rows
        if r["status"] == "passed"
        and abs(r["cutoff"] - cross_cutoff) < 1e-9
        and "psd_freqs" in r
    ]
    if not use:
        return
    n_variants = len(use)
    n_cols = 3
    n_rows = (n_variants + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows), squeeze=False
    )
    freqs = psd_before["freqs"]
    db_before = psd_before["mean_db"]
    for i, row in enumerate(use):
        ax = axes[i // n_cols, i % n_cols]
        ax.plot(freqs, db_before, color="0.6", lw=1.0, label="Before ASR")
        ax.plot(
            row["psd_freqs"],
            row["psd_after_uv2_hz"],
            color="C0",
            lw=1.0,
            label="After ASR",
        )
        bump_before = row.get("psd_bump_before")
        bump_after = row.get("psd_bump_after")
        title = row["variant"]
        if bump_before and bump_after:
            title += (
                f"\nbump: {bump_before['bump_height']:.2g}"
                f" -> {bump_after['bump_height']:.2g}"
            )
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (dB)")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    for j in range(len(use), n_rows * n_cols):
        axes[j // n_cols, j % n_cols].axis("off")
    fig.suptitle(f"{label} — Kim 2025 Fig 9 equivalent (k={cross_cutoff})")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_kim2025_psd_overlay.png", dpi=140)
    plt.close(fig)


def _plot_blink_reduction(
    rows: list[dict[str, Any]],
    out_dir: Path,
    label: str,
    cross_cutoff: float,
) -> None:
    use = [
        r
        for r in rows
        if r["status"] == "passed"
        and abs(r["cutoff"] - cross_cutoff) < 1e-9
        and r.get("blink_before_uv_pp") is not None
    ]
    if not use:
        return
    use = sorted(use, key=lambda r: r["variant"])
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(use))
    width = 0.4
    before = [r["blink_before_uv_pp"] for r in use]
    after = [r["blink_after_uv_pp"] for r in use]
    ax.bar(x - width / 2, before, width, label="Before ASR", color="0.7")
    ax.bar(x + width / 2, after, width, label="After ASR", color="C0")
    ax.set_xticks(x)
    ax.set_xticklabels([r["variant"] for r in use], rotation=20, ha="right")
    ax.set_ylabel("Frontal Pp 95th percentile (µV)")
    ax.set_title(f"{label} — Blum 2019 Fig 3 equivalent (k={cross_cutoff})")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig_blum2019_blink_reduction.png", dpi=140)
    plt.close(fig)


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    scalar_keys = sorted(
        {
            k
            for r in rows
            for k, v in r.items()
            if k
            not in (
                "psd_freqs",
                "psd_before_uv2_hz",
                "psd_after_uv2_hz",
                "psd_bump_before",
                "psd_bump_after",
                "traceback",
            )
            and not isinstance(v, list | dict)
        }
    )
    with path.open("w", newline="", encoding="utf-8") as fid:
        writer = csv.DictWriter(fid, fieldnames=scalar_keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in scalar_keys})


def _write_summary(summary: dict[str, Any], out_root: Path) -> None:
    (out_root / "summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2), encoding="utf-8"
    )
    lines = [
        "# ASR Paper-Driven Validation",
        "",
        f"Generated: {summary['created_utc']}",
        "",
        "## Datasets",
        "",
        "| Label | EEG ch | Duration | sfreq | Frontal | Rows |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for ds in summary["results"]:
        lines.append(
            f"| {ds['label']} | {ds['n_eeg']} | {ds['duration_s']:.0f}s "
            f"| {ds['sfreq']:.0f}Hz | {', '.join(ds['frontal_channels'])} "
            f"| {ds['n_rows']} |"
        )
    lines.extend(["", "## Plots", ""])
    for ds in summary["results"]:
        lines.append(f"### {ds['label']}")
        lines.append("")
        for p in ds["plot_paths"]:
            rel = Path(p).relative_to(out_root)
            lines.append(f"- ![{Path(p).stem}]({rel.as_posix()})")
        lines.append("")
    (out_root / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
