"""Run real-data ASR validation and resource reporting.

This script is intentionally separate from the normal pytest suite because it
uses local EEG datasets and records machine-dependent CPU/RSS measurements.
It validates algorithm behavior on real MNE Raw objects plus known injected
bursts, then writes JSON, CSV, and Markdown summaries.
"""

# ruff: noqa: I001

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import math
import os
import platform
import sys
import threading
import time
import tracemalloc
import traceback
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mne
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mne_denoise.asr import ASR, AdaptiveASR, JugglerASR


DEFAULT_VARIANTS = (
    "standard",
    "standard_lowmem",
    "riemannian",
    "adaptive_psw",
    "adaptive_psp",
    "juggler_dbscan",
    "juggler_gev",
)

SCALAR_DIAGNOSTICS = (
    "memory_mode",
    "max_mem_mb",
    "used_memory_bound",
    "estimated_full_cov_bytes",
    "chunk_samples",
    "peak_cov_buffer_bytes",
    "fraction_reconstructed_samples",
    "fraction_reconstructed_windows",
    "mean_components_reconstructed",
    "max_components_reconstructed",
    "n_windows",
    "calibration_clean_window_fraction",
    "calibration_clean_sample_fraction",
    "n_clean_calibration_windows",
    "n_calibration_windows",
    "n_clean_calibration_samples",
    "n_calibration_candidate_samples",
)


@dataclass
class ValidationDataset:
    """Loaded dataset metadata and Raw object."""

    path: Path
    raw: mne.io.BaseRaw
    n_eeg: int


class ResourceMeter:
    """Measure wall time, CPU time, tracemalloc peak, and sampled process RSS."""

    def __init__(self, sample_interval: float = 0.05) -> None:
        self.sample_interval = sample_interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._rss_samples: list[int] = []
        self._started_tracemalloc = False
        self._start_rss: int | None = None
        self._end_rss: int | None = None
        self.summary: dict[str, Any] = {}

    def __enter__(self) -> ResourceMeter:
        self._start_wall = time.perf_counter()
        self._start_cpu = time.process_time()
        self._start_rss = _current_rss_bytes()
        if tracemalloc.is_tracing():
            tracemalloc.reset_peak()
        else:
            tracemalloc.start()
            self._started_tracemalloc = True
        self._thread = threading.Thread(target=self._sample_rss, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._end_rss = _current_rss_bytes()
        _, traced_peak = tracemalloc.get_traced_memory()
        if self._started_tracemalloc:
            tracemalloc.stop()
        end_wall = time.perf_counter()
        end_cpu = time.process_time()
        peak_rss = max(self._rss_samples, default=self._end_rss or self._start_rss or 0)
        start_rss = self._start_rss or peak_rss
        wall_s = end_wall - self._start_wall
        cpu_s = end_cpu - self._start_cpu
        cpu_count = os.cpu_count() or 1
        self.summary = {
            "wall_time_s": wall_s,
            "cpu_time_s": cpu_s,
            "cpu_utilization_pct_one_core": 100.0 * cpu_s / wall_s if wall_s else 0.0,
            "cpu_utilization_pct_all_cores": (
                100.0 * cpu_s / (wall_s * cpu_count) if wall_s else 0.0
            ),
            "rss_start_bytes": self._start_rss,
            "rss_end_bytes": self._end_rss,
            "rss_peak_bytes": peak_rss,
            "rss_peak_delta_bytes": max(0, peak_rss - start_rss),
            "tracemalloc_peak_bytes": traced_peak,
        }

    def _sample_rss(self) -> None:
        while not self._stop.is_set():
            rss = _current_rss_bytes()
            if rss is not None:
                self._rss_samples.append(int(rss))
            self._stop.wait(self.sample_interval)


def main() -> int:
    args = _parse_args()
    data_files = _discover_data_files(args)
    if not data_files:
        raise SystemExit(
            "No supported EEG data files found. Pass --data-file PATH or populate "
            ".cache/asr_datasets / data with .fif, .edf, .vhdr, or .set files."
        )

    output = args.output.resolve()
    results: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "mne_version": mne.__version__,
        "numpy_version": np.__version__,
        "config": {
            "variants": list(args.variants),
            "max_duration_s": args.max_duration,
            "resample_hz": args.resample,
            "highpass_hz": args.highpass,
            "cutoff": args.cutoff,
            "max_mem_mb": args.max_mem_mb,
            "low_mem_mb": args.low_mem_mb,
            "inject_bursts": not args.no_inject_bursts,
            "burst_count": args.burst_count,
            "burst_duration_s": args.burst_duration,
            "burst_amplitude": args.burst_amplitude,
        },
        "datasets": [],
    }

    failed = False
    for data_path in data_files[: args.max_files]:
        dataset = _load_dataset(data_path, args)
        dataset_result = _validate_dataset(dataset, args)
        results["datasets"].append(dataset_result)
        failed = failed or any(
            run["status"] != "passed" for run in dataset_result["runs"]
        )

    _write_reports(results, output)
    print(_markdown_report(results))
    print(f"\nWrote reports:\n- {output}\n- {output.with_suffix('.csv')}")
    print(f"- {output.with_suffix('.md')}")
    if failed and args.fail_on_error:
        return 1
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate ASR variants on local real EEG data and report CPU/RSS."
    )
    parser.add_argument("--data-file", action="append", type=Path, default=[])
    parser.add_argument("--data-root", action="append", type=Path, default=[])
    parser.add_argument("--max-files", type=int, default=1)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=DEFAULT_VARIANTS,
        default=list(DEFAULT_VARIANTS),
    )
    parser.add_argument("--max-duration", type=float, default=120.0)
    parser.add_argument("--resample", type=float, default=250.0)
    parser.add_argument("--highpass", type=float, default=1.0)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--max-mem-mb", type=float, default=512.0)
    parser.add_argument("--low-mem-mb", type=float, default=8.0)
    parser.add_argument("--no-inject-bursts", action="store_true")
    parser.add_argument("--burst-count", type=int, default=8)
    parser.add_argument("--burst-duration", type=float, default=0.5)
    parser.add_argument("--burst-amplitude", type=float, default=12.0)
    parser.add_argument("--random-state", type=int, default=97)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "asr_real_data_validation.json",
    )
    parser.add_argument("--fail-on-error", action="store_true")
    return parser.parse_args()


def _discover_data_files(args: argparse.Namespace) -> list[Path]:
    if args.data_file:
        return [path.resolve() for path in args.data_file]
    roots = args.data_root or [ROOT / ".cache" / "asr_datasets", ROOT / "data"]
    candidates: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for suffix in ("*.vhdr", "*.fif", "*.edf", "*.set"):
            candidates.extend(path for path in root.rglob(suffix) if path.is_file())
    return sorted(candidates, key=_dataset_priority)


def _dataset_priority(path: Path) -> tuple[int, str]:
    text = path.as_posix().lower()
    if "mobile_bci" in text and path.suffix.lower() == ".vhdr":
        return (0, text)
    if "ds003775" in text:
        return (1, text)
    if "erp-core" in text or "mne-erp-core" in text:
        return (2, text)
    return (10, text)


def _load_dataset(path: Path, args: argparse.Namespace) -> ValidationDataset:
    raw = _read_raw(path)
    raw.load_data(verbose=False)
    if args.max_duration and args.max_duration > 0:
        tmax = min(args.max_duration, raw.n_times / raw.info["sfreq"])
        raw.crop(tmin=0.0, tmax=tmax, include_tmax=False)
    if args.resample and args.resample > 0 and raw.info["sfreq"] > args.resample:
        raw.resample(args.resample, verbose=False)
    picks = mne.pick_types(raw.info, eeg=True, meg=False, exclude=[])
    if len(picks) == 0:
        raise RuntimeError(f"No EEG channels found in {path}")
    if args.highpass and args.highpass > 0:
        raw.filter(args.highpass, None, picks=picks, verbose=False)
    return ValidationDataset(path=path, raw=raw, n_eeg=len(picks))


def _read_raw(path: Path) -> mne.io.BaseRaw:
    suffix = path.suffix.lower()
    if suffix == ".fif":
        return mne.io.read_raw_fif(path, preload=False, verbose=False)
    if suffix == ".edf":
        return mne.io.read_raw_edf(path, preload=False, verbose=False)
    if suffix == ".vhdr":
        return mne.io.read_raw_brainvision(path, preload=False, verbose=False)
    if suffix == ".set":
        return mne.io.read_raw_eeglab(path, preload=False, verbose=False)
    raise ValueError(f"Unsupported EEG file suffix: {path.suffix}")


def _validate_dataset(
    dataset: ValidationDataset, args: argparse.Namespace
) -> dict[str, Any]:
    base_raw = dataset.raw
    eeg_picks = mne.pick_types(base_raw.info, eeg=True, meg=False, exclude=[])
    rng = np.random.default_rng(args.random_state)
    if args.no_inject_bursts:
        input_raw = base_raw.copy()
        burst_mask = np.zeros(base_raw.n_times, dtype=bool)
        injection_info = {"enabled": False}
    else:
        input_raw, burst_mask, injection_info = _inject_bursts(
            base_raw,
            eeg_picks=eeg_picks,
            rng=rng,
            n_bursts=args.burst_count,
            burst_duration=args.burst_duration,
            amplitude=args.burst_amplitude,
        )
    result = {
        "path": str(dataset.path),
        "n_channels": len(base_raw.ch_names),
        "n_eeg_channels": int(len(eeg_picks)),
        "n_times": int(base_raw.n_times),
        "sfreq": float(base_raw.info["sfreq"]),
        "duration_s": float(base_raw.n_times / base_raw.info["sfreq"]),
        "injection": injection_info,
        "runs": [],
    }
    for variant in args.variants:
        run = _run_variant(variant, base_raw, input_raw, burst_mask, eeg_picks, args)
        result["runs"].append(run)
    return result


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
    data = corrupt._data
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
    channel_scale = np.median(np.std(eeg, axis=1))
    channel_scale = float(max(channel_scale, np.finfo(float).eps))
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
            "burst_duration_s": float(burst_duration),
            "burst_samples": int(np.sum(mask)),
            "burst_fraction": float(np.mean(mask)),
            "amplitude_scale": float(amplitude),
        },
    )


def _run_variant(
    variant: str,
    original_raw: mne.io.BaseRaw,
    input_raw: mne.io.BaseRaw,
    burst_mask: np.ndarray,
    eeg_picks: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    estimator = _makecore(variant, input_raw.info["sfreq"], args)
    run: dict[str, Any] = {"variant": variant, "status": "passed"}
    try:
        with ResourceMeter() as meter:
            cleaned = estimator.fit_transform(input_raw)
        run.update(meter.summary)
        run.update(
            _compute_validation_metrics(
                original_raw, input_raw, cleaned, burst_mask, eeg_picks
            )
        )
        from mne_denoise.qa import variance_removed, max_abs_change

        run["qa"] = {
            "variance_removed_pct": variance_removed(
                input_raw.get_data(picks="eeg"), cleaned.get_data(picks="eeg")
            ),
            "max_abs_change": max_abs_change(
                input_raw.get_data(picks="eeg"), cleaned.get_data(picks="eeg")
            ),
        }
        run["calibration"] = _select_scalars(
            getattr(estimator, "calibration_info_", {})
        )
        run["diagnostics"] = _select_scalars(getattr(estimator, "diagnostics_", {}))
        run["checks"] = _metadata_checks(input_raw, cleaned)
        run["quality_checks"] = _quality_checks(run, require_burst=np.any(burst_mask))
        if not all(run["quality_checks"].values()):
            run["status"] = "failed_quality"
    except Exception as exc:  # noqa: BLE001 - report all validation failures.
        run["status"] = "failed"
        run["error_type"] = type(exc).__name__
        run["error"] = str(exc)
        run["traceback"] = traceback.format_exc(limit=8)
    return _json_safe(run)


def _makecore(
    variant: str, sfreq: float, args: argparse.Namespace
) -> ASR | AdaptiveASR | JugglerASR:
    common = {
        "sfreq": sfreq,
        "cutoff": args.cutoff,
        "picks": "eeg",
        "max_mem_mb": args.max_mem_mb,
        "copy": True,
        "random_state": args.random_state,
        "verbose": False,
    }
    if variant == "standard":
        return ASR(**common)
    if variant == "standard_lowmem":
        return ASR(**{**common, "max_mem_mb": args.low_mem_mb})
    if variant == "riemannian":
        return ASR(**common, method="riemannian", experimental=True)
    if variant == "adaptive_psw":
        return AdaptiveASR(**{**common, "max_mem_mb": args.low_mem_mb}, variant="psw")
    if variant == "adaptive_psp":
        return AdaptiveASR(**{**common, "max_mem_mb": args.low_mem_mb}, variant="psp")
    if variant == "juggler_dbscan":
        return JugglerASR(
            **{**common, "max_mem_mb": args.low_mem_mb}, strategy="dbscan"
        )
    if variant == "juggler_gev":
        return JugglerASR(**{**common, "max_mem_mb": args.low_mem_mb}, strategy="gev")
    raise ValueError(f"Unknown ASR variant: {variant}")


def _compute_validation_metrics(
    original_raw: mne.io.BaseRaw,
    input_raw: mne.io.BaseRaw,
    cleaned_raw: mne.io.BaseRaw,
    burst_mask: np.ndarray,
    eeg_picks: np.ndarray,
) -> dict[str, Any]:
    original = original_raw.get_data(picks=eeg_picks)
    corrupted = input_raw.get_data(picks=eeg_picks)
    cleaned = cleaned_raw.get_data(picks=eeg_picks)
    metrics: dict[str, Any] = {
        "shape_preserved": cleaned_raw.get_data().shape == input_raw.get_data().shape,
        "finite_output_fraction": float(np.mean(np.isfinite(cleaned))),
        "input_rms": _rms(corrupted),
        "output_rms": _rms(cleaned),
    }
    if np.any(burst_mask):
        artifact_before = _rms(corrupted[:, burst_mask] - original[:, burst_mask])
        artifact_after = _rms(cleaned[:, burst_mask] - original[:, burst_mask])
        background = ~burst_mask
        background_rmse = (
            _rms(cleaned[:, background] - original[:, background])
            if np.any(background)
            else 0.0
        )
        background_rms = _rms(original[:, background]) if np.any(background) else 0.0
        metrics.update(
            {
                "artifact_rms_before": artifact_before,
                "artifact_rms_after": artifact_after,
                "artifact_attenuation_db": 20.0
                * math.log10((artifact_before + 1e-30) / (artifact_after + 1e-30)),
                "background_distortion_ratio": background_rmse
                / max(background_rms, np.finfo(float).eps),
            }
        )
    return metrics


def _metadata_checks(
    input_raw: mne.io.BaseRaw, cleaned_raw: mne.io.BaseRaw
) -> dict[str, bool]:
    return {
        "raw_object_preserved": isinstance(cleaned_raw, mne.io.BaseRaw),
        "concrete_type_preserved": type(cleaned_raw) is type(input_raw),
        "channel_order_preserved": cleaned_raw.ch_names == input_raw.ch_names,
        "sfreq_preserved": cleaned_raw.info["sfreq"] == input_raw.info["sfreq"],
        "n_times_preserved": cleaned_raw.n_times == input_raw.n_times,
        "bads_preserved": cleaned_raw.info["bads"] == input_raw.info["bads"],
        "annotations_preserved": len(cleaned_raw.annotations)
        == len(input_raw.annotations),
    }


def _quality_checks(run: dict[str, Any], *, require_burst: bool) -> dict[str, bool]:
    metadata = run.get("checks", {})
    required_metadata = {
        key: value
        for key, value in metadata.items()
        if key != "concrete_type_preserved"
    }
    checks = {
        "shape_preserved": bool(run.get("shape_preserved")),
        "finite_output": float(run.get("finite_output_fraction", 0.0)) == 1.0,
        "metadata_preserved": all(bool(value) for value in required_metadata.values()),
    }
    if require_burst:
        checks["artifact_energy_reduced"] = (
            float(run.get("artifact_attenuation_db", -np.inf)) > 0.0
        )
        # `background_distortion_ratio` is RMS(cleaned-original) / RMS(original)
        # at non-burst samples. The old 5.0 cap was vestigial: across the
        # 2026-05-25 sweep every variant landed in 0.85-1.00, so it never
        # fired. 1.5 is a meaningful explosion bound — anything above 1.0
        # means ASR added more diff energy than the recording's own background
        # carried, i.e. it produced a signal that is, in the RMS sense, more
        # different from the source than the source itself. The 0.5 of slack
        # accommodates the upper end of healthy-but-aggressive ASR runs
        # (max observed 1.00 on the 120 s ERP sweep).
        #
        # NOTE: this check intentionally does NOT discriminate the SSVEP-style
        # subtle over-correction (the 2026-05-25 smoking gun): SSVEP's
        # background_distortion_ratio (0.85-0.99) sits inside the same band as
        # the working ERP/EDF runs because ASR rewrites significant background
        # on real EEG in both cases (validated empirically — see the
        # commit/PR notes; per-channel ratio, per-channel correlation, and
        # background energy preservation all overlap between the two regimes).
        # Subtle over-correction is properly caught by `artifact_energy_reduced`
        # (atten_dB < 0 on SSVEP vs +4.9 dB on working ERP) — that is the
        # canonical signal, not this bound.
        checks["background_distortion_bounded"] = (
            float(run.get("background_distortion_ratio", np.inf)) < 1.5
        )
    return checks


def _select_scalars(mapping: dict[str, Any]) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for key in SCALAR_DIAGNOSTICS:
        if key in mapping:
            value = mapping[key]
            if np.isscalar(value) or value is None:
                selected[key] = value
    return selected


def _write_reports(results: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_safe(results), indent=2), encoding="utf-8")
    output.with_suffix(".md").write_text(_markdown_report(results), encoding="utf-8")
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as fid:
        rows = _flat_rows(results)
        writer = csv.DictWriter(
            fid, fieldnames=sorted({k for row in rows for k in row})
        )
        writer.writeheader()
        writer.writerows(rows)


def _flat_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in results["datasets"]:
        for run in dataset["runs"]:
            row = {
                "dataset": dataset["path"],
                "duration_s": dataset["duration_s"],
                "n_eeg_channels": dataset["n_eeg_channels"],
                "variant": run["variant"],
                "status": run["status"],
                "wall_time_s": run.get("wall_time_s"),
                "cpu_time_s": run.get("cpu_time_s"),
                "rss_peak_mb": _bytes_to_mb(run.get("rss_peak_bytes")),
                "rss_peak_delta_mb": _bytes_to_mb(run.get("rss_peak_delta_bytes")),
                "tracemalloc_peak_mb": _bytes_to_mb(run.get("tracemalloc_peak_bytes")),
                "artifact_attenuation_db": run.get("artifact_attenuation_db"),
                "background_distortion_ratio": run.get("background_distortion_ratio"),
                "finite_output_fraction": run.get("finite_output_fraction"),
            }
            row.update(
                {
                    f"diag_{key}": value
                    for key, value in run.get("diagnostics", {}).items()
                }
            )
            row.update(
                {
                    f"cal_{key}": value
                    for key, value in run.get("calibration", {}).items()
                }
            )
            rows.append(_json_safe(row))
    return rows


def _markdown_report(results: dict[str, Any]) -> str:
    lines = [
        "# ASR Real-Data Validation",
        "",
        f"Generated: {results['created_utc']}",
        "",
    ]
    for dataset in results["datasets"]:
        lines.extend(
            [
                f"## {Path(dataset['path']).name}",
                "",
                (
                    f"- Duration: {dataset['duration_s']:.2f} s; "
                    f"EEG channels: {dataset['n_eeg_channels']}; "
                    f"sfreq: {dataset['sfreq']:.2f} Hz"
                ),
                "",
                (
                    "| Variant | Status | Wall s | CPU s | RSS peak MB | "
                    "RSS delta MB | Py alloc MB | Proc mode | Cal mode | "
                    "Atten dB | Bg dist |"
                ),
                "|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|",
            ]
        )
        for run in dataset["runs"]:
            diag = run.get("diagnostics", {})
            cal = run.get("calibration", {})
            lines.append(
                "| {variant} | {status} | {wall} | {cpu} | {rss_peak} | "
                "{rss_delta} | {py} | {proc_mode} | {cal_mode} | {atten} | "
                "{dist} |".format(
                    variant=run["variant"],
                    status=run["status"],
                    wall=_fmt(run.get("wall_time_s")),
                    cpu=_fmt(run.get("cpu_time_s")),
                    rss_peak=_fmt(_bytes_to_mb(run.get("rss_peak_bytes"))),
                    rss_delta=_fmt(_bytes_to_mb(run.get("rss_peak_delta_bytes"))),
                    py=_fmt(_bytes_to_mb(run.get("tracemalloc_peak_bytes"))),
                    proc_mode=diag.get("memory_mode", ""),
                    cal_mode=cal.get("memory_mode", ""),
                    atten=_fmt(run.get("artifact_attenuation_db")),
                    dist=_fmt(run.get("background_distortion_ratio")),
                )
            )
        lines.append("")
    return "\n".join(lines)


def _current_rss_bytes() -> int | None:
    if os.name == "nt":
        return _windows_current_rss_bytes()
    statm = Path("/proc/self/statm")
    if statm.exists():
        with suppress(Exception):
            resident_pages = int(statm.read_text(encoding="utf-8").split()[1])
            return resident_pages * os.sysconf("SC_PAGE_SIZE")
    with suppress(Exception):
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value * 1024)
    return None


def _windows_current_rss_bytes() -> int | None:
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    with suppress(Exception):
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        handle = kernel32.GetCurrentProcess()
        ok = psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), ctypes.sizeof(counters)
        )
        if ok:
            return int(counters.WorkingSetSize)
    return None


def _rms(data: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(data, dtype=float) ** 2)))


def _bytes_to_mb(value: Any) -> float | None:
    if value is None:
        return None
    return float(value) / (1024.0 * 1024.0)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return str(value)
    with suppress(Exception):
        return f"{float(value):.3g}"
    return str(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
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
