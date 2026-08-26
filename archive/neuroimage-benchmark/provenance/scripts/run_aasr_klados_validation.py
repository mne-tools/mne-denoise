"""Tier-A AASR validation on the Klados semi-simulated EEG/EOG dataset.

For each of the 41 paired trials in
``refs/asr/datasets/mendeley_klados_eog/data/``:

1. Load the clean (``Pure_Data.mat``) and contaminated (``Contaminated_Data.mat``)
   versions of trial ``i``.
2. Apply 1-50 Hz bandpass (matches the MATLAB AASR demo's ``datafiltering2``).
3. For each of the 4 AASR variants -- Init-ASR, MW-ASR, PSP-ASR, PSW-ASR --
   run ``AdaptiveASR`` at the paper's ``cutoff=20`` and compute paired
   ground-truth metrics:

   - per-channel RMSE and mean RMSE
   - per-channel Pearson correlation and mean correlation
   - SNR improvement in dB
   - wall time

Each (trial, variant) is one row. The 41 x 4 = 164 rows are written to
``reports/paper_validation/aasr/tier_A_klados_per_trial.csv`` and
aggregated (median + IQR per variant) into
``tier_A_klados_aggregate.json``. Per-variant ``mw_diagnostics_`` is preserved
when present, so Stage 3 plots can read it back from the JSON.

This is the **first sprint with true paired ground truth** -- earlier
sprints used synthetic burst injection.
"""

# ruff: noqa: I001

from __future__ import annotations

import argparse
import contextlib
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
import numpy as np
from scipy.io import loadmat, whosmat
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mne_denoise.asr import AdaptiveASR  # noqa: E402

VARIANTS = ("init", "mw", "psp", "psw")


@dataclass
class TrialPair:
    trial_idx: int  # 1-based to match Klados naming
    pure: np.ndarray  # (n_channels, n_samples)
    contaminated: (
        np.ndarray
    )  # (n_channels, n_samples_contam) -- not always equal length
    n_samples_pure: int
    n_samples_contam: int


# ----------------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------------


def _load_pair(
    pure_path: Path,
    contam_path: Path,
    trial_idx: int,
) -> TrialPair | None:
    pure_var = f"sim{trial_idx}_resampled"
    contam_var = f"sim{trial_idx}_con"
    try:
        pure = loadmat(pure_path, squeeze_me=False, variable_names=[pure_var])[
            pure_var
        ].astype(np.float64)
    except (KeyError, ValueError, OSError) as exc:
        # OSError("could not read bytes") shows up sporadically on individual
        # variables in the Klados Mendeley archive (zlib stream quirk).
        # Skip the trial rather than crashing the whole run.
        print(
            f"[klados] WARN trial {trial_idx} pure load failed: {type(exc).__name__}: {exc}"
        )
        return None
    try:
        contam = loadmat(contam_path, squeeze_me=False, variable_names=[contam_var])[
            contam_var
        ].astype(np.float64)
    except (KeyError, ValueError, OSError) as exc:
        print(
            f"[klados] WARN trial {trial_idx} contam load failed: {type(exc).__name__}: {exc}"
        )
        return None
    return TrialPair(
        trial_idx=trial_idx,
        pure=pure,
        contaminated=contam,
        n_samples_pure=pure.shape[1],
        n_samples_contam=contam.shape[1],
    )


def _bandpass(data: np.ndarray, sfreq: float, low: float, high: float) -> np.ndarray:
    """Zero-phase 1-50 Hz Butterworth (4th order). Matches demo's datafiltering2."""
    sos = butter(4, [low, high], btype="bandpass", fs=sfreq, output="sos")
    # sosfiltfilt operates along axis=-1 by default; our shape is (channels, samples)
    return sosfiltfilt(sos, data, axis=-1)


# ----------------------------------------------------------------------------
# Per-variant runners
# ----------------------------------------------------------------------------


def _run_variant(
    variant: str,
    contaminated: np.ndarray,
    *,
    sfreq: float,
    cutoff: float,
    init_window_s: float,
    psw_window_s: float,
    mw_window_s: float,
) -> tuple[np.ndarray, AdaptiveASR, float]:
    init_samples = max(1, int(round(init_window_s * sfreq)))
    psw_samples = max(1, int(round(psw_window_s * sfreq)))

    if variant == "init":
        asr = AdaptiveASR(sfreq=sfreq, cutoff=cutoff, variant="psp", verbose=False)
        first = contaminated[:, : min(init_samples, contaminated.shape[1])]
        t0 = time.perf_counter()
        asr.fit(first)
        cleaned = asr.transform(contaminated)
        dt = time.perf_counter() - t0
    elif variant == "psp":
        asr = AdaptiveASR(sfreq=sfreq, cutoff=cutoff, variant="psp", verbose=False)
        t0 = time.perf_counter()
        cleaned = asr.fit_transform(contaminated)
        dt = time.perf_counter() - t0
    elif variant == "psw":
        asr = AdaptiveASR(sfreq=sfreq, cutoff=cutoff, variant="psw", verbose=False)
        t0 = time.perf_counter()
        first = contaminated[:, : min(psw_samples, contaminated.shape[1])]
        asr.fit(first)
        cursor = psw_samples
        while cursor < contaminated.shape[1]:
            stop = min(cursor + psw_samples, contaminated.shape[1])
            chunk = contaminated[:, cursor:stop]
            if chunk.shape[1] >= asr.blocksize:
                # too-short tail or per-window failure -- skip the update
                with contextlib.suppress(Exception):
                    asr.partial_fit(chunk)
            cursor = stop
        cleaned = asr.transform(contaminated)
        dt = time.perf_counter() - t0
    elif variant == "mw":
        asr = AdaptiveASR(
            sfreq=sfreq,
            cutoff=cutoff,
            variant="mw",
            mw_window_length=mw_window_s,
            verbose=False,
        )
        t0 = time.perf_counter()
        cleaned = asr.fit_transform(contaminated)
        dt = time.perf_counter() - t0
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return cleaned, asr, dt


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------


def _paired_metrics(
    pure: np.ndarray,
    contaminated: np.ndarray,
    cleaned: np.ndarray,
) -> dict[str, Any]:
    n_samples = min(pure.shape[1], cleaned.shape[1])
    pure_a = pure[:, :n_samples]
    cont_a = contaminated[:, :n_samples]
    clean_a = cleaned[:, :n_samples]

    # Per-channel and mean RMSE between cleaned and pure
    rmse_per_ch = np.sqrt(np.mean((clean_a - pure_a) ** 2, axis=1))
    rmse_per_ch_contam = np.sqrt(np.mean((cont_a - pure_a) ** 2, axis=1))

    # Per-channel Pearson correlation
    corr_per_ch = np.zeros(pure_a.shape[0])
    for c in range(pure_a.shape[0]):
        if np.std(pure_a[c]) < 1e-12 or np.std(clean_a[c]) < 1e-12:
            corr_per_ch[c] = np.nan
        else:
            corr_per_ch[c] = np.corrcoef(clean_a[c], pure_a[c])[0, 1]

    # Pre-cleaning Pearson (contaminated vs pure) for context
    corr_per_ch_contam = np.zeros(pure_a.shape[0])
    for c in range(pure_a.shape[0]):
        if np.std(pure_a[c]) < 1e-12 or np.std(cont_a[c]) < 1e-12:
            corr_per_ch_contam[c] = np.nan
        else:
            corr_per_ch_contam[c] = np.corrcoef(cont_a[c], pure_a[c])[0, 1]

    # SNR improvement (dB): 10 log10( var(pure) / var(cleaned - pure) )
    var_pure = float(np.var(pure_a))
    var_residual_clean = float(np.var(clean_a - pure_a))
    var_residual_contam = float(np.var(cont_a - pure_a))
    snr_clean = 10.0 * np.log10(var_pure / max(var_residual_clean, 1e-30))
    snr_contam = 10.0 * np.log10(var_pure / max(var_residual_contam, 1e-30))

    return {
        "rmse_per_ch": rmse_per_ch.tolist(),
        "mean_rmse": float(rmse_per_ch.mean()),
        "rmse_per_ch_contam": rmse_per_ch_contam.tolist(),
        "mean_rmse_contam": float(rmse_per_ch_contam.mean()),
        "rmse_reduction_pct": float(
            (rmse_per_ch_contam.mean() - rmse_per_ch.mean())
            / max(rmse_per_ch_contam.mean(), 1e-30)
            * 100.0
        ),
        "corr_per_ch": corr_per_ch.tolist(),
        "mean_correlation": float(np.nanmean(corr_per_ch)),
        "corr_per_ch_contam": corr_per_ch_contam.tolist(),
        "mean_correlation_contam": float(np.nanmean(corr_per_ch_contam)),
        "snr_after_db": float(snr_clean),
        "snr_before_db": float(snr_contam),
        "snr_improvement_db": float(snr_clean - snr_contam),
        "n_samples_compared": int(n_samples),
    }


def _serialize_mw_diagnostics(asr: AdaptiveASR) -> list[dict[str, Any]]:
    diagnostics = getattr(asr, "mw_diagnostics_", [])
    out = []
    for d in diagnostics:
        item: dict[str, Any] = {
            "window_idx": int(d.get("window_idx", -1)),
            "window_start": int(d.get("window_start", -1)),
            "window_stop": int(d.get("window_stop", -1)),
            "n_samples": int(d.get("n_samples", 0)),
            "status": str(d.get("status", "")),
        }
        if d.get("status") == "passed":
            T = np.asarray(d["T"])
            item["T_frobenius"] = float(np.linalg.norm(T))
            thr = np.asarray(d["thresholds"]).ravel()
            item["thresholds_mean"] = float(thr.mean())
            item["thresholds_max"] = float(thr.max())
            item["rank"] = int(d["rank"])
            item["clean_window_fraction"] = d.get("clean_window_fraction")
        else:
            item["error_type"] = d.get("error_type")
            item["error"] = d.get("error")
        out.append(item)
    return out


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------


def _plot_boxplots(rows: list[dict[str, Any]], out_dir: Path) -> dict[str, dict]:
    out_payload: dict[str, dict] = {}
    metrics = [
        ("mean_rmse", "Per-trial mean RMSE (uV)", "rmse_boxplot"),
        (
            "mean_correlation",
            "Per-trial Pearson correlation (cleaned vs pure)",
            "correlation_boxplot",
        ),
        ("snr_improvement_db", "SNR improvement (dB)", "snr_boxplot"),
    ]
    variant_order = list(VARIANTS)
    for key, ylabel, slug in metrics:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        data = []
        labels = []
        for v in variant_order:
            vals = [
                r[key] for r in rows if r["variant"] == v and r["status"] == "passed"
            ]
            data.append(vals)
            labels.append(v)
        ax.boxplot(data, labels=labels, showmeans=True)
        ax.set_xlabel("AASR variant")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Tsai et al. — {ylabel} across 41 Klados paired trials")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        png = out_dir / f"fig_tsai_per_variant_{slug}.png"
        fig.savefig(png, dpi=140)
        plt.close(fig)
        payload = {
            v: {"values": data[i], "n": len(data[i])}
            for i, v in enumerate(variant_order)
        }
        png.with_suffix(".json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        out_payload[slug] = payload
    return out_payload


def _plot_wall_time(rows: list[dict[str, Any]], out_dir: Path) -> dict:
    variant_order = list(VARIANTS)
    medians = []
    for v in variant_order:
        vals = [
            r["wall_time_s"]
            for r in rows
            if r["variant"] == v and r["status"] == "passed"
        ]
        medians.append(float(np.median(vals)) if vals else 0.0)
    fig, ax = plt.subplots(figsize=(7, 4.0))
    ax.bar(variant_order, medians, color="steelblue", edgecolor="white")
    ax.set_xlabel("AASR variant")
    ax.set_ylabel("Median wall time per trial (s)")
    ax.set_title("Tsai et al. — Per-variant wall time on Klados (41 trials, k=20)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    png = out_dir / "fig_tsai_per_variant_wall_time.png"
    fig.savefig(png, dpi=140)
    plt.close(fig)
    payload = dict(zip(variant_order, medians))
    png.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _plot_mw_trajectory(rows: list[dict[str, Any]], out_dir: Path) -> dict | None:
    # Choose a representative trial -- use the one with the most MW windows
    mw_rows = [
        r
        for r in rows
        if r["variant"] == "mw" and r["status"] == "passed" and r.get("mw_diagnostics")
    ]
    if not mw_rows:
        return None
    best = max(mw_rows, key=lambda r: len(r["mw_diagnostics"]))
    diag = best["mw_diagnostics"]
    passed = [d for d in diag if d.get("status") == "passed"]
    if len(passed) < 2:
        return None
    T_norms = [d["T_frobenius"] for d in passed]
    indices = [d["window_idx"] for d in passed]
    deltas = [abs(T_norms[i] - T_norms[i - 1]) for i in range(1, len(T_norms))]
    delta_idx = indices[1:]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    axes[0].plot(indices, T_norms, marker="o", color="steelblue")
    axes[0].set_xlabel("MW window index")
    axes[0].set_ylabel("||T||_F")
    axes[0].set_title(f"Trial {best['trial_idx']}: threshold-matrix norm per window")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(delta_idx, deltas, marker="s", color="C3")
    axes[1].set_xlabel("MW window index")
    axes[1].set_ylabel("||T_t - T_{t-1}||")
    axes[1].set_title("First-difference adaptation trajectory")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    png = out_dir / "fig_tsai_mw_adaptation_trajectory.png"
    fig.savefig(png, dpi=140)
    plt.close(fig)
    payload = {
        "trial_idx": best["trial_idx"],
        "window_index": indices,
        "T_frobenius": T_norms,
        "delta_T_frobenius_index": delta_idx,
        "delta_T_frobenius": deltas,
    }
    png.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _plot_waveform_overlay(
    pair: TrialPair,
    cleaned_by_variant: dict[str, np.ndarray],
    contaminated: np.ndarray,
    sfreq: float,
    out_dir: Path,
    seconds_to_show: float = 15.0,
) -> dict:
    n_total = pair.pure.shape[1]
    show_samples = min(int(seconds_to_show * sfreq), n_total)
    t = np.arange(show_samples) / sfreq
    # Pick three frontal-ish channels (assume 19-ch 10-20 ordering: Fp1, Fp2, Fz are usually first three)
    ch_idx = [0, 1, 2]
    payload: dict[str, Any] = {
        "trial_idx": pair.trial_idx,
        "seconds_shown": seconds_to_show,
        "channels_shown": ch_idx,
    }
    fig, axes = plt.subplots(
        len(VARIANTS),
        len(ch_idx),
        figsize=(4 * len(ch_idx), 2.0 * len(VARIANTS)),
        squeeze=False,
        sharex=True,
    )
    for r, v in enumerate(VARIANTS):
        if v not in cleaned_by_variant:
            continue
        cleaned = cleaned_by_variant[v]
        for c, ch in enumerate(ch_idx):
            ax = axes[r, c]
            ax.plot(
                t,
                pair.pure[ch, -show_samples:] * 1e6,
                color="0.6",
                lw=0.8,
                label="Pure",
            )
            ax.plot(
                t,
                contaminated[ch, -show_samples:] * 1e6,
                color="C3",
                lw=0.8,
                alpha=0.7,
                label="Contaminated",
            )
            ax.plot(
                t,
                cleaned[ch, -show_samples:] * 1e6,
                color="C0",
                lw=1.0,
                label=v.upper(),
            )
            if r == 0:
                ax.set_title(f"channel idx {ch}", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{v.upper()}\n(uV)", fontsize=9)
            if r == len(VARIANTS) - 1:
                ax.set_xlabel("Time (s)")
            ax.grid(True, alpha=0.3)
            if r == 0 and c == 0:
                ax.legend(loc="upper right", fontsize=7)
    fig.suptitle(
        f"AASR_demo.ipynb-style overlay -- last {seconds_to_show:.0f} s, trial {pair.trial_idx}"
    )
    fig.tight_layout()
    png = out_dir / f"fig_tsai_waveform_overlay_trial{pair.trial_idx:02d}.png"
    fig.savefig(png, dpi=140)
    plt.close(fig)
    png.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "refs" / "asr" / "datasets" / "mendeley_klados_eog" / "data",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports" / "paper_validation" / "aasr",
    )
    p.add_argument("--cutoff", type=float, default=20.0)
    p.add_argument("--sfreq", type=float, default=200.0)
    p.add_argument("--highpass", type=float, default=1.0)
    p.add_argument("--lowpass", type=float, default=50.0)
    p.add_argument("--init-window-s", type=float, default=20.0)
    p.add_argument("--psw-window-s", type=float, default=20.0)
    p.add_argument("--mw-window-s", type=float, default=20.0)
    p.add_argument(
        "--variants",
        nargs="+",
        default=list(VARIANTS),
        choices=VARIANTS,
    )
    p.add_argument(
        "--max-trials",
        type=int,
        default=41,
        help="Cap on the number of paired trials (paper-faithful default = 41).",
    )
    p.add_argument(
        "--demo-trial",
        type=int,
        default=2,
        help="Trial index used for the demo-style waveform overlay (default 2 matches AASR_demo.ipynb).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pure_path = args.data_dir / "Pure_Data.mat"
    contam_path = args.data_dir / "Contaminated_Data.mat"
    if not pure_path.exists() or not contam_path.exists():
        raise SystemExit(
            f"Klados files missing under {args.data_dir}. "
            "See refs/asr/datasets/mendeley_klados_eog/README.md for the download instructions."
        )

    # Build the list of paired trial indices that exist in both files
    pure_vars = {n for n, _, _ in whosmat(pure_path)}
    n_paired = sum(
        1 for i in range(1, args.max_trials + 1) if f"sim{i}_resampled" in pure_vars
    )
    print(
        f"[klados] paired trials available: {n_paired} of {args.max_trials} requested"
    )

    rows: list[dict[str, Any]] = []
    saved_demo = False
    for trial_idx in range(1, args.max_trials + 1):
        pair = _load_pair(pure_path, contam_path, trial_idx)
        if pair is None:
            continue
        # Align lengths -- many trials have contaminated longer than pure; truncate
        n_common = min(pair.n_samples_pure, pair.n_samples_contam)
        pure = pair.pure[:, :n_common]
        contam_raw = pair.contaminated[:, :n_common]

        # 1-50 Hz bandpass on both (matches demo)
        pure_filt = _bandpass(pure, args.sfreq, args.highpass, args.lowpass)
        contam_filt = _bandpass(contam_raw, args.sfreq, args.highpass, args.lowpass)

        # The MNE/ASR pipeline expects amplitudes in volts; Klados data is in uV-equivalent
        # floats (no documented scaling); we leave it in native units and report
        # SNR/correlation/RMSE in those units. ICLabel Stage C re-scales if needed.

        cleaned_by_variant: dict[str, np.ndarray] = {}
        for variant in args.variants:
            try:
                cleaned, asr_inst, dt = _run_variant(
                    variant,
                    contam_filt,
                    sfreq=args.sfreq,
                    cutoff=args.cutoff,
                    init_window_s=args.init_window_s,
                    psw_window_s=args.psw_window_s,
                    mw_window_s=args.mw_window_s,
                )
            except Exception as exc:  # noqa: BLE001
                rows.append(
                    {
                        "trial_idx": trial_idx,
                        "variant": variant,
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(limit=4),
                    }
                )
                continue

            metrics = _paired_metrics(pure_filt, contam_filt, cleaned)
            row = {
                "trial_idx": trial_idx,
                "variant": variant,
                "status": "passed",
                "wall_time_s": dt,
                "n_samples": pure_filt.shape[1],
                **metrics,
            }
            if variant == "mw":
                row["mw_diagnostics"] = _serialize_mw_diagnostics(asr_inst)
            rows.append(row)
            print(
                f"[klados t{trial_idx:02d} {variant:>4s}] "
                f"corr={metrics['mean_correlation']:.3f} "
                f"snr_imp={metrics['snr_improvement_db']:+.2f}dB "
                f"rmse_red={metrics['rmse_reduction_pct']:+.1f}% "
                f"t={dt:.2f}s"
            )
            cleaned_by_variant[variant] = cleaned
            del asr_inst, cleaned

        # Save the demo-trial waveform overlay
        if trial_idx == args.demo_trial and cleaned_by_variant and not saved_demo:
            _plot_waveform_overlay(
                TrialPair(
                    trial_idx=trial_idx,
                    pure=pure_filt,
                    contaminated=contam_filt,
                    n_samples_pure=pure_filt.shape[1],
                    n_samples_contam=contam_filt.shape[1],
                ),
                cleaned_by_variant,
                contam_filt,
                args.sfreq,
                args.output_dir,
            )
            saved_demo = True
        del pure_filt, contam_filt, contam_raw, pure
        gc.collect()

    # ------------------------------------------------------------------
    # Persist per-trial CSV + aggregate JSON
    # ------------------------------------------------------------------
    csv_path = args.output_dir / "tier_A_klados_per_trial.csv"
    scalar_keys = [
        "trial_idx",
        "variant",
        "status",
        "wall_time_s",
        "n_samples",
        "mean_rmse",
        "mean_rmse_contam",
        "rmse_reduction_pct",
        "mean_correlation",
        "mean_correlation_contam",
        "snr_before_db",
        "snr_after_db",
        "snr_improvement_db",
        "n_samples_compared",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=scalar_keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in scalar_keys})

    # Aggregate stats per variant
    aggregate: dict[str, Any] = {}
    for v in VARIANTS:
        ok = [r for r in rows if r["variant"] == v and r["status"] == "passed"]
        if not ok:
            aggregate[v] = {"n": 0}
            continue

        def _stats(key: str, ok=ok) -> dict:
            vals = np.array([r[key] for r in ok], dtype=np.float64)
            return {
                "n": int(vals.size),
                "median": float(np.median(vals)),
                "iqr_low": float(np.percentile(vals, 25)),
                "iqr_high": float(np.percentile(vals, 75)),
                "min": float(vals.min()),
                "max": float(vals.max()),
            }

        aggregate[v] = {
            "n": len(ok),
            "wall_time_s": _stats("wall_time_s"),
            "mean_rmse": _stats("mean_rmse"),
            "rmse_reduction_pct": _stats("rmse_reduction_pct"),
            "mean_correlation": _stats("mean_correlation"),
            "snr_improvement_db": _stats("snr_improvement_db"),
        }

    json_path = args.output_dir / "tier_A_klados_aggregate.json"
    json_path.write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "cutoff": args.cutoff,
                "sfreq": args.sfreq,
                "highpass": args.highpass,
                "lowpass": args.lowpass,
                "variants": list(args.variants),
                "n_trials_attempted": args.max_trials,
                "n_rows": len(rows),
                "aggregate": aggregate,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Full per-trial JSON (large — includes per-channel arrays + mw_diagnostics)
    full_json_path = args.output_dir / "tier_A_klados_full.json"
    full_json_path.write_text(
        json.dumps(rows, indent=2, default=lambda o: None),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Tier B figures (Stage 3 outputs) -- emitted in the same run
    # ------------------------------------------------------------------
    boxplot_payload = _plot_boxplots(rows, args.output_dir)
    walltime_payload = _plot_wall_time(rows, args.output_dir)
    mw_trajectory_payload = _plot_mw_trajectory(rows, args.output_dir)

    figures_index = {
        "boxplots": boxplot_payload,
        "wall_time": walltime_payload,
        "mw_trajectory": mw_trajectory_payload,
        "waveform_overlay_emitted_for_trial": args.demo_trial if saved_demo else None,
    }
    (args.output_dir / "stage3_figures.json").write_text(
        json.dumps(figures_index, indent=2), encoding="utf-8"
    )

    print(f"\n[klados] wrote {csv_path}")
    print(f"        {json_path}")
    print(f"        {full_json_path}")
    print(f"        {args.output_dir / 'stage3_figures.json'}")
    n_ok = sum(1 for r in rows if r["status"] == "passed")
    n_total = len(rows)
    print(f"\nStatus: {n_ok}/{n_total} (trial, variant) cells passed")
    return 0 if n_ok == n_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
