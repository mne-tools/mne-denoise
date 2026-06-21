"""MW-ASR window-length × mw_mode sweep on Klados paired trials.

The AASR sprint discovered MW-ASR (variant='mw', mw_window_length=20.0,
mw_mode='final_state' [the only mode at the time]) produces a *negative*
median RMSE reduction of -15.7% across the 40 paired Klados trials —
i.e., MW makes cleaned signals farther from ground truth than the
contaminated input was.

This script sweeps:
  - mw_window_length ∈ {5, 10, 20, 40, 60} s
  - mw_mode ∈ {'final_state', 'sliding'}

and reports per-(mode, length) the median RMSE-reduction-pct, median
correlation, median SNR-improvement-dB across the 40 paired trials.

The plan's acceptance criterion: identify at least one (mode, length)
configuration where median RMSE reduction goes *positive* (Klados
cleaning actually helps), OR document that no such configuration
exists in the swept range.
"""

# ruff: noqa: I001

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat, whosmat

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mne_denoise.asr import AdaptiveASR

KLADOS_DIR = ROOT / "refs" / "asr" / "datasets" / "mendeley_klados_eog" / "data"
SFREQ = 200.0
CUTOFF = 20.0
WINDOW_LENGTHS = [5.0, 10.0, 20.0, 40.0, 60.0]
MODES = ["final_state", "sliding"]


def _load_pair(trial_idx: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (pure, contaminated) for a 1-indexed paired trial, or None if unloadable."""
    pure_var = f"sim{trial_idx}_resampled"
    con_var = f"sim{trial_idx}_con"
    try:
        pure_data = loadmat(
            KLADOS_DIR / "Pure_Data.mat",
            squeeze_me=False,
            variable_names=[pure_var],
        )[pure_var]
        con_data = loadmat(
            KLADOS_DIR / "Contaminated_Data.mat",
            squeeze_me=False,
            variable_names=[con_var],
        )[con_var]
    except Exception as exc:  # noqa: BLE001
        print(f"  [trial {trial_idx}] load failed: {type(exc).__name__}: {exc}")
        return None
    # Trials are stored (channels, samples). Truncate to the shorter length.
    n = min(pure_data.shape[1], con_data.shape[1])
    return (
        np.ascontiguousarray(pure_data[:, :n], dtype=np.float64),
        np.ascontiguousarray(con_data[:, :n], dtype=np.float64),
    )


def _run_one(
    pure: np.ndarray, contaminated: np.ndarray, length_s: float, mode: str
) -> dict:
    """Run one MW configuration on one trial and return per-trial metrics."""
    asr = AdaptiveASR(
        sfreq=SFREQ,
        cutoff=CUTOFF,
        variant="mw",
        mw_window_length=length_s,
        mw_mode=mode,
        picks=None,
        max_mem_mb=256,
        verbose=False,
    )
    t0 = time.perf_counter()
    if mode == "sliding":
        cleaned = asr.fit_transform(contaminated)
    else:
        asr.fit(contaminated)
        cleaned = asr.transform(contaminated)
    dt = time.perf_counter() - t0

    cleaned = np.asarray(cleaned, dtype=np.float64)
    cleaned = cleaned[:, : pure.shape[1]]
    contaminated_trim = contaminated[:, : pure.shape[1]]

    rmse_contaminated = float(np.sqrt(np.mean((contaminated_trim - pure) ** 2)))
    rmse_cleaned = float(np.sqrt(np.mean((cleaned - pure) ** 2)))
    rmse_reduction_pct = (
        (rmse_contaminated - rmse_cleaned) / max(rmse_contaminated, np.finfo(float).eps)
    ) * 100.0

    corr = float(np.corrcoef(cleaned.ravel(), pure.ravel())[0, 1])

    var_pure = float(np.var(pure))
    var_resid = float(np.var(cleaned - pure))
    snr_after = 10.0 * np.log10(var_pure / max(var_resid, np.finfo(float).eps))
    var_resid_contaminated = float(np.var(contaminated_trim - pure))
    snr_before = 10.0 * np.log10(
        var_pure / max(var_resid_contaminated, np.finfo(float).eps)
    )
    snr_improvement_db = snr_after - snr_before

    return {
        "length_s": length_s,
        "mode": mode,
        "rmse_contaminated": rmse_contaminated,
        "rmse_cleaned": rmse_cleaned,
        "rmse_reduction_pct": rmse_reduction_pct,
        "mean_correlation": corr,
        "snr_improvement_db": snr_improvement_db,
        "wall_time_s": dt,
    }


def main() -> int:
    out_dir = ROOT / "reports" / "paper_validation" / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Enumerate paired trials
    pure_vars = [n for n, _, _ in whosmat(KLADOS_DIR / "Pure_Data.mat")]
    trial_ids = sorted(
        int(v.replace("sim", "").replace("_resampled", ""))
        for v in pure_vars
        if v.startswith("sim") and v.endswith("_resampled")
    )
    print(
        f"Klados paired trials available: {len(trial_ids)} (sim1..sim{trial_ids[-1]})"
    )

    records = []
    for trial_id in trial_ids:
        pair = _load_pair(trial_id)
        if pair is None:
            continue
        pure, contaminated = pair
        for length_s in WINDOW_LENGTHS:
            for mode in MODES:
                try:
                    rec = _run_one(pure, contaminated, length_s, mode)
                except Exception as exc:  # noqa: BLE001
                    rec = {
                        "length_s": length_s,
                        "mode": mode,
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                rec["trial_id"] = int(trial_id)
                records.append(rec)
        print(
            f"  [trial {trial_id:02d}] done across "
            f"{len(WINDOW_LENGTHS) * len(MODES)} (length, mode) combos"
        )

    # Aggregate per (length, mode)
    summary = {}
    for length_s in WINDOW_LENGTHS:
        for mode in MODES:
            cell = [
                r
                for r in records
                if r["length_s"] == length_s
                and r["mode"] == mode
                and "rmse_reduction_pct" in r
            ]
            if not cell:
                summary[f"{mode}@{length_s}s"] = {"n": 0}
                continue
            summary[f"{mode}@{length_s}s"] = {
                "n": len(cell),
                "rmse_reduction_pct_median": float(
                    np.median([r["rmse_reduction_pct"] for r in cell])
                ),
                "correlation_median": float(
                    np.median([r["mean_correlation"] for r in cell])
                ),
                "snr_improvement_db_median": float(
                    np.median([r["snr_improvement_db"] for r in cell])
                ),
                "wall_time_s_median": float(
                    np.median([r["wall_time_s"] for r in cell])
                ),
            }

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "n_trials_attempted": len(trial_ids),
        "window_lengths_s": WINDOW_LENGTHS,
        "modes": MODES,
        "summary": summary,
        "records": records,
    }
    (out_dir / "mw_window_length_sweep.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    # Plot: median RMSE reduction across window length, one line per mode
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for mode, marker in (("final_state", "o"), ("sliding", "s")):
        xs = []
        ys = []
        for length_s in WINDOW_LENGTHS:
            key = f"{mode}@{length_s}s"
            if summary[key].get("n", 0) > 0:
                xs.append(length_s)
                ys.append(summary[key]["rmse_reduction_pct_median"])
        ax.plot(xs, ys, marker=marker, lw=1.5, label=mode)
    ax.axhline(0, color="red", lw=1.0, alpha=0.5, label="break-even (0%)")
    ax.set_xlabel("mw_window_length (s)")
    ax.set_ylabel("Median RMSE reduction (%) across 40 Klados trials")
    ax.set_title(
        "MW-ASR window-length sweep: positive = cleaning helps; negative = MW over-cleans"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "mw_window_length_sweep.png", dpi=140)
    plt.close(fig)

    # Markdown
    md_lines = [
        "# MW-ASR window-length × mw_mode sweep",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Trials attempted: {len(trial_ids)}",
        "",
        "## Median per-trial metrics by (mw_window_length, mw_mode)",
        "",
        "| length (s) | mode | n | median RMSE-red (%) | median corr | median SNR-imp (dB) | median t (s) |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for length_s in WINDOW_LENGTHS:
        for mode in MODES:
            s = summary[f"{mode}@{length_s}s"]
            if s.get("n", 0) == 0:
                md_lines.append(f"| {length_s:.0f} | {mode} | 0 | — | — | — | — |")
                continue
            md_lines.append(
                f"| {length_s:.0f} | {mode} | {s['n']} | "
                f"{s['rmse_reduction_pct_median']:+.2f} | "
                f"{s['correlation_median']:.4f} | "
                f"{s['snr_improvement_db_median']:+.3f} | "
                f"{s['wall_time_s_median']:.2f} |"
            )

    # Headline finding
    md_lines.append("")
    md_lines.append("## Headline finding")
    md_lines.append("")
    positive_configs = [
        (k, v)
        for k, v in summary.items()
        if v.get("n", 0) > 0 and v["rmse_reduction_pct_median"] > 0
    ]
    if positive_configs:
        best = max(positive_configs, key=lambda x: x[1]["rmse_reduction_pct_median"])
        md_lines.append(
            f"**Best configuration**: `{best[0]}` with median RMSE reduction = "
            f"**+{best[1]['rmse_reduction_pct_median']:.2f}%** across {best[1]['n']} trials. "
            f"MW-ASR CAN produce a positive RMSE reduction on Klados — pick this configuration."
        )
    else:
        md_lines.append(
            "**No configuration produced a positive median RMSE reduction.** "
            "Across the swept window lengths and both modes, MW-ASR consistently "
            "over-cleans on the Klados semi-simulated EOG benchmark. "
            "Recommendation: use PSP-ASR or PSW-ASR instead; consider sliding "
            "mode only if per-segment temporal locality is required."
        )

    md_lines.append("")
    md_lines.append("![MW sweep](mw_window_length_sweep.png)")
    (out_dir / "mw_window_length_sweep.md").write_text(
        "\n".join(md_lines), encoding="utf-8"
    )
    print(f"\nWrote mw_window_length_sweep.{{json,png,md}} under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
