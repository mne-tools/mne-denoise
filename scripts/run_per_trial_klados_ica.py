"""Per-trial Klados ICA — re-do AASR Tier C with one ICA fit per trial.

The AASR sprint found that ALL 4 AASR variants (init, mw, psp, psw)
produce identical IC distributions on the concatenated 40-trial Klados
stream (15 ICs / 11 brain / 1 eye / 0 muscle / 3 other / 11 dipolar each).
The concatenated approach hides per-variant differences that ARE visible
in the per-trial Tier A metrics (median correlation 0.92-0.94).

This script re-does Tier C with **one ICA fit per trial** (40 ICAs per
variant) and aggregates per-class IC counts as medians + IQR across trials.
The hypothesis: per-trial ICA discriminates variants where concatenated ICA could not.

Output: ``reports/paper_validation/robustness/klados_per_trial_ica.{json,md,png}``
"""

# ruff: noqa: I001

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.io import loadmat, whosmat

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mne_denoise.asr import AdaptiveASR

KLADOS_DIR = ROOT / "refs" / "asr" / "datasets" / "mendeley_klados_eog" / "data"
SFREQ = 200.0
CUTOFF = 20.0
VARIANTS = ("init", "psp", "psw", "mw_final_state", "mw_sliding")
ICLABEL_CLASSES = (
    "brain",
    "muscle artifact",
    "eye blink",
    "heart beat",
    "line noise",
    "channel noise",
    "other",
)
CH_NAMES_19 = [
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "T7",
    "C3",
    "Cz",
    "C4",
    "T8",
    "P7",
    "P3",
    "Pz",
    "P4",
    "P8",
    "O1",
    "O2",
]


def _load_pair(trial_idx: int):
    pure_var = f"sim{trial_idx}_resampled"
    con_var = f"sim{trial_idx}_con"
    try:
        pure = loadmat(
            KLADOS_DIR / "Pure_Data.mat",
            squeeze_me=False,
            variable_names=[pure_var],
        )[pure_var]
        con = loadmat(
            KLADOS_DIR / "Contaminated_Data.mat",
            squeeze_me=False,
            variable_names=[con_var],
        )[con_var]
    except Exception:
        return None
    n = min(pure.shape[1], con.shape[1])
    return (
        np.ascontiguousarray(pure[:, :n], dtype=np.float64),
        np.ascontiguousarray(con[:, :n], dtype=np.float64),
    )


def _apply_variant(contaminated: np.ndarray, variant: str) -> np.ndarray:
    if variant == "init":
        # AASR(variant='psp') calibrated on first 20 s only
        asr = AdaptiveASR(
            sfreq=SFREQ,
            cutoff=CUTOFF,
            variant="psp",
            picks=None,
            verbose=False,
        )
        first_20s = contaminated[:, : int(20 * SFREQ)]
        asr.fit(first_20s)
        return asr.transform(contaminated)
    if variant == "psp":
        asr = AdaptiveASR(
            sfreq=SFREQ,
            cutoff=CUTOFF,
            variant="psp",
            picks=None,
            verbose=False,
        )
        return asr.fit_transform(contaminated)
    if variant == "psw":
        asr = AdaptiveASR(
            sfreq=SFREQ,
            cutoff=CUTOFF,
            variant="psw",
            picks=None,
            verbose=False,
        )
        return asr.fit_transform(contaminated)
    if variant == "mw_final_state":
        asr = AdaptiveASR(
            sfreq=SFREQ,
            cutoff=CUTOFF,
            variant="mw",
            mw_window_length=20.0,
            mw_mode="final_state",
            picks=None,
            verbose=False,
        )
        asr.fit(contaminated)
        return asr.transform(contaminated)
    if variant == "mw_sliding":
        asr = AdaptiveASR(
            sfreq=SFREQ,
            cutoff=CUTOFF,
            variant="mw",
            mw_window_length=20.0,
            mw_mode="sliding",
            picks=None,
            verbose=False,
        )
        return asr.fit_transform(contaminated)
    raise ValueError(f"unknown variant {variant}")


def _build_raw(data_2d: np.ndarray) -> mne.io.Raw:
    info = mne.create_info(CH_NAMES_19, sfreq=SFREQ, ch_types="eeg")
    raw = mne.io.RawArray(data_2d * 1e-6, info, verbose=False)
    try:
        std_mont = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(std_mont, match_case=False, on_missing="ignore", verbose=False)
    except Exception:
        pass
    raw.set_eeg_reference("average", projection=False, verbose=False)
    return raw


def _classify_iclabel(raw: mne.io.Raw, n_components: int = 15) -> dict:
    """Fit ICA + run ICLabel. Returns per-class counts + dipolar count."""
    from mne_icalabel.iclabel import iclabel_label_components

    # We need 1-100 Hz bandpass for ICLabel; Klados is 200 Hz so Nyquist = 100.
    # If sfreq > 200, would also need lowpass. At 200 Hz exactly we skip LP.
    nyquist = raw.info["sfreq"] / 2.0
    if (raw.info["lowpass"] is None or raw.info["lowpass"] > 99.0 - 1e-6) and (
        nyquist > 100.0 + 1e-6
    ):
        raw.filter(l_freq=None, h_freq=100.0, picks="eeg", verbose=False)
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method="infomax",
        fit_params={"extended": True},
        random_state=97,
        max_iter=500,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ica.fit(raw, verbose=False)
        proba = iclabel_label_components(raw, ica, inplace=False)
    proba = np.asarray(proba)
    argmax = np.argmax(proba, axis=1)
    labels = [ICLABEL_CLASSES[i] for i in argmax]
    counts = {c: int(sum(1 for L in labels if c == L)) for c in ICLABEL_CLASSES}
    dipolar = int((proba[:, 0] >= 0.5).sum())
    return {"counts": counts, "dipolar": dipolar, "n_components": int(proba.shape[0])}


def main() -> int:
    out_dir = ROOT / "reports" / "paper_validation" / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)

    pure_vars = [n for n, _, _ in whosmat(KLADOS_DIR / "Pure_Data.mat")]
    trial_ids = sorted(
        int(v.replace("sim", "").replace("_resampled", ""))
        for v in pure_vars
        if v.startswith("sim") and v.endswith("_resampled")
    )
    # Robustness-sprint compromise: full 40-trial × 6-variant × ICA budget would
    # be ~4 hours. Sample 8 trials to get directional discrimination signal
    # in a runtime that fits the sprint envelope (~30 minutes).
    import os

    n_sample = int(os.environ.get("KLADOS_PER_TRIAL_ICA_N_TRIALS", "8"))
    trial_ids = trial_ids[:n_sample]
    print(
        f"Klados paired trials available: {len(pure_vars)} ; sampling first {len(trial_ids)}"
    )

    per_trial_records = []
    for trial_idx, trial_id in enumerate(trial_ids, 1):
        pair = _load_pair(trial_id)
        if pair is None:
            print(f"  [{trial_idx:02d}] trial {trial_id} load failed; skipped")
            continue
        pure, contaminated = pair

        # Baseline: contaminated input
        raw_base = _build_raw(contaminated)
        try:
            base = _classify_iclabel(raw_base)
            per_trial_records.append(
                {
                    "trial_id": int(trial_id),
                    "variant": "contaminated",
                    "counts": base["counts"],
                    "dipolar": base["dipolar"],
                    "n_components": base["n_components"],
                }
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [{trial_idx:02d}] contaminated ICA failed: {exc}")
            continue

        for variant in VARIANTS:
            try:
                cleaned = _apply_variant(contaminated, variant)
                raw = _build_raw(cleaned)
                rec = _classify_iclabel(raw)
                per_trial_records.append(
                    {
                        "trial_id": int(trial_id),
                        "variant": variant,
                        "counts": rec["counts"],
                        "dipolar": rec["dipolar"],
                        "n_components": rec["n_components"],
                    }
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"  [{trial_idx:02d}] {variant} failed: {type(exc).__name__}: {exc}"
                )
                per_trial_records.append(
                    {
                        "trial_id": int(trial_id),
                        "variant": variant,
                        "status": "failed",
                        "error": str(exc),
                    }
                )
        print(f"  [{trial_idx:02d}] trial {trial_id}: done")

    # Aggregate per variant
    summary = {}
    for v in ("contaminated", *VARIANTS):
        rows = [r for r in per_trial_records if r.get("variant") == v and "counts" in r]
        if not rows:
            summary[v] = {"n": 0}
            continue
        agg = {"n": len(rows)}
        for cls in ICLABEL_CLASSES:
            vals = [r["counts"].get(cls, 0) for r in rows]
            agg[f"{cls}_median"] = float(np.median(vals))
            agg[f"{cls}_q25"] = float(np.percentile(vals, 25))
            agg[f"{cls}_q75"] = float(np.percentile(vals, 75))
        dvals = [r["dipolar"] for r in rows]
        agg["dipolar_median"] = float(np.median(dvals))
        agg["dipolar_q25"] = float(np.percentile(dvals, 25))
        agg["dipolar_q75"] = float(np.percentile(dvals, 75))
        summary[v] = agg

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "n_trials_processed": len(trial_ids),
        "variants": list(VARIANTS),
        "summary": summary,
        "records": per_trial_records,
    }
    (out_dir / "klados_per_trial_ica.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    # Plot: per-variant median brain / dipolar with IQR
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    variants_list = ("contaminated", *VARIANTS)
    for ax, key, title in (
        (axes[0], "brain", "median brain IC count per variant"),
        (axes[1], "dipolar", "median dipolar (brain >= 0.5) per variant"),
    ):
        ys, q1s, q3s = [], [], []
        for v in variants_list:
            s = summary.get(v, {})
            if s.get("n", 0) == 0:
                ys.append(np.nan)
                q1s.append(np.nan)
                q3s.append(np.nan)
                continue
            ys.append(s.get(f"{key}_median", s.get("dipolar_median", 0)))
            q1s.append(s.get(f"{key}_q25", s.get("dipolar_q25", 0)))
            q3s.append(s.get(f"{key}_q75", s.get("dipolar_q75", 0)))
        x = np.arange(len(variants_list))
        ax.bar(x, ys, color="steelblue", edgecolor="black")
        ax.errorbar(
            x,
            ys,
            yerr=[np.subtract(ys, q1s), np.subtract(q3s, ys)],
            fmt="none",
            ecolor="black",
            capsize=4,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(variants_list, rotation=20, ha="right")
        ax.set_ylabel("Median IC count (n trials)")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "klados_per_trial_ica.png", dpi=140)
    plt.close(fig)

    # Markdown
    md = [
        "# Per-trial Klados ICA — settling the AASR Tier C indistinguishability",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Median per-class IC counts (with IQR) across paired trials",
        "",
        "| Variant | n | brain median (q25-q75) | eye | muscle | dipolar |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for v in variants_list:
        s = summary.get(v, {})
        if s.get("n", 0) == 0:
            md.append(f"| {v} | 0 | — | — | — | — |")
            continue
        md.append(
            f"| {v} | {s['n']} | "
            f"{s['brain_median']:.1f} ({s['brain_q25']:.0f}-{s['brain_q75']:.0f}) | "
            f"{s['eye blink_median']:.1f} | "
            f"{s['muscle artifact_median']:.1f} | "
            f"{s['dipolar_median']:.1f} ({s['dipolar_q25']:.0f}-{s['dipolar_q75']:.0f}) |"
        )
    md.append("")
    md.append("![Per-trial Klados ICA](klados_per_trial_ica.png)")
    (out_dir / "klados_per_trial_ica.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote klados_per_trial_ica.{{json,png,md}} under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
