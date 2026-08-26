"""Cross-variant × cross-substrate matrix for the robustness sprint.

For each (substrate, variant, cutoff) cell, compute headline metrics:
- %var_reduced (background, excluding any injected burst region)
- %win_modified (fraction of windows reconstructed)
- wall_time_s

The substrates are the union of what the prior sprints + this robustness
sprint produced:
- SME (24 ch Blum 2019 sample)
- ERP-CORE Flankers sub-001 (30 ch)
- 4-channel synthetic
- 64-channel synthetic
- Real-EOG s01 (Ehrlich 30-ch, 200 Hz, ~58 min — cropped to 120 s)
- 2-h long-recording synthetic (cropped to 120 s for runtime budget)

Each cell runs in try/except so one failure doesn't mask others.
The output is ``cross_variant_matrix.{json,csv}`` plus a heatmap PNG of
%var_reduced per (substrate × variant) at the median cutoff.
"""

# ruff: noqa: I001

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mne_denoise.asr import ASR, AdaptiveASR, JugglerASR

from scripts.run_asr_paper_validation import (  # noqa: E402
    DatasetSpec,
    _inject_bursts,
    _load_dataset,
    _reader_for_path,
)


# A small substrate cap keeps the matrix run under ~30 min.
SUBSTRATE_DURATION_S = 120.0
RESAMPLE_HZ = 250.0

SUBSTRATES = [
    {
        "label": "sme_1_1",
        "path": ROOT
        / "refs/asr/repos/rASRMatlab/sampleData/filtered/sme_1_1.xdf_filt.set",
        "kind": "real_eeg",
    },
    {
        "label": "erp_core_s001",
        "path": ROOT
        / "data/MNE-ERP-CORE-data/ERP-CORE_Subject-001_Task-Flankers_eeg.fif",
        "kind": "real_eeg",
    },
    {
        "label": "ehrlich_s01_real_eog",
        "path": ROOT / ".cache/asr_datasets/real_eog/s01_raw.fif",
        "kind": "real_artifact",
    },
    {
        "label": "low_channel_4ch",
        "path": ROOT / ".cache/asr_datasets/low_channel_4ch.fif",
        "kind": "synthetic_low_ch",
    },
    {
        "label": "high_channel_64ch",
        "path": ROOT / ".cache/asr_datasets/high_channel_64ch.fif",
        "kind": "synthetic_high_ch",
    },
    {
        "label": "long_recording_2h",
        "path": ROOT / ".cache/asr_datasets/long_recording_2h.fif",
        "kind": "synthetic_long",
    },
]

VARIANTS = [
    "standard",
    "riemannian",
    "riemannian_windowed",
    "adaptive_psp",
    "adaptive_psw",
    "adaptive_mw_final_state",
    "adaptive_mw_sliding",
    "juggler_dbscan",
    "juggler_gev",
]
CUTOFFS = [5.0, 20.0, 50.0]


def _load_substrate(spec: dict) -> mne.io.Raw:
    path = Path(spec["path"])
    if not path.exists():
        raise FileNotFoundError(f"substrate missing: {path}")
    suffix = path.suffix.lower()
    if suffix == ".fif":
        raw = mne.io.read_raw_fif(path, preload=True, verbose=False)
    elif suffix == ".set":
        raw = mne.io.read_raw_eeglab(path, preload=True, verbose=False)
    else:
        # Fall back to the existing helper
        ds = DatasetSpec(
            path=path,
            label=spec["label"],
            reader=_reader_for_path(path),
            max_duration_s=SUBSTRATE_DURATION_S,
            resample_hz=RESAMPLE_HZ,
            highpass_hz=1.0,
        )
        return _load_dataset(ds)
    # Crop + resample if needed (skip for already-prepped fif files)
    if raw.n_times / raw.info["sfreq"] > SUBSTRATE_DURATION_S:
        raw.crop(tmin=0.0, tmax=SUBSTRATE_DURATION_S, include_tmax=False)
    if raw.info["sfreq"] > RESAMPLE_HZ + 1.0 and spec["kind"] == "real_eeg":
        raw.resample(RESAMPLE_HZ, verbose=False)
    return raw


def _maybe_inject(raw: mne.io.Raw, substrate_kind: str) -> mne.io.Raw:
    """Inject bursts only into substrates that don't already have artifacts."""
    if substrate_kind in ("real_artifact", "synthetic_long"):
        return raw  # already has artifacts
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    rng = np.random.default_rng(97)
    raw_in, _, _ = _inject_bursts(
        raw,
        eeg_picks=eeg_picks,
        rng=rng,
        n_bursts=8,
        burst_duration=0.5,
        amplitude=12.0,
    )
    return raw_in


def _build_variant_asr(variant: str, sfreq: float, cutoff: float):
    common = {
        "sfreq": sfreq,
        "cutoff": cutoff,
        "picks": "eeg",
        "max_mem_mb": 512,
        "copy": True,
        "random_state": 97,
        "verbose": False,
    }
    if variant == "standard":
        return ASR(**common)
    if variant == "riemannian":
        return ASR(**common, method="riemannian", experimental=True)
    if variant == "riemannian_windowed":
        return ASR(**common, method="riemannian_windowed", experimental=True)
    if variant == "adaptive_psp":
        return AdaptiveASR(**common, variant="psp")
    if variant == "adaptive_psw":
        return AdaptiveASR(**common, variant="psw")
    if variant == "adaptive_mw_final_state":
        return AdaptiveASR(
            **common, variant="mw", mw_window_length=20.0, mw_mode="final_state"
        )
    if variant == "adaptive_mw_sliding":
        return AdaptiveASR(
            **common, variant="mw", mw_window_length=20.0, mw_mode="sliding"
        )
    if variant == "juggler_dbscan":
        return JugglerASR(**common, strategy="dbscan")
    if variant == "juggler_gev":
        return JugglerASR(**common, strategy="gev")
    raise ValueError(f"unknown variant {variant}")


def _run_cell(raw: mne.io.Raw, variant: str, cutoff: float) -> dict:
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    est = _build_variant_asr(variant, raw.info["sfreq"], cutoff)
    t0 = time.perf_counter()
    if variant == "adaptive_mw_sliding":
        cleaned = est.fit_transform(raw)
    else:
        est.fit(raw)
        cleaned = est.transform(raw)
    dt = time.perf_counter() - t0
    before = raw.get_data(picks=eeg_picks)
    after = cleaned.get_data(picks=eeg_picks)
    var_red = (np.var(before) - np.var(after)) / max(
        np.var(before), np.finfo(float).eps
    )
    diag = getattr(est, "diagnostics_", {})
    return {
        "wall_time_s": float(dt),
        "pct_var_reduced": float(var_red * 100),
        "pct_windows_modified": float(
            diag.get("fraction_reconstructed_windows", 0.0) * 100
        ),
        "n_channels": int(len(eeg_picks)),
    }


def main() -> int:
    out_dir = ROOT / "reports" / "paper_validation" / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for spec in SUBSTRATES:
        print(f"\n=== substrate: {spec['label']} ({spec['kind']}) ===")
        try:
            raw_clean = _load_substrate(spec)
            raw_in = _maybe_inject(raw_clean, spec["kind"])
        except Exception as exc:  # noqa: BLE001
            print(f"  substrate load failed: {exc}")
            for variant in VARIANTS:
                for k in CUTOFFS:
                    rows.append(
                        {
                            "substrate": spec["label"],
                            "variant": variant,
                            "cutoff": k,
                            "status": "substrate_failed",
                            "error": str(exc),
                        }
                    )
            continue
        for variant in VARIANTS:
            for k in CUTOFFS:
                cell = {
                    "substrate": spec["label"],
                    "kind": spec["kind"],
                    "variant": variant,
                    "cutoff": float(k),
                }
                try:
                    metrics = _run_cell(raw_in, variant, k)
                    cell.update({"status": "passed", **metrics})
                except Exception as exc:  # noqa: BLE001
                    cell.update(
                        {
                            "status": "failed",
                            "error_type": type(exc).__name__,
                            "error": str(exc)[:200],
                        }
                    )
                rows.append(cell)
                marker = "OK" if cell["status"] == "passed" else "XX"
                pct = cell.get("pct_var_reduced", float("nan"))
                t = cell.get("wall_time_s", float("nan"))
                print(
                    f"  [{marker}] {variant:>24s} k={k:>5.1f}  "
                    f"%var={pct:>5.1f}%  t={t:>5.1f}s"
                )

    # JSON
    (out_dir / "cross_variant_matrix.json").write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "n_cells": len(rows),
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    # CSV
    keys = sorted({k for r in rows for k in r})
    with (out_dir / "cross_variant_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Heatmap: %var_reduced at k=20
    substrates = [s["label"] for s in SUBSTRATES]
    matrix = np.full((len(substrates), len(VARIANTS)), np.nan)
    for r in rows:
        if r.get("status") != "passed" or abs(r["cutoff"] - 20.0) > 1e-6:
            continue
        i = substrates.index(r["substrate"])
        j = VARIANTS.index(r["variant"])
        matrix[i, j] = r["pct_var_reduced"]
    fig, ax = plt.subplots(figsize=(12, max(4, len(substrates) * 0.45)))
    im = ax.imshow(matrix, cmap="RdYlBu_r", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(np.arange(len(VARIANTS)))
    ax.set_xticklabels(VARIANTS, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(substrates)))
    ax.set_yticklabels(substrates, fontsize=9)
    ax.set_title("Cross-variant matrix — %var_reduced at k=20 (NaN = failed)")
    for i in range(len(substrates)):
        for j in range(len(VARIANTS)):
            v = matrix[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="0.4")
            else:
                ax.text(
                    j,
                    i,
                    f"{v:.0f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if v > 50 else "black",
                )
    fig.colorbar(im, ax=ax, label="%var_reduced")
    fig.tight_layout()
    fig.savefig(out_dir / "cross_variant_matrix.png", dpi=140)
    plt.close(fig)
    print(f"\nWrote cross_variant_matrix.{{json,csv,png}} under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
