"""rASR-vs-standard comparison plots — Blum 2019 Figs 3 / 4 / 5 lookalikes.

Reads the standard-ASR and rASR report directories produced by
``scripts/run_asr_paper_validation.py``. For each dataset present in BOTH
directories, emits three side-by-side comparison figures plus their numerical
sidecars:

- ``fig_blum2019_rASR_vs_standard_blink_reduction.png`` — **Blum 2019 Fig 3**:
  grouped bar chart of frontal peak-to-peak amplitude before/after, comparing
  standard ASR and rASR at each cutoff swept in Stage 2.
- ``fig_blum2019_rASR_vs_standard_topo_diff.png`` — **Blum 2019 Fig 4**:
  scalp topography of ``(% var reduced rASR) − (% var reduced standard)`` at
  ``--cutoff``. Needs the per-channel variance numbers from the Stage 3
  topomap-sidecar JSONs.
- ``fig_blum2019_rASR_vs_standard_computation_time.png`` — **Blum 2019 Fig 5**:
  log-log plot of wall time per ``fit_transform`` vs cutoff, both variants.

Outputs are written to the rASR report directory by default.
"""

# ruff: noqa: I001

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--standard-dir",
        type=Path,
        default=ROOT / "reports" / "paper_validation" / "standard_asr",
    )
    parser.add_argument(
        "--rasr-dir",
        type=Path,
        default=ROOT / "reports" / "paper_validation" / "rasr",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=20.0,
        help="Cutoff used for the topo-diff figure (Fig 4 lookalike).",
    )
    return parser.parse_args()


def _load_rows(report_dir: Path, dataset_label: str) -> list[dict]:
    p = report_dir / dataset_label / "rows.json"
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


def _pick_variant_rows(rows: list[dict], variant: str) -> list[dict]:
    return sorted(
        [
            r
            for r in rows
            if r.get("variant") == variant and r.get("status") == "passed"
        ],
        key=lambda r: r["cutoff"],
    )


def _plot_blink_reduction_comparison(
    std_rows: list[dict],
    rasr_rows: list[dict],
    out_dir: Path,
    label: str,
) -> dict | None:
    std_use = [r for r in std_rows if r.get("blink_reduction_pct") is not None]
    rasr_use = [r for r in rasr_rows if r.get("blink_reduction_pct") is not None]
    if not std_use and not rasr_use:
        return None

    std_ks = np.array([r["cutoff"] for r in std_use])
    std_red = np.array([r["blink_reduction_pct"] for r in std_use])
    rasr_ks = np.array([r["cutoff"] for r in rasr_use])
    rasr_red = np.array([r["blink_reduction_pct"] for r in rasr_use])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if std_use:
        ax.plot(std_ks, std_red, marker="o", label="Standard ASR", color="C0")
    if rasr_use:
        ax.plot(rasr_ks, rasr_red, marker="s", label="rASR (Riemannian)", color="C3")
    ax.set_xscale("log")
    ax.set_xlabel("Cutoff k (SDs)")
    ax.set_ylabel("Frontal peak-to-peak reduction (%)")
    ax.set_title(f"{label} — Blum 2019 Fig 3 equivalent")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    png = out_dir / "fig_blum2019_rASR_vs_standard_blink_reduction.png"
    fig.savefig(png, dpi=140)
    plt.close(fig)

    payload = {
        "label": label,
        "standard": {
            "cutoffs": std_ks.tolist(),
            "blink_reduction_pct": std_red.tolist(),
        },
        "riemannian": {
            "cutoffs": rasr_ks.tolist(),
            "blink_reduction_pct": rasr_red.tolist(),
        },
    }
    png.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _plot_computation_time_comparison(
    std_rows: list[dict],
    rasr_rows: list[dict],
    out_dir: Path,
    label: str,
) -> dict | None:
    if not std_rows and not rasr_rows:
        return None

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if std_rows:
        ax.plot(
            [r["cutoff"] for r in std_rows],
            [r["wall_time_s"] for r in std_rows],
            marker="o",
            label="Standard ASR",
            color="C0",
        )
    if rasr_rows:
        ax.plot(
            [r["cutoff"] for r in rasr_rows],
            [r["wall_time_s"] for r in rasr_rows],
            marker="s",
            label="rASR (Riemannian)",
            color="C3",
        )
    ax.set_xscale("log")
    ax.set_xlabel("Cutoff k (SDs)")
    ax.set_ylabel("Wall time per fit_transform (s)")
    ax.set_title(f"{label} — Blum 2019 Fig 5 equivalent")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    png = out_dir / "fig_blum2019_rASR_vs_standard_computation_time.png"
    fig.savefig(png, dpi=140)
    plt.close(fig)

    payload = {
        "label": label,
        "standard": {
            "cutoffs": [r["cutoff"] for r in std_rows],
            "wall_time_s": [r["wall_time_s"] for r in std_rows],
        },
        "riemannian": {
            "cutoffs": [r["cutoff"] for r in rasr_rows],
            "wall_time_s": [r["wall_time_s"] for r in rasr_rows],
        },
    }
    png.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _plot_topo_diff(
    std_dir: Path,
    rasr_dir: Path,
    out_dir: Path,
    label: str,
) -> dict | None:
    """Per-channel (% var reduced rASR - % var reduced standard) on scalp."""
    std_json = std_dir / label / "fig_chang2020_topomap_variance_change.json"
    rasr_json = (
        rasr_dir / label / "fig_chang2020_topomap_variance_change_riemannian.json"
    )
    if not std_json.exists() or not rasr_json.exists():
        return {
            "label": label,
            "error": (
                f"Missing topomap JSON sidecar(s): "
                f"std_exists={std_json.exists()}, rasr_exists={rasr_json.exists()}"
            ),
        }

    std = json.loads(std_json.read_text(encoding="utf-8"))
    rasr = json.loads(rasr_json.read_text(encoding="utf-8"))

    # Align by channel name (sets must intersect)
    std_pct = dict(zip(std["channels"], std["pct_variance_reduced"]))
    rasr_pct = dict(zip(rasr["channels"], rasr["pct_variance_reduced"]))
    common = [c for c in std["channels"] if c in rasr_pct]
    if not common:
        return {"label": label, "error": "no common channels"}
    diff = np.array([rasr_pct[c] - std_pct[c] for c in common])

    # Reload the substrate to obtain an info object with channel positions
    # (we don't keep an info pickle in the report dir).
    # Hint paths from the standard-asr summary.json which preserves the original
    # file path. Best-effort.
    summary = json.loads((std_dir / "summary.json").read_text(encoding="utf-8"))
    raw_path = None
    for item in summary.get("results", []):
        if item.get("label") == label:
            raw_path = Path(item["path"])
            break
    info_for_topo = None
    if raw_path and raw_path.exists():
        try:
            suffix = raw_path.suffix.lower()
            if suffix == ".fif":
                raw = mne.io.read_raw_fif(raw_path, preload=False, verbose=False)
            elif suffix == ".set":
                raw = mne.io.read_raw_eeglab(raw_path, preload=False, verbose=False)
            elif suffix == ".vhdr":
                raw = mne.io.read_raw_brainvision(
                    raw_path, preload=False, verbose=False
                )
            else:
                raw = None
            if raw is not None:
                # Apply standard 10-20 montage to maximize chance of usable positions
                try:
                    std_mont = mne.channels.make_standard_montage("standard_1020")
                    raw.set_montage(
                        std_mont, match_case=False, on_missing="ignore", verbose=False
                    )
                except Exception:  # noqa: BLE001
                    pass
                picks = mne.pick_channels(raw.ch_names, include=common, ordered=True)
                info_for_topo = mne.pick_info(raw.info, picks)
        except Exception:  # noqa: BLE001
            info_for_topo = None

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    if info_for_topo is not None:
        vmax = float(max(np.abs(diff).max(), 1.0))
        im, _ = mne.viz.plot_topomap(
            diff,
            info_for_topo,
            axes=ax,
            show=False,
            cmap="RdBu_r",
            vlim=(-vmax, vmax),
            sensors=True,
            contours=6,
        )
        fig.colorbar(im, ax=ax, label="Δ % variance reduced (rASR − standard)")
        ax.set_title(f"{label} — Blum 2019 Fig 4 equivalent")
    else:
        # Bar fallback if positions cannot be resolved
        order = np.argsort(diff)[::-1]
        ax.barh(np.array(common)[order], diff[order], color="purple")
        ax.set_xlabel("Δ % variance reduced (rASR − standard)")
        ax.set_title(f"{label} — per-channel diff (no usable montage)")
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    png = out_dir / "fig_blum2019_rASR_vs_standard_topo_diff.png"
    fig.savefig(png, dpi=140)
    plt.close(fig)

    payload = {
        "label": label,
        "channels": common,
        "diff_pct_variance_reduced": diff.tolist(),
        "info_positions_available": info_for_topo is not None,
        "summary": {
            "min": float(diff.min()),
            "max": float(diff.max()),
            "mean": float(diff.mean()),
            "median": float(np.median(diff)),
            "frontal_mean": float(
                np.mean(
                    [
                        d
                        for c, d in zip(common, diff)
                        if c.lower().startswith(("fp", "af", "fz", "f7", "f8"))
                    ]
                )
            ),
        },
    }
    png.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    args = _parse_args()
    std_dir = args.standard_dir.resolve()
    rasr_dir = args.rasr_dir.resolve()

    if not (std_dir / "summary.json").exists():
        raise SystemExit(f"summary.json missing under {std_dir}")
    if not (rasr_dir / "summary.json").exists():
        raise SystemExit(f"summary.json missing under {rasr_dir}")

    std_summary = json.loads((std_dir / "summary.json").read_text(encoding="utf-8"))
    rasr_summary = json.loads((rasr_dir / "summary.json").read_text(encoding="utf-8"))
    std_labels = {r["label"] for r in std_summary.get("results", [])}
    rasr_labels = {r["label"] for r in rasr_summary.get("results", [])}
    common_labels = sorted(std_labels & rasr_labels)
    if not common_labels:
        raise SystemExit(
            f"No datasets common to both report dirs. "
            f"standard: {sorted(std_labels)} rasr: {sorted(rasr_labels)}"
        )

    out_records = []
    for label in common_labels:
        out_dir = rasr_dir / label
        if not out_dir.exists():
            print(f"[skip] {label}: rASR output dir missing")
            continue
        std_rows = _load_rows(std_dir, label)
        rasr_rows = _load_rows(rasr_dir, label)
        # rASR rows may contain BOTH standard and riemannian variants since
        # the Stage 2 sweep includes both. Pick the riemannian ones.
        rasr_riem = _pick_variant_rows(rasr_rows, "riemannian")
        std_std = _pick_variant_rows(std_rows, "standard")
        # If the rASR-dir's standard sweep gave more data, prefer that.
        rasr_std_in_rasrdir = _pick_variant_rows(rasr_rows, "standard")
        if len(rasr_std_in_rasrdir) >= len(std_std):
            std_std = rasr_std_in_rasrdir

        rec = {"label": label}
        rec["blink"] = _plot_blink_reduction_comparison(
            std_std, rasr_riem, out_dir, label
        )
        rec["timing"] = _plot_computation_time_comparison(
            std_std, rasr_riem, out_dir, label
        )
        rec["topo_diff"] = _plot_topo_diff(std_dir, rasr_dir, out_dir, label)
        out_records.append(rec)
        emitted = [
            k
            for k, v in rec.items()
            if k != "label" and v is not None and "error" not in (v or {})
        ]
        print(f"[{label}] emitted: {emitted}")

    (rasr_dir / "stage3_comparison_figures.json").write_text(
        json.dumps({"cutoff": args.cutoff, "records": out_records}, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {rasr_dir / 'stage3_comparison_figures.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
