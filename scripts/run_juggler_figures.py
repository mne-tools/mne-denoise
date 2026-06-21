"""Juggler-specific figures: DBSCAN cluster viz + GEV mode viz.

The main validation script (``run_asr_paper_validation.py``) already produces
the Kim 2025 Fig 8 lookalike (``fig_kim2025_reference_fraction.png``). This
script adds two diagnostic plots that the main script does NOT generate:

- ``fig_juggler_dbscan_cluster_<label>.png`` — scatter plot of the top-2
  DBSCAN feature dimensions, coloured by cluster label, with the selected
  cluster highlighted. Shows the paper's "tight cluster of clean amplitudes"
  visually.
- ``fig_juggler_gev_mode_<label>.png`` — histogram of the leading amplitude
  with the fitted GEV pdf overlaid and the GEV mode marked. Samples to the
  left of the mode are the reference selection.
- ``fig_juggler_selection_timeline_<label>_<strategy>.png`` — one EEG
  channel's amplitude over time, coloured by whether each sample was
  selected as a reference. Makes the per-sample selection visible against
  the raw signal.

Inputs are the same substrates the validation script ran in Stage 2.
"""

# ruff: noqa: I001

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mne_denoise.asr import select_juggler_reference_samples

from scripts.run_asr_paper_validation import (  # noqa: E402
    DatasetSpec,
    _inject_bursts,
    _load_dataset,
    _reader_for_path,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=ROOT / "reports" / "paper_validation" / "juggler",
    )
    parser.add_argument("--cutoff", type=float, default=20.0)
    parser.add_argument("--random-state", type=int, default=97)
    parser.add_argument("--burst-count", type=int, default=40)
    parser.add_argument("--burst-duration", type=float, default=0.5)
    parser.add_argument("--burst-amplitude", type=float, default=12.0)
    parser.add_argument(
        "--demo-channel",
        type=str,
        default=None,
        help="Channel name to plot in the selection-timeline figure (default: first frontal)",
    )
    return parser.parse_args()


def _dataset_spec_from_summary(item: dict, summary_cfg: dict) -> DatasetSpec:
    path = Path(item["path"])
    cfg_ds = next(
        (d for d in summary_cfg.get("datasets", []) if d.get("label") == item["label"]),
        {},
    )
    return DatasetSpec(
        path=path,
        label=item["label"],
        reader=_reader_for_path(path),
        max_duration_s=float(cfg_ds.get("max_duration_s", 120.0)),
        resample_hz=float(cfg_ds.get("resample_hz", 250.0)),
        highpass_hz=float(cfg_ds.get("highpass_hz", 1.0)),
    )


def _plot_dbscan_cluster(
    contaminated: np.ndarray,
    sfreq: float,
    out_dir: Path,
    label: str,
) -> dict:
    """Scatter the top-2 DBSCAN feature dimensions, colour by cluster label."""
    # Run the selection to collect diagnostics
    _, mask, diag = select_juggler_reference_samples(
        contaminated,
        sfreq,
        strategy="dbscan",
    )
    # Reproduce the top-K feature matrix the same way juggler.py does
    from mne_denoise.asr.core import _apply_statistics_filter, _design_statistics_filter

    b, a = _design_statistics_filter(sfreq, "asr")
    Xs = _apply_statistics_filter(contaminated, b, a)
    Xs -= np.median(Xs, axis=1, keepdims=True)
    amp = np.sort(np.abs(Xs), axis=0)[::-1]
    top_k = min(int(diag["dbscan_top_k"]), contaminated.shape[0])
    features = amp[:top_k].T  # (n_times, top_k)
    labels = diag["juggler_dbscan_labels"]
    selected_label = int(diag["juggler_dbscan_selected_label"])

    fig, ax = plt.subplots(figsize=(6, 5))
    # Noise (label=-1) as grey, other clusters as colour, selected as red
    noise = labels == -1
    selected = labels == selected_label
    others = (~noise) & (~selected)
    if noise.any():
        ax.scatter(
            features[noise, 0],
            features[noise, 1],
            s=4,
            c="0.7",
            alpha=0.4,
            label=f"noise ({int(noise.sum())})",
        )
    if others.any():
        ax.scatter(
            features[others, 0],
            features[others, 1],
            s=4,
            c="C0",
            alpha=0.6,
            label=f"other clusters ({int(others.sum())})",
        )
    if selected.any():
        ax.scatter(
            features[selected, 0],
            features[selected, 1],
            s=4,
            c="C3",
            alpha=0.8,
            label=f"selected cluster ({int(selected.sum())})",
        )
    ax.set_xlabel("Top-1 amplitude (µV)")
    ax.set_ylabel("Top-2 amplitude (µV)")
    ax.set_title(
        f"{label} — Juggler DBSCAN feature space\n"
        f"eps={diag['juggler_dbscan_eps']:.2g}  min_samples={diag['juggler_dbscan_min_samples']}  "
        f"retained {mask.mean() * 100:.1f}%"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    png = out_dir / f"fig_juggler_dbscan_cluster_{label}.png"
    fig.savefig(png, dpi=140)
    plt.close(fig)

    payload = {
        "label": label,
        "n_clusters": int(np.unique(labels[labels >= 0]).size),
        "selected_label": selected_label,
        "selected_size": int(selected.sum()),
        "noise_size": int(noise.sum()),
        "reference_fraction": float(mask.mean()),
        "eps": float(diag["juggler_dbscan_eps"]),
        "min_samples": int(diag["juggler_dbscan_min_samples"]),
    }
    png.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _plot_gev_mode(
    contaminated: np.ndarray,
    sfreq: float,
    out_dir: Path,
    label: str,
) -> dict:
    """Histogram of leading amplitude with fitted GEV pdf overlaid."""
    _, mask, diag = select_juggler_reference_samples(
        contaminated,
        sfreq,
        strategy="gev",
    )
    leading = diag["leading_amplitude"]
    shape = diag["juggler_gev_shape"]
    loc = diag["juggler_gev_loc"]
    scale = diag["juggler_gev_scale"]
    mode = diag["juggler_gev_mode"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(
        leading,
        bins=80,
        density=True,
        color="steelblue",
        edgecolor="white",
        alpha=0.7,
        label=f"leading amplitude (n={leading.size})",
    )
    x = np.linspace(float(leading.min()), float(leading.max()), 1000)
    pdf = stats.genextreme.pdf(x, shape, loc=loc, scale=scale)
    ax.plot(x, pdf, color="C3", lw=2.0, label="fitted GEV pdf")
    ax.axvline(
        mode, color="black", linestyle="--", lw=1.2, label=f"GEV mode = {mode:.2g}"
    )
    ax.axvspan(
        float(leading.min()),
        mode,
        color="green",
        alpha=0.10,
        label=f"reference (retained {mask.mean() * 100:.1f}%)",
    )
    ax.set_xlabel("Leading-channel amplitude (µV)")
    ax.set_ylabel("Density")
    ax.set_title(
        f"{label} — Juggler GEV mode selection\n"
        f"shape={shape:.3f}, loc={loc:.2g}, scale={scale:.2g}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    png = out_dir / f"fig_juggler_gev_mode_{label}.png"
    fig.savefig(png, dpi=140)
    plt.close(fig)

    payload = {
        "label": label,
        "shape": float(shape),
        "loc": float(loc),
        "scale": float(scale),
        "mode": float(mode),
        "reference_fraction": float(mask.mean()),
        "n_samples": int(leading.size),
    }
    png.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _plot_selection_timeline(
    raw_in: mne.io.BaseRaw,
    sfreq: float,
    strategy: str,
    out_dir: Path,
    label: str,
    demo_channel: str | None,
) -> dict | None:
    """One EEG channel over time, coloured by selected (green) vs rejected (red)."""
    eeg_picks = mne.pick_types(raw_in.info, eeg=True, exclude=[])
    if len(eeg_picks) == 0:
        return None
    data = raw_in.get_data(picks=eeg_picks)

    _, mask, _ = select_juggler_reference_samples(data, sfreq, strategy=strategy)

    ch_names = [raw_in.ch_names[i] for i in eeg_picks]
    if demo_channel is None or demo_channel not in ch_names:
        # Prefer frontal channels for visibility
        preferred = ["Fp1", "Fp2", "Fz", "Cz", "Pz"]
        demo_channel = next((c for c in preferred if c in ch_names), ch_names[0])
    chan_idx = ch_names.index(demo_channel)
    sig_uv = data[chan_idx] * 1e6
    t = np.arange(sig_uv.size) / sfreq

    # Plot the first 30 seconds for legibility
    horizon = int(min(30.0 * sfreq, sig_uv.size))
    t_h = t[:horizon]
    sig_h = sig_uv[:horizon]
    mask_h = mask[:horizon]

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(t_h, sig_h, color="0.5", lw=0.6, label="EEG", alpha=0.7)
    # Highlight selected samples
    sel_idx = np.where(mask_h)[0]
    if sel_idx.size > 0:
        # Plot selected samples as faint green dots
        ax.plot(
            t_h[sel_idx],
            sig_h[sel_idx],
            ".",
            color="C2",
            ms=1.2,
            alpha=0.5,
            label=f"selected reference ({mask.mean() * 100:.1f}% overall)",
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"{demo_channel} (µV)")
    ax.set_title(
        f"{label} — Juggler {strategy.upper()} per-sample reference selection (first 30 s)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    png = out_dir / f"fig_juggler_selection_timeline_{label}_{strategy}.png"
    fig.savefig(png, dpi=140)
    plt.close(fig)

    payload = {
        "label": label,
        "strategy": strategy,
        "demo_channel": demo_channel,
        "horizon_seconds": float(horizon / sfreq),
        "reference_fraction_overall": float(mask.mean()),
    }
    png.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _process_dataset(
    summary_item: dict,
    summary: dict,
    out_dir: Path,
    args: argparse.Namespace,
) -> dict:
    ds = _dataset_spec_from_summary(summary_item, summary.get("config", {}))
    raw = _load_dataset(ds)

    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    rng = np.random.default_rng(args.random_state)
    inject_info = summary_item.get("injection", {})
    if inject_info.get("enabled", True):
        raw_in, _, _ = _inject_bursts(
            raw,
            eeg_picks=eeg_picks,
            rng=rng,
            n_bursts=args.burst_count,
            burst_duration=args.burst_duration,
            amplitude=args.burst_amplitude,
        )
    else:
        raw_in = raw.copy()
    data = raw_in.get_data(picks=eeg_picks)

    dbscan_payload = _plot_dbscan_cluster(
        data, raw_in.info["sfreq"], out_dir, summary_item["label"]
    )
    gev_payload = _plot_gev_mode(
        data, raw_in.info["sfreq"], out_dir, summary_item["label"]
    )
    timeline_dbscan = _plot_selection_timeline(
        raw_in,
        raw_in.info["sfreq"],
        "dbscan",
        out_dir,
        summary_item["label"],
        args.demo_channel,
    )
    timeline_gev = _plot_selection_timeline(
        raw_in,
        raw_in.info["sfreq"],
        "gev",
        out_dir,
        summary_item["label"],
        args.demo_channel,
    )
    return {
        "label": summary_item["label"],
        "dbscan": dbscan_payload,
        "gev": gev_payload,
        "timeline_dbscan": timeline_dbscan,
        "timeline_gev": timeline_gev,
    }


def main() -> int:
    args = _parse_args()
    reports_dir = args.reports_dir.resolve()
    summary_path = reports_dir / "summary.json"
    if not summary_path.exists():
        raise SystemExit(
            f"summary.json not found at {summary_path}. "
            "Run scripts/run_asr_paper_validation.py first."
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    out_records = []
    for item in summary.get("results", []):
        out_dir = reports_dir / item["label"]
        if not out_dir.exists():
            print(f"[skip] {item['label']}: dataset output dir missing")
            continue
        try:
            rec = _process_dataset(item, summary, out_dir, args)
            out_records.append(rec)
            print(
                f"[{item['label']}] dbscan_frac={rec['dbscan']['reference_fraction'] * 100:.1f}% "
                f"gev_frac={rec['gev']['reference_fraction'] * 100:.1f}%"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[{item['label']}] FAILED: {type(exc).__name__}: {exc}")
            out_records.append(
                {
                    "label": item["label"],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    (reports_dir / "stage3_figures.json").write_text(
        json.dumps({"records": out_records}, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {reports_dir / 'stage3_figures.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
