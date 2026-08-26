"""Augment Tier-B figures for Chang 2020 + Mullen 2013.

The main validation script (``scripts/run_asr_paper_validation.py``) already
produces five Tier-B figures per dataset:

- ``fig_chang2020_cutoff_sweep.png``   (Chang 2020 Fig 2)
- ``fig_blum2019_computation_time.png``
- ``fig_kim2025_reference_fraction.png``
- ``fig_kim2025_psd_overlay.png``       (PSD before/after per variant)
- ``fig_blum2019_blink_reduction.png``

This script adds the two figures that script does **not** generate, which
together complete the standard-ASR Tier B target for Chang 2020 + Mullen 2013:

- ``fig_chang2020_topomap_variance_change.png``   (Chang 2020 Fig 6 lookalike)
- ``fig_mullen2013_burst_demo.png``               (raw vs cleaned trace overlay)

Both pull the dataset spec from the validation script's ``summary.json`` /
``rows.json`` outputs and *reproduce* the input deterministically (same
``random_state=97`` burst injection). No new datasets fetched; runs are local.

Sibling ``.json`` files are emitted next to each PNG so the figures are
auditable (numbers behind every panel).
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mne_denoise.asr import ASR

# We deliberately reuse the loader + burst injector from the validation script
# so the substrate is byte-identical to what produced the Tier-A metrics.
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
        default=ROOT / "reports" / "paper_validation" / "standard_asr",
        help="Output root from run_asr_paper_validation.py",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=20.0,
        help="Cutoff at which to recompute ASR for topo + burst demo",
    )
    parser.add_argument(
        "--method",
        choices=("standard", "riemannian"),
        default="standard",
        help="ASR backend used when re-running for the topo + burst figures.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=97,
        help="Must match the random_state used in Stage 2 to reproduce bursts",
    )
    parser.add_argument(
        "--burst-count",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--burst-duration",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--burst-amplitude",
        type=float,
        default=12.0,
    )
    parser.add_argument(
        "--max-bursts-to-plot",
        type=int,
        default=3,
        help="How many burst windows to show in the Mullen 2013 demo",
    )
    parser.add_argument(
        "--demo-channels",
        type=int,
        default=4,
        help="How many channels to overlay in the Mullen 2013 demo (front + side)",
    )
    return parser.parse_args()


def _dataset_spec_from_summary(
    item: dict, max_duration_s: float, resample_hz: float, highpass_hz: float
) -> DatasetSpec:
    path = Path(item["path"])
    return DatasetSpec(
        path=path,
        label=item["label"],
        reader=_reader_for_path(path),
        max_duration_s=max_duration_s,
        resample_hz=resample_hz,
        highpass_hz=highpass_hz,
    )


def _ensure_montage(raw: mne.io.BaseRaw) -> bool:
    """Best-effort montage assignment. Returns True if positions are usable."""
    info = raw.info
    montage = info.get_montage()
    if montage is not None:
        # Verify at least one EEG channel has non-NaN xyz
        for ch in info["chs"]:
            loc = ch.get("loc")
            if (
                loc is not None
                and np.isfinite(loc[:3]).all()
                and np.linalg.norm(loc[:3]) > 0
            ):
                return True
    # Fall back to the standard 10-20 montage by channel name
    try:
        std = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(std, match_case=False, on_missing="ignore", verbose=False)
        for ch in raw.info["chs"]:
            loc = ch.get("loc")
            if (
                loc is not None
                and np.isfinite(loc[:3]).all()
                and np.linalg.norm(loc[:3]) > 0
            ):
                return True
    except Exception:  # noqa: BLE001
        pass
    return False


def _plot_topomap_variance_change(
    raw_in: mne.io.BaseRaw,
    raw_clean: mne.io.BaseRaw,
    out_dir: Path,
    label: str,
    cutoff: float,
    method: str = "standard",
) -> dict | None:
    eeg_picks = mne.pick_types(raw_in.info, eeg=True, exclude=[])
    if len(eeg_picks) == 0:
        return None
    before = raw_in.get_data(picks=eeg_picks)
    after = raw_clean.get_data(picks=eeg_picks)
    var_b = np.var(before, axis=1)
    var_a = np.var(after, axis=1)
    # % variance reduced per channel
    pct_reduced = (var_b - var_a) / np.maximum(var_b, np.finfo(float).eps) * 100.0
    ch_names = [raw_in.ch_names[i] for i in eeg_picks]

    info = mne.pick_info(raw_in.info, eeg_picks)
    positions_ok = _ensure_montage(raw_in)
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    if positions_ok:
        # Update info object too after montage assignment
        info = mne.pick_info(raw_in.info, eeg_picks)
        # Symmetric vmin/vmax around 0 so red = reduced, blue = increased
        vmax = float(max(np.abs(pct_reduced).max(), 1.0))
        im, _ = mne.viz.plot_topomap(
            pct_reduced,
            info,
            axes=ax,
            show=False,
            cmap="RdBu_r",
            vlim=(-vmax, vmax),
            sensors=True,
            contours=6,
        )
        fig.colorbar(im, ax=ax, label="% variance reduced")
        ax.set_title(
            f"{label} — Chang 2020 Fig 6 equivalent (k={cutoff:g})\n"
            f"per-channel variance reduction"
        )
    else:
        # Bar fallback when no electrode positions are known
        order = np.argsort(pct_reduced)[::-1]
        ax.barh(
            np.array(ch_names)[order],
            pct_reduced[order],
            color="steelblue",
        )
        ax.set_xlabel("% variance reduced")
        ax.set_title(
            f"{label} — per-channel variance reduction (k={cutoff:g})\n"
            "(no montage — Chang Fig 6 topo unavailable)"
        )
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    png = out_dir / f"fig_chang2020_topomap_variance_change_{method}.png"
    fig.savefig(png, dpi=140)
    plt.close(fig)
    payload = {
        "cutoff": float(cutoff),
        "channels": ch_names,
        "pct_variance_reduced": pct_reduced.tolist(),
        "positions_available": bool(positions_ok),
    }
    png.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _plot_burst_demo(
    raw_in: mne.io.BaseRaw,
    raw_clean: mne.io.BaseRaw,
    burst_mask: np.ndarray,
    out_dir: Path,
    label: str,
    cutoff: float,
    max_bursts: int,
    n_channels: int,
    method: str = "standard",
) -> dict | None:
    eeg_picks = mne.pick_types(raw_in.info, eeg=True, exclude=[])
    if len(eeg_picks) == 0 or not np.any(burst_mask):
        return None
    sfreq = raw_in.info["sfreq"]

    # Pick channels: prefer frontal + one occipital for variety
    ch_names = [raw_in.ch_names[i] for i in eeg_picks]
    preferred = ["Fp1", "Fp2", "Fz", "Cz", "Pz", "Oz", "FC3", "AF3"]
    chosen = [c for c in preferred if c in ch_names]
    if len(chosen) < n_channels:
        # backfill with the first available channels not already chosen
        for c in ch_names:
            if c not in chosen:
                chosen.append(c)
            if len(chosen) >= n_channels:
                break
    chosen = chosen[:n_channels]
    chan_idx_global = [raw_in.ch_names.index(c) for c in chosen]

    before = raw_in.get_data(picks=chan_idx_global) * 1e6  # to µV
    after = raw_clean.get_data(picks=chan_idx_global) * 1e6

    # Identify burst windows (contiguous mask runs)
    diff = np.diff(burst_mask.astype(int), prepend=0, append=0)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    windows = list(zip(starts.tolist(), ends.tolist()))[:max_bursts]
    if not windows:
        return None

    fig, axes = plt.subplots(
        len(windows),
        n_channels,
        figsize=(3.3 * n_channels, 2.2 * len(windows)),
        squeeze=False,
        sharey="row",
    )
    pad = int(0.5 * sfreq)  # 0.5 s margin around each burst
    for r, (s, e) in enumerate(windows):
        ws = max(0, s - pad)
        we = min(raw_in.n_times, e + pad)
        t = (np.arange(ws, we) - s) / sfreq  # 0 at burst onset
        for c, ch in enumerate(chosen):
            ax = axes[r, c]
            ax.plot(t, before[c, ws:we], color="0.5", lw=0.9, label="Raw + burst")
            ax.plot(t, after[c, ws:we], color="C0", lw=1.0, label="ASR cleaned")
            ax.axvspan(0.0, (e - s) / sfreq, color="orange", alpha=0.15)
            if r == 0:
                ax.set_title(ch, fontsize=10)
            if c == 0:
                ax.set_ylabel(f"Burst {r + 1}\nµV", fontsize=9)
            if r == len(windows) - 1:
                ax.set_xlabel("Time relative to burst onset (s)")
            ax.grid(True, alpha=0.3)
            if r == 0 and c == 0:
                ax.legend(loc="upper right", fontsize=7)
    fig.suptitle(
        f"{label} — Mullen 2013 burst-suppression demo (k={cutoff:g})\n"
        f"orange = injected burst window"
    )
    fig.tight_layout()
    png = out_dir / f"fig_mullen2013_burst_demo_{method}.png"
    fig.savefig(png, dpi=140)
    plt.close(fig)
    payload = {
        "cutoff": float(cutoff),
        "channels": chosen,
        "n_bursts_plotted": len(windows),
        "window_samples": [(int(s), int(e)) for s, e in windows],
        "sfreq": float(sfreq),
    }
    png.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _process_dataset(
    summary_item: dict,
    summary: dict,
    out_dir: Path,
    args: argparse.Namespace,
) -> dict:
    config = summary.get("config", {})
    max_duration = float(config.get("datasets", [{}])[0].get("max_duration_s", 120.0))
    resample_hz = float(config.get("datasets", [{}])[0].get("resample_hz", 250.0))
    highpass_hz = float(config.get("datasets", [{}])[0].get("highpass_hz", 1.0))

    # Recover per-dataset preprocessing values (may differ between entries)
    for ds_cfg in config.get("datasets", []):
        if ds_cfg.get("label") == summary_item["label"]:
            max_duration = float(ds_cfg.get("max_duration_s", max_duration))
            resample_hz = float(ds_cfg.get("resample_hz", resample_hz))
            highpass_hz = float(ds_cfg.get("highpass_hz", highpass_hz))
            break

    ds = _dataset_spec_from_summary(
        summary_item, max_duration, resample_hz, highpass_hz
    )
    raw = _load_dataset(ds)

    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    rng = np.random.default_rng(args.random_state)
    inject_info = summary_item.get("injection", {})
    if inject_info.get("enabled", True):
        raw_in, burst_mask, _ = _inject_bursts(
            raw,
            eeg_picks=eeg_picks,
            rng=rng,
            n_bursts=args.burst_count,
            burst_duration=args.burst_duration,
            amplitude=args.burst_amplitude,
        )
    else:
        raw_in = raw.copy()
        burst_mask = np.zeros(raw.n_times, dtype=bool)

    # ASR @ cutoff (this matches the Stage 2 run for the requested method)
    asr_kwargs = {
        "sfreq": raw_in.info["sfreq"],
        "cutoff": args.cutoff,
        "picks": "eeg",
        "max_mem_mb": 512,
        "copy": True,
        "random_state": args.random_state,
        "verbose": False,
    }
    if args.method == "riemannian":
        asr_kwargs.update(method="riemannian", experimental=True)
    est = ASR(**asr_kwargs)
    raw_clean = est.fit_transform(raw_in)

    topo_payload = _plot_topomap_variance_change(
        raw_in,
        raw_clean,
        out_dir,
        summary_item["label"],
        args.cutoff,
        method=args.method,
    )
    burst_payload = _plot_burst_demo(
        raw_in,
        raw_clean,
        burst_mask,
        out_dir,
        summary_item["label"],
        args.cutoff,
        args.max_bursts_to_plot,
        args.demo_channels,
        method=args.method,
    )
    return {
        "label": summary_item["label"],
        "cutoff": float(args.cutoff),
        "topo_emitted": topo_payload is not None,
        "burst_emitted": burst_payload is not None,
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
            record = _process_dataset(item, summary, out_dir, args)
            out_records.append(record)
            print(
                f"[{item['label']}] topo={record['topo_emitted']} "
                f"burst={record['burst_emitted']}"
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
        json.dumps({"cutoff": args.cutoff, "records": out_records}, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {reports_dir / 'stage3_figures.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
