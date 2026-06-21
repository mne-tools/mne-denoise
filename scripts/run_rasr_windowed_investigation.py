"""Side-by-side comparison: ``method="riemannian"`` vs ``method="riemannian_windowed"``.

Reproduces the cutoff-invariance bug in the original rASR backend and shows
the per-window backend's monotone behavior on SME + ERP-CORE.

Output: ``reports/paper_validation/robustness/rasr_windowed_fix.{md,json,png}``
"""

# ruff: noqa: I001

from __future__ import annotations

import json
import sys
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

from mne_denoise.asr import ASR

from scripts.run_asr_paper_validation import (  # noqa: E402
    DatasetSpec,
    _inject_bursts,
    _load_dataset,
    _reader_for_path,
)


SUBSTRATES = [
    (
        "sme_1_1.xdf_filt",
        ROOT / "refs/asr/repos/rASRMatlab/sampleData/filtered/sme_1_1.xdf_filt.set",
    ),
    (
        "ERP-CORE_Subject-001_Task-Flankers_eeg",
        ROOT / "data/MNE-ERP-CORE-data/ERP-CORE_Subject-001_Task-Flankers_eeg.fif",
    ),
]

CUTOFFS = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
BURST_COUNT = 8


def _build_ds(label: str, path: Path) -> DatasetSpec:
    return DatasetSpec(
        path=path,
        label=label,
        reader=_reader_for_path(path),
        max_duration_s=120.0,
        resample_hz=250.0,
        highpass_hz=1.0,
    )


def _run_one(raw_in, method: str, cutoff: float) -> dict:
    eeg_picks = mne.pick_types(raw_in.info, eeg=True, exclude=[])
    asr = ASR(
        sfreq=raw_in.info["sfreq"],
        cutoff=cutoff,
        method=method,
        experimental=True,
        picks="eeg",
        max_mem_mb=512,
        copy=True,
        random_state=97,
        verbose=False,
    )
    cleaned = asr.fit_transform(raw_in)
    before = raw_in.get_data(picks=eeg_picks)
    after = cleaned.get_data(picks=eeg_picks)
    diag = asr.diagnostics_
    return {
        "method": method,
        "cutoff": float(cutoff),
        "pct_windows_modified": float(diag.get("fraction_reconstructed_windows", 0.0))
        * 100,
        "pct_var_reduced": float(
            (np.var(before) - np.var(after)) / max(np.var(before), np.finfo(float).eps)
        )
        * 100,
        "covariance_geometry": diag.get("covariance_geometry"),
    }


def main() -> int:
    out_dir = ROOT / "reports" / "paper_validation" / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for label, path in SUBSTRATES:
        print(f"\n=== {label} ===")
        ds = _build_ds(label, path)
        raw = _load_dataset(ds)
        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        rng = np.random.default_rng(97)
        raw_in, _, _ = _inject_bursts(
            raw,
            eeg_picks=eeg_picks,
            rng=rng,
            n_bursts=BURST_COUNT,
            burst_duration=0.5,
            amplitude=12.0,
        )
        for method in ("riemannian", "riemannian_windowed"):
            for k in CUTOFFS:
                row = _run_one(raw_in, method, k)
                row["substrate"] = label
                records.append(row)
                print(
                    f"  {method:>22s} k={k:>5.1f}  "
                    f"%win={row['pct_windows_modified']:>5.1f}%  "
                    f"%var={row['pct_var_reduced']:>5.1f}%"
                )

    json_path = out_dir / "rasr_windowed_fix.json"
    json_path.write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, label_target in zip(axes, [s[0] for s in SUBSTRATES]):
        for method, marker, color in (
            ("riemannian", "o", "C3"),
            ("riemannian_windowed", "s", "C0"),
        ):
            xs = sorted(
                r["cutoff"]
                for r in records
                if r["substrate"] == label_target and r["method"] == method
            )
            rs = sorted(
                [
                    r
                    for r in records
                    if r["substrate"] == label_target and r["method"] == method
                ],
                key=lambda r: r["cutoff"],
            )
            ys = [r["pct_windows_modified"] for r in rs]
            ax.plot(xs, ys, marker=marker, color=color, lw=1.5, label=method)
        ax.set_xscale("log")
        ax.set_xlabel("Cutoff k (SDs)")
        ax.set_ylabel("% windows reconstructed")
        ax.set_title(label_target)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
    fig.suptitle(
        "rASR cutoff-invariance bug + fix: riemannian (flat, bug) vs "
        "riemannian_windowed (monotone, fixed)"
    )
    fig.tight_layout()
    png_path = out_dir / "rasr_windowed_fix.png"
    fig.savefig(png_path, dpi=140)
    plt.close(fig)

    md_lines = [
        "# rASR cutoff-invariance — bug + fix",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Numbers",
        "",
        "| Substrate | Method | k=1 | k=5 | k=20 | k=100 | span |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for substrate, _ in [(s[0], s[1]) for s in SUBSTRATES]:
        for method in ("riemannian", "riemannian_windowed"):
            rs = sorted(
                [
                    r
                    for r in records
                    if r["substrate"] == substrate and r["method"] == method
                ],
                key=lambda r: r["cutoff"],
            )
            by_k = {r["cutoff"]: r["pct_windows_modified"] for r in rs}
            vals = list(by_k.values())
            span = max(vals) - min(vals)
            md_lines.append(
                f"| `{substrate}` | `{method}` | "
                f"{by_k.get(1.0, 0):.1f}% | {by_k.get(5.0, 0):.1f}% | "
                f"{by_k.get(20.0, 0):.1f}% | {by_k.get(100.0, 0):.1f}% | "
                f"**{span:.1f}%** |"
            )
    md_lines.extend(
        [
            "",
            "Read: the `span` column is the headline. Original `riemannian` shows "
            "**flat** (span ~ 0) — the bug. New `riemannian_windowed` shows a "
            "large monotone span — the fix.",
            "",
            f"![rASR fix]({png_path.name})",
        ]
    )
    (out_dir / "rasr_windowed_fix.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(
        f"\nWrote {json_path}\n      {png_path}\n      {out_dir / 'rasr_windowed_fix.md'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
