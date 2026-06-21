"""AASR algorithm-mapping: Python AdaptiveASR vs MATLAB AASR (Tsai et al.).

For each ``tests/parity/matlab_reference/aasr_case_reference_*.mat`` fixture:

1. Load the MATLAB reference (M, T, thresholds, patterns, cleaned) plus the
   input chunks (data, update_chunk_1..n_updates).
2. Replay our Python ``AdaptiveASR`` against the same sequence:

   - ``fit(update_chunk_1)``
   - ``partial_fit(update_chunk_i)`` for i = 2..n_updates
   - ``reset_process_state()``
   - ``transform(data)``

3. Compute Python-vs-MATLAB numerical agreement:

   - **M_relerr**: covariance matrix relative error.
   - **T_basis_corr**: threshold-matrix alignment after a sign-correction.
   - **T_relerr**: threshold-matrix relative error (sign-corrected).
   - **thresholds_relerr** / **thresholds_corr**: scalar threshold vector match.
   - **cleaned_relerr** / **cleaned_corr**: cleaned-signal relative error / Pearson.

4. Emit a per-fixture markdown table + JSON record + a histogram PNG of
   ``cleaned_relerr`` to ``reports/paper_validation/aasr/``. The markdown
   includes a MATLAB-to-Python function-correspondence table so a future
   reader can navigate from MATLAB ``ASR.m`` lines to Python
   ``mne_denoise/asr/adaptive.py`` symbols.

Run-only depends on `scipy`, `numpy`, `matplotlib`, and the in-repo
``mne_denoise.asr`` package. No MATLAB engine needed.
"""

# ruff: noqa: I001

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mne_denoise.asr import AdaptiveASR  # noqa: E402


# ----------------------------------------------------------------------------
# Helpers mirroring tests/parity/test_aasr_parity.py exactly
# ----------------------------------------------------------------------------


def _array(ref: dict, key: str) -> np.ndarray:
    return np.asarray(ref[key], dtype=np.float64)


def _scalar(ref: dict, key: str) -> float:
    return float(np.asarray(ref[key]).squeeze())


def _str(ref: dict, key: str) -> str:
    value = ref[key]
    if isinstance(value, str):
        return value
    return str(np.asarray(value).squeeze())


def _relative_error(a: np.ndarray, b: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(b)), float(np.finfo(float).eps))
    return float(np.linalg.norm(a - b) / denom)


def _threshold_basis_corr(T_py: np.ndarray, T_mat: np.ndarray) -> float:
    """Cosine of T_py and T_mat after per-row sign flips (gauge fix)."""
    py_n = T_py / max(np.linalg.norm(T_py), np.finfo(float).eps)
    mat_n = T_mat / max(np.linalg.norm(T_mat), np.finfo(float).eps)
    signs = np.sign(np.sum(py_n * mat_n, axis=1))
    signs[signs == 0] = 1.0
    return float(np.abs(np.sum(py_n * signs[:, np.newaxis] * mat_n)))


def _threshold_relerr_signed(T_py: np.ndarray, T_mat: np.ndarray) -> float:
    signs = np.sign(np.sum(T_py * T_mat, axis=1))
    signs[signs == 0] = 1.0
    return _relative_error(T_py * signs[:, np.newaxis], T_mat)


# ----------------------------------------------------------------------------
# Per-fixture analysis
# ----------------------------------------------------------------------------


def _process_fixture(ref_path: Path) -> dict:
    ref = loadmat(ref_path, squeeze_me=True)
    case_id = ref_path.stem.replace("aasr_case_reference_", "")
    variant = _str(ref, "variant")
    n_updates = int(round(_scalar(ref, "n_updates")))

    asr = AdaptiveASR(
        sfreq=_scalar(ref, "sfreq"),
        cutoff=_scalar(ref, "cutoff"),
        variant=variant,
        verbose=False,
    )
    asr.fit(_array(ref, "update_chunk_1"))
    for update_idx in range(2, n_updates + 1):
        asr.partial_fit(_array(ref, f"update_chunk_{update_idx}"))

    asr.reset_process_state()
    py_cleaned = asr.transform(_array(ref, "data"))

    M_py = np.asarray(asr.M_)
    T_py = np.asarray(asr.T_)
    thr_py = np.asarray(asr.thresholds_).ravel()

    M_mat = _array(ref, "M")
    T_mat = _array(ref, "T")
    thr_mat = _array(ref, "thresholds").ravel()
    mat_cleaned = _array(ref, "cleaned")

    M_relerr = _relative_error(M_py, M_mat)
    T_basis_corr = _threshold_basis_corr(T_py, T_mat)
    T_relerr = _threshold_relerr_signed(T_py, T_mat)
    thr_relerr = _relative_error(thr_py, thr_mat)
    thr_corr = float(np.corrcoef(thr_py, thr_mat)[0, 1])
    cleaned_relerr = _relative_error(py_cleaned, mat_cleaned)
    cleaned_corr = float(np.corrcoef(py_cleaned.ravel(), mat_cleaned.ravel())[0, 1])

    return {
        "case_id": case_id,
        "variant": variant,
        "n_updates": n_updates,
        "fixture": str(ref_path.relative_to(ROOT)),
        "n_channels": int(M_py.shape[0]),
        "sfreq": _scalar(ref, "sfreq"),
        "cutoff": _scalar(ref, "cutoff"),
        # Calibration deltas
        "M_relerr": M_relerr,
        "T_basis_corr": T_basis_corr,
        "T_relerr": T_relerr,
        "thresholds_relerr": thr_relerr,
        "thresholds_corr": thr_corr,
        # Process deltas
        "cleaned_relerr": cleaned_relerr,
        "cleaned_corr": cleaned_corr,
    }


# ----------------------------------------------------------------------------
# Output writers
# ----------------------------------------------------------------------------


def _plot_cleaned_relerr_histogram(records: list[dict], out_path: Path) -> None:
    values = [r["cleaned_relerr"] for r in records if "cleaned_relerr" in r]
    if not values:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values, bins=20, color="steelblue", edgecolor="white")
    ax.axvline(1e-5, color="red", linestyle="--", lw=1.0, label="parity gate (1e-5)")
    ax.axvline(
        max(values),
        color="0.4",
        linestyle=":",
        lw=1.0,
        label=f"max = {max(values):.2e}",
    )
    ax.set_xlabel("Cleaned-signal relative error (Python vs MATLAB AASR)")
    ax.set_ylabel("Fixture count")
    ax.set_title("AASR algorithm-mapping: per-fixture cleaned-signal agreement")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _write_markdown(
    records: list[dict], json_path: Path, png_path: Path, md_path: Path
) -> None:
    lines: list[str] = []
    lines.append("# AASR algorithm-mapping: Python AdaptiveASR vs MATLAB AASR")
    lines.append("")
    lines.append(
        f"Generated: {datetime.now(timezone.utc).isoformat()}    "
        f"Fixtures analysed: {len(records)}"
    )
    lines.append("")

    # Per-fixture results table
    lines.append("## Per-fixture numerical agreement")
    lines.append("")
    lines.append(
        "| Case | variant | n_upd | M relerr | T basis corr | T relerr | "
        "Thr relerr | Cleaned relerr | Cleaned corr |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in records:
        if "M_relerr" not in r:
            continue
        lines.append(
            f"| `{r['case_id']}` | {r['variant']} | {r['n_updates']} | "
            f"{r['M_relerr']:.2e} | {r['T_basis_corr']:.6f} | "
            f"{r['T_relerr']:.2e} | {r['thresholds_relerr']:.2e} | "
            f"{r['cleaned_relerr']:.2e} | {r['cleaned_corr']:.6f} |"
        )
    lines.append("")

    # Aggregate
    def _agg(key: str) -> str:
        vals = [r[key] for r in records if key in r and not np.isnan(r[key])]
        if not vals:
            return "n/a"
        return f"min={min(vals):.4g}, max={max(vals):.4g}, median={float(np.median(vals)):.4g}"

    lines.append("## Aggregate")
    lines.append("")
    lines.append(f"- **M relerr**: {_agg('M_relerr')}")
    lines.append(f"- **T basis correlation**: {_agg('T_basis_corr')}")
    lines.append(f"- **T relerr (sign-corrected)**: {_agg('T_relerr')}")
    lines.append(f"- **Cleaned signal relerr**: {_agg('cleaned_relerr')}")
    lines.append(f"- **Cleaned signal correlation**: {_agg('cleaned_corr')}")
    lines.append("")

    # Function-correspondence table
    lines.append("## MATLAB ↔ Python function correspondence")
    lines.append("")
    lines.append(
        "Use this table to navigate from a MATLAB symbol in "
        "`refs/asr/repos/AASR/` to the equivalent Python symbol in "
        "`mne_denoise/asr/`."
    )
    lines.append("")
    lines.append(
        "| MATLAB (refs/asr/repos/AASR/) | Python (mne_denoise/asr/) | Notes |"
    )
    lines.append("|---|---|---|")
    lines.append(
        "| `ASR.m::subspace(data)` | `calibrate_asr(...) + _build_adaptive_learner` | "
        "Robust covariance via `block_geometric_median`, eigen-decomp, "
        "threshold matrix from generalized-Gaussian fit, then init Hebbian net. |"
    )
    lines.append(
        "| `ASR.m::update(data)` | `AdaptiveASR.partial_fit(data)` (variant='psp') | "
        "Streams new chunk through the Hebbian similarity matcher; target = `state.M`. |"
    )
    lines.append(
        "| `ASR_PSW.m::update(data)` | `AdaptiveASR.partial_fit(data)` (variant='psw') | "
        "Same shape as PSP but with anti-Hebbian update; target = identity. |"
    )
    lines.append(
        "| `ASR.m::reconstruct(data)` | `AdaptiveASR.transform(data)` | "
        "Runs `_process_adaptive_chunk` with the current state. |"
    )
    lines.append(
        "| `FSM.m::fit_next(x)` | `_AdaptiveSimilarityMatcher.fit_next(x)` | "
        "Hebbian forward-backward network update (Tsai eq. (4)-(6)). |"
    )
    lines.append(
        "| `FSM_PSW.m::fit_next(x)` | same Python class, target=identity | "
        "Anti-Hebbian variant: covariance update subtracts identity instead of running M. |"
    )
    lines.append(
        "| `test_asr_process.m` | `_process_adaptive_chunk` | "
        "Streaming reconstruction kernel: 0.5 s window, 0.25 s lookahead, raised-cosine blending. |"
    )
    lines.append(
        "| `test_eeg_dist_revi.m` | `fit_rms_distribution` | "
        "Generalized Gaussian fit by KL-divergence grid search; defaults match Tsai's: "
        "MinCleanFraction=0.25, MaxDropoutFraction=0.1, FitQuantiles=[0.022, 0.6]. |"
    )
    lines.append(
        "| `block_geometric_median.m` | block geometric median inside `calibrate_asr` | "
        "Weiszfeld with blocksize=10 + tol=1e-5. |"
    )
    lines.append(
        "| `datafiltering2.m` (`eegfiltfft(1, 50)`) | `_design_aasr_filter` (Yule-Walker 8th IIR) | "
        "Magnitude template `[3, 0.75, 0.33, 0.33, 1, 1, 3, 3]` at `[0, 2, 3, 13, 16, 40, fmax, fnyq]` Hz. |"
    )
    lines.append("")

    # Variant-mapping table for the demo
    lines.append("## AASR_demo.ipynb cell ↔ Python call")
    lines.append("")
    lines.append("| Demo cell | Paper variant | Python call |")
    lines.append("|---|---|---|")
    lines.append(
        "| Cell 2 | **PSP-ASR** (offline) | "
        "`AdaptiveASR(variant='psp', cutoff=20).fit_transform(data)` |"
    )
    lines.append(
        "| Cell 3 | **Init-ASR** (calibrate first 20 s only) | "
        "`AdaptiveASR(variant='psp', cutoff=20).fit(data[:, :int(20*sfreq)]).transform(data)` |"
    )
    lines.append(
        "| Cell 4 | **MW-ASR** (sliding-window subspace, no Hebbian carry-over) | "
        "`AdaptiveASR(variant='mw', mw_window_length=20.0, cutoff=20).fit_transform(data)` "
        "*(MW variant added in this sprint's Stage 1)* |"
    )
    lines.append(
        "| Cell 5 | **PSW-ASR** (sliding-window, anti-Hebbian update) | "
        "`AdaptiveASR(variant='psw', cutoff=20).fit(w0).partial_fit(w1)..."
        ".transform(data)` |"
    )
    lines.append("")

    # Artefacts
    lines.append("## Artefacts")
    lines.append("")
    lines.append(f"- JSON record: `{json_path.name}`")
    lines.append(f"- Per-fixture cleaned-relerr histogram: `{png_path.name}`")
    md_path.write_text("\n".join(lines), encoding="utf-8")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixtures-dir",
        type=Path,
        default=ROOT / "tests" / "parity" / "matlab_reference",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports" / "paper_validation" / "aasr",
    )
    parser.add_argument(
        "--pattern",
        default="aasr_case_reference_*.mat",
        help="Glob for the reference fixtures to analyse",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    fixtures = sorted(args.fixtures_dir.glob(args.pattern))
    if not fixtures:
        raise SystemExit(
            f"No fixtures matching {args.pattern} under {args.fixtures_dir}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for fx in fixtures:
        print(f"[aasr-mapping] processing {fx.name}")
        try:
            records.append(_process_fixture(fx))
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            records.append(
                {
                    "case_id": fx.stem.replace("aasr_case_reference_", ""),
                    "fixture": str(fx.relative_to(ROOT)),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    json_path = args.output_dir / "algorithm_mapping.json"
    md_path = args.output_dir / "algorithm_mapping.md"
    png_path = args.output_dir / "fig_aasr_cleaned_relerr_histogram.png"

    json_path.write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "n_fixtures": len(records),
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    _plot_cleaned_relerr_histogram([r for r in records if "M_relerr" in r], png_path)
    _write_markdown(records, json_path, png_path, md_path)

    ok = sum(1 for r in records if "cleaned_relerr" in r and r["cleaned_relerr"] < 1e-5)
    bad = len(records) - ok
    print(f"\nWrote {md_path}")
    print(f"      {json_path}")
    print(f"      {png_path}")
    print(f"Fixtures with cleaned_relerr < 1e-5: {ok}/{len(records)}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
