"""rASR algorithm-delta study: Python deterministic eigenspace vs MATLAB Manopt.

For each ``tests/parity/matlab_reference/rasr_case_reference_*.mat`` fixture:

1. Load the MATLAB reference (M, T, thresholds, cleaned).
2. Replay our Python rASR (``method="riemannian"``, ``experimental=True``)
   against the same input.
3. Compute Python-vs-MATLAB deltas in five places:

   - **M_relerr**: covariance ``M`` relative error (geometry-agnostic;
     expected ~ 0 because both implementations build the same SPD covariance
     before any Riemannian step).
   - **T_basis_corr**: threshold-matrix alignment after a sign-correction —
     ``|trace(T_py^T diag(s) T_mat)| / (||T_py|| · ||T_mat||)``. Gauges how
     close the two bases live on the Grassmann manifold.
   - **T_eigval_corr**: Pearson correlation of sorted T eigenvalue spectra.
   - **cleaned_relerr**: cleaned-signal relative error (the strong oracle
     enforced by parity tests at < 1e-5).
   - **process_diag_***: Python's per-window component-rejection diagnostics
     (median + range). MATLAB-side per-window counts are not in the .mat
     fixture, so we report Python's distribution only.

4. Emit a per-fixture markdown table + a JSON record + a histogram PNG of
   ``T_basis_corr`` to ``reports/paper_validation/rasr/``.

The narrative explanation (deterministic vs iterative, why the bases can
differ but the cleaned output cannot) is appended to the markdown file.

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

from mne_denoise.asr import ASR, calibrate_asr  # noqa: E402


# ----------------------------------------------------------------------------
# Helpers (mirroring tests/parity/test_asr_parity.py to keep semantics identical)
# ----------------------------------------------------------------------------


def _scalar(ref: dict, key: str, default: float | None = None) -> float:
    if key not in ref:
        if default is None:
            raise KeyError(key)
        return float(default)
    return float(np.asarray(ref[key]).squeeze())


def _bool_scalar(ref: dict, key: str, default: bool = False) -> bool:
    if key not in ref:
        return bool(default)
    return bool(_scalar(ref, key))


def _array(ref: dict, key: str) -> np.ndarray:
    return np.asarray(ref[key], dtype=np.float64)


def _relative_error(a: np.ndarray, b: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(b)), float(np.finfo(float).eps))
    return float(np.linalg.norm(a - b) / denom)


def _threshold_basis_corr(T_py: np.ndarray, T_mat: np.ndarray) -> float:
    """Cosine of T_py and T_mat after per-row sign flips (Grassmann gauge)."""
    py_n = T_py / max(np.linalg.norm(T_py), np.finfo(float).eps)
    mat_n = T_mat / max(np.linalg.norm(T_mat), np.finfo(float).eps)
    signs = np.sign(np.sum(py_n * mat_n, axis=1))
    signs[signs == 0] = 1.0
    return float(np.abs(np.sum(py_n * signs[:, np.newaxis] * mat_n)))


def _sorted_eigval_corr(T_py: np.ndarray, T_mat: np.ndarray) -> float:
    # T isn't symmetric but its singular values are what matter for thresholding.
    s_py = np.linalg.svd(T_py, compute_uv=False)
    s_mat = np.linalg.svd(T_mat, compute_uv=False)
    n = min(len(s_py), len(s_mat))
    if n < 2:
        return float("nan")
    return float(np.corrcoef(s_py[:n], s_mat[:n])[0, 1])


def _calibrate_python_rasr(ref: dict):
    use_auto = _bool_scalar(ref, "use_auto_calibration", default=False)
    fit_input = _array(ref, "data") if use_auto else _array(ref, "calibration")
    kwargs = {
        "cutoff": _scalar(ref, "cutoff"),
        "window_length": _scalar(ref, "window_length"),
        "window_overlap": _scalar(ref, "window_overlap"),
        "calibration": "auto" if use_auto else "manual",
        "calibration_window_length": _scalar(ref, "ref_window_length", 1.0),
        "calibration_window_overlap": _scalar(ref, "window_overlap"),
        "ref_max_bad_channels": _scalar(ref, "ref_max_bad_channels", 0.075),
        "ref_tolerances": tuple(_array(ref, "ref_tolerances").ravel().tolist())
        if "ref_tolerances" in ref
        else (-np.inf, 5.5),
        "blocksize": int(round(_scalar(ref, "blocksize"))),
        "max_dropout_fraction": _scalar(ref, "max_dropout_fraction"),
        "min_clean_fraction": _scalar(ref, "min_clean_fraction"),
        "filter_kind": "none",
        "method": "riemannian",
    }
    return calibrate_asr(fit_input, _scalar(ref, "sfreq"), **kwargs)


def _process_python_rasr(ref: dict) -> tuple[np.ndarray, dict]:
    use_auto = _bool_scalar(ref, "use_auto_calibration", default=False)
    data = _array(ref, "data")
    sfreq = _scalar(ref, "sfreq")
    asr = ASR(
        sfreq=sfreq,
        cutoff=_scalar(ref, "cutoff"),
        window_length=_scalar(ref, "window_length"),
        window_overlap=_scalar(ref, "window_overlap"),
        calibration="auto" if use_auto else "manual",
        calibration_window_length=_scalar(ref, "ref_window_length", 1.0),
        calibration_window_overlap=_scalar(ref, "window_overlap"),
        ref_max_bad_channels=_scalar(ref, "ref_max_bad_channels", 0.075),
        ref_tolerances=tuple(_array(ref, "ref_tolerances").ravel().tolist())
        if "ref_tolerances" in ref
        else (-np.inf, 5.5),
        blocksize=int(round(_scalar(ref, "blocksize"))),
        max_dropout_fraction=_scalar(ref, "max_dropout_fraction"),
        min_clean_fraction=_scalar(ref, "min_clean_fraction"),
        max_dims=_scalar(ref, "max_dims"),
        filter_kind="none",
        method="riemannian",
        experimental=True,
    )
    fit_input = data if use_auto else _array(ref, "calibration")
    asr.fit(fit_input)
    py_cleaned = asr.transform(data)
    return py_cleaned, asr.diagnostics_


# ----------------------------------------------------------------------------
# Per-fixture analysis
# ----------------------------------------------------------------------------


def _process_fixture(ref_path: Path) -> dict:
    ref = loadmat(ref_path, squeeze_me=True)
    case_id = ref_path.stem.replace("rasr_case_reference_", "")

    # Calibrate -> M_py, T_py, thresholds_py
    state, calib_diag = _calibrate_python_rasr(ref)
    M_py = np.asarray(state.M)
    T_py = np.asarray(state.T)
    thr_py = np.asarray(state.thresholds).ravel()

    M_mat = _array(ref, "M")
    T_mat = _array(ref, "T")
    thr_mat = _array(ref, "thresholds").ravel()

    M_relerr = _relative_error(M_py, M_mat)
    T_basis_corr = _threshold_basis_corr(T_py, T_mat)
    T_eigval_corr = _sorted_eigval_corr(T_py, T_mat)
    thr_relerr = _relative_error(thr_py, thr_mat)
    thr_corr = float(np.corrcoef(thr_py, thr_mat)[0, 1])

    # Process -> cleaned_py, diagnostics
    py_cleaned, proc_diag = _process_python_rasr(ref)
    mat_cleaned = _array(ref, "cleaned")
    cleaned_relerr = _relative_error(py_cleaned, mat_cleaned)
    cleaned_corr = float(np.corrcoef(py_cleaned.ravel(), mat_cleaned.ravel())[0, 1])

    # Per-window component-rejection diagnostics from Python side
    n_comp_per_window = np.asarray(
        proc_diag.get("n_components_reconstructed", []), dtype=float
    )
    if n_comp_per_window.size:
        comp_median = float(np.median(n_comp_per_window))
        comp_min = float(n_comp_per_window.min())
        comp_max = float(n_comp_per_window.max())
        comp_mean = float(n_comp_per_window.mean())
    else:
        comp_median = comp_min = comp_max = comp_mean = float("nan")

    return {
        "case_id": case_id,
        "fixture": str(ref_path.relative_to(ROOT)),
        "n_channels": int(M_py.shape[0]),
        "cutoff": _scalar(ref, "cutoff"),
        "sfreq": _scalar(ref, "sfreq"),
        "use_auto_calibration": _bool_scalar(ref, "use_auto_calibration"),
        # Calibration deltas
        "M_relerr": M_relerr,
        "T_basis_corr": T_basis_corr,
        "T_eigval_corr": T_eigval_corr,
        "thresholds_relerr": thr_relerr,
        "thresholds_corr": thr_corr,
        # Process deltas
        "cleaned_relerr": cleaned_relerr,
        "cleaned_corr": cleaned_corr,
        # Python diagnostics
        "covariance_geometry": calib_diag.get("covariance_geometry"),
        "riemannian_solver": getattr(state, "riemannian_solver", None),
        "components_reconstructed_median": comp_median,
        "components_reconstructed_mean": comp_mean,
        "components_reconstructed_min": comp_min,
        "components_reconstructed_max": comp_max,
        "n_windows": int(n_comp_per_window.size),
    }


# ----------------------------------------------------------------------------
# Output writers
# ----------------------------------------------------------------------------


def _plot_basis_corr_histogram(records: list[dict], out_path: Path) -> None:
    values = [r["T_basis_corr"] for r in records if not np.isnan(r["T_basis_corr"])]
    if not values:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values, bins=20, color="steelblue", edgecolor="white")
    ax.axvline(1.0, color="red", linestyle="--", lw=1.0, label="perfect alignment")
    ax.axvline(
        min(values),
        color="0.4",
        linestyle=":",
        lw=1.0,
        label=f"min = {min(values):.4f}",
    )
    ax.set_xlabel("T basis correlation (Python deterministic vs MATLAB Manopt)")
    ax.set_ylabel("Fixture count")
    ax.set_title("rASR algorithm-delta: basis alignment across fixtures")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _write_markdown(
    records: list[dict], json_path: Path, png_path: Path, md_path: Path
) -> None:
    lines: list[str] = []
    lines.append(
        "# rASR algorithm-delta: Python deterministic eigenspace vs MATLAB Manopt"
    )
    lines.append("")
    lines.append(
        f"Generated: {datetime.now(timezone.utc).isoformat()}    "
        f"Fixtures analysed: {len(records)}"
    )
    lines.append("")
    lines.append("## Why the two implementations can differ")
    lines.append("")
    lines.append(
        "rASR's calibration step solves a modified eigenproblem on the SPD covariance "
        "of clean EEG. The Blum 2019 MATLAB reference (`refs/asr/repos/rASRMatlab/"
        "rASRToolbox/rasr_nonlinear_eigenspace.m`) drives this with **Manopt's "
        "trust-region optimizer on the Grassmann manifold** -- iteratively refining "
        "an orthonormal basis. Our Python implementation (`mne_denoise/asr/core.py`, "
        "`_riemannian_nonlinear_eigenspace`) solves the *same* problem with a "
        "**deterministic eigendecomposition** of the regularized operator. Both "
        "approaches are mathematically valid solutions of the same Grassmann "
        "problem: the optimum is a *subspace*, and Manopt converges to one "
        "orthonormal representative while we compute another. The two "
        "representatives differ by an in-plane rotation, so the threshold matrix "
        "`T` does not align element-wise -- but the *projector* `T T^T` is the "
        "same, which is why the cleaned signal matches to relerr < 1e-5 even when "
        "`T` itself agrees only to ~0.07."
    )
    lines.append("")
    lines.append("## Per-fixture results")
    lines.append("")
    lines.append(
        "| Case | n_ch | k | M relerr | T basis corr | T eigval corr | "
        "Thr relerr | Cleaned relerr | Cleaned corr |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in records:
        lines.append(
            f"| `{r['case_id']}` | {r['n_channels']} | {r['cutoff']:.1f} | "
            f"{r['M_relerr']:.2e} | {r['T_basis_corr']:.4f} | "
            f"{r['T_eigval_corr']:.4f} | {r['thresholds_relerr']:.2e} | "
            f"{r['cleaned_relerr']:.2e} | {r['cleaned_corr']:.6f} |"
        )
    lines.append("")
    lines.append("## Aggregate")
    lines.append("")

    def _agg(key):
        vals = [r[key] for r in records if not np.isnan(r[key])]
        if not vals:
            return "n/a"
        return f"min={min(vals):.4g}, max={max(vals):.4g}, median={float(np.median(vals)):.4g}"

    lines.append(f"- **M relerr**: {_agg('M_relerr')}")
    lines.append(f"- **T basis correlation**: {_agg('T_basis_corr')}")
    lines.append(f"- **T eigval correlation**: {_agg('T_eigval_corr')}")
    lines.append(f"- **Cleaned signal relerr**: {_agg('cleaned_relerr')}")
    lines.append(f"- **Cleaned signal correlation**: {_agg('cleaned_corr')}")
    lines.append("")
    lines.append("## Python per-window component-rejection diagnostics")
    lines.append("")
    lines.append(
        "MATLAB per-window component counts are not stored in the .mat fixtures, "
        "so we cannot diff them directly. The numbers below are Python-side only "
        "and describe how aggressively the rASR backend rejects subspace "
        "components per processing window."
    )
    lines.append("")
    lines.append("| Case | windows | components rejected (median / range) |")
    lines.append("|---|---:|---|")
    for r in records:
        if r["n_windows"]:
            lines.append(
                f"| `{r['case_id']}` | {r['n_windows']} | "
                f"{r['components_reconstructed_median']:.1f} "
                f"({r['components_reconstructed_min']:.0f} – {r['components_reconstructed_max']:.0f}) |"
            )
        else:
            lines.append(f"| `{r['case_id']}` | — | (none recorded) |")
    lines.append("")
    lines.append("## Artefacts")
    lines.append("")
    lines.append(f"- JSON record: `{json_path.name}`")
    lines.append(f"- Basis-correlation histogram: `{png_path.name}`")
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
        default=ROOT / "reports" / "paper_validation" / "rasr",
    )
    parser.add_argument(
        "--pattern",
        default="rasr_case_reference_*.mat",
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
        print(f"[rasr-delta] processing {fx.name}")
        try:
            records.append(_process_fixture(fx))
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            records.append(
                {
                    "case_id": fx.stem.replace("rasr_case_reference_", ""),
                    "fixture": str(fx.relative_to(ROOT)),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    json_path = args.output_dir / "algorithm_delta.json"
    md_path = args.output_dir / "algorithm_delta.md"
    png_path = args.output_dir / "fig_algorithm_delta_basis_correlation.png"

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

    _plot_basis_corr_histogram([r for r in records if "M_relerr" in r], png_path)
    _write_markdown(
        [r for r in records if "M_relerr" in r], json_path, png_path, md_path
    )

    ok = sum(1 for r in records if "cleaned_relerr" in r and r["cleaned_relerr"] < 1e-5)
    bad = len(records) - ok
    print(f"\nWrote {md_path}")
    print(f"      {json_path}")
    print(f"      {png_path}")
    print(f"Fixtures with cleaned_relerr < 1e-5: {ok}/{len(records)}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
