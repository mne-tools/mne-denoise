"""Benchmark gate for GuidedASR: neural preservation vs artifact removal.

Builds a synthetic substrate with a KNOWN neural target (a high-variance 10 Hz
oscillation with a fixed spatial pattern, present as transient bursts) and a
KNOWN artifact (high-amplitude broadband bursts with a different pattern). ASR
is calibrated on a quiet baseline that contains neither, so the variance-only
keep/reject rule tends to over-clean the neural bursts.

Three methods are compared at a matched cutoff:

* ``standard``  -- ``ASR(method="standard")``
* ``rasr``      -- ``ASR(method="riemannian_windowed")``
* ``guided``    -- ``GuidedASR(reconstruction="soft", preserve_biases=[10 Hz],
  artifact_biases=[EMG band])``

Metrics on the neural-event region:

* **neural preservation** -- Pearson correlation between the cleaned and the
  ground-truth projections onto the neural spatial pattern (higher = better).
* **artifact removal** -- percent of injected-artifact variance removed in the
  artifact region (higher = better).

Decision rule (the gate): GuidedASR must show **higher neural preservation at
equal-or-better artifact removal** than plain rASR. Otherwise the bias-scoring
layer is not justified and only the ``reconstruction="soft"`` option should be
contributed to the base ``ASR``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mne_denoise.asr import ASR, GuidedASR  # noqa: E402
from mne_denoise.dss.denoisers import BandpassBias, PeakFilterBias  # noqa: E402


def _build_substrate(seed: int, sfreq: float = 250.0):
    rng = np.random.default_rng(seed)
    n_ch = 16
    dur = 60.0
    n = int(dur * sfreq)
    t = np.arange(n) / sfreq

    # 1/f-ish background present everywhere (the calibration baseline).
    background = np.zeros((n_ch, n))
    for f in (3.0, 6.0, 11.0, 17.0, 23.0, 31.0):
        amp = 1.0 / f
        for c in range(n_ch):
            background[c] += amp * np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28))
    background += 0.15 * rng.standard_normal((n_ch, n))

    # Neural target: 10 Hz, fixed spatial pattern, transient high-variance bursts.
    p_neural = rng.standard_normal(n_ch)
    p_neural /= np.linalg.norm(p_neural)
    neural_ts = np.zeros(n)
    neural_idx = np.zeros(n, dtype=bool)
    for start in np.linspace(0.55 * n, 0.95 * n, 5).astype(int):
        seg = slice(start, min(start + int(1.0 * sfreq), n))
        neural_ts[seg] += 4.0 * np.sin(2 * np.pi * 10.0 * t[seg])
        neural_idx[seg] = True
    neural = np.outer(p_neural, neural_ts)

    clean = background + neural  # ground truth (no artifact)

    # Artifact: EMG-like broadband bursts, different spatial pattern.
    p_art = rng.standard_normal(n_ch)
    p_art /= np.linalg.norm(p_art)
    artifact = np.zeros((n_ch, n))
    artifact_idx = np.zeros(n, dtype=bool)
    for start in np.linspace(0.30 * n, 0.50 * n, 4).astype(int):
        seg = slice(start, min(start + int(0.4 * sfreq), n))
        burst = rng.standard_normal(seg.stop - seg.start)
        artifact[:, seg] += 7.0 * np.outer(p_art, burst)
        artifact_idx[seg] = True

    contaminated = clean + artifact
    calib = background[:, : int(15.0 * sfreq)]  # quiet baseline, no events
    return {
        "clean": clean,
        "contaminated": contaminated,
        "calib": calib,
        "p_neural": p_neural,
        "p_art": p_art,
        "neural_idx": neural_idx,
        "artifact_idx": artifact_idx,
        "sfreq": sfreq,
    }


def _metrics(cleaned, sub) -> dict:
    clean = sub["clean"]
    contaminated = sub["contaminated"]
    p_neural = sub["p_neural"]
    nidx = sub["neural_idx"]
    aidx = sub["artifact_idx"]

    # Neural preservation: corr of neural-pattern projection over the event region
    proj_clean = p_neural @ clean
    proj_cleaned = p_neural @ np.asarray(cleaned)
    a = proj_cleaned[nidx]
    b = proj_clean[nidx]
    neural_corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0

    # Artifact removal: fraction of injected-artifact variance removed
    resid_before = (contaminated - clean)[:, aidx]
    resid_after = (np.asarray(cleaned) - clean)[:, aidx]
    denom = float(np.sum(resid_before**2))
    artifact_removal = (
        100.0 * (1.0 - float(np.sum(resid_after**2)) / denom) if denom > 0 else 0.0
    )
    return {
        "neural_preservation_corr": round(neural_corr, 4),
        "artifact_removal_pct": round(artifact_removal, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "guided_asr" / "benchmark.json",
    )
    parser.add_argument("--tolerance", type=float, default=2.0)
    args = parser.parse_args()

    sub = _build_substrate(args.seed)
    sf = sub["sfreq"]
    X = sub["contaminated"]
    calib = sub["calib"]
    common = {
        "sfreq": sf,
        "cutoff": args.cutoff,
        "calibration": "manual",
        "picks": None,
        "verbose": False,
    }

    results = {}

    std = ASR(method="standard", **common)
    std.fit(calib)
    results["standard"] = _metrics(std.transform(X), sub)

    rasr = ASR(method="riemannian_windowed", **common)
    rasr.fit(calib)
    results["rasr"] = _metrics(rasr.transform(X), sub)

    guided = GuidedASR(
        reconstruction="soft",
        experimental=True,
        preserve_biases=[PeakFilterBias(10.0, sf)],
        artifact_biases=[BandpassBias((30.0, 80.0), sf)],
        **common,
    )
    guided.fit(X, calibration=calib)  # bias subspaces from X, thresholds from calib
    results["guided"] = _metrics(guided.transform(X), sub)

    g, r = results["guided"], results["rasr"]
    gate_pass = bool(
        g["neural_preservation_corr"] > r["neural_preservation_corr"]
        and g["artifact_removal_pct"] >= r["artifact_removal_pct"] - args.tolerance
    )
    verdict = {
        "gate": "PASS" if gate_pass else "FAIL",
        "rule": (
            "GuidedASR neural preservation > rASR at artifact removal within "
            f"{args.tolerance} pp"
        ),
        "results": results,
        "cutoff": args.cutoff,
        "seed": args.seed,
    }

    print(f"\n{'method':<10} {'neural_preservation':>20} {'artifact_removal_%':>20}")
    print("-" * 52)
    for name in ("standard", "rasr", "guided"):
        m = results[name]
        print(
            f"{name:<10} {m['neural_preservation_corr']:>20.4f} "
            f"{m['artifact_removal_pct']:>20.2f}"
        )
    print(f"\nGATE: {verdict['gate']}  ({verdict['rule']})")
    if not gate_pass:
        print(
            "  -> Bias-scoring layer not justified on this substrate; consider "
            "contributing only reconstruction='soft' to the base ASR."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print(f"\nWrote {args.output}")
    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

