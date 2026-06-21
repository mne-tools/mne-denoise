"""Compare local ASR implementations against external Python reference repos.

This script is intentionally a local benchmark harness, not a pass/fail parity
test. Standard ASR is compared against the local ``python-meegkit`` checkout.
The experimental Riemannian backend is compared qualitatively against the local
``timeflux_rasr`` checkout when its optional dependencies are importable.

Expected local layout
---------------------
The benchmark looks for cloned references under ``refs/asr/repos``:

- ``python-meegkit``
- ``timeflux_rasr``

The timeflux benchmark also accepts a local ``.cache/pydeps`` directory for
isolated validation-only dependencies such as ``pyriemann``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from mne_denoise.asr import ASR

REFS = ROOT / "refs" / "asr" / "repos"


def _prepend_if_exists(path: Path) -> None:
    if path.exists():
        sys.path.insert(0, str(path))


def _make_synthetic_eeg(
    *,
    sfreq: float,
    duration: float,
    n_channels: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_times = int(round(sfreq * duration))
    times = np.arange(n_times) / sfreq
    data = np.zeros((n_channels, n_times), dtype=np.float64)
    for ch_idx in range(n_channels):
        phase = rng.uniform(0, 2 * np.pi)
        data[ch_idx] = (
            0.5 * np.sin(2 * np.pi * 10 * times + phase)
            + 0.2 * np.sin(2 * np.pi * 6 * times + 0.5 * phase)
            + 0.05 * rng.standard_normal(n_times)
        )
    return data


def _burst_corrupted_copy(
    brain: np.ndarray,
    *,
    sfreq: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    data = brain.copy()
    burst_mask = np.zeros(brain.shape[1], dtype=bool)
    spatial = rng.standard_normal((brain.shape[0], 2))
    spatial /= np.linalg.norm(spatial, axis=0, keepdims=True)
    for onset, stop in ((4.0, 4.8), (8.0, 8.6)):
        start = int(round(onset * sfreq))
        end = int(round(stop * sfreq))
        burst_mask[start:end] = True
        data[:, start:end] += spatial @ (8.0 * rng.standard_normal((2, end - start)))
    return data, burst_mask


def _summarize(
    clean: np.ndarray, brain: np.ndarray, burst_mask: np.ndarray
) -> dict[str, float]:
    return {
        "burst_mse": float(np.mean((clean[:, burst_mask] - brain[:, burst_mask]) ** 2)),
        "clean_mse": float(
            np.mean((clean[:, ~burst_mask] - brain[:, ~burst_mask]) ** 2)
        ),
        "overall_corr": float(np.corrcoef(clean.ravel(), brain.ravel())[0, 1]),
    }


def _run_meegkit_reference() -> None:
    repo = REFS / "python-meegkit"
    if not repo.exists():
        print("[skip] python-meegkit repo not found under refs/asr/repos")
        return
    _prepend_if_exists(repo)
    from meegkit.asr import ASR as MeegkitASR

    sfreq = 250.0
    brain = _make_synthetic_eeg(sfreq=sfreq, duration=12.0, n_channels=8, seed=42)
    data, burst_mask = _burst_corrupted_copy(brain, sfreq=sfreq, seed=99)
    calibration = brain[:, : int(5 * sfreq)]

    ours = ASR(
        sfreq=sfreq,
        cutoff=20.0,
        calibration="manual",
        filter_kind="asr",
        verbose=False,
    )
    ours.fit(calibration)
    ours_clean = ours.transform(data)

    meegkit = MeegkitASR(
        sfreq=sfreq,
        cutoff=20.0,
        blocksize=10,
        win_len=0.5,
        win_overlap=0.66,
        method="euclid",
    )
    meegkit.fit(calibration)
    meegkit_clean = meegkit.transform(data)

    ours_metrics = _summarize(ours_clean, brain, burst_mask)
    meegkit_metrics = _summarize(meegkit_clean, brain, burst_mask)
    relerr = float(
        np.linalg.norm(ours_clean - meegkit_clean) / np.linalg.norm(meegkit_clean)
    )
    corr = float(np.corrcoef(ours_clean.ravel(), meegkit_clean.ravel())[0, 1])

    print("[standard] mne-denoise vs python-meegkit")
    print(f"  ours:     {ours_metrics}")
    print(f"  meegkit:  {meegkit_metrics}")
    print(f"  pairwise corr={corr:.4f} relerr={relerr:.4f}")


def _run_timeflux_reference() -> None:
    repo = REFS / "timeflux_rasr"
    if not repo.exists():
        print("[skip] timeflux_rasr repo not found under refs/asr/repos")
        return

    _prepend_if_exists(ROOT / ".cache" / "pydeps")
    try:
        import pyriemann.utils.covariance as covariance_utils
    except Exception as exc:  # pragma: no cover - local-only helper
        print(f"[skip] pyriemann unavailable for timeflux_rasr benchmark: {exc}")
        return

    # ``timeflux_rasr`` imports a private helper removed from newer pyriemann.
    if not hasattr(covariance_utils, "_check_est"):
        covariance_utils._check_est = lambda estimator: estimator

    _prepend_if_exists(repo)
    try:
        from timeflux_rasr.estimators.blending import Blending
        from timeflux_rasr.estimators.rasr import RASR
        from timeflux_rasr.helpers.utils import epoch
    except Exception as exc:  # pragma: no cover - local-only helper
        print(f"[skip] timeflux_rasr import failed: {exc}")
        return

    sfreq = 250.0
    brain_long = _make_synthetic_eeg(sfreq=sfreq, duration=80.0, n_channels=8, seed=123)
    brain = brain_long[:, : int(12.0 * sfreq)].copy()
    data, burst_mask = _burst_corrupted_copy(brain, sfreq=sfreq, seed=321)

    ours = ASR(
        sfreq=sfreq,
        cutoff=20.0,
        calibration="manual",
        method="riemannian",
        experimental=True,
        filter_kind="none",
        verbose=False,
    )
    ours.fit(brain_long)
    ours_clean = ours.transform(data)

    window_samples = int(round(0.5 * sfreq))
    interval = 32
    calibration_epochs = epoch(
        brain_long.T, size=window_samples, interval=interval, axis=0
    )
    test_epochs = epoch(data.T, size=window_samples, interval=interval, axis=0)

    reference = RASR(rejection_cutoff=20.0, max_dimension=0.66)
    reference.fit(calibration_epochs)
    reference_epochs = reference.transform(test_epochs)
    blender = Blending(window_overlap=window_samples - interval, merge=True)
    blender.fit(reference_epochs)
    reference_clean = blender.transform(reference_epochs).T

    n_valid = min(reference_clean.shape[1], ours_clean.shape[1], brain.shape[1])
    reference_clean = reference_clean[:, :n_valid]
    ours_clean = ours_clean[:, :n_valid]
    brain = brain[:, :n_valid]
    burst_mask = burst_mask[:n_valid]

    ours_metrics = _summarize(ours_clean, brain, burst_mask)
    reference_metrics = _summarize(reference_clean, brain, burst_mask)
    relerr = float(
        np.linalg.norm(ours_clean - reference_clean) / np.linalg.norm(reference_clean)
    )
    corr = float(np.corrcoef(ours_clean.ravel(), reference_clean.ravel())[0, 1])

    print("[riemannian] mne-denoise vs timeflux_rasr")
    print(
        "  Qualitative only: timeflux_rasr is epoched/trial-based, not a parity oracle."
    )
    print(f"  ours:      {ours_metrics}")
    print(f"  timeflux:  {reference_metrics}")
    print(f"  pairwise corr={corr:.4f} relerr={relerr:.4f}")


def main() -> None:
    _run_meegkit_reference()
    _run_timeflux_reference()


if __name__ == "__main__":
    main()
