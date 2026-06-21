"""Optional ASR cross-checks against local Python reference repositories."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from mne_denoise.asr import ASR

ROOT = Path(__file__).resolve().parents[2]
REFS = ROOT / "refs" / "asr" / "repos"


def _prepend_if_exists(path: Path) -> None:
    if path.exists():
        sys.path.insert(0, str(path))


def _load_meegkit_asr():
    repo = REFS / "python-meegkit"
    if not repo.exists():
        pytest.skip("python-meegkit reference repo not found under refs/asr/repos")
    _prepend_if_exists(repo)
    try:
        from meegkit.asr import ASR as MeegkitASR
    except ModuleNotFoundError as exc:
        pytest.skip(f"python-meegkit reference dependency not available: {exc.name}")

    return MeegkitASR


def _make_synthetic_burst_data():
    sfreq = 250.0
    duration = 12.0
    n_channels = 8
    n_times = int(sfreq * duration)
    times = np.arange(n_times) / sfreq
    rng = np.random.default_rng(42)

    brain = np.zeros((n_channels, n_times), dtype=np.float64)
    for ch_idx in range(n_channels):
        phase = rng.uniform(0, 2 * np.pi)
        brain[ch_idx] = (
            0.5 * np.sin(2 * np.pi * 10 * times + phase)
            + 0.2 * np.sin(2 * np.pi * 6 * times + 0.5 * phase)
            + 0.05 * rng.standard_normal(n_times)
        )

    data = brain.copy()
    burst_mask = np.zeros(n_times, dtype=bool)
    spatial = rng.standard_normal((n_channels, 2))
    spatial /= np.linalg.norm(spatial, axis=0, keepdims=True)
    for onset, stop in ((4.0, 4.8), (8.0, 8.6)):
        start = int(onset * sfreq)
        end = int(stop * sfreq)
        burst_mask[start:end] = True
        data[:, start:end] += spatial @ (8.0 * rng.standard_normal((2, end - start)))

    return data, brain, burst_mask, sfreq


def test_standard_asr_tracks_python_meegkit_reference_behavior():
    """Standard ASR stays close to python-meegkit on a shared synthetic fixture."""
    MeegkitASR = _load_meegkit_asr()
    data, brain, burst_mask, sfreq = _make_synthetic_burst_data()
    calibration = brain[:, : int(5 * sfreq)]

    ours = ASR(
        sfreq=sfreq,
        cutoff=20.0,
        calibration="manual",
        filter_kind="asr",
        verbose=False,
    )
    ours.fit(calibration)
    our_clean = ours.transform(data)

    reference = MeegkitASR(
        sfreq=sfreq,
        cutoff=20.0,
        blocksize=10,
        win_len=0.5,
        win_overlap=0.66,
        method="euclid",
    )
    reference.fit(calibration)
    reference_clean = reference.transform(data)

    before_burst_mse = np.mean((data[:, burst_mask] - brain[:, burst_mask]) ** 2)
    our_burst_mse = np.mean((our_clean[:, burst_mask] - brain[:, burst_mask]) ** 2)
    our_clean_mse = np.mean((our_clean[:, ~burst_mask] - brain[:, ~burst_mask]) ** 2)
    ref_burst_mse = np.mean(
        (reference_clean[:, burst_mask] - brain[:, burst_mask]) ** 2
    )
    ref_clean_mse = np.mean(
        (reference_clean[:, ~burst_mask] - brain[:, ~burst_mask]) ** 2
    )

    corr = np.corrcoef(our_clean.ravel(), reference_clean.ravel())[0, 1]
    relerr = np.linalg.norm(our_clean - reference_clean) / np.linalg.norm(
        reference_clean
    )

    assert our_clean.shape == data.shape
    assert reference_clean.shape == data.shape
    assert np.isfinite(our_clean).all()
    assert np.isfinite(reference_clean).all()
    assert our_burst_mse < before_burst_mse * 0.01
    assert ref_burst_mse < before_burst_mse * 0.01
    assert our_burst_mse <= ref_burst_mse * 1.5
    assert our_clean_mse <= ref_clean_mse * 1.25
    assert corr > 0.9
    assert relerr < 0.4
