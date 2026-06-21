"""Export a riemannian_windowed fixture for the MATLAB cross-check.

The robustness sprint added ``ASR(method="riemannian_windowed")``. Its
*processing* step is byte-identical to standard ASR (proven in
``test_riemannian_windowed_processing_identical_to_standard``), and standard
ASR's processing is already MATLAB-parity-tested. This script produces a
fixture that lets a MATLAB run *directly* confirm that transitive chain:

1. Calibrate a riemannian_windowed state on a deterministic synthetic input.
2. Process the input through ``process_asr(method="riemannian_windowed")``.
3. Export: the input data, the calibration state (M, T, B, A), all the
   processing parameters, and the Python-cleaned output, to
   ``riemannian_windowed_xcheck_input.mat``.

The companion ``generate_riemannian_windowed_reference.m`` injects the same
(M, T, B, A) into clean_rawdata's ``asr_process`` and saves the MATLAB-cleaned
output. ``test_riemannian_windowed_parity.py`` then compares the two.

Run from the repo root:
    python tests/parity/matlab_reference/generate_riemannian_windowed_input.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.io import savemat

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mne_denoise.asr.core import calibrate_asr, process_asr

HERE = Path(__file__).resolve().parent
OUT = HERE / "riemannian_windowed_xcheck_input.mat"

SFREQ = 250.0
CUTOFF = 20.0
WINDOW_LENGTH = 0.5
WINDOW_OVERLAP = 0.66
MAX_DIMS = 0.66
BLOCKSIZE = 10


def _make_fixture() -> tuple[np.ndarray, np.ndarray]:
    """Deterministic 8-ch fixture: oscillatory brain + a few bursts."""
    rng = np.random.default_rng(20260528)
    n_ch, n_t = 8, 6000  # 24 s @ 250 Hz
    t = np.arange(n_t) / SFREQ
    data = np.zeros((n_ch, n_t))
    for c in range(n_ch):
        phase = rng.uniform(0, 2 * np.pi)
        data[c] = 0.6 * np.sin(2 * np.pi * 10.0 * t + phase) + 0.15 * np.sin(
            2 * np.pi * 6.5 * t + phase * 0.8
        )
    data += 0.05 * rng.standard_normal(data.shape)
    # Calibration slice = first half (manual calibration; no clean-window pick)
    calibration = data[:, : n_t // 2].copy()
    # Inject bursts into the FULL stream (after the calibration slice was taken)
    for s in np.linspace(int(0.55 * n_t), n_t - 300, 5).astype(int):
        spatial = rng.standard_normal(n_ch)
        spatial /= max(np.linalg.norm(spatial), 1e-12)
        temporal = rng.standard_normal(200)
        data[:, s : s + 200] += 10.0 * np.outer(spatial, temporal)
    return data, calibration


def main() -> int:
    data, calibration = _make_fixture()

    state, _ = calibrate_asr(
        calibration,
        SFREQ,
        cutoff=CUTOFF,
        blocksize=BLOCKSIZE,
        window_length=WINDOW_LENGTH,
        window_overlap=WINDOW_OVERLAP,
        calibration="manual",
        method="riemannian_windowed",
        filter_kind="asr",
    )
    py_cleaned, _ = process_asr(
        data,
        SFREQ,
        state,
        window_length=WINDOW_LENGTH,
        window_overlap=WINDOW_OVERLAP,
        max_dims=MAX_DIMS,
        method="riemannian_windowed",
    )

    payload = {
        "data": data,
        "calibration": calibration,
        "sfreq": float(SFREQ),
        "cutoff": float(CUTOFF),
        "window_length": float(WINDOW_LENGTH),
        "window_overlap": float(WINDOW_OVERLAP),
        "max_dims": float(MAX_DIMS),
        "blocksize": float(BLOCKSIZE),
        # Calibration state to inject into MATLAB asr_process:
        "M": np.asarray(state.M, dtype=np.float64),
        "T": np.asarray(state.T, dtype=np.float64),
        "B": np.asarray(state.filter_b, dtype=np.float64).ravel(),
        "A": np.asarray(state.filter_a, dtype=np.float64).ravel(),
        # Python-cleaned output (the thing MATLAB must reproduce):
        "py_cleaned": np.asarray(py_cleaned, dtype=np.float64),
    }
    savemat(OUT, payload)
    print(f"Wrote {OUT}")
    print(f"  data {data.shape}, calibration {calibration.shape}")
    print(f"  M {state.M.shape}, T {state.T.shape}, B {state.filter_b.shape}")
    print(f"  py_cleaned {py_cleaned.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
