"""Generate deterministic adaptive-ASR MATLAB parity fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import savemat

OUTPUT_DIR = Path(__file__).resolve().parent
SFREQ = 250.0
N_CHANNELS = 8
DURATION = 24.0
CHUNK_SECONDS = 8.0
CUTOFF = 20.0


def _make_case(
    seed: int, *, burst_scale: float, drift_scale: float
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_times = int(SFREQ * DURATION)
    chunk_len = int(SFREQ * CHUNK_SECONDS)
    t = np.arange(n_times, dtype=np.float64) / SFREQ

    brain = np.zeros((N_CHANNELS, n_times), dtype=np.float64)
    for ch_idx in range(N_CHANNELS):
        phase = rng.uniform(0.0, 2.0 * np.pi)
        brain[ch_idx] = (
            0.6 * np.sin(2.0 * np.pi * 10.0 * t + phase)
            + 0.15 * np.sin(2.0 * np.pi * 6.5 * t + 0.25 * phase)
            + 0.05 * rng.standard_normal(n_times)
        )

    data = brain.copy()
    spatial = rng.standard_normal((N_CHANNELS, 3))
    spatial /= np.linalg.norm(spatial, axis=0, keepdims=True)
    bursts = (
        (2.5, 3.3, 0),
        (9.0, 10.0, 1),
        (17.0, 18.1, 2),
        (20.4, 21.2, 0),
    )
    for onset, stop, source_idx in bursts:
        start = int(onset * SFREQ)
        stop_samp = int(stop * SFREQ)
        source = rng.standard_normal((1, stop_samp - start)) * burst_scale
        data[:, start:stop_samp] += spatial[:, [source_idx]] @ source

    blink = np.exp(-0.5 * ((t - 6.0) / 0.25) ** 2) + np.exp(
        -0.5 * ((t - 14.5) / 0.35) ** 2
    )
    data[:2] += drift_scale * blink[np.newaxis, :]

    return {
        "data": data.astype(np.float64),
        "update_chunk_1": data[:, :chunk_len].astype(np.float64),
        "update_chunk_2": data[:, chunk_len : 2 * chunk_len].astype(np.float64),
        "update_chunk_3": data[:, 2 * chunk_len : 3 * chunk_len].astype(np.float64),
        "sfreq": np.array(SFREQ, dtype=np.float64),
        "cutoff": np.array(CUTOFF, dtype=np.float64),
        "window_length": np.array(0.5, dtype=np.float64),
        "clean_window_length": np.array(1.0, dtype=np.float64),
        "clean_window_overlap": np.array(0.66, dtype=np.float64),
        "clean_max_bad_channels": np.array(0.2, dtype=np.float64),
        "clean_tolerances": np.array([-3.5, 5.0], dtype=np.float64),
        "blocksize": np.array(10, dtype=np.float64),
        "max_dims": np.array(0.66, dtype=np.float64),
        "learning_rate": np.array(0.2, dtype=np.float64),
        "tau_psp": np.array(0.8, dtype=np.float64),
        "tau_psw": np.array(1e-5, dtype=np.float64),
    }


def main() -> None:
    cases = {
        "moderate": _make_case(20260506, burst_scale=7.0, drift_scale=1.2),
        "strong": _make_case(20260507, burst_scale=10.0, drift_scale=2.0),
    }
    for case_name, payload in cases.items():
        savemat(
            OUTPUT_DIR / f"aasr_case_input_{case_name}.mat",
            {"case_name": case_name, **payload},
            do_compression=True,
        )


if __name__ == "__main__":
    main()
