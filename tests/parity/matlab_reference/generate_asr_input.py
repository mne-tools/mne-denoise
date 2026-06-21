"""Generate deterministic ASR parity input data for MATLAB reference runs.

Run from the repository root:

    python tests/parity/matlab_reference/generate_asr_input.py

This writes:

- ``asr_input_fixture.mat`` for the legacy single-case parity scripts
- ``asr_case_input_<name>.mat`` files for the expanded parity matrix
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import savemat

CASE_MATRIX: tuple[dict[str, object], ...] = (
    {
        "name": "manual_strong_cutoff20",
        "sfreq": 250.0,
        "n_channels": 8,
        "n_times": 3000,
        "n_calibration": 1500,
        "seed": 42,
        "artifact_scale": 7.0,
        "cutoff": 20.0,
        "blocksize": 10.0,
        "window_length": 0.5,
        "window_overlap": 0.66,
        "max_dropout_fraction": 0.1,
        "min_clean_fraction": 0.25,
        "max_dims": 0.66,
        "maxmem": 64.0,
        "use_auto_calibration": False,
        "ref_max_bad_channels": 0.075,
        "ref_tolerances": (-3.5, 5.5),
        "ref_window_length": 1.0,
    },
    {
        "name": "manual_moderate_cutoff5",
        "sfreq": 250.0,
        "n_channels": 8,
        "n_times": 3000,
        "n_calibration": 1500,
        "seed": 101,
        "artifact_scale": 1.5,
        "cutoff": 5.0,
        "blocksize": 10.0,
        "window_length": 0.5,
        "window_overlap": 0.66,
        "max_dropout_fraction": 0.1,
        "min_clean_fraction": 0.25,
        "max_dims": 0.66,
        "maxmem": 64.0,
        "use_auto_calibration": False,
        "ref_max_bad_channels": 0.075,
        "ref_tolerances": (-3.5, 5.5),
        "ref_window_length": 1.0,
    },
    {
        "name": "manual_moderate_cutoff20",
        "sfreq": 250.0,
        "n_channels": 8,
        "n_times": 3000,
        "n_calibration": 1500,
        "seed": 101,
        "artifact_scale": 1.5,
        "cutoff": 20.0,
        "blocksize": 10.0,
        "window_length": 0.5,
        "window_overlap": 0.66,
        "max_dropout_fraction": 0.1,
        "min_clean_fraction": 0.25,
        "max_dims": 0.66,
        "maxmem": 64.0,
        "use_auto_calibration": False,
        "ref_max_bad_channels": 0.075,
        "ref_tolerances": (-3.5, 5.5),
        "ref_window_length": 1.0,
    },
    {
        "name": "manual_strong_sfreq200_maxdims05",
        "sfreq": 200.0,
        "n_channels": 8,
        "n_times": 2400,
        "n_calibration": 1200,
        "seed": 73,
        "artifact_scale": 6.0,
        "cutoff": 20.0,
        "blocksize": 10.0,
        "window_length": 0.5,
        "window_overlap": 0.66,
        "max_dropout_fraction": 0.1,
        "min_clean_fraction": 0.25,
        "max_dims": 0.5,
        "maxmem": 64.0,
        "use_auto_calibration": False,
        "ref_max_bad_channels": 0.075,
        "ref_tolerances": (-3.5, 5.5),
        "ref_window_length": 1.0,
    },
    {
        "name": "auto_moderate_cutoff5",
        "sfreq": 250.0,
        "n_channels": 8,
        "n_times": 4000,
        "n_calibration": 2000,
        "seed": 211,
        "artifact_scale": 1.5,
        "cutoff": 5.0,
        "blocksize": 10.0,
        "window_length": 0.5,
        "window_overlap": 0.66,
        "max_dropout_fraction": 0.1,
        "min_clean_fraction": 0.25,
        "max_dims": 0.66,
        "maxmem": 64.0,
        "use_auto_calibration": True,
        "ref_max_bad_channels": 0.075,
        "ref_tolerances": (-3.5, 5.5),
        "ref_window_length": 1.0,
    },
    {
        "name": "auto_moderate_cutoff20",
        "sfreq": 250.0,
        "n_channels": 8,
        "n_times": 4000,
        "n_calibration": 2000,
        "seed": 211,
        "artifact_scale": 1.5,
        "cutoff": 20.0,
        "blocksize": 10.0,
        "window_length": 0.5,
        "window_overlap": 0.66,
        "max_dropout_fraction": 0.1,
        "min_clean_fraction": 0.25,
        "max_dims": 0.66,
        "maxmem": 64.0,
        "use_auto_calibration": True,
        "ref_max_bad_channels": 0.075,
        "ref_tolerances": (-3.5, 5.5),
        "ref_window_length": 1.0,
    },
    {
        "name": "auto_strong_sfreq200_maxdims05",
        "sfreq": 200.0,
        "n_channels": 8,
        "n_times": 3200,
        "n_calibration": 1600,
        "seed": 307,
        "artifact_scale": 4.0,
        "cutoff": 20.0,
        "blocksize": 10.0,
        "window_length": 0.5,
        "window_overlap": 0.66,
        "max_dropout_fraction": 0.1,
        "min_clean_fraction": 0.25,
        "max_dims": 0.5,
        "maxmem": 64.0,
        "use_auto_calibration": True,
        "ref_max_bad_channels": 0.075,
        "ref_tolerances": (-3.5, 5.5),
        "ref_window_length": 1.0,
    },
)


def generate_asr_fixture(
    *,
    sfreq: float,
    n_channels: int,
    n_times: int,
    n_calibration: int,
    seed: int,
    artifact_scale: float,
    cutoff: float,
    blocksize: float,
    window_length: float,
    window_overlap: float,
    max_dropout_fraction: float,
    min_clean_fraction: float,
    max_dims: float,
    maxmem: float,
    use_auto_calibration: bool,
    ref_max_bad_channels: float,
    ref_tolerances: tuple[float, float],
    ref_window_length: float,
) -> dict[str, np.ndarray | float]:
    """Return deterministic calibration and processing data for one case."""
    rng = np.random.default_rng(seed)
    times = np.arange(n_times) / sfreq
    cal_times = np.arange(n_calibration) / sfreq

    brain = np.zeros((n_channels, n_times), dtype=np.float64)
    calibration = np.zeros((n_channels, n_calibration), dtype=np.float64)
    for ch_idx in range(n_channels):
        phase = rng.uniform(0, 2 * np.pi)
        brain[ch_idx] = (
            0.4 * np.sin(2 * np.pi * 10 * times + phase)
            + 0.2 * np.sin(2 * np.pi * 6 * times + 0.4 * phase)
            + 0.04 * rng.standard_normal(n_times)
        )
        calibration[ch_idx] = (
            0.4 * np.sin(2 * np.pi * 10 * cal_times + phase)
            + 0.2 * np.sin(2 * np.pi * 6 * cal_times + 0.4 * phase)
            + 0.04 * rng.standard_normal(n_calibration)
        )

    data = brain.copy()
    spatial = rng.standard_normal((n_channels, 2))
    spatial /= np.linalg.norm(spatial, axis=0, keepdims=True)
    for onset, stop in (
        (0.30 * times[-1], 0.30 * times[-1] + 0.55),
        (0.60 * times[-1], 0.60 * times[-1] + 0.55),
    ):
        start = int(round(onset * sfreq))
        end = min(n_times, int(round(stop * sfreq)))
        source = artifact_scale * rng.standard_normal((2, end - start))
        data[:, start:end] += spatial @ source

    data -= data.mean(axis=1, keepdims=True)
    calibration -= calibration.mean(axis=1, keepdims=True)

    return {
        "data": data,
        "calibration": calibration,
        "sfreq": float(sfreq),
        "cutoff": float(cutoff),
        "blocksize": float(blocksize),
        "window_length": float(window_length),
        "window_overlap": float(window_overlap),
        "max_dropout_fraction": float(max_dropout_fraction),
        "min_clean_fraction": float(min_clean_fraction),
        "max_dims": float(max_dims),
        "maxmem": float(maxmem),
        "filter_b": np.array([1.0], dtype=np.float64),
        "filter_a": np.array([1.0], dtype=np.float64),
        "use_auto_calibration": float(use_auto_calibration),
        "ref_max_bad_channels": float(ref_max_bad_channels),
        "ref_tolerances": np.asarray(ref_tolerances, dtype=np.float64),
        "ref_window_length": float(ref_window_length),
    }


def iter_case_payloads() -> tuple[tuple[str, dict[str, np.ndarray | float]], ...]:
    """Return all case names and serialized payloads."""
    payloads: list[tuple[str, dict[str, np.ndarray | float]]] = []
    for case in CASE_MATRIX:
        name = str(case["name"])
        payload = generate_asr_fixture(
            sfreq=float(case["sfreq"]),
            n_channels=int(case["n_channels"]),
            n_times=int(case["n_times"]),
            n_calibration=int(case["n_calibration"]),
            seed=int(case["seed"]),
            artifact_scale=float(case["artifact_scale"]),
            cutoff=float(case["cutoff"]),
            blocksize=float(case["blocksize"]),
            window_length=float(case["window_length"]),
            window_overlap=float(case["window_overlap"]),
            max_dropout_fraction=float(case["max_dropout_fraction"]),
            min_clean_fraction=float(case["min_clean_fraction"]),
            max_dims=float(case["max_dims"]),
            maxmem=float(case["maxmem"]),
            use_auto_calibration=bool(case["use_auto_calibration"]),
            ref_max_bad_channels=float(case["ref_max_bad_channels"]),
            ref_tolerances=tuple(case["ref_tolerances"]),  # type: ignore[arg-type]
            ref_window_length=float(case["ref_window_length"]),
        )
        payloads.append((name, payload))
    return tuple(payloads)


def main() -> None:
    """Write the legacy single-case fixture plus the expanded case matrix."""
    out_dir = Path(__file__).resolve().parent
    payloads = iter_case_payloads()

    legacy_path = out_dir / "asr_input_fixture.mat"
    savemat(legacy_path, payloads[0][1])
    print(f"Wrote {legacy_path}")

    for name, payload in payloads:
        case_path = out_dir / f"asr_case_input_{name}.mat"
        savemat(case_path, payload)
        print(f"Wrote {case_path}")


if __name__ == "__main__":
    main()
