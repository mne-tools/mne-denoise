"""Tests for causal replay deployment metrics."""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np

SCRIPT = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "run_streaming_replay.py"
SPEC = importlib.util.spec_from_file_location("run_streaming_replay", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_boundary_discontinuity_uses_adjacent_boundary_samples():
    smooth = np.arange(24.0)[None]
    assert np.isclose(MODULE._boundary_discontinuity(smooth, 6), 1.0)
    jumped = smooth.copy()
    jumped[:, 6:] += 20
    assert MODULE._boundary_discontinuity(jumped, 6) > 10


def test_settling_time_requires_consecutive_blocks():
    reference = np.array([1.0, 0.0])
    history = [
        np.array([0.0, 1.0]),
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([1.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([1.0, 0.0]),
    ]
    settling = MODULE._settling_time(
        history,
        reference,
        start_block=1,
        block_size=64,
        sfreq=256,
        threshold_degrees=10,
        consecutive_blocks=3,
    )
    assert np.isclose(settling, 0.5)
