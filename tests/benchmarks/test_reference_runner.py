"""Regression tests for the locked reference-quality benchmark runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_runner():
    path = Path(__file__).resolve().parents[2] / "scripts" / "run_icanclean_reference_ground_truth.py"
    spec = importlib.util.spec_from_file_location("reference_ground_truth_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_band_power_does_not_require_numpy_2_trapezoid(monkeypatch):
    """Fir's NumPy 1.26 must support the frozen reference simulation."""
    monkeypatch.delattr(np, "trapezoid", raising=False)
    runner = _load_runner()
    times = np.arange(0.0, 4.0, 1.0 / 256.0)
    data = np.vstack(
        [
            np.sin(2.0 * np.pi * 10.0 * times),
            np.sin(2.0 * np.pi * 20.0 * times),
        ]
    )

    power = runner._band_power(data, sfreq=256.0)

    assert np.isfinite(power)
    assert power > 0.0
