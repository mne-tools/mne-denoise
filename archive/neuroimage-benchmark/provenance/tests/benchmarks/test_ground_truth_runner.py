"""Regression tests for the frozen ground-truth benchmark runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_runner():
    path = Path(__file__).resolve().parents[2] / "scripts" / "run_ground_truth_arm.py"
    spec = importlib.util.spec_from_file_location("ground_truth_arm_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_amari_requires_square_identifiable_unmixing():
    runner = _load_runner()
    mixing = np.zeros((8, 8))

    assert runner._supports_amari(np.zeros((8, 8)), mixing)
    assert not runner._supports_amari(np.zeros((6, 8)), mixing)
    assert not runner._supports_amari(np.zeros((8, 7)), mixing)
