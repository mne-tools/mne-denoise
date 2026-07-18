"""Tests for the Kaya motor-imagery inventory."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "inventory_kaya2018_mi.py"
    spec = importlib.util.spec_from_file_location("inventory_kaya2018_mi", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_event_onsets_count_runs_once() -> None:
    """Sustained marker runs must count as one event each."""
    module = _load_module()
    marker = np.array([0, 1, 1, 0, 1, 2, 2, 0, 2, 0])
    np.testing.assert_array_equal(module._event_onsets(marker, 1), [1, 4])
    np.testing.assert_array_equal(module._event_onsets(marker, 2), [5, 8])


def test_parse_recording_name() -> None:
    """The public archive naming fields must be preserved."""
    module = _load_module()
    parsed = module._parse_name(
        Path("HaLT-SubjectM-161124-6St-LRHandLegTongue.mat")
    )
    assert parsed == {
        "paradigm": "HaLT",
        "subject": "M",
        "session": "161124-6St-LRHandLegTongue",
    }
