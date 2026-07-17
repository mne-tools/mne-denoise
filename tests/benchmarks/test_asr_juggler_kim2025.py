"""Tests for the procedural Kim et al. Juggler simulation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import yaml

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_asr_juggler_kim2025.py"
SPEC = importlib.util.spec_from_file_location("run_asr_juggler_kim2025", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
CONFIG = Path(__file__).resolve().parents[2] / "configs/benchmarks/asr_juggler_kim2025.yaml"


def _config():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_kim2025_simulation_has_reported_layout():
    simulation = MODULE.simulate_kim2025(
        _config(), motion_1_interval_s=0.1, seed=11, smoke=True
    )

    assert simulation["clean"].shape == (100, 4096)
    assert simulation["mixture"].shape == simulation["clean"].shape
    assert np.any(simulation["motion_1"])
    assert np.any(simulation["blink"])
    assert np.any(simulation["motion_2"])
    assert np.array_equal(
        simulation["any"],
        simulation["motion_1"] | simulation["blink"] | simulation["motion_2"],
    )
    assert not np.allclose(simulation["mixture"], simulation["clean"])


def test_kim2025_estimators_use_reported_parameters():
    cfg = _config()
    standard = MODULE._method("asr_original_5pct", sfreq=256.0, cutoff=5, cfg=cfg)
    dbscan = MODULE._method("juggler_dbscan", sfreq=256.0, cutoff=5, cfg=cfg)
    gev = MODULE._method("juggler_gev", sfreq=256.0, cutoff=5, cfg=cfg)

    assert standard.calibration == "auto"
    assert standard.ref_max_bad_channels == 0.05
    assert standard.blocksize == 10
    assert standard.filter_kind == "asr"
    assert standard.window_criterion == 0.0
    assert dbscan.strategy == "dbscan"
    assert dbscan.dbscan_top_k == 5
    assert dbscan.dbscan_eps == "paper"
    assert dbscan.dbscan_min_samples == "paper"
    assert gev.strategy == "gev"
