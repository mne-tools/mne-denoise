"""Config submission-validator tests."""

import pathlib

import pytest

from mne_denoise.benchmarks import config as cfgmod

CONFIG_DIR = pathlib.Path(__file__).resolve().parents[2] / "configs" / "benchmarks"
ARMS = sorted(CONFIG_DIR.glob("*.yaml"))


def test_config_dir_has_benchmark_arms():
    assert len(ARMS) >= 10


@pytest.mark.parametrize("path", ARMS, ids=lambda p: p.stem)
def test_configs_parse(path):
    cfg = cfgmod.load_arm_config(path)
    assert cfg.get("arm") and cfg.get("runner")


@pytest.mark.parametrize("path", ARMS, ids=lambda p: p.stem)
def test_all_configs_are_submission_ready(path):
    cfg = cfgmod.load_arm_config(path)
    assert cfgmod.validate_for_submission(cfg) == []


def test_validator_flags_each_placeholder_kind():
    bad = {
        "arm": "x", "runner": "y",
        "a": None, "b": "pending", "c": "pending_feasibility",
        "d": "TBD", "e": "auto",
        "nested": {"deep": "pending"}, "lst": ["ok", "TBD"],
    }
    paths = {i.path for i in cfgmod.validate_for_submission(bad)}
    assert {"a", "b", "c", "d", "e", "nested.deep", "lst[1]"} <= paths


def test_fully_specified_config_is_ready():
    good = {
        "arm": "demo", "runner": "line_noise", "status": "frozen",
        "dataset": {"id": "ds", "version": "1.0.0", "subjects": ["sub-01"]},
        "methods_under_test": ["demo"],
        "metrics": {"primary_target": "R_f0", "primary_preservation": "sideband"},
        "equivalence_margin": {"value": 0.1, "direction": "non_inferiority"},
        "unit_of_inference": "subject",
        "storage": {"always_save": ["metrics"]},
    }
    assert cfgmod.is_submission_ready(good)
    assert cfgmod.validate_for_submission(good) == []


def test_missing_top_level_keys_flagged():
    issues = cfgmod.validate_for_submission({"foo": 1})
    paths = {i.path for i in issues}
    assert "arm" in paths and "runner" in paths
