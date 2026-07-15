"""Tests for the manuscript-facing software and evidence inventory."""

from __future__ import annotations

import importlib
import pathlib

import yaml

INVENTORY = (
    pathlib.Path(__file__).resolve().parents[2]
    / "configs"
    / "software"
    / "method_inventory.yaml"
)


def _resolve(path: str):
    module_name, attribute = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attribute)


def test_inventory_has_required_status_and_evidence_fields():
    inventory = yaml.safe_load(INVENTORY.read_text(encoding="utf-8"))
    method_ids = {method["id"] for method in inventory["methods"]}
    assert {
        "dss",
        "iterative_dss",
        "zapline",
        "zapline_plus",
        "asr_standard_and_riemannian",
        "adaptive_asr",
        "juggler_asr",
        "guided_asr",
        "icanclean",
        "sound",
        "ssp_sir",
        "continuous_dss",
    } <= method_ids
    assert inventory["integration_commit"].startswith("40322136")
    for method in inventory["methods"]:
        assert method["id"]
        assert method["status"] in inventory["status_definitions"]
        assert method["evidence_tier"] in inventory["evidence_tiers"]
        assert method["benchmark_state"]


def test_all_native_inventory_paths_are_importable():
    inventory = yaml.safe_load(INVENTORY.read_text(encoding="utf-8"))
    paths = [method["import_path"] for method in inventory["methods"]]
    for group in inventory["operators"].values():
        paths.extend(method["import_path"] for method in group)
    for path in paths:
        assert _resolve(path) is not None, path
