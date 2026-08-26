"""Tests for the dataset-manifest builder."""

from __future__ import annotations

import importlib.util
import pathlib

SCRIPT = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "benchmarks" / "build_dataset_manifest.py"
SPEC = importlib.util.spec_from_file_location("build_dataset_manifest", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_manifest_inventory_and_hash_modes(tmp_path):
    (tmp_path / "sub-01").mkdir()
    (tmp_path / "sub-01" / "recording.bin").write_bytes(b"123")
    (tmp_path / "dataset_description.json").write_text('{"Name": "demo"}')
    metadata = MODULE.build_manifest(
        tmp_path,
        dataset_id="demo",
        version="1",
        doi=None,
        license_="CC0",
        hash_mode="metadata",
    )
    by_name = {item["path"]: item for item in metadata["files"]}
    assert metadata["subjects"] == ["sub-01"]
    assert "sha256" in by_name["dataset_description.json"]
    assert "sha256" not in by_name["sub-01/recording.bin"]

    all_hashed = MODULE.build_manifest(
        tmp_path,
        dataset_id="demo",
        version="1",
        doi=None,
        license_="CC0",
        hash_mode="all",
    )
    assert all("sha256" in item for item in all_hashed["files"])


def test_manifest_include_patterns_lock_only_selected_files(tmp_path):
    (tmp_path / "NMM10_Clean_1.set").write_bytes(b"development")
    (tmp_path / "NMM10_Clean_2.set").write_bytes(b"locked header")
    (tmp_path / "NMM10_Clean_2.fdt").write_bytes(b"locked samples")
    manifest = MODULE.build_manifest(
        tmp_path,
        dataset_id="ds004784-repeat2",
        version="1.0.4",
        doi="10.18112/openneuro.ds004784.v1.0.4",
        license_="CC0",
        hash_mode="all",
        include=("NMM10_*_2.set", "NMM10_*_2.fdt"),
    )
    assert manifest["include_patterns"] == ["NMM10_*_2.set", "NMM10_*_2.fdt"]
    assert [item["path"] for item in manifest["files"]] == [
        "NMM10_Clean_2.fdt",
        "NMM10_Clean_2.set",
    ]
    assert all("sha256" in item for item in manifest["files"])
    repeated = MODULE.build_manifest(
        tmp_path,
        dataset_id="ds004784-repeat2",
        version="1.0.4",
        doi="10.18112/openneuro.ds004784.v1.0.4",
        license_="CC0",
        hash_mode="all",
        include=("NMM10_*_2.set", "NMM10_*_2.fdt"),
    )
    assert repeated["content_sha256"] == manifest["content_sha256"]
