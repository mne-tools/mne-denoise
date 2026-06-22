"""Dataset registry, resolver, and validation tests (no network/data needed)."""

import json

import pytest

from mne_denoise.benchmarks import datasets as D


def test_registry_has_six_datasets():
    expected = {"ds003620", "ds004505", "ds000117", "erp_core_n170",
                "eegdenoisenet", "ds004784"}
    assert expected <= set(D.REGISTRY)


def test_get_spec_unknown_raises():
    with pytest.raises(KeyError):
        D.get_spec("nope")


def test_candidate_order_datasets_root_first(tmp_path):
    env = {"DATASETS_ROOT": str(tmp_path / "shared"), "SCRATCH": str(tmp_path / "scr")}
    cands = D.candidate_roots("ds003620", env=env)
    assert cands[0] == tmp_path / "shared" / "openneuro" / "ds003620"
    assert cands[1].as_posix().endswith("/project/rrg-kjerbi/datasets/openneuro/ds003620") \
        or "rrg-kjerbi" in cands[1].as_posix()
    assert cands[-1].as_posix().startswith((tmp_path / "scr").as_posix())


def test_resolve_prefers_existing_datasets_root(tmp_path):
    rel = "openneuro/ds003620"
    shared = tmp_path / "shared" / rel
    shared.mkdir(parents=True)
    env = {"DATASETS_ROOT": str(tmp_path / "shared"), "SCRATCH": str(tmp_path / "scr")}
    assert D.resolve_dataset_root("ds003620", env=env) == shared


def test_resolve_falls_back_to_scratch_target(tmp_path):
    env = {"DATASETS_ROOT": str(tmp_path / "shared"), "SCRATCH": str(tmp_path / "scr")}
    got = D.resolve_dataset_root("ds003620", env=env)   # nothing exists
    assert (tmp_path / "scr").as_posix() in got.as_posix()
    assert got.name == D.get_spec("ds003620").version  # <version> leaf


def _make_openneuro_tree(root, n_subjects):
    root.mkdir(parents=True)
    (root / "dataset_description.json").write_text(json.dumps({"Name": "x"}))
    for i in range(1, n_subjects + 1):
        (root / f"sub-{i:02d}").mkdir()


def test_validate_openneuro_ok(tmp_path):
    spec = D.get_spec("ds000117")
    root = tmp_path / "ds000117"
    _make_openneuro_tree(root, spec.expected_subjects)
    assert D.validate_dataset(root, "ds000117") == []


def test_validate_flags_wrong_subject_count_and_missing_description(tmp_path):
    root = tmp_path / "ds000117"
    root.mkdir()
    (root / "sub-01").mkdir()
    issues = D.validate_dataset(root, "ds000117")
    assert any("subject count" in s for s in issues)
    assert any("dataset_description" in s for s in issues)


def test_validate_missing_root(tmp_path):
    issues = D.validate_dataset(tmp_path / "absent", "ds003620")
    assert issues and "does not exist" in issues[0]
