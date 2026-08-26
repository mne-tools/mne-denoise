"""Tests for immutable benchmark provenance records."""

from __future__ import annotations

import json

import pytest

from mne_denoise.benchmarks.provenance import (
    AttemptRecorder,
    RunRecord,
    inventory_results,
    protocol_fingerprint,
    sha256_value,
)


def _record() -> RunRecord:
    return RunRecord(
        run_id="arm_unit_method_001",
        arm="arm",
        method="method",
        unit_id="unit",
        status="created",
        started_at_utc="2026-07-14T00:00:00Z",
    )


def test_hash_is_order_independent_for_mappings():
    assert sha256_value({"a": 1, "b": 2}) == sha256_value({"b": 2, "a": 1})


def test_protocol_fingerprint_changes_with_inputs():
    a = protocol_fingerprint(
        git_commit="a", config_hash="b", dataset_hash="c", environment_hash_="d"
    )
    b = protocol_fingerprint(
        git_commit="a", config_hash="changed", dataset_hash="c", environment_hash_="d"
    )
    assert a != b


def test_attempt_recorder_writes_completion(tmp_path):
    with AttemptRecorder(tmp_path / "attempt", _record()):
        pass
    payload = json.loads((tmp_path / "attempt" / "terminal_status.json").read_text())
    assert payload["status"] == "completed"
    assert payload["runtime_seconds"] >= 0
    assert payload["peak_memory_mb"] is None or payload["peak_memory_mb"] > 0


def test_attempt_recorder_writes_failure(tmp_path):
    with pytest.raises(RuntimeError, match="boom"):
        with AttemptRecorder(tmp_path / "attempt", _record()):
            raise RuntimeError("boom")
    payload = json.loads((tmp_path / "attempt" / "terminal_status.json").read_text())
    assert payload["status"] == "failed"
    assert payload["error_type"] == "RuntimeError"
    assert "boom" in payload["error_message"]


def test_inventory_counts_terminal_statuses(tmp_path):
    for idx in range(2):
        with AttemptRecorder(tmp_path / f"run-{idx}", _record()):
            pass
    report = inventory_results(tmp_path)
    assert report["counts"] == {"completed": 2}
    assert len(report["records"]) == 2
