"""Tests for audited factorial-shard merging."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "merge_benchmark_shards.py"
SPEC = importlib.util.spec_from_file_location("merge_benchmark_shards", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write_shard(root: Path, *, row_status: str, terminal_status: str) -> Path:
    shard = root / "shard-00000"
    attempt = shard / "unit-1" / "asr"
    attempt.mkdir(parents=True)
    with (shard / "raw_metrics.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("unit_id", "method", "status"), delimiter="\t"
        )
        writer.writeheader()
        writer.writerow({"unit_id": "unit-1", "method": "asr", "status": row_status})
    (attempt / "terminal_status.json").write_text(
        json.dumps(
            {"unit_id": "unit-1", "method": "asr", "status": terminal_status}
        ),
        encoding="utf-8",
    )
    return shard


def test_merge_keeps_method_failure_in_denominator(tmp_path):
    shards = tmp_path / "shards"
    _write_shard(shards, row_status="failed", terminal_status="failed")
    output = tmp_path / "raw_metrics.tsv"

    assert MODULE.main(
        [str(shards), "--expected-shards", "1", "--output", str(output)]
    ) == 0

    rows = list(csv.DictReader(output.open(encoding="utf-8"), delimiter="\t"))
    assert rows == [{"method": "asr", "status": "failed", "unit_id": "unit-1"}]


@pytest.mark.parametrize(
    ("row_status", "terminal_status"),
    [("success", "failed"), ("failed", "completed")],
)
def test_merge_rejects_terminal_metrics_disagreement(
    tmp_path, row_status, terminal_status
):
    shards = tmp_path / "shards"
    _write_shard(
        shards, row_status=row_status, terminal_status=terminal_status
    )

    with pytest.raises(RuntimeError, match="disagrees"):
        MODULE.main([str(shards), "--expected-shards", "1"])


def test_merge_rejects_unavailable_dependency(tmp_path):
    shards = tmp_path / "shards"
    _write_shard(
        shards, row_status="unavailable_dependency", terminal_status="failed"
    )

    with pytest.raises(RuntimeError, match="unavailable dependency"):
        MODULE.main([str(shards), "--expected-shards", "1"])
