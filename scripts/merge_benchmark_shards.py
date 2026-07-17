#!/usr/bin/env python
"""Audit and merge stable benchmark shards without copying attempt directories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards_root", type=Path)
    parser.add_argument("--expected-shards", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    root = args.shards_root.resolve()
    shard_dirs = [root / f"shard-{index:05d}" for index in range(args.expected_shards)]
    missing = [str(path) for path in shard_dirs if not path.is_dir()]
    if missing:
        raise RuntimeError(f"missing shard directories: {missing[:10]}")

    rows: list[dict[str, str]] = []
    shard_records = []
    attempts: set[tuple[str, str]] = set()
    allowed_terminal = {
        "completed",
        "failed",
        "skipped_outside_intended_regime",
    }
    for index, shard in enumerate(shard_dirs):
        raw = shard / "raw_metrics.tsv"
        if not raw.is_file():
            raise RuntimeError(f"missing shard metrics: {raw}")
        with raw.open(encoding="utf-8", newline="") as stream:
            shard_rows = list(csv.DictReader(stream, delimiter="\t"))
        rows_by_attempt = {}
        for row in shard_rows:
            key = (str(row.get("unit_id")), str(row.get("method")))
            if key in rows_by_attempt:
                raise RuntimeError(f"shard {index}: duplicate metrics row: {key}")
            rows_by_attempt[key] = row
        terminals = sorted(shard.rglob("terminal_status.json"))
        if len(terminals) != len(shard_rows):
            raise RuntimeError(
                f"shard {index}: terminal/row mismatch "
                f"({len(terminals)} != {len(shard_rows)})"
            )
        for path in terminals:
            terminal = json.loads(path.read_text(encoding="utf-8"))
            if terminal.get("status") not in allowed_terminal:
                raise RuntimeError(
                    f"shard {index}: non-acceptable terminal state in {path}: "
                    f"{terminal.get('status')}"
                )
            key = (str(terminal.get("unit_id")), str(terminal.get("method")))
            row = rows_by_attempt.get(key)
            if row is None:
                raise RuntimeError(
                    f"shard {index}: terminal has no matching metrics row: {key}"
                )
            terminal_status = str(terminal.get("status"))
            row_status = str(row.get("status"))
            if row_status == "unavailable_dependency":
                raise RuntimeError(
                    f"shard {index}: unavailable dependency in metrics row: {key}"
                )
            if terminal_status == "failed" and row_status != "failed":
                raise RuntimeError(
                    f"shard {index}: failed terminal disagrees with metrics row "
                    f"for {key}: {row_status}"
                )
            if terminal_status != "failed" and row_status == "failed":
                raise RuntimeError(
                    f"shard {index}: successful terminal disagrees with failed metrics "
                    f"row for {key}"
                )
            if key in attempts:
                raise RuntimeError(f"duplicate attempt across shards: {key}")
            attempts.add(key)
        rows.extend(shard_rows)
        shard_records.append(
            {
                "index": index,
                "rows": len(shard_rows),
                "raw_metrics_sha256": _sha256(raw),
                "terminal_set_sha256": hashlib.sha256(
                    "".join(_sha256(path) for path in terminals).encode()
                ).hexdigest(),
            }
        )

    fields = sorted({key for row in rows for key in row})
    output = (args.output or (root / "raw_metrics.tsv")).resolve()
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "shards_root": str(root),
        "expected_shards": args.expected_shards,
        "attempts": len(rows),
        "unique_unit_method_attempts": len(attempts),
        "merged_raw_metrics": str(output),
        "merged_raw_metrics_sha256": _sha256(output),
        "shards": shard_records,
    }
    manifest_path = root / "shard_merge_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
