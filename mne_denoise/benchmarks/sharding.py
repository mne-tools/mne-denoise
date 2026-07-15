"""Stable sharding helpers for benchmark arrays."""

from __future__ import annotations

import argparse
import hashlib


def add_shard_arguments(parser: argparse.ArgumentParser) -> None:
    """Add zero-based stable-shard options to a benchmark parser."""
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Total number of stable shards (default: one unsharded run).",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard index; use SLURM_ARRAY_TASK_ID with a 0-based array.",
    )


def validate_shard(index: int, count: int) -> tuple[int, int]:
    """Validate and normalize a shard specification."""
    index, count = int(index), int(count)
    if count < 1:
        raise ValueError("shard_count must be at least one")
    if index < 0 or index >= count:
        raise ValueError(f"shard_index must satisfy 0 <= index < {count}")
    return index, count


def unit_in_shard(unit_id: str, index: int, count: int) -> bool:
    """Return whether a unit belongs to a stable content-hashed shard."""
    index, count = validate_shard(index, count)
    digest = hashlib.sha256(str(unit_id).encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], byteorder="big", signed=False) % count
    return bucket == index


def args_select_unit(args, unit_id: str) -> bool:
    """Return whether ``unit_id`` belongs to the shard declared on ``args``."""
    return unit_in_shard(unit_id, args.shard_index, args.shard_count)
