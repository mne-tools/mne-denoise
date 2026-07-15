import argparse

import pytest

from mne_denoise.benchmarks.sharding import (
    add_shard_arguments,
    unit_in_shard,
    validate_shard,
)


def test_each_unit_is_assigned_once_and_stably():
    units = [f"unit-{index:04d}" for index in range(500)]
    first = [[unit for unit in units if unit_in_shard(unit, shard, 17)] for shard in range(17)]
    second = [[unit for unit in units if unit_in_shard(unit, shard, 17)] for shard in range(17)]
    assert first == second
    assigned = [unit for shard in first for unit in shard]
    assert sorted(assigned) == units
    assert len(assigned) == len(set(assigned))


@pytest.mark.parametrize("index,count", [(-1, 2), (2, 2), (0, 0)])
def test_invalid_shard_rejected(index, count):
    with pytest.raises(ValueError):
        validate_shard(index, count)


def test_cli_defaults_are_unsharded():
    parser = argparse.ArgumentParser()
    add_shard_arguments(parser)
    args = parser.parse_args([])
    assert args.shard_index == 0
    assert args.shard_count == 1
    assert unit_in_shard("anything", args.shard_index, args.shard_count)
