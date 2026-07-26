"""Execution-only sharding for the low-density arm.

The low-density muscle arm is dominated by EEMD-CCA, whose per-channel cost made the
whole six-subject unit exceed its wall clock. Sharding by subject, channel density and
method class keeps the frozen configuration byte-identical (so the protocol hash is
unchanged) and only redistributes the same work across scheduler tasks. These tests pin
the property that makes that legitimate: the shards partition the unsharded work exactly.
"""
from __future__ import annotations

import pathlib
import sys

import pytest
import yaml

_REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "scripts"))

import run_lowdensity_arm as L  # noqa: E402

CONFIG = _REPO / "configs" / "benchmarks" / "lowdensity_muscle.yaml"


@pytest.fixture()
def cfg():
    with open(CONFIG) as handle:
        return yaml.safe_load(handle)


def _unsharded_units(cfg):
    """(density, method) pairs the unsharded runner would attempt."""
    return [
        (n_ch, method)
        for n_ch in L.shard_grid(cfg, env={})
        for method in L._cfg_methods(L._gate_heavy(cfg, n_ch))
    ]


def test_defaults_are_unsharded(cfg):
    assert L.shard_grid(cfg, env={}) == [64, 32, 16, 8, 4]
    sharded, shard = L.shard_class(cfg, env={})
    assert shard is None
    assert L._cfg_methods(sharded) == L._cfg_methods(cfg)


def test_shards_partition_the_frozen_protocol(cfg):
    """Every unsharded (density, method) unit is covered exactly once by the shard plan."""
    covered = []
    for n_ch in L.shard_grid(cfg, env={}):
        for cls in ("light", "heavy"):
            env = {"LD_CHANNEL_GRID": str(n_ch), "LD_METHOD_CLASS": cls}
            grid = L.shard_grid(cfg, env=env)
            sharded, shard = L.shard_class(cfg, env=env)
            assert grid == [n_ch]
            try:
                methods = L._cfg_methods(L._density_cfg(sharded, n_ch, shard))
            except ValueError:
                continue  # class is gated out at this density; the shard is never scheduled
            covered.extend((n_ch, method) for method in methods)

    expected = _unsharded_units(cfg)
    assert sorted(covered) == sorted(expected)
    assert len(covered) == len(set(covered))


def test_light_and_heavy_are_complementary(cfg):
    light, _ = L.shard_class(cfg, env={"LD_METHOD_CLASS": "light"})
    heavy, _ = L.shard_class(cfg, env={"LD_METHOD_CLASS": "heavy"})
    light_m, heavy_m = L._cfg_methods(light), L._cfg_methods(heavy)
    assert set(light_m).isdisjoint(heavy_m)
    assert sorted(light_m + heavy_m) == sorted(L._cfg_methods(cfg))
    assert heavy_m == ["eemd_cca"]


def test_light_shard_preserves_configuration_order(cfg):
    """Method order is unchanged, so a sharded run reproduces the unsharded sequence."""
    light, _ = L.shard_class(cfg, env={"LD_METHOD_CLASS": "light"})
    full = L._cfg_methods(cfg)
    assert L._cfg_methods(light) == [m for m in full if m not in L.HEAVY]


def test_heavy_shard_above_the_gate_fails_closed(cfg):
    """Requesting EEMD above its density gate is a scheduling error, not an empty success."""
    sharded, shard = L.shard_class(cfg, env={"LD_METHOD_CLASS": "heavy"})
    for n_ch in (64, 32):
        with pytest.raises(ValueError, match="selects no method"):
            L._density_cfg(sharded, n_ch, shard)
    for n_ch in (16, 8, 4):
        assert L._cfg_methods(L._density_cfg(sharded, n_ch, shard)) == ["eemd_cca"]


@pytest.mark.parametrize("value", ["128", "64,17", "0"])
def test_grid_outside_the_frozen_protocol_is_rejected(cfg, value):
    with pytest.raises(ValueError, match="not a subset"):
        L.shard_grid(cfg, env={"LD_CHANNEL_GRID": value})


def test_unknown_method_class_is_rejected(cfg):
    with pytest.raises(ValueError, match="must be 'heavy' or 'light'"):
        L.shard_class(cfg, env={"LD_METHOD_CLASS": "fast"})


def test_expected_attempt_count_per_subject(cfg):
    """Five densities of seven light methods plus three eligible EEMD densities."""
    assert len(_unsharded_units(cfg)) == 5 * 7 + 3 * 1
