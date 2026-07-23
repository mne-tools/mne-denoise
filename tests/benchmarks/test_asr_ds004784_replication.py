"""Tests for the published and locked ds004784 ASR campaigns."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from mne_denoise.benchmarks.config import assert_submission_ready
from mne_denoise.benchmarks.provenance import environment_record

REPO = pathlib.Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "benchmarks" / "asr_ds004784_replication.yaml"
SCRIPT = REPO / "scripts" / "run_asr_ds004784_replication.py"
SPEC = importlib.util.spec_from_file_location("run_asr_ds004784_replication", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _config():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_protocol_is_submission_ready():
    assert_submission_ready(_config(), source=str(CONFIG))


def test_provenance_records_eeglab_reader_version():
    assert "pymatreader" in environment_record()["packages"]


def test_exact_published_cutoff_grid_matches_released_matlab():
    values = MODULE.exact_cutoffs(_config())
    assert len(values) == 436
    assert values[:3] == [1.0, 1.05, 1.1]
    assert values[180:184] == [10.0, 10.5, 11.0, 11.5]
    assert values[-3:] == [248.0, 249.0, 250.0]


def test_phantom_protocol_uses_clean_rawdata_reference_tolerances():
    method = _config()["published_protocol"]["method"]
    assert method["reference_window_tolerances"] == ["-inf", 5.5]
    assert method["calibration_blocksize"] == "clean_rawdata"
    assert method["max_mem_mb"] == 64
    assert method["processing_mode"] == "clean_rawdata"


def test_band_power_supports_numpy_one_compatible_stack():
    time = np.arange(4096) / 512.0
    data = np.sin(2 * np.pi * 10.0 * time)[None]
    assert MODULE._band_power(data, 512.0) > 0.0


def test_synchronized_data_scores_interval_without_cropping_fit_input():
    data = np.arange(20.0).reshape(2, 10)
    synchronized = MODULE._synchronized_data(data, (2, 7))
    np.testing.assert_array_equal(synchronized, data[:, 2:7])
    assert data.shape == (2, 10)


@pytest.mark.parametrize("sync_samples", [(-1, 3), (3, 3), (3, 11)])
def test_synchronized_data_rejects_invalid_interval(sync_samples):
    with pytest.raises(ValueError, match="invalid synchronization interval"):
        MODULE._synchronized_data(np.zeros((2, 10)), sync_samples)


def test_calibration_metrics_preserve_sample_and_window_denominators():
    model = SimpleNamespace(
        calibration_info_={
            "clean_sample_mask": np.array([True, False, True, True]),
            "clean_window_mask": np.array([True, False, True]),
            "blocksize_requested": "clean_rawdata",
            "blocksize_effective": 265,
        }
    )
    metrics = MODULE._calibration_metrics(model)
    assert metrics["calibration_reference_samples"] == 3
    assert metrics["calibration_candidate_samples"] == 4
    assert metrics["calibration_reference_fraction"] == pytest.approx(0.75)
    assert metrics["calibration_reference_mask_sha256"] == hashlib.sha256(
        np.packbits(np.array([True, False, True, True]), bitorder="little").tobytes()
    ).hexdigest()
    assert metrics["calibration_clean_windows"] == 2
    assert metrics["calibration_candidate_windows"] == 3
    assert metrics["calibration_clean_window_fraction"] == pytest.approx(2 / 3)
    assert metrics["calibration_blocksize_requested"] == "clean_rawdata"
    assert metrics["calibration_blocksize_effective"] == 265


def test_calibration_metrics_prefer_juggler_reference_mask():
    model = SimpleNamespace(
        calibration_info_={
            "reference_sample_mask": np.array([True, False, False, True]),
            "clean_sample_mask": np.ones(2, dtype=bool),
        }
    )
    metrics = MODULE._calibration_metrics(model)
    assert metrics["calibration_reference_samples"] == 2
    assert metrics["calibration_candidate_samples"] == 4
    assert metrics["calibration_reference_fraction"] == pytest.approx(0.5)


def test_processing_metrics_preserve_compatibility_provenance():
    model = SimpleNamespace(processing_mode="clean_rawdata")
    metrics = MODULE._processing_metrics(
        model,
        {
            "clean_rawdata_splits": 9990,
            "stepsize_samples": 128,
            "window_length_samples": 256,
            "memory_mode": "rolling",
            "used_memory_bound": True,
        },
    )
    assert metrics == {
        "processing_mode": "clean_rawdata",
        "processing_clean_rawdata_splits": 9990,
        "processing_stepsize_samples": 128,
        "processing_window_length_samples": 256,
        "processing_memory_mode": "rolling",
        "processing_used_memory_bound": True,
    }


def test_published_reference_cells_cover_raw_external_and_target_selection():
    cells = MODULE.published_reference_cells(_config())
    assert len(cells) == 18
    all_external = next(
        cell
        for cell in cells
        if cell.condition == "All" and cell.calibration_source == "external_clean"
    )
    assert all_external.cutoff == 6.8
    assert all_external.expected_dqs == pytest.approx(27.57306879404443)


def test_locked_family_campaign_has_one_control_and_all_intended_cells():
    cells = MODULE.family_cells(_config())
    assert len(cells) == 678
    assert sum(cell.method == "none" for cell in cells) == 6
    assert {cell.repeat for cell in cells} == {2}
    methods = {cell.method for cell in cells}
    assert methods == {
        "none",
        "asr_standard",
        "rasr_windowed",
        "rasr_legacy",
        "adaptive_psp",
        "adaptive_psw",
        "adaptive_mw_final_state",
        "adaptive_mw_sliding",
        "juggler_dbscan",
        "juggler_gev",
        "guided_asr",
    }


def test_guided_family_estimator_explicitly_opts_into_soft_reconstruction():
    config = _config()
    cell = next(
        cell
        for cell in MODULE.family_cells(config)
        if cell.method == "guided_asr" and cell.condition == "All"
    )
    estimator = MODULE._estimator(config, cell, sfreq=512.0)
    assert estimator.experimental is True
    assert estimator.reconstruction == "soft"


def test_adaptive_updates_match_published_complete_interval_semantics():
    class Recorder:
        blocksize = 10

        def __init__(self):
            self.chunks = []
            self.reset_count = 0

        def partial_fit(self, chunk):
            self.chunks.append(chunk.copy())

        def reset_process_state(self):
            self.reset_count += 1

    model = Recorder()
    data = np.arange(2 * 41_472, dtype=float).reshape(2, 41_472)
    updates, omitted = MODULE._adaptive_updates(
        model, data, sfreq=512.0, chunk_s=20.0
    )

    assert updates == 4
    assert [chunk.shape[1] for chunk in model.chunks] == [10_241] * 4
    np.testing.assert_array_equal(model.chunks[0], data[:, :10_241])
    np.testing.assert_array_equal(model.chunks[1], data[:, 10_240:20_481])
    assert omitted == 511
    assert model.reset_count == 1


def _write_cell(root: pathlib.Path, cell, *, commit: str, dirty: bool = False):
    cell_dir = root / "family_replication" / cell.unit_id
    cell_dir.mkdir(parents=True)
    terminal = {
        "status": "completed",
        "git_commit": commit,
        "git_dirty": dirty,
        "config_hash": "config",
        "dataset_manifest_hash": "dataset",
        "environment_hash": "environment",
        "runtime_seconds": 1.0,
        "peak_memory_mb": 2.0,
        "slurm_job_id": "1",
        "slurm_array_task_id": "0",
    }
    metrics = {
        "campaign": cell.campaign,
        "unit_id": cell.unit_id,
        "technical_repeat": cell.repeat,
        "condition": cell.condition,
        "method": cell.method,
        "calibration_source": cell.calibration_source,
        "cutoff": cell.cutoff,
        "status": "success",
    }
    (cell_dir / "terminal_status.json").write_text(
        json.dumps(terminal), encoding="utf-8"
    )
    (cell_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")


def test_merge_rejects_mixed_execution_provenance(tmp_path, monkeypatch):
    cells = MODULE.family_cells(_config())[:2]
    monkeypatch.setattr(MODULE, "campaign_cells", lambda config, campaign: cells)
    _write_cell(tmp_path, cells[0], commit="one")
    _write_cell(tmp_path, cells[1], commit="two")
    args = SimpleNamespace(
        config=str(CONFIG),
        campaign="family_replication",
        output_root=str(tmp_path),
        allow_incomplete=False,
    )
    with pytest.raises(RuntimeError, match="provenance-invalid"):
        MODULE.merge_campaign(args)
    summary = json.loads(
        (tmp_path / "family_replication" / "merge_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["invalid_provenance_signatures"]["git_commit"] == ["one", "two"]
