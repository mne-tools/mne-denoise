"""Tests for exact ASR paper-protocol primitives."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.io import loadmat

from scripts.asr_paper_protocols import (
    build_tsai_demo_sequence,
    paper_rmse_and_snr,
    tsai_demo_update_slices,
    tsai_fft_bandpass,
)

MATLAB_REFERENCE_DIR = Path(__file__).parents[1] / "parity" / "matlab_reference"
REPO_ROOT = Path(__file__).resolve().parents[2]


def test_tsai_fft_bandpass_matches_public_matlab_reference():
    payload = loadmat(MATLAB_REFERENCE_DIR / "aasr_filter_input.mat")
    reference = loadmat(MATLAB_REFERENCE_DIR / "aasr_filter_reference.mat")[
        "filtered"
    ]

    filtered = tsai_fft_bandpass(
        payload["data"], float(np.asarray(payload["sfreq"]).squeeze())
    )

    np.testing.assert_allclose(filtered, reference, rtol=1e-13, atol=1e-13)


def test_tsai_fft_bandpass_retains_only_strict_interior_bins():
    sfreq = 200.0
    n_times = 2000
    times = np.arange(n_times) / sfreq
    data = np.vstack(
        [
            np.sin(2 * np.pi * 1.0 * times)
            + np.sin(2 * np.pi * 10.0 * times)
            + np.sin(2 * np.pi * 50.0 * times),
            np.cos(2 * np.pi * 10.0 * times),
        ]
    )

    filtered = tsai_fft_bandpass(data, sfreq)

    np.testing.assert_allclose(filtered[0], np.sin(2 * np.pi * 10.0 * times), atol=1e-12)
    np.testing.assert_allclose(filtered[1], np.cos(2 * np.pi * 10.0 * times), atol=1e-12)


def test_tsai_demo_sequence_matches_public_notebook_layout():
    sfreq = 10.0
    clean = np.arange(300, dtype=float)[np.newaxis, :]
    contaminated = clean + 1000.0

    sequence = build_tsai_demo_sequence(
        clean,
        contaminated,
        sfreq=sfreq,
        crop_start_s=2.0,
        crop_duration_s=24.0,
    )

    clean_segment = clean[:, 20:260]
    contaminated_segment = contaminated[:, 20:260]
    clean_48 = np.concatenate((clean_segment, clean_segment), axis=1)
    contaminated_48 = np.concatenate(
        (contaminated_segment, contaminated_segment), axis=1
    )
    np.testing.assert_array_equal(
        sequence.contaminated,
        np.concatenate(
            (contaminated_48, clean_48, contaminated_48, clean_48, contaminated_48),
            axis=1,
        ),
    )
    np.testing.assert_array_equal(
        sequence.clean,
        np.concatenate((clean_48,) * 5, axis=1),
    )
    assert sequence.clean.shape[1] == 2400


def test_tsai_demo_update_slices_preserve_matlab_inclusive_endpoints():
    slices = tsai_demo_update_slices(48_000, sfreq=200.0, update_window_s=20.0)
    assert len(slices) == 11
    assert slices[0] == slice(0, 4001)
    assert slices[1] == slice(4000, 8001)
    assert slices[-1] == slice(40_000, 44_001)


def test_paper_rmse_and_snr_are_channelwise_energy_metrics():
    clean = np.array([[1.0, -1.0], [2.0, -2.0]])
    processed = np.array([[2.0, 0.0], [2.0, -2.0]])

    rmse, snr = paper_rmse_and_snr(clean, processed)

    np.testing.assert_allclose(rmse[0], 1.0)
    np.testing.assert_allclose(snr[0], 0.0)
    np.testing.assert_allclose(rmse[1], 0.0)
    assert np.isfinite(snr[1])


def test_tsai_public_motor_imagery_execution_is_audited():
    registry = yaml.safe_load(
        (REPO_ROOT / "configs/protocols/asr_paper_replications_v1.yaml").read_text()
    )
    replication = registry["studies"]["tsai_2023"][
        "motor_imagery_public_replication"
    ]
    execution = replication["execution"]
    assert replication["public_halt_subject_count"] == 12
    assert replication["paper_reported_subject_count"] == 13
    assert replication["public_halt_trial_count"] == 9224
    assert execution["attempted_cells"] == execution["successful_cells"] == 960
    assert len(execution["aggregate_sha256"]) == 64
    assert len(execution["per_subject_csv_sha256"]) == 64


def test_ds004784_is_attributed_to_the_published_2023_protocol():
    registry = yaml.safe_load(
        (REPO_ROOT / "configs/protocols/asr_paper_replications_v1.yaml").read_text()
    )
    richer = registry["studies"]["richer_2020"]
    downey = registry["studies"]["downey_ferris_2023"]
    assert richer["original_data"]["status"] == "blocked_external"
    assert downey["original_data"]["doi"].endswith("ds004784.v1.0.4")
    assert downey["published_protocol"]["expected_cutoff_count"] == 436
    freeze_path = REPO_ROOT / downey["published_protocol"]["freeze_manifest"]
    assert freeze_path.is_file()
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    config_path = REPO_ROOT / downey["published_protocol"]["protocol"]
    assert freeze["protocol_id"] == "asr_ds004784_replication_v4"
    assert freeze["configs"][0]["sha256"] == hashlib.sha256(
        config_path.read_bytes()
    ).hexdigest()
    assert downey["split"]["locked_family_replication"] == "technical_repeat_2"


def test_asr_variant_sensitivity_execution_is_complete_and_audited():
    registry = yaml.safe_load(
        (REPO_ROOT / "configs/protocols/asr_paper_replications_v1.yaml").read_text()
    )
    execution = registry["validation_campaigns"]["asr_variant_sensitivity"][
        "execution"
    ]
    assert execution["attempted_cells"] == execution["successful_cells"] == 36000
    assert execution["failed_cells"] == 0
    assert execution["unique_unit_method_attempts"] == 36000
    assert len(execution["merged_raw_metrics_sha256"]) == 10
    assert all(len(value) == 64 for value in execution["merged_raw_metrics_sha256"])


@pytest.mark.parametrize(
    "data,sfreq,low,high",
    [
        (np.zeros(100), 200.0, 1.0, 50.0),
        (np.zeros((2, 100)), 200.0, 0.0, 50.0),
        (np.zeros((2, 100)), 200.0, 50.0, 1.0),
    ],
)
def test_tsai_fft_bandpass_rejects_invalid_inputs(data, sfreq, low, high):
    with pytest.raises(ValueError):
        tsai_fft_bandpass(data, sfreq, low, high)
