"""Tests for exact ASR paper-protocol primitives."""

from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from scripts.asr_paper_protocols import (
    build_tsai_demo_sequence,
    paper_rmse_and_snr,
    tsai_demo_update_slices,
    tsai_fft_bandpass,
)

MATLAB_REFERENCE_DIR = Path(__file__).parents[1] / "parity" / "matlab_reference"


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
