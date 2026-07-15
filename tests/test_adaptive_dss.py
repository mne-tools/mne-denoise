"""Tests for segmented, automatically selected adaptive DSS."""

import numpy as np
import pytest

from mne_denoise.dss import (
    DSS,
    BandpassBias,
    FixedWindowSegmenter,
    eigenvalue_ratio_selection,
    max_gap_selection,
)


def test_selection_heuristics_report_gap_location():
    eigenvalues = np.array([12.0, 9.0, 1.0, 0.9])
    assert eigenvalue_ratio_selection(eigenvalues, 3.0) == 2
    assert max_gap_selection(eigenvalues, 1.2) == 2


def test_segmented_dss_removes_nonstationary_narrowband_component():
    rng = np.random.default_rng(42)
    sfreq = 200.0
    n_channels, n_times = 8, 1600
    time = np.arange(n_times) / sfreq
    clean = 0.3 * rng.standard_normal((n_channels, n_times))
    clean += rng.normal(size=(n_channels, 1)) @ np.sin(2 * np.pi * 10 * time)[None]
    line = np.sin(2 * np.pi * 50 * time)
    first = rng.normal(size=n_channels)
    second = rng.normal(size=n_channels)
    artifact = np.zeros_like(clean)
    artifact[:, : n_times // 2] = first[:, None] * line[: n_times // 2]
    artifact[:, n_times // 2 :] = second[:, None] * line[n_times // 2 :]
    contaminated = clean + artifact
    model = DSS(
        BandpassBias((48, 52), sfreq),
        n_components=4,
        n_select=1,
        smooth=int(sfreq / 50),
        segmented=True,
        segmenter=FixedWindowSegmenter(sfreq, window_len=2.0),
        crossfade=0.1,
        return_type="raw",
        verbose=False,
    )
    cleaned = model.fit_transform(contaminated)
    assert cleaned.shape == contaminated.shape
    assert len(model.segment_results_) == 4
    assert all(result["n_selected"] == 1 for result in model.segment_results_)
    before = abs(np.fft.rfft(contaminated, axis=1)[:, 400]).mean()
    after = abs(np.fft.rfft(cleaned, axis=1)[:, 400]).mean()
    assert after < before


def test_segmented_dss_rejects_fit_without_transform():
    model = DSS(
        BandpassBias((8, 12), 100.0),
        segmented=True,
        segmenter=FixedWindowSegmenter(100.0, 1.0),
    )
    with pytest.raises(RuntimeError, match="fit_transform"):
        model.fit(np.ones((3, 300)))
