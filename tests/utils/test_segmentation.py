"""Tests for segmentation utilities."""

from __future__ import annotations

import numpy as np

from mne_denoise.dss.segmentation import CovarianceSegmenter, FixedWindowSegmenter


def _assert_contiguous_cover(segments, n_times):
    assert segments[0][0] == 0
    assert segments[-1][1] == n_times
    for previous, current in zip(segments[:-1], segments[1:], strict=True):
        assert previous[1] == current[0]


def test_fixed_window_segmentation_contract():
    """Fixed windows cover recordings and merge only undersized trailing chunks."""
    sfreq = 250.0
    segmenter = FixedWindowSegmenter(sfreq=sfreq, window_len=10.0)
    cases = [(30, 3), (5, 1), (21, 2), (26, 3), (35, 4)]
    for duration, expected_count in cases:
        data = np.random.default_rng(duration).standard_normal(
            (4, int(duration * sfreq))
        )
        segments = segmenter.segment(data)
        assert len(segments) == expected_count, duration
        _assert_contiguous_cover(segments, data.shape[1])

    regular = segmenter.segment(np.ones((4, int(30 * sfreq))))
    assert {end - start for start, end in regular} == {int(10 * sfreq)}
    merged = segmenter.segment(np.ones((4, int(21 * sfreq))))
    assert merged[-1][1] == int(21 * sfreq)


def test_covariance_segmenter_stationarity_contract():
    """Stationary data stays covered while a strong covariance change splits."""
    sfreq = 250.0
    rng = np.random.default_rng(42)
    stationary = rng.standard_normal((8, int(60 * sfreq)))
    segmenter = CovarianceSegmenter(sfreq=sfreq, min_chunk_len=10.0)
    stationary_segments = segmenter.segment(stationary)
    assert len(stationary_segments) >= 1
    _assert_contiguous_cover(stationary_segments, stationary.shape[1])
    assert all(
        end - start >= int(10 * sfreq) * 0.8 for start, end in stationary_segments
    )

    half = int(30 * sfreq)
    nonstationary = np.concatenate(
        [rng.standard_normal((8, half)) * 0.1, rng.standard_normal((8, half)) * 20.0],
        axis=1,
    )
    split_segments = CovarianceSegmenter(sfreq=sfreq, min_chunk_len=5.0).segment(
        nonstationary
    )
    assert len(split_segments) >= 2
    _assert_contiguous_cover(split_segments, nonstationary.shape[1])


def test_covariance_segmenter_edge_and_minimum_contract():
    """Short, single-window, and zero-variance inputs remain valid and covered."""
    sfreq = 100.0
    rng = np.random.default_rng(0)
    short = rng.standard_normal((4, int(5 * sfreq)))
    assert CovarianceSegmenter(sfreq=sfreq, min_chunk_len=30.0).segment(short) == [
        (0, short.shape[1])
    ]

    one_window = rng.standard_normal((4, int(1.5 * sfreq)))
    assert CovarianceSegmenter(sfreq=sfreq, cov_win_len=1.0, min_chunk_len=0.5).segment(
        one_window
    ) == [(0, one_window.shape[1])]

    flat = rng.standard_normal((4, int(20 * sfreq)))
    flat[:, int(6 * sfreq) : int(13 * sfreq)] = 0.0
    segments = CovarianceSegmenter(
        sfreq=sfreq, cov_win_len=1.0, min_chunk_len=2.0
    ).segment(flat)
    _assert_contiguous_cover(segments, flat.shape[1])


def test_covariance_segmenter_prominence_controls_split_count():
    """Higher prominence demands a more decisive covariance change."""
    sfreq = 250.0
    rng = np.random.default_rng(1)
    half = int(60 * sfreq)
    scale = np.array([5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])[:, None]
    data = np.concatenate(
        [rng.standard_normal((8, half)), rng.standard_normal((8, half)) * scale],
        axis=1,
    )

    lenient = CovarianceSegmenter(sfreq=sfreq, min_chunk_len=10.0, prominence=0.25)
    strict = CovarianceSegmenter(sfreq=sfreq, min_chunk_len=10.0, prominence=3.0)
    assert len(lenient.segment(data)) > len(strict.segment(data))
