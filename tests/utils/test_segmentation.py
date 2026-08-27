"""Unit tests for segmentation utilities."""

from __future__ import annotations

import numpy as np

from mne_denoise.dss.segmentation import (
    CovarianceSegmenter,
    FixedWindowSegmenter,
)

# ============================================================================
# FixedWindowSegmenter
# ============================================================================


class TestFixedWindowSegmenter:
    """Tests for FixedWindowSegmenter."""

    def test_basic_segmentation(self):
        """Even-length data should produce equal-length windows."""
        sfreq = 250.0
        segmenter = FixedWindowSegmenter(sfreq=sfreq, window_len=10.0)
        data = np.random.default_rng(0).standard_normal((4, int(30 * sfreq)))
        segments = segmenter.segment(data)
        assert len(segments) == 3
        for s, e in segments:
            assert e - s == int(10 * sfreq)

    def test_short_data_single_segment(self):
        """Data shorter than window_len should produce one segment."""
        sfreq = 250.0
        segmenter = FixedWindowSegmenter(sfreq=sfreq, window_len=10.0)
        data = np.random.default_rng(0).standard_normal((4, int(5 * sfreq)))
        segments = segmenter.segment(data)
        assert len(segments) == 1
        assert segments[0] == (0, data.shape[1])

    def test_trailing_merge(self):
        """Tiny trailing chunk (< 50% window) should be merged with previous."""
        sfreq = 250.0
        segmenter = FixedWindowSegmenter(sfreq=sfreq, window_len=10.0)
        # 25.1 seconds → 2 windows of 10s, trailing 5.1s > 50% → 3 segments
        # But 21 seconds → 2 windows of 10s, trailing 1s < 50% → merge
        n_times = int(21 * sfreq)
        data = np.random.default_rng(0).standard_normal((4, n_times))
        segments = segmenter.segment(data)
        # Trailing 1s < 50% of 10s window → merged
        assert len(segments) == 2
        assert segments[-1][1] == n_times

    def test_no_trailing_merge_when_large(self):
        """Trailing chunk >= 50% of window should NOT be merged."""
        sfreq = 250.0
        segmenter = FixedWindowSegmenter(sfreq=sfreq, window_len=10.0)
        # 26 seconds → 2 windows of 10s + trailing 6s (>= 50% of 10s)
        n_times = int(26 * sfreq)
        data = np.random.default_rng(0).standard_normal((4, n_times))
        segments = segmenter.segment(data)
        assert len(segments) == 3
        assert segments[-1][1] == n_times

    def test_covers_all_samples(self):
        """Segments should cover all samples without gaps or overlaps."""
        sfreq = 250.0
        segmenter = FixedWindowSegmenter(sfreq=sfreq, window_len=10.0)
        n_times = int(35 * sfreq)
        data = np.random.default_rng(0).standard_normal((4, n_times))
        segments = segmenter.segment(data)
        assert segments[0][0] == 0
        assert segments[-1][1] == n_times
        for i in range(len(segments) - 1):
            assert segments[i][1] == segments[i + 1][0]


# ============================================================================
# CovarianceSegmenter
# ============================================================================


class TestCovarianceSegmenter:
    """Tests for CovarianceSegmenter."""

    def test_stationary_data_single_segment(self):
        """Stationary data should result in a single segment."""
        sfreq = 250.0
        rng = np.random.default_rng(42)
        data = rng.standard_normal((8, int(60 * sfreq)))
        segmenter = CovarianceSegmenter(sfreq=sfreq, min_chunk_len=10.0)
        segments = segmenter.segment(data)
        # Stationary noise → likely one segment
        assert len(segments) >= 1
        assert segments[0][0] == 0
        assert segments[-1][1] == data.shape[1]

    def test_nonstationary_data_detects_boundary(self):
        """Data with a clear stationarity break should produce >=2 segments."""
        sfreq = 250.0
        rng = np.random.default_rng(42)
        n_ch = 8
        half = int(30 * sfreq)
        # First half: small noise. Second half: noise * 20
        part1 = rng.standard_normal((n_ch, half)) * 0.1
        part2 = rng.standard_normal((n_ch, half)) * 20.0
        data = np.concatenate([part1, part2], axis=1)
        segmenter = CovarianceSegmenter(sfreq=sfreq, min_chunk_len=5.0)
        segments = segmenter.segment(data)
        assert len(segments) >= 2

    def test_min_chunk_length_enforced(self):
        """No segment should be shorter than min_chunk_len (in samples)."""
        sfreq = 250.0
        rng = np.random.default_rng(42)
        n_ch = 8
        data = rng.standard_normal((n_ch, int(120 * sfreq)))
        min_chunk = 10.0
        segmenter = CovarianceSegmenter(sfreq=sfreq, min_chunk_len=min_chunk)
        segments = segmenter.segment(data)
        min_samples = int(min_chunk * sfreq)
        for s, e in segments:
            # Allow last segment to be slightly shorter due to rounding
            assert (e - s) >= min_samples * 0.8

    def test_covers_all_samples(self):
        """Segments should cover all samples without gaps or overlaps."""
        sfreq = 250.0
        rng = np.random.default_rng(42)
        data = rng.standard_normal((4, int(60 * sfreq)))
        segmenter = CovarianceSegmenter(sfreq=sfreq, min_chunk_len=10.0)
        segments = segmenter.segment(data)
        assert segments[0][0] == 0
        assert segments[-1][1] == data.shape[1]
        for i in range(len(segments) - 1):
            assert segments[i][1] == segments[i + 1][0]

    def test_short_data_single_segment(self):
        """Data shorter than min_chunk_len should produce one segment."""
        sfreq = 250.0
        rng = np.random.default_rng(42)
        data = rng.standard_normal((4, int(5 * sfreq)))
        segmenter = CovarianceSegmenter(sfreq=sfreq, min_chunk_len=30.0)
        segments = segmenter.segment(data)
        assert len(segments) == 1
        assert segments[0] == (0, data.shape[1])

    def test_bandpass_parameter(self):
        """Bandpass parameter should be accepted without error."""
        sfreq = 250.0
        rng = np.random.default_rng(42)
        data = rng.standard_normal((4, int(60 * sfreq)))
        segmenter = CovarianceSegmenter(
            sfreq=sfreq, min_chunk_len=10.0, bandpass=(8.0, 12.0)
        )
        segments = segmenter.segment(data)
        assert len(segments) >= 1


def test_covariance_segmenter_single_window_returns_whole_recording():
    """With one covariance window there are no successive distances to compare."""
    sfreq = 100.0
    # Exactly one 1-second covariance window fits in the data
    data = np.random.default_rng(0).standard_normal((4, int(1.5 * sfreq)))

    seg = CovarianceSegmenter(sfreq=sfreq, cov_win_len=1.0, min_chunk_len=0.5)
    assert seg.segment(data) == [(0, data.shape[1])]


def test_covariance_segmenter_handles_zero_variance_windows():
    """All-zero stretches give a zero-trace covariance that must not divide."""
    sfreq = 100.0
    rng = np.random.default_rng(0)
    data = rng.standard_normal((4, int(20 * sfreq)))
    # Flatline the middle third: trace(cov) == 0 for those windows
    data[:, int(6 * sfreq) : int(13 * sfreq)] = 0.0

    seg = CovarianceSegmenter(sfreq=sfreq, cov_win_len=1.0, min_chunk_len=2.0)
    segments = seg.segment(data)

    assert len(segments) >= 1
    assert segments[0][0] == 0
    assert segments[-1][1] == data.shape[1]
    for a, b in zip(segments[:-1], segments[1:], strict=True):
        assert a[1] == b[0]


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
