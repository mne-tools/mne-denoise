"""Unit tests for the shared raised-cosine blending helpers."""

from __future__ import annotations

import numpy as np
import pytest

from mne_denoise._blending import overlap_add_combine, raised_cosine_ramp

# ---------------------------------------------------------------------------
# raised_cosine_ramp
# ---------------------------------------------------------------------------


def test_ramp_is_monotonic_and_bounded():
    ramp = raised_cosine_ramp(64)
    assert ramp.shape == (64,)
    assert np.all(np.diff(ramp) > 0)
    assert ramp[0] > 0
    assert ramp[-1] == pytest.approx(1.0)


def test_ramp_crosses_half_at_midpoint():
    """The taper reaches 0.5 halfway through the transition."""
    width = 50
    ramp = raised_cosine_ramp(width)
    # Sample index width//2 - 1 corresponds to argument pi/2 exactly.
    assert ramp[width // 2 - 1] == pytest.approx(0.5)


def test_ramp_and_complement_partition_unity():
    ramp = raised_cosine_ramp(32)
    assert ramp + (1.0 - ramp) == pytest.approx(np.ones(32))


def test_ramp_width_one():
    assert raised_cosine_ramp(1) == pytest.approx(np.array([1.0]))


@pytest.mark.parametrize("width", [0, -5])
def test_ramp_rejects_nonpositive_width(width):
    with pytest.raises(ValueError, match="width must be positive"):
        raised_cosine_ramp(width)


def test_ramp_matches_reference_formula():
    """Guard the exact formula the ASR reconstruction relied on inline."""
    width = 17
    expected = (1.0 - np.cos(np.pi * np.arange(1, width + 1) / width)) / 2.0
    assert raised_cosine_ramp(width) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# overlap_add_combine
# ---------------------------------------------------------------------------


def _tiling_chunks(data, bounds, n_overlap, n_times):
    """Build overlap_add_combine input by extending each segment."""
    chunks = []
    for i, (start, end) in enumerate(bounds):
        ext_start = start if i == 0 else max(0, start - n_overlap)
        ext_end = end if i == len(bounds) - 1 else min(n_times, end + n_overlap)
        chunks.append(
            {
                "data": data[:, ext_start:ext_end],
                "ext_start": ext_start,
                "ext_end": ext_end,
                "start": start,
                "end": end,
            }
        )
    return chunks


def test_identical_chunks_reproduce_input():
    """If every chunk carries the same signal, blending is the identity."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal((4, 1000))
    bounds = [(0, 300), (300, 700), (700, 1000)]
    chunks = _tiling_chunks(data, bounds, n_overlap=50, n_times=1000)

    out = overlap_add_combine((4, 1000), chunks)
    assert out == pytest.approx(data)


def test_single_chunk_is_passthrough():
    rng = np.random.default_rng(1)
    data = rng.standard_normal((3, 200))
    chunks = [{"data": data, "ext_start": 0, "ext_end": 200, "start": 0, "end": 200}]
    assert overlap_add_combine((3, 200), chunks) == pytest.approx(data)


def test_zero_overlap_is_hard_concatenation():
    rng = np.random.default_rng(2)
    a = rng.standard_normal((2, 100))
    b = rng.standard_normal((2, 100))
    chunks = [
        {"data": a, "ext_start": 0, "ext_end": 100, "start": 0, "end": 100},
        {"data": b, "ext_start": 100, "ext_end": 200, "start": 100, "end": 200},
    ]
    out = overlap_add_combine((2, 200), chunks)
    assert out == pytest.approx(np.concatenate([a, b], axis=1))


def test_overlap_removes_boundary_discontinuity():
    """Blending two mismatched constants must beat hard concatenation."""
    n_times, n_overlap = 400, 60
    left = np.full((1, n_times), 1.0)
    right = np.full((1, n_times), 5.0)
    bounds = [(0, 200), (200, 400)]

    chunks = []
    for i, (src, (start, end)) in enumerate(zip([left, right], bounds, strict=True)):
        ext_start = start if i == 0 else max(0, start - n_overlap)
        ext_end = end if i == len(bounds) - 1 else min(n_times, end + n_overlap)
        chunks.append(
            {
                "data": src[:, ext_start:ext_end],
                "ext_start": ext_start,
                "ext_end": ext_end,
                "start": start,
                "end": end,
            }
        )

    blended = overlap_add_combine((1, n_times), chunks)[0]
    hard = np.concatenate([left[0, :200], right[0, 200:]])

    assert np.max(np.abs(np.diff(blended))) < np.max(np.abs(np.diff(hard)))
    # Output stays inside the range spanned by the two inputs
    assert blended.min() >= 1.0 - 1e-9
    assert blended.max() <= 5.0 + 1e-9
    # Far from the boundary each side is untouched
    assert blended[0] == pytest.approx(1.0)
    assert blended[-1] == pytest.approx(5.0)


def test_gap_in_coverage_does_not_divide_by_zero():
    """Chunks that fail to tile must not produce inf/nan."""
    data = np.ones((2, 50))
    chunks = [{"data": data, "ext_start": 0, "ext_end": 50, "start": 0, "end": 50}]
    out = overlap_add_combine((2, 100), chunks)
    assert np.all(np.isfinite(out))
    assert out[:, 50:] == pytest.approx(np.zeros((2, 50)))
