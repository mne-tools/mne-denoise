"""JugglerASR."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from mne_denoise.asr import JugglerASR, select_juggler_reference_samples
from mne_denoise.asr._filters import _design_statistics_filter, _lfilter_channels

SFREQ = 250.0


def _make_synthetic_eeg(
    sfreq: float = 250.0,
    duration_s: float = 60.0,
    n_channels: int = 16,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Build clean synthetic EEG: 10 Hz + 6 Hz oscillation + Gaussian noise."""
    if rng is None:
        rng = np.random.default_rng(42)
    n_samples = int(sfreq * duration_s)
    t = np.arange(n_samples) / sfreq
    data = np.zeros((n_channels, n_samples))
    for ch in range(n_channels):
        phase = rng.uniform(0, 2 * np.pi)
        data[ch] = 0.6 * np.sin(2 * np.pi * 10.0 * t + phase) + 0.15 * np.sin(
            2 * np.pi * 6.5 * t + phase * 0.8
        )
    data += 0.05 * rng.standard_normal(data.shape)
    return data


def _inject_bursts(
    data: np.ndarray,
    n_bursts: int = 8,
    burst_duration_s: float = 0.5,
    amplitude: float = 12.0,
    sfreq: float = 250.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Inject n_bursts high-amplitude bursts at evenly-spaced intervals."""
    if rng is None:
        rng = np.random.default_rng(97)
    burst_len = int(round(burst_duration_s * sfreq))
    n_times = data.shape[1]
    starts = np.linspace(burst_len, n_times - burst_len, n_bursts).astype(int)
    contaminated = data.copy()
    mask = np.zeros(n_times, dtype=bool)
    channel_scale = float(np.median(np.std(data, axis=1)))
    for start in starts:
        stop = min(start + burst_len, n_times)
        actual_len = stop - start
        spatial = rng.standard_normal(data.shape[0])
        spatial /= max(np.linalg.norm(spatial), np.finfo(float).eps)
        temporal = rng.standard_normal(actual_len)
        contaminated[:, start:stop] += (
            amplitude * channel_scale * np.outer(spatial, temporal)
        )
        mask[start:stop] = True
    return contaminated, mask


def test_juggler_summary_identifies_strategy_and_fit_state(caplog):
    """JugglerASR INFO includes strategy and the fitted operating point."""
    data = _make_synthetic_eeg(sfreq=SFREQ, duration_s=15.0, n_channels=8)
    with caplog.at_level(logging.INFO, logger="mne_denoise"):
        JugglerASR(
            sfreq=SFREQ,
            cutoff=20.0,
            strategy="dbscan",
            random_state=42,
            verbose=False,
        ).fit(data, verbose=True)
    summaries = [
        record for record in caplog.records if record.message.startswith("JugglerASR:")
    ]
    assert len(summaries) == 1
    for token in (
        "strategy=dbscan",
        "method=",
        "channels=",
        "sfreq=",
        "cutoff=",
        "rank=",
    ):
        assert token in summaries[0].message


def test_juggler_clean_input_keeps_most_samples():
    """Clean EEG with no bursts should retain a large fraction of samples."""
    sfreq = 250.0
    clean = _make_synthetic_eeg(sfreq=sfreq, duration_s=30.0, n_channels=16)
    _, mask, diag = select_juggler_reference_samples(
        clean,
        sfreq,
        strategy="dbscan",
    )
    keep = float(np.mean(mask))
    assert keep > 0.5, f"DBSCAN on clean input retained only {keep * 100:.1f}%"
    assert diag["reference_selected_samples"] == int(mask.sum())
    assert diag["reference_selection_strategy"] == "dbscan"


def test_juggler_gev_on_clean_input():
    """GEV strategy should retain a non-trivial fraction on clean EEG."""
    sfreq = 250.0
    clean = _make_synthetic_eeg(sfreq=sfreq, duration_s=30.0, n_channels=16)
    _, mask, diag = select_juggler_reference_samples(
        clean,
        sfreq,
        strategy="gev",
    )
    keep = float(np.mean(mask))
    assert keep > 0.05, f"GEV retained only {keep * 100:.1f}%"
    assert diag["reference_selection_strategy"] == "gev"
    assert np.isfinite(diag["juggler_gev_mode"])
    assert diag["juggler_gev_scale"] > 0


def test_juggler_min_reference_fraction_raises():
    """min_reference_fraction floor should trigger on heavily-contaminated input."""
    sfreq = 250.0
    clean = _make_synthetic_eeg(sfreq=sfreq, duration_s=30.0, n_channels=16)
    contaminated, _ = _inject_bursts(
        clean,
        n_bursts=50,
        burst_duration_s=1.0,
        amplitude=50.0,
        sfreq=sfreq,
    )
    # Set the floor unreasonably high (99%) so the call MUST raise.
    with pytest.raises(RuntimeError, match="retained too little data"):
        select_juggler_reference_samples(
            contaminated,
            sfreq,
            strategy="gev",
            min_reference_fraction=0.99,
        )


@pytest.mark.parametrize("strategy", ["dbscan", "gev"])
def test_juggler_asr_fit_transform_round_trip(strategy):
    """Both reference-selection strategies feed a fitted ASR state."""
    sfreq = 250.0
    clean = _make_synthetic_eeg(sfreq=sfreq, duration_s=30.0, n_channels=16)
    contaminated, _ = _inject_bursts(clean, sfreq=sfreq)

    asr = JugglerASR(
        sfreq=sfreq,
        cutoff=20.0,
        strategy=strategy,
        random_state=42,
        verbose=False,
    )
    asr.fit(contaminated)
    cleaned = asr.transform(contaminated)

    assert cleaned.shape == contaminated.shape
    assert np.all(np.isfinite(cleaned))
    assert asr.calibration_info_["reference_selection_strategy"] == strategy
    assert 0.0 < asr.calibration_info_["reference_selected_fraction"] <= 1.0


def test_juggler_get_calibration_mask_after_fit():
    """get_calibration_mask should return a sample-wise bool array of length n_times."""
    sfreq = 250.0
    clean = _make_synthetic_eeg(sfreq=sfreq, duration_s=20.0, n_channels=8)
    asr = JugglerASR(
        sfreq=sfreq,
        cutoff=20.0,
        strategy="dbscan",
        random_state=42,
        verbose=False,
    )
    asr.fit(clean)
    mask = asr.get_calibration_mask()
    assert asr.calibration_mask_kind_ == "sample"
    assert mask.dtype == bool
    assert mask.shape == (clean.shape[1],)
    assert mask.sum() == int(asr.calibration_info_["reference_selected_samples"])


def test_juggler_rejects_invalid_selection_parameters():
    """Selector and estimator reject invalid Juggler-specific options."""
    data = _make_synthetic_eeg(sfreq=SFREQ, duration_s=20.0, n_channels=8)
    cases = [
        ("strategy", {"strategy": "bogus"}, "strategy must be"),
        ("dbscan_top_k", {"dbscan_top_k": 0}, "dbscan_top_k"),
        ("gev_grid_size", {"gev_grid_size": 8}, "gev_grid_size"),
        (
            "min_reference_fraction",
            {"min_reference_fraction": 1.5},
            "min_reference_fraction",
        ),
    ]
    for _label, kwargs, message in cases:
        with pytest.raises(ValueError, match=message):
            select_juggler_reference_samples(data, SFREQ, **kwargs)
        with pytest.raises(ValueError, match=message):
            JugglerASR(sfreq=SFREQ, verbose=False, **kwargs).fit(data)


def test_juggler_dbscan_deterministic():
    """JugglerASR(dbscan) with same random_state should be deterministic."""
    sfreq = 250.0
    clean = _make_synthetic_eeg(sfreq=sfreq, duration_s=20.0, n_channels=8)
    contaminated, _ = _inject_bursts(clean, sfreq=sfreq)
    asr1 = JugglerASR(
        sfreq=sfreq,
        cutoff=20.0,
        strategy="dbscan",
        random_state=42,
        verbose=False,
    )
    asr2 = JugglerASR(
        sfreq=sfreq,
        cutoff=20.0,
        strategy="dbscan",
        random_state=42,
        verbose=False,
    )
    asr1.fit(contaminated)
    asr2.fit(contaminated)
    np.testing.assert_allclose(asr1.M_, asr2.M_, rtol=1e-12)
    np.testing.assert_allclose(asr1.T_, asr2.T_, rtol=1e-12)
    np.testing.assert_array_equal(
        asr1.get_calibration_mask(), asr2.get_calibration_mask()
    )


@pytest.mark.parametrize("strategy", ["dbscan", "gev"])
def test_select_juggler_reference_samples_rejects_burst_samples(
    synthetic_burst_data,
    strategy,
):
    """Juggler reference selectors prefer low-amplitude samples."""
    data, _, burst_mask, sfreq = synthetic_burst_data
    reference, sample_mask, diagnostics = select_juggler_reference_samples(
        data,
        sfreq,
        strategy=strategy,
    )

    assert reference.shape[0] == data.shape[0]
    assert reference.shape[1] == int(np.sum(sample_mask))
    assert sample_mask.shape == (data.shape[1],)
    assert np.mean(sample_mask[burst_mask]) < np.mean(sample_mask[~burst_mask])
    assert diagnostics["reference_selection_strategy"] == strategy
    assert diagnostics["reference_selected_samples"] == int(np.sum(sample_mask))


def test_juggler_reference_contains_continuously_filtered_samples(
    synthetic_burst_data,
):
    """Pointwise selection must not re-filter concatenated sample fragments."""
    data, _, _, sfreq = synthetic_burst_data
    reference, sample_mask, diagnostics = select_juggler_reference_samples(
        data,
        sfreq,
        strategy="gev",
    )
    b, a = _design_statistics_filter(sfreq, "asr")
    filtered, zf = _lfilter_channels(data, b, a)

    np.testing.assert_allclose(reference, filtered[:, sample_mask])
    np.testing.assert_allclose(diagnostics["selection_filter_zi"], zf)


def test_juggler_rejects_mismatched_statistics_filters(synthetic_burst_data):
    """Selection and reconstruction must use the same statistics filter."""
    data, _, _, sfreq = synthetic_burst_data
    model = JugglerASR(
        sfreq=sfreq,
        filter_kind="asr",
        selection_filter_kind="none",
        verbose=False,
    )
    with pytest.raises(ValueError, match="filter_kind and selection_filter_kind"):
        model.fit(data)
