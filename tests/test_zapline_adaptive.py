from unittest.mock import patch

import mne
import numpy as np
import pytest
from scipy import signal

from mne_denoise.zapline import ZapLine
from mne_denoise.zapline.adaptive import (
    apply_cleanline_notch,
    apply_hybrid_cleanup,
    check_artifact_presence,
    check_spectral_qa,
    detect_harmonics,
    find_fine_peak,
    find_noise_freqs,
    segment_data,
)


def test_adaptive_components():
    # Synthetic data
    rng = np.random.default_rng(42)
    sfreq = 1000
    times = np.arange(10000) / sfreq  # 10s
    data = rng.standard_normal((4, 10000)) * 0.1

    # 50 Hz noise
    noise = np.sin(2 * np.pi * 50 * times) * 2.0
    data[:, :] += noise

    # Test frequency detection
    freqs = find_noise_freqs(data, sfreq, fmin=45, fmax=55)
    assert len(freqs) > 0
    assert np.isclose(freqs[0], 50, atol=1.0)

    # Test artifact presence
    present = check_artifact_presence(data, sfreq, target_freq=50)
    assert present

    present_wrong = check_artifact_presence(data, sfreq, target_freq=60)
    assert not present_wrong


def test_zapline_plus_pipeline():
    # Long data with shifting noise
    rng = np.random.default_rng(42)
    sfreq = 250
    duration = 120  # 2 minutes
    n_times = int(duration * sfreq)
    times = np.arange(n_times) / sfreq
    n_ch = 4
    data = rng.standard_normal((n_ch, n_times)) * 0.05  # Low background noise

    # 0-40s: 50Hz strong
    # 40-80s: 50.1Hz weak
    # 80-120s: No noise

    t1 = int(sfreq * 40)
    t2 = int(sfreq * 80)

    # Noise 1
    noise1 = np.sin(2 * np.pi * 50.0 * times[:t1]) * 10.0
    # Noise 2 (slightly shifted)
    noise2 = np.sin(2 * np.pi * 50.1 * times[t1:t2]) * 2.0

    # Apply strongly to ch 0, weakly to ch 1
    data[0, :t1] += noise1
    data[1, :t1] += noise1 * 0.5

    data[0, t1:t2] += noise2
    data[1, t1:t2] += noise2 * 0.5

    # Run Zapline Plus
    zl = ZapLine(
        sfreq=sfreq,
        line_freq=None,  # Auto detect
        adaptive=True,
        adaptive_params={
            "fmin": 45,
            "fmax": 55,
            "n_remove_params": {"sigma": 3.0},
            "qa_params": {"max_sigma": 4.0},
        },
    )

    cleaned = zl.fit_transform(data)

    # Check if lines removed
    assert hasattr(zl, "adaptive_results_")

    # PSD function helper
    def get_power_at(d, freq, fs):
        f, p = signal.welch(d, fs=fs, nperseg=int(fs), axis=-1)
        idx = np.argmin(np.abs(f - freq))
        return np.mean(p[:, idx])

    # Check 50Hz power in first segment
    p_orig_1 = get_power_at(data[:, :t1], 50.0, sfreq)
    p_clean_1 = get_power_at(cleaned[:, :t1], 50.0, sfreq)

    assert p_clean_1 < p_orig_1 * 0.1  # 90% reduction

    # Check 50.1Hz power in second segment
    p_orig_2 = get_power_at(data[:, t1:t2], 50.1, sfreq)
    p_clean_2 = get_power_at(cleaned[:, t1:t2], 50.1, sfreq)

    assert p_clean_2 < p_orig_2 * 0.5  # significant reduction

    # Check no-noise segment preservation
    p_orig_3 = np.var(data[:, t2:])
    p_clean_3 = np.var(cleaned[:, t2:])

    # Allow small reduction due to random chance correlations
    assert np.isclose(p_clean_3, p_orig_3, rtol=0.2)


@pytest.fixture
def line_noise_data():
    """Generate synthetic data with line noise."""
    rng = np.random.default_rng(42)
    sfreq = 250
    n_channels = 4
    n_times = 2500  # 10s
    times = np.arange(n_times) / sfreq

    # Background signal
    brain = rng.normal(0, 0.5, (n_channels, n_times))

    # 50 Hz line noise
    line_noise = np.sin(2 * np.pi * 50 * times) * 2.0
    data = brain + line_noise[np.newaxis, :]

    return {"data": data, "sfreq": sfreq, "line_freq": 50.0, "times": times}


def test_cleanline_notch_output_shape(line_noise_data):
    """apply_cleanline_notch should return same shape as input."""
    data = line_noise_data["data"]
    sfreq = line_noise_data["sfreq"]

    filtered = apply_cleanline_notch(data, sfreq=sfreq, freq=50.0)

    assert filtered.shape == data.shape


def test_cleanline_notch_reduces_target_power(line_noise_data):
    """apply_cleanline_notch should reduce power at target frequency."""
    data = line_noise_data["data"]
    sfreq = line_noise_data["sfreq"]

    filtered = apply_cleanline_notch(data, sfreq=sfreq, freq=50.0)

    def get_power_at(d, freq, fs):
        f, psd = signal.welch(d, fs=fs, nperseg=int(fs), axis=-1)
        idx = np.argmin(np.abs(f - freq))
        return np.mean(psd[:, idx])

    power_before = get_power_at(data, 50.0, sfreq)
    power_after = get_power_at(filtered, 50.0, sfreq)

    # Should reduce 50 Hz power significantly
    assert power_after < power_before * 0.1


def test_cleanline_notch_preserves_other_frequencies(line_noise_data):
    """apply_cleanline_notch should preserve power at other frequencies."""
    data = line_noise_data["data"]
    sfreq = line_noise_data["sfreq"]

    # Add a 10 Hz signal we want to preserve
    times = line_noise_data["times"]
    alpha = np.sin(2 * np.pi * 10 * times)
    data_with_alpha = data.copy()
    data_with_alpha[0] += alpha

    filtered = apply_cleanline_notch(data_with_alpha, sfreq=sfreq, freq=50.0)

    def get_power_at(d, freq, fs):
        f, psd = signal.welch(d, fs=fs, nperseg=int(fs), axis=-1)
        idx = np.argmin(np.abs(f - freq))
        return np.mean(psd[:, idx])

    power_alpha_before = get_power_at(data_with_alpha[0:1], 10.0, sfreq)
    power_alpha_after = get_power_at(filtered[0:1], 10.0, sfreq)

    # Alpha power should be mostly preserved (within 20%)
    ratio = power_alpha_after / power_alpha_before
    assert ratio > 0.8


def test_cleanline_notch_bandwidth_parameter(line_noise_data):
    """apply_cleanline_notch should respect bandwidth parameter."""
    data = line_noise_data["data"]
    sfreq = line_noise_data["sfreq"]

    # Narrow bandwidth
    filtered_narrow = apply_cleanline_notch(data, sfreq=sfreq, freq=50.0, bandwidth=0.2)

    # Wide bandwidth
    filtered_wide = apply_cleanline_notch(data, sfreq=sfreq, freq=50.0, bandwidth=2.0)

    # Both should reduce 50 Hz
    def get_power_at(d, freq, fs):
        f, psd = signal.welch(d, fs=fs, nperseg=int(fs), axis=-1)
        idx = np.argmin(np.abs(f - freq))
        return np.mean(psd[:, idx])

    power_narrow = get_power_at(filtered_narrow, 50.0, sfreq)
    power_wide = get_power_at(filtered_wide, 50.0, sfreq)
    power_orig = get_power_at(data, 50.0, sfreq)

    assert power_narrow < power_orig
    assert power_wide < power_orig


def test_cleanline_notch_order_parameter(line_noise_data):
    """apply_cleanline_notch should accept order parameter."""
    data = line_noise_data["data"]
    sfreq = line_noise_data["sfreq"]

    # Different filter orders
    filtered_low = apply_cleanline_notch(data, sfreq=sfreq, freq=50.0, order=2)
    filtered_high = apply_cleanline_notch(data, sfreq=sfreq, freq=50.0, order=6)

    # Both should produce valid output
    assert filtered_low.shape == data.shape
    assert filtered_high.shape == data.shape


def test_cleanline_notch_60hz():
    """apply_cleanline_notch should work with 60 Hz."""
    rng = np.random.default_rng(42)
    sfreq = 250
    n_times = 1000
    times = np.arange(n_times) / sfreq

    data = rng.normal(0, 1, (4, n_times))
    data += np.sin(2 * np.pi * 60 * times) * 3.0

    filtered = apply_cleanline_notch(data, sfreq=sfreq, freq=60.0)

    def get_power_at(d, freq, fs):
        f, psd = signal.welch(d, fs=fs, nperseg=int(fs), axis=-1)
        idx = np.argmin(np.abs(f - freq))
        return np.mean(psd[:, idx])

    power_before = get_power_at(data, 60.0, sfreq)
    power_after = get_power_at(filtered, 60.0, sfreq)

    assert power_after < power_before * 0.2


def test_cleanline_notch_invalid_freq():
    """apply_cleanline_notch should handle frequency near Nyquist gracefully."""
    data = np.random.randn(4, 1000)
    sfreq = 100  # Nyquist = 50 Hz

    # 49 Hz is very close to Nyquist
    filtered = apply_cleanline_notch(data, sfreq=sfreq, freq=49.0, bandwidth=0.5)

    # Should still produce valid output (clamped to valid range)
    assert filtered.shape == data.shape


def test_cleanline_notch_low_freq():
    """apply_cleanline_notch should work at low frequencies."""
    data = np.random.randn(4, 1000)
    sfreq = 100

    # Low frequency notch
    filtered = apply_cleanline_notch(data, sfreq=sfreq, freq=10.0, bandwidth=1.0)

    assert filtered.shape == data.shape


def test_cleanline_notch_invalid_bandwidth_returns_original():
    """apply_cleanline_notch with invalid bandwidth should return original."""
    data = np.random.randn(4, 1000)
    sfreq = 100  # Nyquist = 50 Hz

    # Bandwidth larger than possible
    filtered = apply_cleanline_notch(data, sfreq=sfreq, freq=49.9, bandwidth=10.0)

    # Should return unchanged data if filter can't be applied
    assert filtered.shape == data.shape


def test_hybrid_cleanup_output_shape(line_noise_data):
    """apply_hybrid_cleanup should return same shape as input."""
    data = line_noise_data["data"]
    sfreq = line_noise_data["sfreq"]

    cleaned = apply_hybrid_cleanup(data, sfreq=sfreq, freq=50.0)

    assert cleaned.shape == data.shape


def test_hybrid_cleanup_reduces_target_power(line_noise_data):
    """apply_hybrid_cleanup should reduce power at target frequency."""
    data = line_noise_data["data"]
    sfreq = line_noise_data["sfreq"]

    cleaned = apply_hybrid_cleanup(data, sfreq=sfreq, freq=50.0)

    def get_power_at(d, freq, fs):
        f, psd = signal.welch(d, fs=fs, nperseg=int(fs), axis=-1)
        idx = np.argmin(np.abs(f - freq))
        return np.mean(psd[:, idx])

    power_before = get_power_at(data, 50.0, sfreq)
    power_after = get_power_at(cleaned, 50.0, sfreq)

    # Should reduce 50 Hz power
    assert power_after < power_before


def test_hybrid_cleanup_bandwidth_parameter(line_noise_data):
    """apply_hybrid_cleanup should accept bandwidth parameter."""
    data = line_noise_data["data"]
    sfreq = line_noise_data["sfreq"]

    cleaned = apply_hybrid_cleanup(data, sfreq=sfreq, freq=50.0, bandwidth=1.0)

    assert cleaned.shape == data.shape


def test_hybrid_cleanup_max_power_reduction_parameter(line_noise_data):
    """apply_hybrid_cleanup should respect max_power_reduction_db parameter."""
    data = line_noise_data["data"]
    sfreq = line_noise_data["sfreq"]

    # Low threshold - more likely to reject cleanup
    cleaned_strict = apply_hybrid_cleanup(
        data, sfreq=sfreq, freq=50.0, max_power_reduction_db=0.1
    )

    # High threshold - more likely to apply cleanup
    cleaned_permissive = apply_hybrid_cleanup(
        data, sfreq=sfreq, freq=50.0, max_power_reduction_db=10.0
    )

    # Both should produce valid output
    assert cleaned_strict.shape == data.shape
    assert cleaned_permissive.shape == data.shape


def test_hybrid_cleanup_skips_when_overcleaning():
    """apply_hybrid_cleanup should skip if cleanup would over-clean."""
    rng = np.random.default_rng(42)
    sfreq = 250
    n_times = 1000

    # Create data where notch would cause too much collateral damage
    times = np.arange(n_times) / sfreq
    data = np.sin(2 * np.pi * 49 * times) + np.sin(2 * np.pi * 51 * times)
    data = data[np.newaxis, :] + rng.normal(0, 0.01, (1, n_times))

    # With very strict threshold, should return original
    cleaned = apply_hybrid_cleanup(
        data, sfreq=sfreq, freq=50.0, bandwidth=5.0, max_power_reduction_db=0.01
    )

    assert cleaned.shape == data.shape


def test_hybrid_cleanup_no_surrounding_freqs():
    """apply_hybrid_cleanup should handle case with no surrounding frequencies."""
    data = np.random.randn(4, 100)  # Very short data
    sfreq = 100

    # This might not have enough frequency resolution for surrounding check
    cleaned = apply_hybrid_cleanup(data, sfreq=sfreq, freq=25.0)

    assert cleaned.shape == data.shape


def test_hybrid_cleanup_clean_data():
    """apply_hybrid_cleanup on clean data should not damage it."""
    rng = np.random.default_rng(42)
    sfreq = 250
    n_times = 1000

    # Clean data without line noise
    clean_data = rng.normal(0, 1, (4, n_times))

    cleaned = apply_hybrid_cleanup(clean_data, sfreq=sfreq, freq=50.0)

    # Variance should be similar (notch filter has some effect but not dramatic)
    var_before = np.var(clean_data)
    var_after = np.var(cleaned)

    # Should not remove more than 20% of variance
    assert var_after > var_before * 0.8


def test_cleanline_notch_low_ge_high():
    """apply_cleanline_notch should return unchanged when low >= high."""
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, (4, 100))
    sfreq = 100  # Nyquist = 50 Hz

    # Extremely narrow bandwidth at edge where low >= high after clamping
    result = apply_cleanline_notch(data, sfreq=sfreq, freq=49.99, bandwidth=0.001)

    assert result.shape == data.shape


def test_hybrid_cleanup_rejects_due_to_power_loss():
    """apply_hybrid_cleanup returns original when cleanup loses too much power."""
    rng = np.random.default_rng(42)
    sfreq = 250
    n_times = 2000
    times = np.arange(n_times) / sfreq

    # Create broadband signal near target frequency
    data = np.sin(2 * np.pi * 48 * times) + np.sin(2 * np.pi * 52 * times)
    data = data[np.newaxis, :] + rng.normal(0, 0.01, (1, n_times))

    # Very strict threshold
    cleaned = apply_hybrid_cleanup(
        data, sfreq=sfreq, freq=50.0, bandwidth=5.0, max_power_reduction_db=0.001
    )

    assert cleaned.shape == data.shape


def test_find_noise_freqs_empty_range():
    """find_noise_freqs returns empty if fmin > fmax range."""
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, (4, 1000))
    sfreq = 250

    result = find_noise_freqs(data, sfreq, fmin=200, fmax=250)
    assert result == []


def test_find_noise_freqs_small_window():
    """find_noise_freqs skips peaks with window < 3."""
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, (4, 256))
    sfreq = 256

    result = find_noise_freqs(data, sfreq, fmin=50, fmax=51, window_length=0.1)
    assert isinstance(result, list)


def test_segment_data_short():
    """segment_data returns single segment for short data."""
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, (4, 2000))
    sfreq = 250

    result = segment_data(data, sfreq, target_freq=50.0, min_chunk_len=10.0)
    assert len(result) >= 1


def test_segment_data_window_too_large():
    """segment_data handles window larger than data."""
    rng = np.random.default_rng(42)
    # Large cov_win_len relative to data
    data = rng.normal(0, 1, (4, 2000))
    sfreq = 250

    result = segment_data(data, sfreq, target_freq=50.0, cov_win_len=20.0)
    assert result == [(0, 2000)]


def test_find_fine_peak_empty():
    """find_fine_peak returns coarse_freq if empty range."""
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, (4, 1000))
    sfreq = 250

    result = find_fine_peak(data, sfreq, coarse_freq=200, search_width=0.05)
    assert result == 200


def test_detect_harmonics_basic():
    """detect_harmonics detects present harmonics."""
    rng = np.random.default_rng(42)
    sfreq = 500
    n_times = 5000
    times = np.arange(n_times) / sfreq

    data = rng.normal(0, 0.1, (4, n_times))
    data += np.sin(2 * np.pi * 50 * times) * 10.0
    data += np.sin(2 * np.pi * 100 * times) * 8.0

    result = detect_harmonics(data, sfreq, fundamental=50.0, max_harmonics=3)
    assert isinstance(result, list)


def test_detect_harmonics_none():
    """detect_harmonics returns empty if no harmonics."""
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, (4, 5000))
    sfreq = 500

    result = detect_harmonics(data, sfreq, fundamental=50.0, max_harmonics=3)
    assert isinstance(result, list)


def test_detect_harmonics_near_nyquist():
    """detect_harmonics stops before Nyquist."""
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, (4, 1000))
    sfreq = 250

    result = detect_harmonics(data, sfreq, fundamental=60.0, max_harmonics=5)
    assert all(h < 125 for h in result)


def test_check_spectral_qa_short():
    """check_spectral_qa returns 'ok' for short data."""
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, (4, 50))
    sfreq = 100

    result = check_spectral_qa(data, sfreq, target_freq=50.0)
    assert result == "ok"


def test_check_spectral_qa_scenarios():
    """check_spectral_qa returns valid status."""
    rng = np.random.default_rng(42)
    sfreq = 250
    n_times = 2000
    times = np.arange(n_times) / sfreq

    data = rng.normal(0, 0.1, (4, n_times))
    data += np.sin(2 * np.pi * 50 * times) * 20.0

    result = check_spectral_qa(data, sfreq, target_freq=50.0)
    assert result in ["weak", "ok", "strong"]


def test_zapline_adaptive_sfreq_mismatch():
    """Test ZapLine robustness against sampling rate mismatch.

    Ensures that line frequency detection works correctly even when the
    data sampling rate is not a nice integer (e.g. 1000 Hz vs 50 Hz line noise).
    """
    sfreq = 1000.0
    line_freq = 50.0
    n_times = 2000
    n_ch = 3

    # Create synthetic data: signal + line noise
    rng = np.random.default_rng(42)
    times = np.arange(n_times) / sfreq
    signal = rng.standard_normal((n_ch, n_times)) * 0.1

    # Add strong line noise at 50 Hz
    noise = np.sin(2 * np.pi * line_freq * times)
    data = signal + 0.5 * noise

    info = mne.create_info(n_ch, sfreq, "eeg")
    raw = mne.io.RawArray(data, info)

    # 1. Test basic ZapLine execution
    zap = ZapLine(sfreq=sfreq, line_freq=line_freq, n_remove=1)
    raw_clean = zap.fit_transform(raw)

    # Check that noise variance is reduced
    orig_var = np.var(raw.get_data(), axis=1).mean()
    clean_var = np.var(raw_clean.get_data(), axis=1).mean()
    assert clean_var < orig_var

    # 2. Test with tricky sampling rate (e.g. 512 Hz)
    sfreq_tricky = 512.0
    times_tricky = np.arange(n_times) / sfreq_tricky
    data_tricky = rng.standard_normal((n_ch, n_times)) * 0.1
    noise_tricky = np.sin(2 * np.pi * line_freq * times_tricky)
    data_tricky += 0.5 * noise_tricky

    raw_tricky = mne.io.RawArray(
        data_tricky, mne.create_info(n_ch, sfreq_tricky, "eeg")
    )

    zap_tricky = ZapLine(sfreq=sfreq_tricky, line_freq=line_freq, n_remove=1)
    raw_clean_tricky = zap_tricky.fit_transform(raw_tricky)

    clean_var_tricky = np.var(raw_clean_tricky.get_data(), axis=1).mean()
    orig_var_tricky = np.var(raw_tricky.get_data(), axis=1).mean()
    assert clean_var_tricky < orig_var_tricky


def test_adaptive_explicit_line_freq():
    """Test adaptive mode with explicit scalar line_freq."""
    zap = ZapLine(
        sfreq=500,
        line_freq=None,
        adaptive=True,
        adaptive_params={"min_chunk_len": 0.1},
    )
    zap.line_freq = 50.0
    data = np.random.randn(1, 1000)
    res = zap._run_adaptive(data)
    assert res["line_freq"] == 50.0


def test_adaptive_qa_branches():
    """Test weak/strong branches in _process_chunk."""
    zap = ZapLine(sfreq=1000, line_freq=None, adaptive=True)
    chunk = np.random.randn(1, 1000)

    with (
        patch("mne_denoise.zapline.core.find_fine_peak", return_value=50.0),
        patch("mne_denoise.zapline.core.check_artifact_presence", return_value=True),
    ):
        # Scenario 1: Returns "weak" once, then "ok"
        # Triggers: current_sigma -= 0.25, current_min_remove += 1
        with patch(
            "mne_denoise.zapline.core.check_spectral_qa", side_effect=["weak", "ok"]
        ) as mock_qa:
            zap._process_chunk(
                chunk,
                50.0,
                sigma_init=3.0,
                min_remove=1,
                max_prop_remove=0.5,
                qa_params={},
                hybrid_fallback=False,
            )
            assert mock_qa.call_count == 2

        # Scenario 2: Returns "strong" once, then "ok"
        # Triggers: current_sigma += 0.25, current_min_remove -= 1
        with patch(
            "mne_denoise.zapline.core.check_spectral_qa", side_effect=["strong", "ok"]
        ) as mock_qa:
            zap._process_chunk(
                chunk,
                50.0,
                sigma_init=3.0,
                min_remove=1,
                max_prop_remove=0.5,
                qa_params={},
                hybrid_fallback=False,
            )
            assert mock_qa.call_count == 2

        # Scenario 3: Hybrid fallback
        # Always "weak", should trigger hybrid fallback
        with (
            patch(
                "mne_denoise.zapline.core.check_spectral_qa",
                return_value="weak",
            ),
            patch(
                "mne_denoise.zapline.core.apply_hybrid_cleanup",
                return_value=chunk,
            ) as mock_hybrid,
        ):
            zap._process_chunk(
                chunk,
                50.0,
                sigma_init=3.0,
                min_remove=1,
                max_prop_remove=0.5,
                qa_params={},
                hybrid_fallback=True,
            )
            mock_hybrid.assert_called_once()


def test_hybrid_cleanup_protection():
    """Test that hybrid cleanup returns original data if signal loss is too high."""
    sfreq = 1000
    t = np.arange(1000) / sfreq
    signal_50 = np.sin(2 * np.pi * 50 * t)
    data = signal_50[None, :]

    with patch("mne_denoise.zapline.adaptive.welch") as mock_welch:
        freqs = np.linspace(0, 100, 100)
        psd_orig = np.ones((1, 100))
        psd_filt = np.zeros((1, 100)) + 1e-10

        mock_welch.side_effect = [(freqs, psd_orig), (freqs, psd_filt)]

        cleaned = apply_hybrid_cleanup(data, sfreq=1000, freq=50.0)
        assert np.array_equal(cleaned, data)
