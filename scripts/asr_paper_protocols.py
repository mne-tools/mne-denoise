"""Protocol primitives used by the ASR paper-replication runners."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TsaiDemoSequence:
    """Prepared 240-second sequence from the public AASR demonstration."""

    clean: np.ndarray
    contaminated: np.ndarray
    crop_start: int
    crop_stop: int
    base_segment_samples: int


def tsai_fft_bandpass(
    data: np.ndarray,
    sfreq: float,
    low: float = 1.0,
    high: float = 50.0,
) -> np.ndarray:
    """Apply the Fourier-bin bandpass used by ``datafiltering2.m``.

    The public AASR demonstration retains only positive-frequency FFT bins
    strictly inside the requested passband and doubles the real inverse FFT.
    This function preserves that convention, including its endpoint handling,
    so replication runners do not silently substitute a Butterworth or FIR
    filter.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("data must have shape (n_channels, n_times)")
    sfreq = float(sfreq)
    low = float(low)
    high = float(high)
    if not 0.0 < low < high <= sfreq / 2.0:
        raise ValueError("Require 0 < low < high <= Nyquist")
    if data.shape[1] < 2:
        raise ValueError("data must contain at least two samples")

    n_times = data.shape[1]
    frequency_bins = np.arange(n_times, dtype=np.float64) * sfreq / n_times
    # MATLAB indices returned by min(abs(fv - cutoff)).  Keeping the 1-based
    # values here makes the subsequent slices match the reference exactly.
    low_index_matlab = int(np.argmin(np.abs(frequency_bins - low))) + 1
    high_index_matlab = int(np.argmin(np.abs(frequency_bins - high))) + 1

    spectrum = np.fft.fft(data, axis=1)
    spectrum[:, :low_index_matlab] = 0.0
    spectrum[:, n_times - low_index_matlab - 1 :] = 0.0
    spectrum[:, high_index_matlab - 1 :] = 0.0
    return 2.0 * np.real(np.fft.ifft(spectrum, axis=1))


def build_tsai_demo_sequence(
    clean: np.ndarray,
    contaminated: np.ndarray,
    *,
    sfreq: float = 200.0,
    crop_start_s: float = 2.0,
    crop_duration_s: float = 24.0,
) -> TsaiDemoSequence:
    """Build the alternating contaminated/clean sequence in ``AASR_demo``.

    The source signals are first cropped from 2 to 26 seconds, duplicated to
    48 seconds, and then arranged as contaminated-clean-contaminated-clean-
    contaminated. The clean target is the duplicated clean signal repeated
    five times.
    """
    clean = np.asarray(clean, dtype=np.float64)
    contaminated = np.asarray(contaminated, dtype=np.float64)
    if clean.ndim != 2 or contaminated.ndim != 2:
        raise ValueError("clean and contaminated must be two-dimensional")
    if clean.shape[0] != contaminated.shape[0]:
        raise ValueError("clean and contaminated must have the same channel count")

    start = int(round(float(crop_start_s) * float(sfreq)))
    segment_samples = int(round(float(crop_duration_s) * float(sfreq)))
    stop = start + segment_samples
    if clean.shape[1] < stop or contaminated.shape[1] < stop:
        raise ValueError(
            "clean and contaminated must cover the requested 2-26 second crop"
        )

    clean_segment = clean[:, start:stop]
    contaminated_segment = contaminated[:, start:stop]
    clean_48 = np.concatenate((clean_segment, clean_segment), axis=1)
    contaminated_48 = np.concatenate(
        (contaminated_segment, contaminated_segment), axis=1
    )
    sequence_contaminated = np.concatenate(
        (
            contaminated_48,
            clean_48,
            contaminated_48,
            clean_48,
            contaminated_48,
        ),
        axis=1,
    )
    sequence_clean = np.concatenate((clean_48,) * 5, axis=1)
    return TsaiDemoSequence(
        clean=sequence_clean,
        contaminated=sequence_contaminated,
        crop_start=start,
        crop_stop=stop,
        base_segment_samples=segment_samples,
    )


def tsai_demo_update_slices(
    n_times: int,
    sfreq: float = 200.0,
    update_window_s: float = 20.0,
) -> tuple[slice, ...]:
    """Return the overlapping update slices used by ``AASR_demo``.

    The MATLAB loop uses inclusive endpoints (``seg(i):seg(i+1)``), so
    adjacent 20-second chunks share one sample. It also intentionally omits
    the tail following the last complete interval.
    """
    n_times = int(n_times)
    step = int(round(float(sfreq) * float(update_window_s)))
    if n_times <= 0 or step <= 0:
        raise ValueError("n_times and the update-window length must be positive")
    starts = np.arange(0, n_times, step, dtype=int)
    return tuple(
        slice(int(start), int(next_start) + 1)
        for start, next_start in zip(starts[:-1], starts[1:])
    )


def paper_rmse_and_snr(
    clean: np.ndarray,
    processed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute channel-wise RMSE and reconstruction SNR used by Tsai et al."""
    clean = np.asarray(clean, dtype=np.float64)
    processed = np.asarray(processed, dtype=np.float64)
    if clean.shape != processed.shape or clean.ndim != 2:
        raise ValueError("clean and processed must have identical 2D shapes")
    residual = processed - clean
    rmse = np.sqrt(np.mean(residual**2, axis=1))
    signal_energy = np.sum(clean**2, axis=1)
    noise_energy = np.sum(residual**2, axis=1)
    energy_scale = np.maximum(signal_energy, 1.0)
    snr = 10.0 * np.log10(
        np.maximum(signal_energy, np.finfo(float).tiny)
        / np.maximum(noise_energy, np.finfo(float).eps * energy_scale)
    )
    return rmse, snr
