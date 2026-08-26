"""Known-target substrates shared by submission benchmark runners."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from scipy import signal

_EPS = float(np.finfo(np.float64).eps)


def locked_seed(global_seed: int, *cell: object) -> int:
    """Derive a stable 32-bit seed without opening another simulation cell."""
    payload = "|".join([str(global_seed), *(str(value) for value in cell)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big", signed=False)


def relative_rmse(estimate: np.ndarray, truth: np.ndarray) -> float:
    """Root-mean-square error normalized by the RMS of known truth."""
    estimate = np.asarray(estimate, dtype=float)
    truth = np.asarray(truth, dtype=float)
    return float(
        np.sqrt(np.mean((estimate - truth) ** 2))
        / max(np.sqrt(np.mean(truth**2)), _EPS)
    )


def clean_correlation(estimate: np.ndarray, truth: np.ndarray) -> float:
    """Pearson correlation after flattening a known-target recording."""
    return float(np.corrcoef(np.ravel(estimate), np.ravel(truth))[0, 1])


def _colored_sources(
    n_sources: int, n_times: int, sfreq: float, rng: np.random.Generator
) -> np.ndarray:
    white = rng.standard_normal((n_sources, n_times))
    sos = signal.butter(3, [0.7, min(45.0, sfreq * 0.45)], fs=sfreq, btype="band", output="sos")
    colored = signal.sosfiltfilt(sos, white, axis=1)
    time = np.arange(n_times) / sfreq
    for index in range(n_sources):
        frequency = [6.0, 10.0, 14.0, 20.0][index % 4]
        envelope = 0.6 + 0.4 * signal.sawtooth(
            2 * np.pi * (0.03 + index * 0.002) * time, width=0.5
        )
        colored[index] += envelope * np.sin(
            2 * np.pi * frequency * time + rng.uniform(0, 2 * np.pi)
        )
    colored -= colored.mean(axis=1, keepdims=True)
    colored /= np.maximum(colored.std(axis=1, keepdims=True), _EPS)
    return colored


def clean_multichannel_eeg(
    *, n_channels: int, n_times: int, sfreq: float, seed: int
) -> np.ndarray:
    """Generate spatially mixed, band-limited clean EEG in arbitrary units."""
    rng = np.random.default_rng(seed)
    n_sources = min(12, n_channels)
    sources = _colored_sources(n_sources, n_times, sfreq, rng)
    mixing = rng.standard_normal((n_channels, n_sources))
    mixing /= np.maximum(np.linalg.norm(mixing, axis=0, keepdims=True), _EPS)
    data = mixing @ sources + 0.03 * rng.standard_normal((n_channels, n_times))
    data -= data.mean(axis=1, keepdims=True)
    data /= np.maximum(data.std(axis=1, keepdims=True), _EPS)
    return data


@dataclass
class TransientMixture:
    """One paired clean/contaminated ASR validation unit."""

    calibration: np.ndarray
    clean: np.ndarray
    contaminated: np.ndarray
    artifact: np.ndarray
    artifact_mask: np.ndarray
    artifact_type: str
    artifact_to_signal_db: float
    seed: int


def transient_mixture(
    *,
    artifact_type: str,
    artifact_to_signal_db: float,
    n_channels: int,
    sfreq: float,
    duration_s: float,
    calibration_s: float,
    seed: int,
    burst_duration_s: float = 1.0,
    calibration_contamination_fraction: float = 0.0,
) -> TransientMixture:
    """Inject a known transient field into independently scored clean EEG."""
    rng = np.random.default_rng(locked_seed(seed, "transient_artifact"))
    n_times = int(round(duration_s * sfreq))
    n_calibration = int(round(calibration_s * sfreq))
    full = clean_multichannel_eeg(
        n_channels=n_channels,
        n_times=n_calibration + n_times,
        sfreq=sfreq,
        seed=locked_seed(seed, "clean_eeg"),
    )
    calibration, clean = full[:, :n_calibration], full[:, n_calibration:]
    artifact = np.zeros_like(clean)
    mask = np.zeros(n_times, dtype=bool)
    event_length = max(3, int(round(burst_duration_s * sfreq)))
    available = max(1, n_times - event_length - 2)
    event_starts = np.linspace(
        int(0.1 * available), int(0.9 * available), 6, dtype=int
    )

    for start in event_starts:
        stop = min(n_times, start + event_length)
        length = stop - start
        local_time = np.arange(length) / sfreq
        if artifact_type == "blink":
            waveform = signal.windows.tukey(length, 0.8) * np.sin(
                np.linspace(0, np.pi, length)
            )
            topography = np.exp(-np.arange(n_channels) / max(2, n_channels / 8))
            artifact[:, start:stop] += np.outer(topography, waveform)
        elif artifact_type == "emg_burst":
            carrier = rng.standard_normal((n_channels, length))
            highpass = signal.butter(3, 30.0, fs=sfreq, btype="high", output="sos")
            carrier = signal.sosfilt(highpass, carrier, axis=1)
            active = rng.choice(n_channels, max(2, n_channels // 6), replace=False)
            artifact[active, start:stop] += carrier[active] * signal.windows.tukey(length)
        elif artifact_type == "electrode_pop":
            active = rng.choice(n_channels, max(1, n_channels // 16), replace=False)
            decay = np.exp(-local_time / max(0.05, burst_duration_s / 4))
            signs = rng.choice([-1.0, 1.0], size=active.size)
            artifact[active, start:stop] += signs[:, None] * decay
        elif artifact_type == "low_rank_covariance_burst":
            rank = min(3, n_channels)
            patterns = rng.standard_normal((n_channels, rank))
            patterns, _ = np.linalg.qr(patterns)
            bursts = rng.standard_normal((rank, length)) * signal.windows.tukey(length)
            artifact[:, start:stop] += patterns @ bursts
        else:
            raise ValueError(f"unknown artifact_type {artifact_type!r}")
        mask[start:stop] = True

    desired = 10.0 ** (artifact_to_signal_db / 20.0)
    clean_rms = np.sqrt(np.mean(clean[:, mask] ** 2))
    artifact_rms = np.sqrt(np.mean(artifact[:, mask] ** 2))
    artifact *= desired * clean_rms / max(artifact_rms, _EPS)
    contaminated = clean + artifact

    if calibration_contamination_fraction > 0:
        count = int(round(calibration_contamination_fraction * n_calibration))
        count = min(max(count, 1), n_calibration)
        cal_artifact = np.resize(artifact, (n_channels, count))
        calibration = calibration.copy()
        calibration[:, -count:] += cal_artifact
    return TransientMixture(
        calibration,
        clean,
        contaminated,
        artifact,
        mask,
        artifact_type,
        float(artifact_to_signal_db),
        int(seed),
    )


@dataclass
class ReferenceMixture:
    """Known neural and artifact fields plus a noisy external reference."""

    clean: np.ndarray
    contaminated: np.ndarray
    artifact: np.ndarray
    reference_fit: np.ndarray
    reference_score: np.ndarray
    seed: int


def reference_mixture(
    *,
    reference_snr_db: float,
    reference_count: int,
    neural_leakage_fraction: float,
    coupling: str,
    n_channels: int,
    n_times: int,
    sfreq: float,
    seed: int,
) -> ReferenceMixture:
    """Create a reference-aware problem with a disjoint scoring reference."""
    rng = np.random.default_rng(locked_seed(seed, "reference_artifact"))
    clean = clean_multichannel_eeg(
        n_channels=n_channels,
        n_times=n_times,
        sfreq=sfreq,
        seed=locked_seed(seed, "clean_eeg"),
    )
    n_artifacts = min(4, reference_count)
    sources = rng.standard_normal((n_artifacts, n_times))
    high = signal.butter(3, [20.0, min(90.0, 0.45 * sfreq)], fs=sfreq, btype="band", output="sos")
    sources = signal.sosfiltfilt(high, sources, axis=1)
    sources *= 0.5 + signal.windows.tukey(n_times, 0.2)[None, :]
    primary_mixing = rng.normal(size=(n_channels, n_artifacts))
    if coupling == "drifting":
        end_mixing = rng.normal(size=primary_mixing.shape)
        alpha = np.linspace(0.0, 1.0, n_times)
        artifact = np.einsum("ca,at->ct", primary_mixing, sources) * (1 - alpha) + np.einsum(
            "ca,at->ct", end_mixing, sources
        ) * alpha
    elif coupling == "stationary":
        artifact = primary_mixing @ sources
    else:
        raise ValueError("coupling must be 'stationary' or 'drifting'")
    artifact *= np.sqrt(np.mean(clean**2)) / max(np.sqrt(np.mean(artifact**2)), _EPS)
    contaminated = clean + artifact

    reference_mixing = rng.normal(size=(reference_count, n_artifacts))
    reference_clean = reference_mixing @ sources
    noise = rng.normal(size=reference_clean.shape)
    noise_scale = np.sqrt(np.mean(reference_clean**2)) / (
        10.0 ** (reference_snr_db / 20.0) * max(np.sqrt(np.mean(noise**2)), _EPS)
    )
    neural_projection = rng.normal(size=(reference_count, n_channels)) @ clean
    neural_projection *= np.sqrt(np.mean(reference_clean**2)) / max(
        np.sqrt(np.mean(neural_projection**2)), _EPS
    )
    reference_fit = (
        reference_clean + noise_scale * noise + neural_leakage_fraction * neural_projection
    )
    score_noise = rng.normal(size=reference_clean.shape)
    reference_score = reference_clean + noise_scale * score_noise
    return ReferenceMixture(
        clean,
        contaminated,
        artifact,
        reference_fit,
        reference_score,
        int(seed),
    )


__all__ = [
    "ReferenceMixture",
    "TransientMixture",
    "clean_correlation",
    "clean_multichannel_eeg",
    "locked_seed",
    "reference_mixture",
    "relative_rmse",
    "transient_mixture",
]
