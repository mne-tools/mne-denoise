"""Build a ~2-hour 32-channel synthetic Raw for the robustness sprint.

Tiles a synthetic-EEG generator across a long span, injects 1 burst per
second to keep ASR's clean-windows criterion active throughout. Pickles
to ``.cache/asr_datasets/long_recording_2h.fif`` for Stage 3 / Stage 4 use.

Run-only depends on numpy + MNE; ~2 minutes on a laptop.
"""

# ruff: noqa: I001

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / ".cache" / "asr_datasets" / "long_recording_2h.fif"


def build_long_recording(
    duration_s: float = 2 * 3600.0,
    sfreq: float = 250.0,
    n_channels: int = 32,
    bursts_per_second: float = 1.0,
    burst_duration_s: float = 0.4,
    burst_amplitude_scale: float = 8.0,
    seed: int = 137,
) -> mne.io.Raw:
    """Build a synthetic Raw of the requested duration."""
    rng = np.random.default_rng(seed)
    n_samples = int(round(sfreq * duration_s))
    t = np.arange(n_samples) / sfreq

    print(
        f"Synthesising {duration_s / 60:.1f}-minute base EEG "
        f"({n_channels} ch x {n_samples} samples = "
        f"{n_channels * n_samples * 8 / 1e6:.1f} MB)…"
    )

    # Brain — 10 + 6 Hz oscillation per channel + 1/f noise
    data = np.zeros((n_channels, n_samples), dtype=np.float64)
    for ch in range(n_channels):
        phase = rng.uniform(0, 2 * np.pi)
        data[ch] = 0.6 * np.sin(2 * np.pi * 10.0 * t + phase) + 0.15 * np.sin(
            2 * np.pi * 6.5 * t + phase * 0.8
        )
    data += 0.05 * rng.standard_normal(data.shape)

    # 1 burst per second across the entire recording
    n_bursts = int(bursts_per_second * duration_s)
    burst_len = int(round(burst_duration_s * sfreq))
    channel_scale = float(np.median(np.std(data, axis=1)))
    print(
        f"Injecting {n_bursts} bursts of {burst_duration_s}s at amplitude "
        f"{burst_amplitude_scale}x channel-median-std…"
    )
    burst_starts = rng.choice(
        np.arange(0, n_samples - burst_len, dtype=int),
        size=n_bursts,
        replace=False,
    )
    for start in burst_starts:
        stop = start + burst_len
        spatial = rng.standard_normal(n_channels)
        spatial /= max(np.linalg.norm(spatial), np.finfo(float).eps)
        temporal = rng.standard_normal(burst_len)
        data[:, start:stop] += (
            burst_amplitude_scale * channel_scale * np.outer(spatial, temporal)
        )

    ch_names = [f"EEG{i:03d}" for i in range(n_channels)]
    info = mne.create_info(ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data * 1e-6, info, verbose=False)  # to volts
    # Apply standard 1 Hz highpass to mimic the typical ASR-ready preprocessing
    raw.filter(l_freq=1.0, h_freq=None, verbose=False)
    return raw


def main() -> int:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw = build_long_recording()
    raw.save(OUT_PATH, overwrite=True, verbose=False)
    print(f"\nWrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")
    print(
        f"  n_eeg={len(raw.ch_names)}  sfreq={raw.info['sfreq']}Hz  "
        f"duration={raw.n_times / raw.info['sfreq'] / 60:.1f} min"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
