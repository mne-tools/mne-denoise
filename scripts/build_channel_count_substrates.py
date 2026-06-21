"""Build 4-channel and 64-channel synthetic substrates for the robustness sprint.

The same noise model as ``build_long_recording_substrate`` but parameterized
to produce two short, fixed-burst Raws at the extremes of typical
channel counts. The robustness Stage 3 cross-variant matrix uses these to
verify each ASR variant degrades gracefully at low / high dimensionality.

Outputs:
- ``.cache/asr_datasets/low_channel_4ch.fif`` — 4-ch, 120-s
- ``.cache/asr_datasets/high_channel_64ch.fif`` — 64-ch, 120-s
"""

# ruff: noqa: I001

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / ".cache" / "asr_datasets"


def build_substrate(
    *,
    n_channels: int,
    duration_s: float = 120.0,
    sfreq: float = 250.0,
    n_bursts: int = 12,
    burst_duration_s: float = 0.5,
    burst_amplitude_scale: float = 10.0,
    seed: int,
) -> mne.io.Raw:
    rng = np.random.default_rng(seed)
    n_samples = int(round(sfreq * duration_s))
    t = np.arange(n_samples) / sfreq
    data = np.zeros((n_channels, n_samples), dtype=np.float64)
    for ch in range(n_channels):
        phase = rng.uniform(0, 2 * np.pi)
        data[ch] = 0.6 * np.sin(2 * np.pi * 10.0 * t + phase) + 0.15 * np.sin(
            2 * np.pi * 6.5 * t + phase * 0.8
        )
    data += 0.05 * rng.standard_normal(data.shape)

    burst_len = int(round(burst_duration_s * sfreq))
    starts = np.linspace(burst_len, n_samples - burst_len, n_bursts).astype(int)
    channel_scale = float(np.median(np.std(data, axis=1)))
    for start in starts:
        stop = min(start + burst_len, n_samples)
        spatial = rng.standard_normal(n_channels)
        spatial /= max(np.linalg.norm(spatial), np.finfo(float).eps)
        temporal = rng.standard_normal(stop - start)
        data[:, start:stop] += (
            burst_amplitude_scale * channel_scale * np.outer(spatial, temporal)
        )

    ch_names = [f"EEG{i:03d}" for i in range(n_channels)]
    info = mne.create_info(ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data * 1e-6, info, verbose=False)
    raw.filter(l_freq=1.0, h_freq=None, verbose=False)
    return raw


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_low = build_substrate(n_channels=4, seed=4001)
    out_low = OUT_DIR / "low_channel_4ch.fif"
    raw_low.save(out_low, overwrite=True, verbose=False)
    print(f"Wrote {out_low} ({out_low.stat().st_size / 1e6:.1f} MB) — 4 ch")

    raw_high = build_substrate(n_channels=64, seed=6401)
    out_high = OUT_DIR / "high_channel_64ch.fif"
    raw_high.save(out_high, overwrite=True, verbose=False)
    print(f"Wrote {out_high} ({out_high.stat().st_size / 1e6:.1f} MB) — 64 ch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
