"""Convert the Ehrlich real-artifact dataset into MNE Raw substrates.

``refs/asr/datasets/dataset-automaticArtifactRemoval/data/s01..s13/`` are
EEGLAB .set/.fdt files with real EOG, facial EMG, and movement artifacts.
This script loads each subject, applies the standard 1 Hz highpass,
and pickles to ``.cache/asr_datasets/real_eog/sub-XX_raw.fif``.

Purpose: settle the "synthetic-bursts-vs-real-artifacts" substrate
question raised by the Juggler GEV inversion (paper found GEV > standard;
our synthetic-burst substrate found the opposite). Real EOG/EMG/movement
artifacts are the appropriate comparator.
"""

# ruff: noqa: I001

from __future__ import annotations

from pathlib import Path

import mne

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = (
    ROOT / "refs" / "asr" / "datasets" / "dataset-automaticArtifactRemoval" / "data"
)
OUT_DIR = ROOT / ".cache" / "asr_datasets" / "real_eog"


def convert_subject(subject_dir: Path) -> Path | None:
    """Load one subject's .set/.fdt and save it back as -raw.fif."""
    set_file = next(subject_dir.glob("*.set"), None)
    if set_file is None:
        return None
    out_path = OUT_DIR / f"{subject_dir.name}_raw.fif"
    if out_path.exists():
        print(f"  [{subject_dir.name}] already present: {out_path.name}")
        return out_path
    try:
        raw = mne.io.read_raw_eeglab(set_file, preload=True, verbose=False)
    except Exception as exc:  # noqa: BLE001
        print(f"  [{subject_dir.name}] read failed: {type(exc).__name__}: {exc}")
        return None
    eeg_picks = mne.pick_types(raw.info, eeg=True, meg=False, exclude=[])
    if len(eeg_picks) == 0:
        print(f"  [{subject_dir.name}] no EEG channels found; skipped")
        return None
    if raw.info["highpass"] is None or raw.info["highpass"] < 0.99:
        raw.filter(l_freq=1.0, h_freq=None, picks=eeg_picks, verbose=False)
    raw.save(out_path, overwrite=True, verbose=False)
    print(
        f"  [{subject_dir.name}] {len(raw.ch_names)} ch, "
        f"sfreq={raw.info['sfreq']:.0f} Hz, "
        f"duration={raw.n_times / raw.info['sfreq']:.0f} s -> {out_path.name}"
    )
    return out_path


def main() -> int:
    if not SRC_DIR.exists():
        raise SystemExit(f"Ehrlich dataset not found at {SRC_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    subject_dirs = sorted(p for p in SRC_DIR.iterdir() if p.is_dir())
    print(f"Found {len(subject_dirs)} subjects under {SRC_DIR}")
    converted = []
    for subject_dir in subject_dirs:
        out = convert_subject(subject_dir)
        if out is not None:
            converted.append(out)
    print(f"\nConverted {len(converted)}/{len(subject_dirs)} subjects to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
