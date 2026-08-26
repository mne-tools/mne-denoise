#!/usr/bin/env python3
"""Download missing files for OpenNeuro ds004505 (Real World Table Tennis).

This script downloads:
1. Missing .fdt files for sourcedata/Merged/ (needed to load EEG data)
2. BIDS metadata for processed data: _channels.tsv, _events.tsv, _events.json,
   _electrodes.tsv for all subjects
3. Deletes known-incomplete .fdt files (partial datalad downloads with hex suffixes)

Usage:
    python scripts/download_ds004505.py [--data-dir PATH] [--subjects SUB ...]

    # Download everything missing (default: all 25 subjects)
    python scripts/download_ds004505.py

    # Download only specific subjects
    python scripts/download_ds004505.py --subjects sub-03 sub-11 sub-12

    # Specify custom data directory
    python scripts/download_ds004505.py --data-dir ./examples/tutorials/data/ds004505
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download missing ds004505 files from OpenNeuro"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to ds004505 dataset root. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Subjects to download (e.g., sub-03 sub-11). Default: all 25.",
    )
    parser.add_argument(
        "--skip-merged",
        action="store_true",
        help="Skip downloading sourcedata/Merged .fdt files",
    )
    parser.add_argument(
        "--skip-bids",
        action="store_true",
        help="Skip downloading BIDS metadata (channels.tsv, events.tsv, etc.)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without actually downloading",
    )
    return parser.parse_args()


def find_data_dir() -> Path:
    """Auto-detect the ds004505 dataset directory."""
    candidates = [
        Path(__file__).resolve().parent.parent
        / "examples"
        / "tutorials"
        / "data"
        / "ds004505",
        Path.cwd() / "examples" / "tutorials" / "data" / "ds004505",
        Path.cwd() / "data" / "ds004505",
    ]
    for c in candidates:
        if (c / "participants.tsv").exists():
            return c
    raise FileNotFoundError(
        "Could not find ds004505 dataset directory. Use --data-dir to specify it."
    )


def check_fdt_status(data_dir: Path, subject: str) -> str:
    """Check .fdt file status for a subject in sourcedata/Merged.

    Returns: 'ok', 'missing', 'incomplete', or 'no_set'
    """
    merged_dir = data_dir / "sourcedata" / "Merged" / subject
    if not merged_dir.exists():
        return "missing"

    set_file = merged_dir / f"{subject}_Merged.set"
    fdt_file = merged_dir / f"{subject}_Merged.fdt"

    if not set_file.exists():
        return "no_set"

    if fdt_file.exists():
        # Verify size
        import scipy.io as sio

        try:
            mat = sio.loadmat(str(set_file), squeeze_me=True, struct_as_record=False)
            nbchan = int(mat["nbchan"])
            pnts = int(mat["pnts"])
            expected = nbchan * pnts * 4  # float32
            actual = fdt_file.stat().st_size
            if abs(actual - expected) < 1024:  # allow tiny tolerance
                return "ok"
            else:
                return "incomplete"
        except Exception:
            return "ok"  # can't verify, assume ok

    # Check for partial downloads with hex suffixes
    partials = list(merged_dir.glob(f"{subject}_Merged.fdt.*"))
    if partials:
        return "incomplete"

    return "missing"


def clean_partial_fdts(data_dir: Path, subject: str, dry_run: bool = False) -> None:
    """Remove known-incomplete .fdt files with hex suffixes."""
    merged_dir = data_dir / "sourcedata" / "Merged" / subject
    if not merged_dir.exists():
        return
    for f in merged_dir.glob(f"{subject}_Merged.fdt.*"):
        if f.suffix != ".fdt":
            if dry_run:
                print(f"  [DRY RUN] Would delete incomplete: {f.name}")
            else:
                print(f"  Deleting incomplete download: {f.name}")
                f.unlink()


def download_with_openneuro(
    dataset_id: str,
    target_dir: str,
    include: list[str],
    dry_run: bool = False,
) -> None:
    """Download files from OpenNeuro using openneuro-py."""
    if dry_run:
        print("  [DRY RUN] Would download:")
        for p in include:
            print(f"    {p}")
        return

    try:
        import openneuro
    except ImportError:
        print(
            "ERROR: openneuro-py not installed. Install with:\n"
            "  pip install openneuro-py\n"
        )
        sys.exit(1)

    openneuro.download(
        dataset=dataset_id,
        target_dir=target_dir,
        include=include,
        verify_size=False,
    )


def main() -> None:
    args = parse_args()

    data_dir = args.data_dir or find_data_dir()
    print(f"Dataset directory: {data_dir}")

    all_subjects = [f"sub-{i:02d}" for i in range(1, 26)]
    subjects = args.subjects or all_subjects

    dataset_id = "ds004505"

    # ── Phase 1: Audit current state ─────────────────────────────────────
    print("\n=== Current data status ===")
    print(
        f"{'Subject':<10s} {'Merged .fdt':<15s} {'channels.tsv':<15s} {'events.tsv':<15s}"
    )
    print("-" * 55)

    need_merged_fdt = []
    need_bids = []

    for sub in subjects:
        # Merged .fdt status
        fdt_status = check_fdt_status(data_dir, sub)

        # BIDS metadata status
        eeg_dir = data_dir / sub / "eeg"
        has_channels = (
            bool(list(eeg_dir.glob("*_channels.tsv"))) if eeg_dir.exists() else False
        )
        has_events = (
            bool(list(eeg_dir.glob("*_events.tsv"))) if eeg_dir.exists() else False
        )

        print(
            f"{sub:<10s} {fdt_status:<15s} "
            f"{'YES' if has_channels else 'MISSING':<15s} "
            f"{'YES' if has_events else 'MISSING':<15s}"
        )

        if fdt_status in ("missing", "incomplete"):
            need_merged_fdt.append(sub)
        if not has_channels or not has_events:
            need_bids.append(sub)

    # ── Phase 2: Clean incomplete downloads ──────────────────────────────
    if need_merged_fdt and not args.skip_merged:
        print("\n=== Cleaning incomplete .fdt downloads ===")
        for sub in need_merged_fdt:
            clean_partial_fdts(data_dir, sub, dry_run=args.dry_run)

    # ── Phase 3: Download missing sourcedata/Merged .fdt files ───────────
    if need_merged_fdt and not args.skip_merged:
        print(f"\n=== Downloading {len(need_merged_fdt)} missing Merged .fdt files ===")
        print("(These are large files, ~4-5 GB each. This may take a while.)\n")

        for sub in need_merged_fdt:
            print(f"Downloading {sub}_Merged.fdt ...")
            include = [f"sourcedata/Merged/{sub}/{sub}_Merged.fdt"]

            # Also grab .set if missing
            set_file = data_dir / "sourcedata" / "Merged" / sub / f"{sub}_Merged.set"
            if not set_file.exists():
                include.append(f"sourcedata/Merged/{sub}/{sub}_Merged.set")

            download_with_openneuro(
                dataset_id, str(data_dir), include, dry_run=args.dry_run
            )
    elif args.skip_merged:
        print("\n(Skipping Merged .fdt downloads)")
    else:
        print("\nAll Merged .fdt files are present.")

    # ── Phase 4: Download BIDS metadata ──────────────────────────────────
    if need_bids and not args.skip_bids:
        print(f"\n=== Downloading BIDS metadata for {len(need_bids)} subjects ===\n")

        include_patterns = []
        for sub in need_bids:
            include_patterns.extend(
                [
                    f"{sub}/eeg/{sub}_task-TableTennis_channels.tsv",
                    f"{sub}/eeg/{sub}_task-TableTennis_events.tsv",
                    f"{sub}/eeg/{sub}_task-TableTennis_events.json",
                    f"{sub}/eeg/{sub}_task-TableTennis_electrodes.tsv",
                    f"{sub}/eeg/{sub}_task-TableTennis_coordsystem.json",
                ]
            )

        print(f"Downloading {len(include_patterns)} metadata files ...")
        download_with_openneuro(
            dataset_id, str(data_dir), include_patterns, dry_run=args.dry_run
        )
    elif args.skip_bids:
        print("\n(Skipping BIDS metadata downloads)")
    else:
        print("\nAll BIDS metadata files are present.")

    # ── Phase 5: Verify ──────────────────────────────────────────────────
    if not args.dry_run:
        print("\n=== Post-download verification ===")
        print(
            f"{'Subject':<10s} {'Merged .fdt':<15s} {'channels.tsv':<15s} {'events.tsv':<15s}"
        )
        print("-" * 55)

        ok_count = 0
        for sub in subjects:
            fdt_status = check_fdt_status(data_dir, sub)
            eeg_dir = data_dir / sub / "eeg"
            has_channels = (
                bool(list(eeg_dir.glob("*_channels.tsv")))
                if eeg_dir.exists()
                else False
            )
            has_events = (
                bool(list(eeg_dir.glob("*_events.tsv"))) if eeg_dir.exists() else False
            )

            all_ok = fdt_status == "ok" and has_channels and has_events
            if all_ok:
                ok_count += 1

            print(
                f"{sub:<10s} {fdt_status:<15s} "
                f"{'YES' if has_channels else 'MISSING':<15s} "
                f"{'YES' if has_events else 'MISSING':<15s}"
                f"{'  OK' if all_ok else ''}"
            )

        print(f"\n{ok_count}/{len(subjects)} subjects fully available.")

    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
