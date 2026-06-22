#!/usr/bin/env python
"""Download EEGdenoiseNet artifact/clean segments into the benchmark dataset tree.

EEGdenoiseNet (Zhang et al. 2021) ships clean-EEG + EOG + EMG ``.npy`` segment
arrays that ``mne_denoise.benchmarks.simulation`` combines into multichannel
mixtures.  Fetched with ``pooch``.  The exact archive URL + hash must be
**verified on Fir** (registry currently ``download_source='pooch:PENDING_URL'``)::

    python scripts/download_eegdenoisenet.py --url <ARCHIVE_URL> --known-hash <sha256> \
        --dest $SCRATCH/mne-denoise/datasets/zenodo/eegdenoisenet

Refuses to invent a URL/hash (see project conventions on unverified identifiers).
"""

from __future__ import annotations

import argparse
import pathlib
import sys

EXPECTED_FILES = ("EEG_all_epochs.npy", "EOG_all_epochs.npy", "EMG_all_epochs.npy")


def download(dest: pathlib.Path, url: str | None, known_hash: str | None,
             dry_run: bool = False) -> int:
    dest = pathlib.Path(dest)
    if not url or url.upper().startswith("PENDING"):
        print("ERROR: EEGdenoiseNet archive URL is not set/verified.\n"
              "       Pass --url <archive> and --known-hash <sha256> "
              "(verify on Zenodo/GitHub), then update datasets.py.",
              file=sys.stderr)
        return 2
    if dry_run:
        print(f"[dry-run] would fetch {url} -> {dest} (hash={known_hash}); "
              f"expect {EXPECTED_FILES}")
        return 0
    import pooch

    dest.mkdir(parents=True, exist_ok=True)
    proc = pooch.Unzip(extract_dir=str(dest)) if url.endswith(".zip") else None
    pooch.retrieve(url=url, known_hash=known_hash, path=str(dest),
                   fname="eegdenoisenet_archive", processor=proc)
    missing = [f for f in EXPECTED_FILES if not list(dest.rglob(f))]
    if missing:
        print(f"[WARN] expected files not found after extract: {missing}", file=sys.stderr)
        return 1
    print(f"[OK] EEGdenoiseNet -> {dest}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dest", required=True)
    p.add_argument("--url", default=None)
    p.add_argument("--known-hash", default=None)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    return download(pathlib.Path(a.dest), a.url, a.known_hash, a.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
