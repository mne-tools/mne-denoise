#!/usr/bin/env python
"""Download ERP CORE (N170) from OSF into the benchmark dataset tree.

ERP CORE is hosted on OSF (not OpenNeuro).  This fetches the N170 component with
``pooch`` (no MATLAB / proprietary tooling).  The exact OSF node id + archive hash
must be **verified on Fir** and recorded in
``mne_denoise.benchmarks.datasets`` (currently ``download_source='osf:PENDING_NODE_ID'``);
pass them here until the registry is updated::

    python scripts/download_erp_core.py --osf-node <NODE> --known-hash <sha256> \
        --dest $SCRATCH/mne-denoise/datasets/osf/erp-core/n170

This script intentionally refuses to invent an OSF id (see project conventions on
not confabulating unverified identifiers).
"""

from __future__ import annotations

import argparse
import pathlib
import sys


def download(dest: pathlib.Path, osf_node: str | None, known_hash: str | None,
             dry_run: bool = False) -> int:
    dest = pathlib.Path(dest)
    if not osf_node or osf_node.upper().startswith("PENDING"):
        print("ERROR: ERP CORE OSF node id is not set/verified.\n"
              "       Pass --osf-node <id> (verify against erpcore on OSF) and, ideally,\n"
              "       --known-hash <sha256>. Update datasets.py once confirmed.",
              file=sys.stderr)
        return 2
    url = f"https://osf.io/{osf_node}/download"
    if dry_run:
        print(f"[dry-run] would fetch {url} -> {dest} (hash={known_hash})")
        return 0
    import pooch

    dest.mkdir(parents=True, exist_ok=True)
    fname = pooch.retrieve(
        url=url, known_hash=known_hash, path=str(dest),
        fname="erp_core_n170.zip",
        processor=pooch.Unzip(extract_dir=str(dest)),
    )
    print(f"[OK] ERP CORE N170 -> {dest} ({fname if isinstance(fname,str) else 'extracted'})")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dest", required=True)
    p.add_argument("--osf-node", default=None)
    p.add_argument("--known-hash", default=None)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    return download(pathlib.Path(a.dest), a.osf_node, a.known_hash, a.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
