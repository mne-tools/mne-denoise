#!/usr/bin/env python
"""Clone a single OSF project's storage via osfclient (for non-component projects).

Use for OSF datasets whose data is in the project's own osfstorage (e.g. EEGEyeNet
``ktv7m``, MEG-MASC ``ag3kj``).  For component-structured projects (ERP CORE
``thsqg``) use download_erp_core.py instead.  Run on a Fir login node (internet).

    python scripts/download_osf.py --dest <dir> --node ktv7m
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys


def download(dest: pathlib.Path, node: str, dry_run: bool = False) -> int:
    dest = pathlib.Path(dest)
    if not node or node.upper().startswith("PENDING"):
        print("ERROR: OSF node id not set.", file=sys.stderr)
        return 2
    if dry_run:
        print(f"[dry-run] osf -p {node} clone {dest}")
        return 0
    if shutil.which("osf") is None:
        print("ERROR: osfclient ('osf') not on PATH. pip install osfclient", file=sys.stderr)
        return 2
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[osf] clone project {node} -> {dest}", flush=True)
    return subprocess.run(["osf", "-p", node, "clone", str(dest)]).returncode


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dest", required=True)
    p.add_argument("--node", required=True)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    return download(pathlib.Path(a.dest), a.node, a.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
