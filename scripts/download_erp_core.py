#!/usr/bin/env python
"""Download the FULL ERP CORE project from OSF via osfclient.

ERP CORE lives on OSF under project node ``thsqg`` (all 7 paradigms, incl. N170).
This clones the entire project with the ``osf`` CLI (osfclient).  Run on a Fir
**login node** (needs internet); ``osfclient`` is pip-installed by the staging
driver. Files land under ``<dest>/`` (osfclient writes provider subdirs, e.g.
``<dest>/osfstorage/...``).

    python scripts/download_erp_core.py --dest <dir> --osf-node thsqg
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys


def download(dest: pathlib.Path, osf_node: str, dry_run: bool = False) -> int:
    dest = pathlib.Path(dest)
    if not osf_node or osf_node.upper().startswith("PENDING"):
        print("ERROR: OSF node id not set (use --osf-node thsqg).", file=sys.stderr)
        return 2
    if dry_run:
        print(f"[dry-run] osf -p {osf_node} clone {dest}")
        return 0
    if shutil.which("osf") is None:
        print("ERROR: osfclient ('osf') not on PATH. Install with: pip install osfclient",
              file=sys.stderr)
        return 2
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[osfclient] cloning FULL OSF project {osf_node} -> {dest}", flush=True)
    # osfclient clone is resumable (skips files already present).
    return subprocess.run(["osf", "-p", osf_node, "clone", str(dest)]).returncode


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dest", required=True)
    p.add_argument("--osf-node", default="thsqg")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    return download(pathlib.Path(a.dest), a.osf_node, a.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
