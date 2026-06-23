#!/usr/bin/env python
"""Download the FULL ERP CORE project from OSF (node thsqg) — all components.

ERP CORE stores each paradigm (N170, MMN, N2pc, N400, P3, ERN, LRP) and extras as
OSF *components* under the project ``thsqg`` (the top project's own storage is
empty).  We enumerate the children via the OSF API (paginated) and run
``osf -p <component> clone`` for each into ``<dest>/<title>/``.  Run on a Fir
**login node** (internet); ``osfclient`` provides the ``osf`` CLI.

    python scripts/download_erp_core.py --dest <dir> --osf-node thsqg
    python scripts/download_erp_core.py --dest <dir> --components N170 MMN   # subset
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import subprocess
import sys
import urllib.request

OSF_API = "https://api.osf.io/v2"


def list_components(node: str, *, attempts: int = 8) -> list[tuple[str, str]]:
    """Return [(component_id, title), ...] for all children of an OSF node (paginated).

    OSF intermittently throttles anonymous callers and returns an empty ``data`` even
    when ``meta.total`` > 0; retry such pages with backoff so a detached run recovers.
    """
    import time

    url = f"{OSF_API}/nodes/{node}/children/?page%5Bsize%5D=50"  # encode [] or OSF returns empty
    out: list[tuple[str, str]] = []
    while url:
        data = None
        for i in range(attempts):
            req = urllib.request.Request(url, headers={
                "Accept": "application/vnd.api+json", "User-Agent": "mne-denoise/1.0"})
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.load(resp)
            except Exception:  # noqa: BLE001
                data = None
            meta = (data or {}).get("meta") or (((data or {}).get("links") or {}).get("meta")) or {}
            total = meta.get("total")
            if data is not None and (data.get("data") or not total):
                break  # got rows, or a genuinely-empty page
            time.sleep(min(60, 8 * (i + 1)))  # OSF throttle / transient-empty backoff
        if not data:
            break
        out.extend((c["id"], c["attributes"]["title"]) for c in data.get("data", []))
        url = (data.get("links") or {}).get("next")
    return out


def _safe(title: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", title).strip("_") or "component"


def download(dest: pathlib.Path, osf_node: str, *, components=None, dry_run=False) -> int:
    dest = pathlib.Path(dest)
    if not osf_node or osf_node.upper().startswith("PENDING"):
        print("ERROR: OSF node id not set.", file=sys.stderr)
        return 2
    if not dry_run and shutil.which("osf") is None:
        print("ERROR: osfclient ('osf') not on PATH. pip install osfclient", file=sys.stderr)
        return 2
    try:
        kids = list_components(osf_node)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: OSF API children query failed: {exc}", file=sys.stderr)
        return 2
    if components:
        want = set(components)
        kids = [(i, t) for (i, t) in kids if t in want or i in want]
    print(f"[erp-core] {len(kids)} component(s) under {osf_node}:")
    for i, t in kids:
        print(f"   {i}  {t}")
    if dry_run:
        return 0
    dest.mkdir(parents=True, exist_ok=True)
    rc = 0
    for i, t in kids:
        sub = dest / _safe(t)
        sub.mkdir(parents=True, exist_ok=True)
        print(f"[erp-core] osf -p {i} clone {sub}   ({t})", flush=True)
        r = subprocess.run(["osf", "-p", i, "clone", str(sub)]).returncode
        if r:
            rc = 1
            print(f"   WARN: component {i} ({t}) returned rc={r}", file=sys.stderr)
    return rc


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dest", required=True)
    p.add_argument("--osf-node", default="thsqg")
    p.add_argument("--components", nargs="*", help="limit to these titles/ids (default: all)")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    return download(pathlib.Path(a.dest), a.osf_node, components=a.components, dry_run=a.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
