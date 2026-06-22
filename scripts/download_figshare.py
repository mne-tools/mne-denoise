#!/usr/bin/env python
"""Download all files of a Figshare article via the public API (stdlib only).

``--article`` is the numeric Figshare id (tail of the DOI, e.g.
10.6084/m9.figshare.16669072 -> article 16669072).  Run on a Fir login node.

    python scripts/download_figshare.py --dest <dir> --article 16669072
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import urllib.request

UA = {"User-Agent": "mne-denoise-benchmark/0.1", "Accept": "application/json"}


def download(dest: pathlib.Path, article: str, dry_run: bool = False) -> int:
    dest = pathlib.Path(dest)
    api = f"https://api.figshare.com/v2/articles/{article}"
    try:
        req = urllib.request.Request(api, headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            meta = json.load(r)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: Figshare API {api} failed: {exc}", file=sys.stderr)
        return 2
    files = meta.get("files", [])
    print(f"[figshare] article {article}: {len(files)} file(s)")
    if dry_run:
        for f in files:
            print(f"   {f.get('name')}  ({f.get('size')} bytes)")
        return 0
    dest.mkdir(parents=True, exist_ok=True)
    rc = 0
    for f in files:
        name = f["name"]
        out = dest / name
        if out.exists() and f.get("size") and out.stat().st_size == f["size"]:
            print(f"   skip {name} (already complete)"); continue
        url = f["download_url"]
        print(f"   downloading {name} ...", flush=True)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA["User-Agent"]})
            with urllib.request.urlopen(req, timeout=120) as resp, open(out, "wb") as fh:
                while chunk := resp.read(1 << 20):
                    fh.write(chunk)
        except Exception as exc:  # noqa: BLE001
            rc = 1
            print(f"   FAIL {name}: {exc}", file=sys.stderr)
    return rc


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dest", required=True)
    p.add_argument("--article", required=True, help="numeric Figshare article id")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    return download(pathlib.Path(a.dest), a.article, a.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
