#!/usr/bin/env python
"""Robust OSF project downloader via the OSF API (stdlib only; no osfclient).

osfclient `clone` is fragile on large projects (dies on a transient HTTP 500, or
``KeyError: 'name'`` on odd entries, with no retry/resume).  This walks osfstorage
through the API instead: recurses folders, descends into child components, retries
each file on 5xx/timeout, skips malformed entries, and skips files already present
at the right size (resume).  Run on a Fir login node (internet).

    python scripts/download_osf.py --dest <dir> --node r7s9b              # single project
    python scripts/download_osf.py --dest <dir> --node ktv7m --dry-run    # incl. child components
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

API = "https://api.osf.io/v2"
UA = {"User-Agent": "mne-denoise-benchmark/0.1"}


def _get_json(url: str, tries: int = 5):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return json.load(r)
        except Exception:  # noqa: BLE001
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))


def _walk(listing_url: str):
    """Yield (relpath, download_url, size) under an osfstorage listing, recursing folders."""
    url = listing_url
    while url:
        d = _get_json(url)
        for it in d["data"]:
            a = it.get("attributes", {})
            name = a.get("name")
            if not name:
                continue  # malformed entry (the osfclient KeyError cause) — skip
            rel = (a.get("materialized_path") or f"/{name}").lstrip("/")
            if a.get("kind") == "file":
                dl = it.get("links", {}).get("download")
                if dl:
                    yield rel, dl, a.get("size") or 0
            elif a.get("kind") == "folder":
                sub = it["relationships"]["files"]["links"]["related"]["href"]
                yield from _walk(sub)
        url = d.get("links", {}).get("next")


def _collect(node: str):
    """All files for a node: root files at top level, each child component under its title/."""
    items = list(_walk(f"{API}/nodes/{node}/files/osfstorage/"))
    children = _get_json(f"{API}/nodes/{node}/children/")["data"]
    for c in children:
        title = c["attributes"]["title"].strip().replace("/", "_")
        for rel, dl, sz in _walk(f"{API}/nodes/{c['id']}/files/osfstorage/"):
            items.append((f"{title}/{rel}", dl, sz))
    return items, len(children)


def _download_file(url: str, out: pathlib.Path, tries: int = 5) -> bool:
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=600) as resp, \
                 open(out, "wb") as fh:
                while chunk := resp.read(1 << 20):
                    fh.write(chunk)
            return True
        except Exception:  # noqa: BLE001
            time.sleep(3 * (i + 1))
    return False


def download(dest: pathlib.Path, node: str, dry_run: bool = False) -> int:
    dest = pathlib.Path(dest)
    if not node or node.upper().startswith("PENDING"):
        print("ERROR: OSF node id not set.", file=sys.stderr)
        return 2
    try:
        files, n_comp = _collect(node)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: OSF API walk failed: {exc}", file=sys.stderr)
        return 2
    tot = sum(s for _, _, s in files)
    print(f"[osf] node {node} (+{n_comp} components): {len(files)} files ~{tot/1e9:.2f} GB")
    if dry_run:
        for rel, _, s in files[:15]:
            print(f"   {rel}  {s/1e6:.1f} MB")
        return 0
    if not files:
        print("ERROR: no files found", file=sys.stderr)
        return 1
    rc, done = 0, 0
    for rel, dl, size in files:
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists() and size and out.stat().st_size == size:
            continue
        if _download_file(dl, out):
            done += 1
        else:
            rc = 1
            print(f"   FAIL {rel}", file=sys.stderr)
    print(f"[osf] downloaded {done} new files (rc={rc})")
    return rc


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
