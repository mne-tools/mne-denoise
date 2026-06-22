#!/usr/bin/env python
"""Robust OSF downloader via the WaterButler API (files.osf.io); stdlib only.

OSF data often lives on addon storage (EEGEyeNet ktv7m keeps its data on a
**googledrive** addon under ``EEGEyeNet-Data/``; its dropbox addon is an empty
duplicate).  The metadata API (api.osf.io) lists addon *folders* but not their
contents, and osfclient dies on a ``KeyError: 'name'``.  WaterButler
(files.osf.io/v1) actually serves addon files, so we walk that: enumerate every
provider, recurse folders, skip malformed entries, retry on 5xx/timeout, build a
frozen JSON manifest, then download with resume (skip files already at size).

    python scripts/download_osf.py --dest <dir> --node ktv7m --dry-run   # build+print manifest only
    python scripts/download_osf.py --dest <dir> --node r7s9b             # download (osfstorage or addons)
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

V2 = "https://api.osf.io/v2"
WB = "https://files.osf.io/v1/resources"
UA = {"User-Agent": "mne-denoise-benchmark/0.1"}


def _get_json(url: str, tries: int = 5):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (404, 410):       # provider/path absent — caller decides
                raise
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))
        except Exception:                  # noqa: BLE001
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))


def _providers(node: str) -> list[str]:
    d = _get_json(f"{V2}/nodes/{node}/files/")
    return [p["attributes"]["provider"] for p in d["data"]]


def _walk(node: str, provider: str, path: str = "/", prefix: str = ""):
    """Yield (relpath, download_url, size) under a WaterButler provider path."""
    try:
        items = _get_json(f"{WB}/{node}/providers/{provider}{path}")["data"]
    except urllib.error.HTTPError:
        return                              # broken/absent addon path — skip
    for it in items:
        a = it.get("attributes", {})
        name = a.get("name")
        if not name:
            continue                        # malformed entry (the osfclient KeyError cause)
        mat = (a.get("materialized") or f"{prefix}/{name}").lstrip("/")
        if a.get("kind") == "file":
            dl = (it.get("links", {}) or {}).get("download") \
                or f"{WB}/{node}/providers/{provider}{a['path']}"
            try:
                sz = int(a.get("size") or 0)
            except (TypeError, ValueError):
                sz = 0
            yield mat, dl, sz
        elif a.get("kind") == "folder":
            yield from _walk(node, provider, a["path"], "/" + mat)


def _collect(node: str, prefix: str = ""):
    items, n_comp = [], 0
    for prov in _providers(node):
        for rel, dl, sz in _walk(node, prov):
            items.append((prefix + rel, dl, sz))
    for c in _get_json(f"{V2}/nodes/{node}/children/")["data"]:
        n_comp += 1
        title = c["attributes"]["title"].strip().replace("/", "_")
        sub, _ = _collect(c["id"], prefix=f"{prefix}{title}/")
        items += sub
    return items, n_comp


def _download_file(url: str, out: pathlib.Path, tries: int = 5) -> bool:
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=900) as resp, \
                 open(out, "wb") as fh:
                while chunk := resp.read(1 << 20):
                    fh.write(chunk)
            return True
        except Exception:                   # noqa: BLE001
            time.sleep(3 * (i + 1))
    return False


def download(dest: pathlib.Path, node: str, dry_run: bool = False) -> int:
    dest = pathlib.Path(dest)
    if not node or node.upper().startswith("PENDING"):
        print("ERROR: OSF node id not set.", file=sys.stderr)
        return 2
    try:
        files, n_comp = _collect(node)
    except Exception as exc:                # noqa: BLE001
        print(f"ERROR: OSF/WaterButler walk failed: {exc}", file=sys.stderr)
        return 2
    # de-dup (same relpath from a broken duplicate addon) keeping the larger size
    best: dict[str, tuple[str, int]] = {}
    for rel, dl, sz in files:
        if rel not in best or sz > best[rel][1]:
            best[rel] = (dl, sz)
    tot = sum(s for _, s in best.values())
    print(f"[osf] node {node} (+{n_comp} components): {len(best)} files ~{tot/1e9:.2f} GB")
    manifest = [{"path": rel, "size": sz} for rel, (_, sz) in sorted(best.items())]
    if dry_run:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "osf_manifest.json").write_text(json.dumps(manifest, indent=1))
        for rel, (_, sz) in list(sorted(best.items()))[:15]:
            print(f"   {rel}  {sz/1e6:.1f} MB")
        print(f"[osf] manifest -> {dest/'osf_manifest.json'} ({len(best)} entries)")
        return 0
    if not best:
        print("ERROR: no files found (all providers empty?)", file=sys.stderr)
        return 1
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "osf_manifest.json").write_text(json.dumps(manifest, indent=1))
    rc, done = 0, 0
    for rel, (dl, size) in sorted(best.items()):
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
