#!/usr/bin/env python
"""Stage a public Figshare collection with an immutable file manifest.

The downloader uses only the Python standard library and is intended for a
Fir login node, where outbound network access is available. Files are kept in
per-article directories to avoid name collisions and are verified against the
size and MD5 digest published by Figshare.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import pathlib
import time
import urllib.request
from datetime import datetime, timezone
from typing import Any

UA = {"User-Agent": "mne-denoise-benchmark/0.1", "Accept": "application/json"}


def _get_json(url: str, *, attempts: int = 5) -> Any:
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.load(response)
        except Exception as exc:  # noqa: BLE001
            error = exc
            if attempt + 1 < attempts:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Figshare request failed after {attempts} attempts: {url}") from error


def _md5(path: pathlib.Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matches(path: pathlib.Path, metadata: dict[str, Any]) -> bool:
    if not path.is_file() or path.stat().st_size != int(metadata["size"]):
        return False
    expected = metadata.get("computed_md5") or metadata.get("supplied_md5")
    return expected is None or _md5(path).lower() == str(expected).lower()


def _download_one(
    root: pathlib.Path,
    article: dict[str, Any],
    file_metadata: dict[str, Any],
) -> dict[str, Any]:
    article_id = str(article["id"])
    destination = root / "articles" / article_id / file_metadata["name"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    status = "verified_existing"
    if not _matches(destination, file_metadata):
        partial = destination.with_name(destination.name + ".part")
        partial.unlink(missing_ok=True)
        request = urllib.request.Request(
            file_metadata["download_url"], headers={"User-Agent": UA["User-Agent"]}
        )
        with urllib.request.urlopen(request, timeout=180) as response, partial.open(
            "wb"
        ) as stream:
            while chunk := response.read(4 * 1024 * 1024):
                stream.write(chunk)
        if not _matches(partial, file_metadata):
            raise RuntimeError(f"download verification failed: {file_metadata['name']}")
        os.replace(partial, destination)
        status = "downloaded_and_verified"
    return {
        "article_id": int(article_id),
        "article_title": article["title"],
        "file_id": int(file_metadata["id"]),
        "name": file_metadata["name"],
        "size": int(file_metadata["size"]),
        "computed_md5": file_metadata.get("computed_md5"),
        "supplied_md5": file_metadata.get("supplied_md5"),
        "relative_path": destination.relative_to(root).as_posix(),
        "status": status,
    }


def _collection_articles(collection_id: str) -> list[dict[str, Any]]:
    url = (
        f"https://api.figshare.com/v2/collections/{collection_id}/articles"
        "?page_size=1000"
    )
    articles = _get_json(url)
    return sorted(articles, key=lambda item: (item["title"], int(item["id"])))


def stage_collection(
    destination: pathlib.Path,
    collection_id: str,
    *,
    workers: int = 4,
    dry_run: bool = False,
) -> dict[str, Any]:
    destination = destination.resolve()
    collection = _get_json(
        f"https://api.figshare.com/v2/collections/{collection_id}"
    )
    articles = _collection_articles(collection_id)
    expanded = []
    for article in articles:
        metadata = _get_json(article["url_public_api"])
        for file_metadata in metadata.get("files", []):
            expanded.append((article, file_metadata))

    total_bytes = sum(int(file_metadata["size"]) for _, file_metadata in expanded)
    print(
        f"[figshare] collection {collection_id}: {len(articles)} articles, "
        f"{len(expanded)} files, {total_bytes / 2**30:.3f} GiB"
    )
    if dry_run:
        return {
            "collection_id": collection_id,
            "article_count": len(articles),
            "file_count": len(expanded),
            "total_bytes": total_bytes,
        }

    destination.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_download_one, destination, article, file_metadata)
            for article, file_metadata in expanded
        ]
        files = []
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            print(f"[{record['status']}] {record['name']}", flush=True)
            files.append(record)

    files.sort(key=lambda item: (item["article_title"], item["name"]))
    manifest = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "collection_id": collection_id,
        "collection_title": collection.get("title"),
        "collection_doi": collection.get("doi"),
        "article_count": len(articles),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "files": files,
    }
    manifest_path = destination / "figshare_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[figshare] wrote {manifest_path}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--dest", type=pathlib.Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    stage_collection(
        args.dest,
        args.collection,
        workers=args.workers,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
