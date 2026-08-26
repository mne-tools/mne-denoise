#!/usr/bin/env python
"""Build a deterministic dataset manifest without loading numerical recordings.

Use ``--hash-mode all`` for locked datasets such as BETA and ds004784 repeat 2.
For very large versioned public datasets, ``metadata`` records the complete relative
path/size inventory and hashes the BIDS/portal metadata that identifies the snapshot.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import pathlib
from datetime import datetime, timezone

METADATA_NAMES = {
    "dataset_description.json",
    "participants.tsv",
    "participants.json",
    "README",
    "README.md",
    "CHANGES",
    "SHA256SUMS",
    "Freq_Phase.mat",
    "Description.pdf",
}


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fid:
        for block in iter(lambda: fid.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(
    root: pathlib.Path,
    *,
    dataset_id: str,
    version: str,
    doi: str | None,
    license_: str | None,
    hash_mode: str,
    include: tuple[str, ...] = (),
    subjects: tuple[str, ...] = (),
) -> dict:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {root}")
    files = []
    total_bytes = 0
    candidates = sorted(p for p in root.rglob("*") if p.is_file())
    if include:
        candidates = [
            path for path in candidates
            if any(fnmatch.fnmatch(path.relative_to(root).as_posix(), pattern) for pattern in include)
        ]
        if not candidates:
            raise FileNotFoundError(
                f"no files under {root} matched include patterns {list(include)!r}"
            )
    for path in candidates:
        rel = path.relative_to(root).as_posix()
        size = path.stat().st_size
        total_bytes += size
        record = {"path": rel, "size_bytes": size}
        should_hash = hash_mode == "all" or (
            hash_mode == "metadata"
            and (path.name in METADATA_NAMES or path.suffix.lower() in {".json", ".tsv"})
        )
        if should_hash:
            record["sha256"] = _sha256(path)
        files.append(record)
    if subjects:
        # Datasets that do not use BIDS sub-* directories (PhysioNet case folders,
        # per-subject archives, recording-level units) carry their unit list here, so
        # the denominator is fixed in the manifest and hashed with it rather than
        # being re-derived by whatever happens to be on disk at run time.
        declared = list(subjects)
        missing = [s for s in declared
                   if not any(f["path"].split("/")[0] == s or s in f["path"] for f in files)]
        if missing:
            raise FileNotFoundError(
                f"declared subjects absent from {root}: {missing}"
            )
    else:
        declared = sorted(p.name for p in root.glob("sub-*") if p.is_dir())
    subjects = declared
    payload = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "version": version,
        "doi": doi,
        "license": license_,
        "root_at_manifest_creation": str(root),
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "hash_mode": hash_mode,
        "include_patterns": list(include),
        "subjects": subjects,
        "file_count": len(files),
        "total_bytes": total_bytes,
        "files": files,
    }
    # The content fingerprint must be portable across computers and stable across
    # repeated inventory runs.  The absolute audit path and creation timestamp are
    # useful provenance, but neither identifies the dataset contents.
    fingerprint_payload = {
        key: value
        for key, value in payload.items()
        if key not in {"generated_at_utc", "root_at_manifest_creation"}
    }
    canonical = json.dumps(
        fingerprint_payload, sort_keys=True, separators=(",", ":"), default=str
    )
    payload["content_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--doi")
    parser.add_argument("--license", dest="license_")
    parser.add_argument("--hash-mode", choices=("metadata", "all"), default="metadata")
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        metavar="GLOB",
        help="Only inventory relative paths matching this glob; repeat as needed.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=[],
        metavar="UNIT",
        help="Declare the unit list explicitly for datasets without BIDS sub-* directories. "
             "Each must be present under the root or the build fails.",
    )
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)
    payload = build_manifest(
        args.root,
        dataset_id=args.dataset_id,
        version=args.version,
        doi=args.doi,
        license_=args.license_,
        hash_mode=args.hash_mode,
        include=tuple(args.include),
        subjects=tuple(args.subjects),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"wrote {args.output}: {payload['file_count']} files, "
        f"{payload['total_bytes']} bytes, hash={payload['content_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
