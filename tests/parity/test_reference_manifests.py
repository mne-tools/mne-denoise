"""Integrity checks for externally generated ASR reference fixtures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REFERENCE_DIR = Path(__file__).parent / "matlab_reference"
MANIFESTS = tuple(sorted(REFERENCE_DIR.glob("*asr_reference_manifest.json")))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.mark.parametrize("manifest_path", MANIFESTS, ids=lambda path: path.stem)
def test_reference_fixture_hashes_match_manifest(manifest_path: Path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared = manifest.get("files", manifest.get("fixtures"))
    assert declared, f"{manifest_path.name} declares no fixture hashes"

    for filename, expected_hash in declared.items():
        fixture_path = REFERENCE_DIR / filename
        assert fixture_path.is_file(), f"Missing declared fixture: {filename}"
        assert _sha256(fixture_path) == expected_hash
