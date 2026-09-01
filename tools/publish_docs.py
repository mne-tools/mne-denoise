"""Publish one documentation build into the versioned Pages layout."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

_ROOT_REDIRECT = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta http-equiv="refresh" content="0; url=stable/">
    <link rel="canonical" href="https://mne.tools/mne-denoise/stable/">
    <title>mne-denoise documentation</title>
  </head>
  <body>
    <p><a href="stable/">Go to the stable documentation</a></p>
  </body>
</html>
"""
_DEFAULT_VERSIONS: list[dict[str, Any]] = [
    {
        "name": "0.0.2 (dev)",
        "version": "dev",
        "url": "https://mne.tools/mne-denoise/dev/",
    },
    {
        "name": "0.0.1 (stable)",
        "version": "0.0.1",
        "url": "https://mne.tools/mne-denoise/stable/",
        "preferred": True,
    },
]
_LEGACY_ROOT_DIRECTORIES = (
    "_modules",
    "_sources",
    "_static",
    "_images",
    "_downloads",
    "_sphinx_design_static",
    "auto_examples",
    "generated",
)
_LEGACY_ROOT_FILES = (
    "api.html",
    "asr.html",
    "bss_cca.html",
    "citing.html",
    "development.html",
    "dss.html",
    "evaluation.html",
    "getting-started.html",
    "icanclean.html",
    "methods.html",
    "sns.html",
    "sound.html",
    "spectrum_interpolation.html",
    "ssa.html",
    "sspsir.html",
    "zapline.html",
    "genindex.html",
    "py-modindex.html",
    "search.html",
    "searchindex.js",
    "objects.inv",
    ".buildinfo",
    "sg_execution_times.html",
)


def _valid_versions(value: object) -> bool:
    """Check the small switcher document used by the site."""
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(entry, dict) for entry in value)
    ):
        return False
    return (
        sum(isinstance(entry, dict) and bool(entry.get("preferred")) for entry in value)
        == 1
    )


def _read_versions(pages: Path, build: Path) -> list[dict[str, Any]]:
    """Read the checked-in switcher list, falling back to the seed list."""
    candidates = [
        build / "_static" / "versions.json",
        pages / "dev" / "_static" / "versions.json",
        pages / "stable" / "_static" / "versions.json",
    ]
    for path in candidates:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if _valid_versions(value):
            return [dict(entry) for entry in value]
    return [dict(entry) for entry in _DEFAULT_VERSIONS]


def _replace_tree(source: Path, target: Path) -> None:
    """Replace one explicitly selected publication directory."""
    if not source.is_dir():
        raise FileNotFoundError(f"Documentation build not found: {source}")
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)


def _write_versions(
    pages: Path, entries: list[dict[str, Any]], directories: tuple[str, ...]
) -> None:
    """Update switcher files only in selected publication directories."""
    payload = json.dumps(entries, indent=2) + "\n"
    for directory in directories:
        path = pages / directory / "_static" / "versions.json"
        if path.parent.is_dir():
            path.write_text(payload, encoding="utf-8")


def _write_root_redirect(pages: Path, build: Path) -> None:
    """Keep the site root as a small redirect to the stable alias."""
    source = build / "_static" / "root-redirect.html"
    target = pages / "index.html"
    if source.is_file():
        shutil.copy2(source, target)
    else:
        target.write_text(_ROOT_REDIRECT, encoding="utf-8")


def migrate_legacy_root(pages: Path) -> None:
    """Remove the old unversioned Sphinx site after stable is available."""
    if not (pages / "stable" / "index.html").is_file():
        raise FileNotFoundError(
            "Cannot migrate the legacy root before stable documentation exists"
        )

    for name in (*_LEGACY_ROOT_DIRECTORIES, *_LEGACY_ROOT_FILES):
        path = pages / name
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        elif path.exists() or path.is_symlink():
            path.unlink()

    (pages / "index.html").write_text(_ROOT_REDIRECT, encoding="utf-8")


def publish_development(pages: Path, build: Path) -> None:
    """Publish a main-branch build under ``/dev/``."""
    entries = _read_versions(pages, build)
    _replace_tree(build, pages / "dev")
    _write_versions(pages, entries, ("dev",))
    if (pages / "stable" / "index.html").is_file():
        _write_root_redirect(pages, build)
    (pages / ".nojekyll").touch()


def publish_release(pages: Path, build: Path, release_version: str) -> None:
    """Publish one release build and copy it to the stable alias."""
    entries = _read_versions(pages, build)
    if not any(
        entry.get("version") == release_version and entry.get("preferred")
        for entry in entries
    ):
        raise ValueError(
            "docs/_static/versions.json must mark the published release as "
            "the preferred stable version"
        )
    _replace_tree(build, pages / release_version)
    _replace_tree(build, pages / "stable")
    _write_versions(pages, entries, ("dev", "stable", release_version))
    _write_root_redirect(pages, build)
    (pages / ".nojekyll").touch()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--target", choices=("dev", "release"), required=True)
    parser.add_argument(
        "--version",
        help="Release version or tag, required when publishing a release.",
    )
    parser.add_argument(
        "--migrate-legacy-root",
        action="store_true",
        help="Remove the old unversioned Sphinx site after a release publish.",
    )
    args = parser.parse_args()
    if args.target == "release":
        if args.version is None:
            parser.error("--version is required for a release")
        if args.version.startswith("v"):
            args.version = args.version[1:]
        if not re.fullmatch(r"\d+\.\d+\.\d+(?:[A-Za-z0-9.+-]*)?", args.version):
            parser.error("--version must look like X.Y.Z or vX.Y.Z")
    elif args.version is not None:
        parser.error("--version is only valid for a release")
    if args.migrate_legacy_root and args.target != "release":
        parser.error("--migrate-legacy-root is only valid for a release")
    return args


def main() -> None:
    """Run the selected publication operation."""
    args = _parse_args()
    pages = args.pages.resolve()
    build = args.build.resolve()
    if args.target == "dev":
        publish_development(pages, build)
    else:
        publish_release(pages, build, args.version)
        if args.migrate_legacy_root:
            migrate_legacy_root(pages)


if __name__ == "__main__":
    main()
