"""Tests for the versioned documentation publisher."""

from __future__ import annotations

import json
from pathlib import Path

from tools.publish_docs import migrate_legacy_root, publish_development, publish_release


def _make_build(
    path: Path, marker: str, versions: list[dict[str, object]] | None = None
) -> Path:
    """Create a minimal documentation build tree."""
    path.mkdir()
    (path / "index.html").write_text(marker, encoding="utf-8")
    static = path / "_static"
    static.mkdir()
    if versions is not None:
        (static / "versions.json").write_text(json.dumps(versions), encoding="utf-8")
    return path


def _release_versions(version: str) -> list[dict[str, object]]:
    """Return a switcher list with one preferred release."""
    return [
        {"name": "development", "version": "dev", "url": "dev/"},
        {
            "name": f"{version} (stable)",
            "version": version,
            "url": "stable/",
            "preferred": True,
        },
    ]


def test_development_publish_before_stable_preserves_root(tmp_path: Path) -> None:
    """Development publishing must not redirect to a missing stable site."""
    pages = tmp_path / "pages"
    pages.mkdir()
    old_index = pages / "index.html"
    old_index.write_text("old root site", encoding="utf-8")
    build = _make_build(tmp_path / "build", "development")

    publish_development(pages, build)

    assert (pages / "dev" / "index.html").read_text(encoding="utf-8") == ("development")
    assert old_index.read_text(encoding="utf-8") == "old root site"
    assert not (pages / "stable").exists()


def test_development_publish_after_stable_updates_root_only(tmp_path: Path) -> None:
    """Development publishing replaces dev and redirects once stable exists."""
    pages = tmp_path / "pages"
    pages.mkdir()
    (pages / "dev").mkdir()
    (pages / "dev" / "index.html").write_text("old dev", encoding="utf-8")
    (pages / "stable").mkdir()
    (pages / "stable" / "index.html").write_text("stable", encoding="utf-8")
    build = _make_build(tmp_path / "build", "new development")

    publish_development(pages, build)

    assert (pages / "dev" / "index.html").read_text(encoding="utf-8") == (
        "new development"
    )
    assert (pages / "stable" / "index.html").read_text(encoding="utf-8") == ("stable")
    assert "stable/" in (pages / "index.html").read_text(encoding="utf-8")


def test_release_publish_preserves_development(tmp_path: Path) -> None:
    """Release publishing creates both release aliases and keeps dev."""
    pages = tmp_path / "pages"
    pages.mkdir()
    (pages / "dev").mkdir()
    (pages / "dev" / "index.html").write_text("development", encoding="utf-8")
    version = "1.2.3"
    build = _make_build(
        tmp_path / "build", "release", versions=_release_versions(version)
    )

    publish_release(pages, build, version)

    assert (pages / version / "index.html").read_text(encoding="utf-8") == ("release")
    assert (pages / "stable" / "index.html").read_text(encoding="utf-8") == ("release")
    assert (pages / "dev" / "index.html").read_text(encoding="utf-8") == ("development")
    assert "stable/" in (pages / "index.html").read_text(encoding="utf-8")


def test_initial_migration_removes_legacy_root_only(tmp_path: Path) -> None:
    """The explicit migration removes old root artifacts and preserves pages."""
    pages = tmp_path / "pages"
    pages.mkdir()
    (pages / "dev").mkdir()
    (pages / "dev" / "index.html").write_text("development", encoding="utf-8")
    (pages / "9.9.9").mkdir()
    (pages / "9.9.9" / "index.html").write_text("archived", encoding="utf-8")
    (pages / ".nojekyll").write_text("", encoding="utf-8")
    (pages / "CNAME").write_text("mne.tools", encoding="utf-8")
    (pages / "site-metadata.json").write_text("{}", encoding="utf-8")
    (pages / "index.html").write_text("old root", encoding="utf-8")

    legacy_directories = (
        "_modules",
        "_sources",
        "_static",
        "_images",
        "_downloads",
        "_sphinx_design_static",
        "auto_examples",
        "generated",
    )
    for name in legacy_directories:
        (pages / name).mkdir()
    legacy_files = (
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
    )
    for name in legacy_files:
        (pages / name).write_text("legacy", encoding="utf-8")

    build = _make_build(
        tmp_path / "build",
        "v0.0.1",
        versions=_release_versions("0.0.1"),
    )
    publish_release(pages, build, "0.0.1")
    migrate_legacy_root(pages)

    assert "stable/" in (pages / "index.html").read_text(encoding="utf-8")
    assert (pages / "dev" / "index.html").read_text(encoding="utf-8") == ("development")
    assert (pages / "9.9.9" / "index.html").read_text(encoding="utf-8") == ("archived")
    assert (pages / ".nojekyll").is_file()
    assert (pages / "CNAME").read_text(encoding="utf-8") == "mne.tools"
    assert (pages / "site-metadata.json").read_text(encoding="utf-8") == "{}"
    for name in (*legacy_directories, *legacy_files):
        assert not (pages / name).exists()
