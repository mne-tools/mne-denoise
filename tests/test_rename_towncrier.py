"""Tests for the local Towncrier fragment helper."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "rename_towncrier.py"
CONFIG = """
[tool.towncrier]
directory = "docs/changes/devel/"

[[tool.towncrier.type]]
directory = "feature"

[[tool.towncrier.type]]
directory = "bugfix"

[[tool.towncrier.type]]
directory = "doc"

[[tool.towncrier.type]]
directory = "removal"

[[tool.towncrier.type]]
directory = "misc"
"""


def _prepare(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(CONFIG, encoding="utf-8")
    (tmp_path / "docs" / "changes" / "devel").mkdir(parents=True, exist_ok=True)


def _run(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    _prepare(tmp_path)
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )


def test_rename_misc_fragment(tmp_path: Path) -> None:
    _prepare(tmp_path)
    fragment = tmp_path / "docs" / "changes" / "devel" / "misc.rst"
    fragment.write_text("Tooling maintenance.\n", encoding="utf-8")

    result = _run(tmp_path, "--pr-number", "123")

    assert result.returncode == 0
    assert not fragment.exists()
    assert (fragment.parent / "123.misc.rst").read_text(encoding="utf-8") == (
        "Tooling maintenance.\n"
    )


def test_rename_succeeds_without_fragments(tmp_path: Path) -> None:
    result = _run(tmp_path, "--pr-number", "123")

    assert result.returncode == 0
    assert "No unnumbered" in result.stdout


def test_rename_refuses_to_overwrite(tmp_path: Path) -> None:
    _prepare(tmp_path)
    fragment_dir = tmp_path / "docs" / "changes" / "devel"
    (fragment_dir / "misc.rst").write_text("new\n", encoding="utf-8")
    (fragment_dir / "123.misc.rst").write_text("old\n", encoding="utf-8")

    result = _run(tmp_path, "--pr-number", "123")

    assert result.returncode == 1
    assert "Refusing to overwrite" in result.stderr
    assert (fragment_dir / "misc.rst").exists()
