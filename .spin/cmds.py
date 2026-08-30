"""Project commands exposed through Spin."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import click


def _run(
    *command: str,
    args: tuple[str, ...] = (),
    env: dict[str, str] | None = None,
) -> None:
    """Run a project command and propagate its exit status."""
    subprocess.run([*command, *args], check=True, env=env)


@click.command(context_settings={"ignore_unknown_options": True})
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def test(args: tuple[str, ...]) -> None:
    """Run the test suite, optionally forwarding pytest arguments."""
    _run("pytest", "-q", args=args)


@click.command()
def lint() -> None:
    """Run all repository hooks."""
    _run("prek", "run", "--all-files")


@click.command()
def docs() -> None:
    """Build the documentation with warnings treated as errors."""
    with tempfile.TemporaryDirectory(prefix="mne-denoise-docs-") as temp_dir:
        temp_root = Path(temp_dir)
        environment = os.environ.copy()
        environment.update(
            {
                "MPLBACKEND": "Agg",
                "MPLCONFIGDIR": str(temp_root / "mplconfig"),
                "HOME": str(temp_root / "home"),
                "MNE_HOME": str(temp_root / "mne"),
                "NUMBA_CACHE_DIR": str(temp_root / "numba"),
                "MNE_DONTWRITE_HOME": "true",
            }
        )
        for directory in ("mplconfig", "home", "mne", "numba"):
            (temp_root / directory).mkdir()
        _run(
            sys.executable,
            "-m",
            "sphinx",
            "-b",
            "html",
            "-W",
            "--keep-going",
            "docs",
            "docs/_build/html",
            env=environment,
        )


@click.command()
def build() -> None:
    """Build and validate clean Python distribution artifacts."""
    dist = Path("dist")
    if dist.is_dir():
        shutil.rmtree(dist)
    elif dist.exists():
        dist.unlink()
    _run(sys.executable, "-m", "build")
    artifacts = tuple(str(path) for path in sorted(dist.iterdir()))
    _run(sys.executable, "-m", "twine", "check", "--strict", args=artifacts)
    _run(sys.executable, "scripts/check_dist.py", "dist")


@click.command()
def check() -> None:
    """Run hooks, tests, and distribution validation."""
    _run(sys.executable, "-m", "spin", "lint")
    _run(sys.executable, "-m", "spin", "test")
    _run(sys.executable, "-m", "spin", "build")
