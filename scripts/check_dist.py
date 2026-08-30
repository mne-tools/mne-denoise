"""Validate the structure and metadata of mne-denoise distributions."""

from __future__ import annotations

import argparse
import tarfile
from email.parser import Parser
from pathlib import Path
from zipfile import ZipFile

from packaging.requirements import Requirement
from packaging.utils import parse_wheel_filename
from packaging.version import Version

CORE_REQUIREMENTS = {"numpy", "scipy", "scikit-learn", "joblib"}
OPTIONAL_REQUIREMENTS = {
    "mne": "mne",
    "progress": "tqdm",
    "viz": "matplotlib",
}
DEVELOPMENT_REQUIREMENTS = {
    "pytest",
    "sphinx",
    "ruff",
    "pre-commit",
    "mypy",
    "build",
    "twine",
    "towncrier",
}
PUBLISHED_EXTRAS = set(OPTIONAL_REQUIREMENTS)
REPOSITORY_TREES = {
    ".github/",
    "docs/",
    "examples/",
    "scripts/",
    "tests/",
}


def _single_artifact(dist_dir: Path, suffix: str) -> Path:
    """Return the only distribution artifact with the requested suffix."""
    artifacts = sorted(
        path
        for path in dist_dir.iterdir()
        if path.is_file() and path.name.endswith(suffix)
    )
    assert len(artifacts) == 1, (
        f"expected exactly one {suffix} artifact in {dist_dir}, found {artifacts}"
    )
    return artifacts[0]


def _wheel_metadata(archive: ZipFile) -> tuple[str, object]:
    """Return the wheel metadata member and parsed email metadata."""
    metadata_members = [
        name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
    ]
    assert len(metadata_members) == 1, (
        f"expected one wheel METADATA member, found {metadata_members}"
    )
    metadata_name = metadata_members[0]
    metadata = Parser().parsestr(archive.read(metadata_name).decode("utf-8"))
    return metadata_name, metadata


def _check_wheel(wheel: Path) -> None:
    """Validate wheel contents and core distribution metadata."""
    project_name, filename_version, _, tags = parse_wheel_filename(wheel.name)
    assert str(project_name) == "mne-denoise"
    assert {str(tag) for tag in tags} == {"py3-none-any"}

    with ZipFile(wheel) as archive:
        names = set(archive.namelist())
        assert "mne_denoise/__init__.py" in names
        assert not any(
            name == tree.rstrip("/") or name.startswith(tree)
            for name in names
            for tree in REPOSITORY_TREES
        )
        assert all(
            name.startswith("mne_denoise/") or ".dist-info/" in name for name in names
        )

        _, metadata = _wheel_metadata(archive)
        assert metadata["Name"] == "mne-denoise"
        assert metadata["Requires-Python"] == ">=3.11"
        assert metadata["License-Expression"] == "BSD-3-Clause"
        assert "LICENSE" in metadata.get_all("License-File", [])

        extras = set(metadata.get_all("Provides-Extra", []))
        assert extras == PUBLISHED_EXTRAS, extras

        requirements = [
            Requirement(value) for value in metadata.get_all("Requires-Dist", [])
        ]
        base_requirements = {
            requirement.name.lower()
            for requirement in requirements
            if requirement.marker is None
        }
        assert base_requirements == CORE_REQUIREMENTS, (
            f"unexpected base requirements: {base_requirements}"
        )

        for extra, expected_requirement in OPTIONAL_REQUIREMENTS.items():
            extra_requirements = {
                requirement.name.lower()
                for requirement in requirements
                if requirement.marker is not None
                and requirement.marker.evaluate({"extra": extra})
            }
            assert extra_requirements == {expected_requirement}, (
                f"unexpected requirements for {extra!r}: {extra_requirements}"
            )

        leaked_requirements = {
            requirement.name.lower()
            for requirement in requirements
            if requirement.name.lower() in DEVELOPMENT_REQUIREMENTS
        }
        assert not leaked_requirements, leaked_requirements

        assert Version(metadata["Version"]) == Version(str(filename_version))

        author_metadata = "\n".join(
            value
            for field in ("Author", "Author-email", "Maintainer", "Maintainer-email")
            for value in metadata.get_all(field, [])
        )
        for author in ("Sina Esmaeili", "Hamza Abdelhedi"):
            assert author in author_metadata


def _check_sdist(sdist: Path) -> None:
    """Validate the minimum source files included in the sdist."""
    with tarfile.open(sdist, mode="r:gz") as archive:
        names = [member.name.rstrip("/") for member in archive.getmembers()]
    roots = {name.split("/", 1)[0] for name in names if name}
    assert len(roots) == 1, f"sdist has unexpected top-level roots: {roots}"
    root = next(iter(roots))
    prefix = f"{root}/"
    relative_names = {
        name.removeprefix(prefix) for name in names if name.startswith(prefix)
    }

    required_files = {
        "pyproject.toml",
        "README.md",
        "LICENSE",
        "mne_denoise/__init__.py",
    }
    assert required_files <= relative_names, (
        f"sdist is missing: {required_files - relative_names}"
    )
    assert any(name == "tests" or name.startswith("tests/") for name in relative_names)


def check_distribution(dist_dir: Path) -> None:
    """Validate the wheel and source distribution in ``dist_dir``."""
    assert dist_dir.is_dir(), f"distribution directory does not exist: {dist_dir}"
    wheel = _single_artifact(dist_dir, ".whl")
    sdist = _single_artifact(dist_dir, ".tar.gz")
    _check_wheel(wheel)
    _check_sdist(sdist)
    print(f"Validated {wheel.name} and {sdist.name}")


def main() -> None:
    """Parse the distribution directory and run all checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dist_dir",
        nargs="?",
        type=Path,
        default=Path("dist"),
        help="directory containing exactly one wheel and one sdist (default: dist)",
    )
    args = parser.parse_args()
    check_distribution(args.dist_dir)


if __name__ == "__main__":
    main()
