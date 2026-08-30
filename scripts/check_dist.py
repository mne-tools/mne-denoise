"""Validate the structure and metadata of mne-denoise distributions."""

from __future__ import annotations

import argparse
import tarfile
from email.parser import Parser
from pathlib import Path
from zipfile import ZipFile

from packaging.requirements import Requirement
from packaging.utils import parse_sdist_filename, parse_wheel_filename
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


def _check_wheel(wheel: Path) -> Version:
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

    return filename_version


def _check_sdist(sdist: Path) -> Version:
    """Validate the minimum source files included in the sdist."""
    project_name, filename_version = parse_sdist_filename(sdist.name)
    assert str(project_name) == "mne-denoise"

    with tarfile.open(sdist, mode="r:gz") as archive:
        members = archive.getmembers()
        names = [member.name.rstrip("/") for member in members]
        roots = {name.split("/", 1)[0] for name in names if name}
        assert len(roots) == 1, f"sdist has unexpected top-level roots: {roots}"
        root = next(iter(roots))

        pkg_info_members = [
            member for member in members if member.name == f"{root}/PKG-INFO"
        ]
        assert len(pkg_info_members) == 1, (
            f"expected one {root}/PKG-INFO member, found {pkg_info_members}"
        )
        pkg_info_member = pkg_info_members[0]
        assert pkg_info_member.isfile(), f"{pkg_info_member.name} is not a regular file"
        pkg_info_file = archive.extractfile(pkg_info_member)
        assert pkg_info_file is not None, f"could not read {pkg_info_member.name}"
        pkg_info = Parser().parsestr(pkg_info_file.read().decode("utf-8"))
        assert pkg_info["Name"] == "mne-denoise"
        assert pkg_info["Version"] is not None
        assert Version(pkg_info["Version"]) == filename_version

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

    return filename_version


def check_distribution(dist_dir: Path, expected_version: Version | None = None) -> None:
    """Validate the wheel and source distribution in ``dist_dir``."""
    assert dist_dir.is_dir(), f"distribution directory does not exist: {dist_dir}"
    wheel = _single_artifact(dist_dir, ".whl")
    sdist = _single_artifact(dist_dir, ".tar.gz")
    wheel_version = _check_wheel(wheel)
    sdist_version = _check_sdist(sdist)
    assert wheel_version == sdist_version, (
        f"wheel/sdist version mismatch: {wheel.name} has {wheel_version}, "
        f"{sdist.name} has {sdist_version}"
    )
    if expected_version is not None:
        assert wheel_version == expected_version, (
            f"expected version {expected_version}, found {wheel_version}"
        )
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
    parser.add_argument(
        "--expected-version",
        type=Version,
        help="require both artifacts to have this semantic version",
    )
    args = parser.parse_args()
    check_distribution(args.dist_dir, expected_version=args.expected_version)


if __name__ == "__main__":
    main()
