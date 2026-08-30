# Release checklist

Package versions are derived from Git tags via hatch-vcs.

## 1. Prepare Release

- [ ] **Update Dependencies**:
    - Check for dependency updates.
    - Run `python -m pip install --upgrade pip` and `python -m pip install -e . --group dev` to ensure environment is fresh.
- [ ] **Run Checks**:
    - [ ] `ruff check .`
    - [ ] `ruff format .`
    - [ ] `pytest` (ensure all pass)
    - [ ] `make -C docs html` (ensure no warnings)
- [ ] **Update Changelog**:
    - Update `CHANGELOG.md` with release date and summary.

## 2. Tag and Publish (GitHub)

- [ ] **Tag Release**:
    - `git tag vX.Y.Z`
    - `git push upstream main`
    - `git push upstream vX.Y.Z`

## 3. Verify Release (GitHub & PyPI)

- [ ] **Verify CI/CD**:
    - Watch functionality of `ci.yml`.
    - Watch `release.yml` (triggered by tag).
    - **PyPI Publishing**: Verify the new version appears on PyPI (automated via Trusted Publishing).

## 4. Post-Release Maintenance

- [ ] **Conda-Forge**:
    - Wait for the regro-cf-autotick-bot to open a PR on the feedstock.
    - Merge the PR to update the conda package.
