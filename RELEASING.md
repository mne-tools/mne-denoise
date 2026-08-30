# Releasing mne-denoise

`mne-denoise` gets its package version from Git tags through `hatch-vcs`. The
canonical release tag is `vX.Y.Z`, for example `v0.0.2` or `v0.0.2rc1`.

Pushing a tag is safe: a tag push alone does **not** publish to PyPI. The
release workflow starts publishing only when the corresponding GitHub Release
is deliberately published.

## Release architecture

The release workflow (`.github/workflows/release.yml`) runs its read-only
`Package` job on pull requests to `main`, pushes to `main`, and published
GitHub Releases. On a published GitHub Release it performs this flow:

```text
published GitHub Release
        ↓
Package: checkout the tagged commit
        ↓
python -m build
        ↓
twine check + check_dist.py + clean wheel installation
        ↓
Actions artifact: release-dists
        ↓
Publish to PyPI: download release-dists
        ↓
PyPI Trusted Publishing / OIDC
```

The wheel and sdist are built once. The PyPI job downloads those exact
validated files and does not check out the repository or rebuild anything.
PyPI is the canonical distribution location; package files are not uploaded
to the GitHub Release.

## Before the release

1. Ensure `main` is current and clean.

   ```bash
   git checkout main
   git pull --ff-only
   git status --short
   ```

2. Confirm that the required Tests and Docs CI checks are green for the
   release commit.

3. Choose the release version `X.Y.Z`.

4. Build and inspect the Towncrier changelog locally. Install the changelog
   group if needed, then run:

   ```bash
   python -m pip install --group changelog
   towncrier build --version X.Y.Z
   git diff -- CHANGELOG.md docs/changes/devel/
   ```

   Review the generated `CHANGELOG.md` and confirm that the intended release
   fragments were consumed. Commit the changelog before tagging.

5. Run the final local distribution validation:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install --group build
   rm -rf dist/
   python -m build
   python -m twine check --strict dist/*
   python scripts/check_dist.py dist
   ```

   Plain `python -m build` is intentional: it builds the sdist first and then
   builds the wheel from the extracted sdist, checking that the sdist is
   complete. Do not replace it with `python -m build --sdist --wheel`.

## Tag the release

After the release changelog is committed and checks are green, create and push
the canonical tag:

```bash
git tag -a vX.Y.Z -m "mne-denoise X.Y.Z"
git push upstream vX.Y.Z
```

Use the project repository remote for `upstream` if the local remote has a
different name. Do not describe this tag push as a publication step. It only
makes the release commit available for review.

## Create and publish the GitHub Release

Create a draft GitHub Release for exactly the tag that was pushed. This can be
done in the GitHub UI or with:

```bash
gh release create vX.Y.Z \
  --draft \
  --generate-notes \
  --title "vX.Y.Z"
```

Before publishing the draft, review:

- the tag and target commit;
- the release title; and
- the generated and manually edited release notes.

Do not upload wheel or sdist files to the GitHub Release. When the draft is
correct, publish the GitHub Release. That deliberate publication is the only
release event that starts the PyPI publishing job. CI does not create, edit,
publish, or attach assets to the GitHub Release.

A draft release may be edited while it is being reviewed. After publication,
do not move or rewrite its tag or replace the package version in place. If a
release is incorrect, publish a corrected new version instead.

## PyPI publication

After the GitHub Release is published:

1. `Package` checks out the release tag with full history, validates that the
   tag is exactly `v<PEP 440 version>`, confirms that `HEAD` is exactly tagged,
   builds the sdist and wheel, and checks that both artifacts have the expected
   release version.
2. The validated wheel and sdist are stored together as the `release-dists`
   Actions artifact.
3. `Publish to PyPI` waits for the `pypi` GitHub Environment rules, if any.
   A required reviewer may approve the deployment.
4. The job downloads `release-dists` and invokes
   `pypa/gh-action-pypi-publish@release/v1` with GitHub OIDC. It does not
   rebuild the package.

The workflow uses no PyPI API token or stored upload credentials, and it does
not call manual `twine upload` or suppress duplicate-publication failures. A
duplicate publication should fail loudly rather than be silently ignored.

PyPI publishing requires the repository's `pypi` GitHub Environment and the
corresponding PyPI Trusted Publisher configuration to remain aligned with
`.github/workflows/release.yml`.

The expected Trusted Publisher configuration is:

- owner: `mne-tools`
- repository: `mne-denoise`
- workflow: `release.yml`
- GitHub environment: `pypi`

If the repository, workflow name, organization, or environment changes, update
the Trusted Publisher configuration before the next release. The `pypi`
Environment should also retain appropriate deployment protections, such as a
`v*` tag rule, trusted reviewer approval, and prevention of self-review when
operationally feasible.

The workflow's `environment: pypi` and `id-token: write` permission do not by
themselves configure the external PyPI publisher or GitHub Environment
protections; those settings must remain configured in their respective
services.

## Verify the release

After PyPI publication, install the released version in a clean environment:

```bash
python -m venv /tmp/mne-denoise-release-check
/tmp/mne-denoise-release-check/bin/python -m pip install --upgrade pip
/tmp/mne-denoise-release-check/bin/python -m pip install "mne-denoise==X.Y.Z"
```

Then verify the installed metadata and package version agree:

```bash
/tmp/mne-denoise-release-check/bin/python - <<'PY'
from importlib.metadata import version
import mne_denoise

print(version("mne-denoise"))
print(mne_denoise.__version__)

assert version("mne-denoise") == mne_denoise.__version__
PY
```

Also verify on PyPI that there is exactly one sdist and one `py3-none-any`
wheel for `X.Y.Z`, and that:

- the upload reports Trusted Publishing as enabled;
- provenance/attestation is present;
- the provenance identifies `mne-tools/mne-denoise`, `release.yml`, the
  `release` trigger, `refs/tags/vX.Y.Z`, and the tagged source commit; and
- the GitHub Release exists for the same tag and target commit.

The PyPA publishing action's provenance attestations remain enabled by
default. Actual PyPI provenance can only be checked after a real publication;
ordinary pull request CI cannot exercise production OIDC.

## Conda-forge

The conda-forge package is maintained in
`conda-forge/mne-denoise-feedstock`, not in this repository's release
workflow.

After a new version is published to PyPI, the conda-forge version-update bot
opens a feedstock pull request with the new source version and hash. Feedstock
CI validates the recipe and, when the feedstock's bot automation is enabled, a
passing version-update pull request may merge automatically and publish the
new conda-forge package.

Before relying on automatic merging, ensure that the feedstock recipe reflects
the current build system, Python support, runtime dependency contract, and
chosen policy for optional integrations.

## Versioning after a release

There is no post-release version-bump commit. Do not set a version to
`X.Y.Z`, change it back to `X.Y.Z.dev0`, or add a manual version file. With
`hatch-vcs`, tagged builds use the tag version and subsequent untagged `main`
builds automatically derive the next development version from Git history.

## Repository release settings

GitHub immutable releases are compatible with this workflow because CI does
not modify a published release or attach package assets after publication.
Enable immutable releases in repository settings once the release process is
configured; this workflow does not change that setting.
