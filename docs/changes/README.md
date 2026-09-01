# Changelog Guide

Changelog fragments are for release-note-worthy changes and should normally be
only 1–2 lines. Pull requests that do not need a release note should use the
`no-changelog-entry-needed` label.

## Adding a Changelog Entry

Use a fragment in `docs/changes/devel/` when a change belongs in the release
notes. The filename should include the pull request number and change type:
`<PR>.<type>.rst`, such as `123.bugfix.rst`.

For a local draft, create an unnumbered stub such as `bugfix.rst` and run:

```bash
python scripts/rename_towncrier.py --pr-number 123
```

The pull-request automation performs this rename when needed. Do not edit
`CHANGELOG.md` in a pull request.

### Available types:

* `feature`: New feature.
* `bugfix`: Bug fix.
* `doc`: Documentation improvement.
* `removal`: Deprecation or removal of a feature.
* `misc`: Internal changes, tooling, etc.

## Building the Changelog

To preview the changelog (requires `towncrier`):

```bash
towncrier build --draft
```
