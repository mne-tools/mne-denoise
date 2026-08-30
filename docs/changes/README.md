# Changelog Guide

We use `towncrier` to manage our changelog. This ensures that changes are documented as they happen, preventing merge conflicts in the changelog file and ensuring high-quality release notes.

## Adding a Changelog Entry

When you make a change (feature, bugfix, documentation update), you should add a fragment file to the `docs/changes/devel/` directory.

The filename should include the pull request number and the change type:
`<PR>.<type>.rst`, such as `123.bugfix.rst`. For a local draft, create an
unnumbered stub such as `bugfix.rst` and run:

```bash
python scripts/rename_towncrier.py --pr-number 123
```

The pull-request automation performs this rename when needed. Do not edit
`CHANGELOG.md` in a pull request.

Format: `<PR>.<TYPE>.rst`

### Available types:

* `feature`: New feature.
* `bugfix`: Bug fix.
* `doc`: Documentation improvement.
* `removal`: Deprecation or removal of a feature.
* `misc`: Internal changes, tooling, etc.

## Example

If you fixed a bug in PR 123, create a file `docs/changes/devel/123.bugfix.rst`:

```rst
Fixed a bug where the ZapLine algorithm would crash on empty data.
```

## Building the Changelog

To preview the changelog (requires `towncrier`):

```bash
towncrier build --draft
```
