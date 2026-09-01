# Contributing to mne-denoise

Thanks for your interest in contributing. Contributions may include bug
fixes, scientific improvements, tests, documentation, examples, or
maintenance.

## Questions, bugs, and feature ideas

Please search existing issues and discussions first.

- Usage and support questions belong on the [MNE Forum](https://mne.discourse.group/).
- Reproducible bugs belong in the [bug report form](https://github.com/mne-tools/mne-denoise/issues/new?template=bug_report.yml).
- Feature or scientific enhancement ideas belong in the [feature request form](https://github.com/mne-tools/mne-denoise/issues/new?template=feature_request.yml).
- Documentation problems belong in the [documentation form](https://github.com/mne-tools/mne-denoise/issues/new?template=documentation.yml).

## Code of Conduct and communication

Contributors must follow the project's [Code of Conduct](https://github.com/mne-tools/mne-denoise/blob/main/CODE_OF_CONDUCT.md).
GitHub technical review is often concise and matter-of-fact; short feedback
should not automatically be interpreted as a lack of interest or friendliness.
Focus on the technical content, while keeping every interaction consistent
with the Code of Conduct.

## AI-assisted contributions

AI-assisted development has been used substantially in mne-denoise and is
welcome when it remains under active human supervision. The contributor is
responsible for every submitted line and must understand, review, and test
AI-assisted work. Scientific algorithm changes deserve particular scrutiny
for correctness, provenance, and licensing.

Material AI assistance must be disclosed in the pull-request description,
including the tool and scope of assistance. Fully automated submissions
without human review are not acceptable, and AI-generated text is not a
substitute for understanding reviewer discussion.

## Development setup

Use an isolated development environment. The [Scientific Python development
guides](https://learn.scientific-python.org/development/) provide general
environment guidance; this repository-specific setup is:

```bash
python -m pip install -e . --group dev
prek install
```

## Development commands

Use the canonical repository commands:

```bash
spin test
spin lint
spin docs
spin build
spin check
```

Focused `pytest` commands are fine during development and debugging.

## Scientific contributions

Changes to scientific algorithms should:

- identify the primary scientific source(s);
- explain intentional deviations or extensions;
- include meaningful numerical or behavioral tests;
- update scientific documentation and docstrings; and
- avoid treating historical output parity alone as evidence of correctness.

See [AGENTS.md](https://github.com/mne-tools/mne-denoise/blob/main/AGENTS.md) for the detailed source hierarchy and test
ownership conventions.

## Tests and public API

New functionality should have appropriate tests. mne-denoise centralizes
shared estimator, MNE-container, public-API, and progress contracts. Before
adding dedicated tests or new public helpers, read [AGENTS.md](https://github.com/mne-tools/mne-denoise/blob/main/AGENTS.md).

## Documentation

Update public-facing documentation, docstrings, and examples when applicable.
Documentation must build without warnings, and primary scientific sources
should support scientific claims. `spin docs` is the canonical full check.

## Changelog

Add a 1–2 line Towncrier fragment under `docs/changes/devel/` for changes that
belong in the release notes. If no changelog entry is warranted, apply the
`no-changelog-entry-needed` label; see [docs/changes/README.md](https://github.com/mne-tools/mne-denoise/blob/main/docs/changes/README.md)
for fragment naming.

## Pull requests

Keep the pull request focused, reference an issue where applicable, and
explain what changed and why. Disclose AI assistance as described above, run
the relevant checks, and respond to review feedback. Maintainers merge changes
after approval.

## Repository architecture for agents and advanced contributors

For repository-specific architecture, public API ownership, shared test
contracts, CI expectations, scientific source hierarchy, and guidance for AI
coding agents, see [AGENTS.md](https://github.com/mne-tools/mne-denoise/blob/main/AGENTS.md).
