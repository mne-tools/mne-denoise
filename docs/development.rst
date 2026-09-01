Development
===========

mne-denoise welcomes contributions to code, tests, documentation, examples,
and scientific methods.

Development setup
-----------------

Install development dependencies:

.. code-block:: console

   python -m pip install -e . --group dev

This installs the editable package together with the tools used by the
development checks.

Install repository hooks:

.. code-block:: console

   prek install

This enables the configured formatting and repository checks before commits.

Common commands
---------------

Tests
~~~~~

Run the test suite. Additional pytest arguments can be forwarded after ``--``.

.. code-block:: console

   spin test

For a focused run, use a separate command such as:

.. code-block:: console

   spin test -- -k <pattern>

Lint and repository hooks
~~~~~~~~~~~~~~~~~~~~~~~~~

Run all configured repository hooks against the full repository.

.. code-block:: console

   spin lint

Documentation
~~~~~~~~~~~~~

Build the complete documentation with Sphinx warnings treated as errors.

.. code-block:: console

   spin docs

Distribution build
~~~~~~~~~~~~~~~~~~

Build and validate the wheel and source distribution.

.. code-block:: console

   spin build

Repository check
~~~~~~~~~~~~~~~~

Run linting, tests, and distribution validation.

.. code-block:: console

   spin check

``spin check`` does not build the documentation; run ``spin docs`` separately
when documentation is affected.

Scientific contributions
------------------------

Changes to scientific methods need more than code that runs. Identify the
primary scientific source, explain intentional deviations or extensions, add
meaningful numerical or behavioral tests, and update the relevant
documentation. Historical output parity alone is not evidence that an
implementation is scientifically correct.

See `AGENTS.md <https://github.com/mne-tools/mne-denoise/blob/main/AGENTS.md>`__
for the detailed scientific source hierarchy and test ownership rules.

AI-assisted development
-----------------------

**AI-assisted development has been used substantially in building and
maintaining mne-denoise, and AI-assisted contributions are welcome.** The
important requirement is active human supervision: a human contributor must
understand, review, test, and take responsibility for every submitted change.

AI tools may assist with implementation, tests, documentation, review, or
exploration, but they are not a substitute for scientific judgment or code
review. Changes to numerical or scientific behavior require the same
provenance, validation, and scrutiny regardless of whether AI was involved.

Fully automated contributions without human review are not accepted. When AI
materially contributes to a pull request, disclose the tool and the scope of
assistance. See the `AI-assisted contributions
<https://github.com/mne-tools/mne-denoise/blob/main/CONTRIBUTING.md#ai-assisted-contributions>`__
section of the contribution guide.

Pull requests and review
------------------------

Keep pull requests focused and explain what changed and why. Run the relevant
checks before requesting review. Scientific changes should make their source,
assumptions, and validation easy for reviewers to inspect.

See `CONTRIBUTING.md <https://github.com/mne-tools/mne-denoise/blob/main/CONTRIBUTING.md>`__
for the complete contribution workflow and review expectations.

CI and changelog shortcuts
--------------------------

For a trivial intermediate commit that does not need GitHub Actions, add
``[skip ci]`` to the commit message. Use this sparingly and push a normal
commit before merge so the required checks run on the final PR state.

Not every pull request needs a changelog entry. If a change is not
release-note-worthy, use the ``no-changelog-entry-needed`` label instead of
adding a Towncrier fragment.

Changelog entries
-----------------

User-facing changes use Towncrier fragments under ``docs/changes/devel/``.
See `docs/changes/README.md <https://github.com/mne-tools/mne-denoise/blob/main/docs/changes/README.md>`__
for naming and local draft instructions.

More contributor guidance
-------------------------

Repository architecture and maintainer guidance are documented in
`AGENTS.md <https://github.com/mne-tools/mne-denoise/blob/main/AGENTS.md>`__.
