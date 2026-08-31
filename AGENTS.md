# AGENTS.md

This file provides repository-specific guidance to coding agents and other automated assistants working with or on `mne-denoise`. It covers two use cases: helping users apply mne-denoise correctly, and modifying the package, tests, documentation, or repository tooling.

## What is mne-denoise

`mne-denoise` provides artifact-suppression and signal-denoising methods for
EEG and MEG. It supports NumPy arrays and optional integration with
MNE-Python containers, and provides scikit-learn-style estimators where that
interface fits the method. The package contains several distinct scientific
methods rather than one universal denoising procedure.

The scientific semantics and assumptions of each method matter more than implementation convenience. mne-denoise should integrate naturally into the MNE ecosystem and reuse generic MNE behavior rather than reproduce functionality that MNE already owns.

## Helping users with mne-denoise

Use the current public API only. Before suggesting an import or workflow, inspect the current public facades and `tests/test_public_api.py`; do not infer the API from filenames or old examples. Importability alone does not make a name public.

Do not hallucinate stale names from historical documentation or branches. In particular:

- Do not suggest an object that exists only because a similar name appeared in
  an earlier version.
- Public imports must come from current documented facades and their declared
  `__all__` values.
- Modules and names beginning with `_` are implementation details and should
  not be recommended to users.

When the context is available, determine the parts of the scientific problem that affect the recommendation:

- EEG or MEG.
- Raw, Epochs, Evoked, or NumPy input.
- Sampling frequency and channel types.
- Known bad channels.
- The target artifact or noise source.
- Continuous versus epoched data.
- Line frequency, when line-noise removal is relevant.
- Whether the goal is artifact suppression, source extraction, component
  enhancement, or another operation.
- Preprocessing already performed, including filtering and referencing.

Do not ask every question in every conversation, but do not ignore these factors when they change the scientific interpretation or API choice. No method is universally best. Explain relevant assumptions, choose the simplest public workflow that fits the goal, and preserve MNE metadata and container semantics.

Do not recommend private helpers when a public estimator or function exists. Distinguish established published methods from mne-denoise-specific extensions and experimental research APIs.

Honor experimental-status and validation warnings in the current public
docstrings and method documentation. Public availability does not imply that
an API is an established or independently validated scientific method. Do not
recommend experimental or unvalidated research APIs as established defaults.
When scientific validation status matters, inspect the current method
documentation and primary literature before making a recommendation. For
example, an API such as GuidedASR should be presented with that qualification
whenever its current documentation marks it experimental.

Artifact attenuation is not evidence that desired neural signal was preserved. Where relevant, encourage users to evaluate both artifact attenuation and preservation of the signal of interest, using suitable controls and scientific knowledge.

Example code for users must:

- Use canonical import paths and current constructor parameter names.
- Use the current `fit`, `transform`, `fit_transform`, or functional API
  semantics.
- Avoid needless NumPy extraction and MNE-object reconstruction when an
  estimator accepts an MNE container directly.
- Never mutate user data unless the public API explicitly documents in-place
  behavior.

## Sources of truth

Scientific truth and software-contract truth are related but not
interchangeable.

### Scientific algorithm semantics

Use this order when resolving a scientific question:

1. The primary scientific publication.
2. An authoritative or reference implementation, when relevant.
3. The current mne-denoise implementation and scientific tests.
4. Historical branches, historical outputs, fixtures, and old documentation
   only when they are needed to resolve an ambiguity.

Historical output parity is not by itself scientific truth. Do not recreate old substrate, parity, or output-generation machinery merely to preserve historical numerical arrays. If the implementation and the publication or reference implementation appear inconsistent, stop and surface the discrepancy rather than silently choosing one interpretation.

### mne-denoise software contracts

Use this order for package behavior and compatibility questions:

1. The intentional public API and its tests.
2. Shared estimator and MNE-container contract tests.
3. Algorithm-specific tests.
4. Current user documentation.

Correct documentation when it contradicts the implementation and tests, unless the discrepancy reveals a scientific bug that needs separate investigation.

## Before writing new code

Search for an existing owner before adding local machinery:

1. Search mne-denoise for an existing implementation or helper.
2. Search public MNE-Python APIs.
3. Search NumPy, SciPy, and scikit-learn functionality where appropriate.
4. Only then introduce new local machinery.

Prefer existing public MNE APIs over local duplication. Generic MNE object, channel, forward, covariance, filter, and container behavior should not be reimplemented here when MNE already owns it. If the needed behavior only exists privately, consider whether an upstream public API or change is more appropriate instead of importing MNE internals.

Keep logic local when mne-denoise semantics genuinely differ. Do not import MNE private helpers casually to remove a few lines, build wrappers solely for conceptual symmetry, or create an abstraction for one caller unless it materially clarifies ownership. Do not add hypothetical parameters or extension points, and do not add dependencies without a concrete need and discussion. Never promote an optional dependency to a required dependency as a side effect.

Implement behavior at the highest existing owner layer. For example, a generic estimator invariant belongs in the package-wide estimator contract suite, not in repeated method-specific tests.

## Keep changes focused

Optimize for reviewer comprehension and long-term maintenance, not diff size or output volume. Prefer the smallest change that fully satisfies the issue, and avoid opportunistic cleanup of unrelated code.

If a task starts requiring a new subsystem, dependency, public API family, or many unrelated files, reconsider its scope. Reuse existing test infrastructure before creating new infrastructure. Avoid future-proofing that has no immediate requirement.

Do not expand a focused task into unrelated architecture, public API,
dependencies, algorithms, or upstream work unless the issue or maintainer
explicitly asks for it. Do not silently turn a documentation or
repository-guidance task into a scientific implementation change.

## Public API

The intentional public API is defined and tested in
`tests/test_public_api.py`. Treat its facade and canonical-path registries,
together with the corresponding `__all__` declarations, as the authoritative
inventory.

Public facades declare `__all__`, and canonical public paths are tested
explicitly. Importability alone does not make a symbol public. A new public API
requires updates to the appropriate facade, `__all__`, API documentation,
tests, and user-facing documentation or the changelog when appropriate.

Private underscore names have no compatibility promise. Experimental APIs
must be clearly identified. Documentation and examples must never invent
public names that are not part of current main.

## Estimator contracts

`tests/_contract_cases.py` is the central estimator capability registry, and `tests/test_estimator_contracts.py` executes the shared cases. The registry is capability-based: it does not promise that every estimator has the same interface or output shape.

The capabilities currently represented are:

- `cloneable`: scikit-learn `clone` preserves the public constructor
  parameters.
- `fit_returns_self`: `fit()` returns the estimator instance.
- `fit_transform_composes`: `fit_transform()` agrees with separate `fit()` then `transform()` calls. AdaptiveASR's deliberately different fit/transform modes are not forced into this group.
- `not_fitted`: the registered pre-fit operation raises the expected `NotFittedError`. DSS and ZapLine intentionally retain their public pre-fit `RuntimeError` behavior and are not placed in this group.
- `numpy_no_mutation`: the public NumPy operation leaves its input unchanged
  and returns the expected array shape.
- `numpy_layout`: supported NumPy layouts, including the applicable epoched
  layout, are preserved.
- `mne_raw`, `mne_epochs`, and `mne_evoked`: the estimator supports the corresponding MNE container in the shared suite.
- `fitted_channel_count`: a fitted estimator rejects a different channel
  count.
- `fitted_channel_order` and `fitted_channel_names`: a fitted MNE estimator
  enforces the channel layout guarantees it declares.
- `sfreq_aware`: a fitted operation rejects a sampling-frequency mismatch
  where its contract requires that check.
- `callback_transparent`: a callback does not change numerical output, receives `ProgressEvent` objects, and propagates callback failures.

Before adding an estimator-specific test for generic behavior, check whether the estimator should instead opt into an existing capability in `_contract_cases.py`. Do not repeat tests such as one input-mutation test per estimator when the central contract owns that property. Algorithm-specific tests should cover algorithm-specific science and behavior.

## MNE container contracts

Read `tests/test_mne_container_contracts.py` before changing public MNE integration. Raw, Epochs, and Evoked are not merely NumPy arrays with labels; their metadata and lifecycle semantics are part of the public contract.

The shared suite currently checks, as applicable:

- The output remains the corresponding MNE container type.
- Transformation returns a new object and does not mutate the input.
- Channel names and order follow the estimator's fitted-layout rules.
- Bad-channel metadata is preserved.
- Untouched channels remain unchanged.
- Raw annotations and `first_samp` are preserved.
- Epochs events, event IDs, baseline, metadata, selection, and drop log are
  preserved.
- Evoked timing, `nave`, and comments are preserved.
- Fitted channel-name/order and sampling-frequency checks are enforced where
  declared by the estimator.

Generic MNE-container behavior belongs in this shared contract suite; algorithm-specific numerical behavior belongs in algorithm tests.

## Progress reporting

The package-wide progress design is implemented in `mne_denoise/progress.py` and tested by `tests/test_progress.py`, `tests/test_progress_api.py`, and `tests/test_progress_tqdm.py`.

Algorithm code emits structured progress events through callback support. `ProgressEvent` is the immutable protocol payload. Callbacks are synchronous runtime observers supplied by keyword; their return values are ignored and their exceptions propagate unchanged. Logging is independent: `verbose` controls package logs, while `callback` controls events.

`TqdmProgress` is an optional presentation adapter that consumes events. Algorithm internals must not own tqdm bars directly. Do not add `print()`-based progress or a second callback interface. New callback-aware methods should follow the existing package convention and should not make callback state an estimator hyperparameter.

## Optional dependencies and base-install boundary

`pyproject.toml` is the authoritative source for dependency declarations,
optional extras, dependency groups, and minimum versions.

The important compatibility boundary is that the base package remains usable
without optional MNE, visualization, or progress dependencies. MNE
integration, visualization, and the tqdm progress adapter remain optional.
Optional dependencies must not be imported eagerly in a way that breaks base
import. `scripts/check_base_install.py` tests this boundary.

Do not raise dependency floors or promote an optional dependency to required
without a concrete compatibility reason.

## Continuous integration

The workflow files are the authoritative source for the exact Python versions,
platforms, and dependency combinations. The matrix in
`.github/workflows/tests.yml` intentionally protects different compatibility
dimensions rather than repeating one test run:

- Lint checks the complete repository hook suite.
- Base-install checks protect the package boundary without optional extras.
- Minimum-dependency jobs protect declared lower bounds.
- Stable jobs cover supported Python and platform combinations.
- Coverage measures the supported test suite.
- MNE-main jobs check compatibility with upcoming MNE changes.
- Scientific-dependency prereleases expose forward-compatibility problems.

`.github/workflows/docs.yml` builds the full gallery against both released MNE
and MNE main. Stable-MNE and MNE-main documentation jobs protect both
documentation compatibility dimensions. Documentation CI prefetches and
caches the real datasets required by the gallery; `scripts/prefetch_docs_data.py`
and the docs workflow are the authoritative inventory. When an example
introduces a new real dataset, update the prefetch mechanism in the same
change.

The release workflow builds and validates distributions and tests an
installed wheel before publishing on a GitHub release. The changelog workflow
checks Towncrier fragments, and dependency review rejects high-severity
dependency findings.

Passing one local environment is not sufficient evidence of repository
compatibility. Do not weaken CI, remove platform coverage, raise dependency
floors just to fix CI, disable tests, or turn required jobs into informational
jobs.

## Repository task commands

Use the current Spin interface in `.spin/cmds.py`:

```bash
spin test
spin test -- -k <pattern>
spin lint
spin docs
spin build
spin check
```

Spin is the canonical repository task interface. Direct `pytest` commands are
fine for focused development and debugging. `prek` is the hook runner; it is
not a replacement for environment or package installation tooling. Do not
introduce another task runner solely to duplicate these commands.

## Testing philosophy

Test scientific, public, and integration contracts rather than private
implementation details. Prefer deterministic behavioral or numerical
assertions to execution-only or shape-only coverage tests. Historical bug
cases are useful when they protect a durable contract, but organize them
under the contract that owns them rather than creating a historical-parity
testing architecture.

Use parametrization when it represents genuinely equivalent cases. Keep
scientifically distinct regimes separate. Use coverage diagnostically; do not
optimize for a percentage at the expense of meaningful tests. Centralize
shared estimator and MNE-container behavior in their contract suites.

## Scientific implementations and validation

Primary scientific literature governs algorithm claims and semantics. Identify
whether behavior is a published method, an established implementation
convention, an mne-denoise-specific extension, or experimental/prototype
behavior. Do not silently describe an extension as if it came from the
original paper.

Scientific tests should exercise meaningful invariants and behavior, not only
reproduce one historical array. If changing scientific behavior, document why
and cite the relevant source. Parity with old package output is insufficient
evidence of correctness. Numerical tolerances should reflect meaningful
floating-point expectations, not be widened merely to make a test pass.

## Documentation and docstrings

Every public function, class, and method docstring is part of the public API
and must match current behavior. Check public docstrings for:

- A one-line summary.
- Current parameter names and types.
- Accepted container types and expected array shape/layout.
- Defaults, return type, return container/shape, and fitted attributes where
  relevant.
- `Raises` entries for meaningful user errors.
- `Notes` where scientific or implementation interpretation is needed.
- References for published scientific methods and `See Also` where genuinely
  useful.
- Examples only when they add value beyond the gallery and other docs.

Use NumPy-style docstrings and the repository's citation conventions. Do not
maintain a complete bibliography manually in every docstring. Do not leave
stale parameter names after API changes, claim Raw/Epochs/Evoked support that
the implementation or contracts do not provide, or misstate copy versus
in-place semantics and `fit`/`transform` lifecycle. Document experimental
status where appropriate.

When changing or reviewing public API, audit public docstrings callable by
callable against the implementation, tests, and relevant scientific sources.
Existing docstrings are not authoritative when they disagree with those
sources of truth.

## Examples and gallery

Gallery examples have CI, runtime, and maintenance costs. One new estimator
does not automatically require one new example. Extend an existing example
when it teaches the same concept; add a gallery item when it teaches a
meaningful scientific capability or workflow. Tiny syntax demonstrations
belong in docstrings or method documentation, and tests are not gallery
examples.

New real-data dependencies must use the documentation data prefetch/cache
mechanism. Add datasets to `scripts/prefetch_docs_data.py` when an example
needs them. Examples must use public APIs and should not preserve obsolete
APIs merely to avoid changing docs. Prefer the existing documentation and
gallery structure unless a change in information architecture is explicitly
part of the task.

## Repository scripts and generated artifacts

Scripts under `scripts/` are operational repository helpers. Do not add
substrate or parity-generation scripts without a demonstrated current need.
Generated changelog, release, and citation artifacts should be maintained by
their owning tooling rather than hand-edited where applicable. Old parity
builders are not part of the current architecture.

## Git and pull-request expectations

Never commit directly to `main`. Keep pull requests focused and use existing
issue or design discussion for substantial behavior changes. User-visible
changes need the repository's changelog mechanism. Do not add AI co-author
trailers unless repository policy explicitly requires them. The human
contributor remains responsible for submitted changes.

Follow `CONTRIBUTING.md` for human-facing contribution policy, communication,
and AI-assisted contribution expectations.
