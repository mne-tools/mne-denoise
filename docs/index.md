# mne-denoise

## Denoising methods for EEG and MEG

::::{div} mdn-hero
`mne-denoise` brings together complementary spatial, spectral, adaptive, and
source-informed denoising methods, with NumPy support and integration with
MNE-Python objects.

Choose a method based on the artifact and the information available in your
recording, and evaluate both what was removed and what was preserved.

::::{div} mdn-hero-actions
:::{button-ref} getting-started
:ref-type: doc
:class: sd-btn-primary
Get started
:::
:::{button-ref} methods
:ref-type: doc
:class: sd-btn-secondary
Choose a method
:::
:::{button-ref} auto_examples/index
:ref-type: doc
:class: sd-btn-secondary
Browse examples
:::
::::
::::

::::{div} mdn-install
```{code-block} bash
pip install mne-denoise
```
::::

## Start with the artifact

No denoising method is appropriate for every recording. Different methods
rely on different kinds of structure — for example clean calibration data,
spatial redundancy, spectral concentration, reference channels, repeated
trials, or a forward model.

{doc}`After cleaning, inspect both artifact attenuation and preservation of the signal of interest. <evaluation>`

## What do you need to clean?

A useful place to start is the type of artifact or structure you want to
address.

::::{grid} 1 1 2 3
:gutter: 3
:class-container: mdn-method-grid

:::{grid-item-card} Transient high-amplitude artifacts
:class-card: mdn-method-card
:shadow: none

Reconstruct periods whose variance departs strongly from a relatively clean
calibration reference.

**Methods:** {doc}`ASR <asr>`

<p class="mdn-card-note">Includes adaptive ASR variants</p>
:::

:::{grid-item-card} Sensor-specific noise
:class-card: mdn-method-card
:shadow: none

Suppress noise specific to individual sensors using spatial redundancy or
forward-model information.

**Methods:** {doc}`SNS <sns>` · {doc}`SOUND <sound>`

<p class="mdn-card-note">Spatial redundancy · Forward-model informed</p>
:::

:::{grid-item-card} Power-line contamination
:class-card: mdn-method-card
:shadow: none

Address narrowband line contamination through spectral interpolation or
DSS-based spatial separation.

**Methods:** {doc}`Spectrum interpolation <spectrum_interpolation>` ·
{doc}`ZapLine <zapline>`
:::

:::{grid-item-card} Reproducible or structured components
:class-card: mdn-method-card
:shadow: none

Extract or suppress components defined by reproducibility, temporal structure,
spectral structure, or repeated-trial dynamics.

**Methods:** {doc}`DSS <dss>` · {doc}`SSA <ssa>`

<p class="mdn-card-note">Includes TimeShiftDSS and other DSS variants</p>
:::

:::{grid-item-card} CCA and reference-informed cleaning
:class-card: mdn-method-card
:shadow: none

Use shared structure between signals, lagged copies, or reference channels to
identify components for cleaning.

**Methods:** {doc}`BSS-CCA <bss_cca>` · {doc}`iCanClean <icanclean>`

<p class="mdn-card-note">Lagged signal structure · Physical or derived references</p>
:::

:::{grid-item-card} TMS-evoked muscle artifact
:class-card: mdn-method-card
:shadow: none

Suppress a fitted artifact subspace and reconstruct the signal using
source-informed geometry.

**Method:** {doc}`SSP-SIR <sspsir>`
:::

::::

## Working with mne-denoise

::::{grid} 1 1 3 3
:gutter: 3
:class-container: mdn-feature-grid

:::{grid-item-card} NumPy and MNE-Python
:class-card: mdn-feature-card
:shadow: none

Work with channel-first NumPy arrays, or supported MNE Raw, Epochs, and Evoked
objects where the method allows it.
:::

:::{grid-item-card} Familiar estimator interfaces
:class-card: mdn-feature-card
:shadow: none

Main methods use `fit`, `transform`, and `fit_transform` when those operations
match the scientific workflow.
:::

:::{grid-item-card} Inspect the result
:class-card: mdn-feature-card
:shadow: none

Many methods expose fitted operators, component information, or diagnostics,
and the package includes utilities for evaluating cleaning and signal
preservation.
:::

::::

## Documentation

::::{grid} 1 1 2 3
:gutter: 3
:class-container: mdn-doc-grid

:::{grid-item-card} Get started
:link: getting-started
:link-type: doc
:class-card: mdn-doc-card
:shadow: none

Installation and a first workflow.
:::

:::{grid-item-card} Methods
:link: methods
:link-type: doc
:class-card: mdn-doc-card
:shadow: none

Find a method from the artifact and assumptions.
:::

:::{grid-item-card} Examples
:link: auto_examples/index
:link-type: doc
:class-card: mdn-doc-card
:shadow: none

Runnable workflows using synthetic and real data.
:::

:::{grid-item-card} API reference
:link: api
:link-type: doc
:class-card: mdn-doc-card
:shadow: none

Exact parameters, outputs, attributes, and public functions.
:::

:::{grid-item-card} Evaluation
:link: evaluation
:link-type: doc
:class-card: mdn-doc-card
:shadow: none

Assess cleaning together with signal preservation.
:::

:::{grid-item-card} Contribute
:link: development
:link-type: doc
:class-card: mdn-doc-card
:shadow: none

Report problems or contribute code, documentation, or scientific improvements.
:::

::::

### Using mne-denoise in research?

::::{div} mdn-footer-note
Please cite the software version you used and the primary scientific paper for
each method.

{doc}`Citation <citing>` · [GitHub](https://github.com/mne-tools/mne-denoise)
::::

```{toctree}
:hidden:
:maxdepth: 2

Get started <getting-started>
Methods <methods>
Examples <auto_examples/index>
API <api>
Evaluation <evaluation>
Contribute <development>
```
