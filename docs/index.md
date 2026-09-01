---
html_theme.sidebar_primary.remove: true
html_theme.sidebar_secondary.remove: true
---

# mne-denoise

::::::{div} mdn-hero

## Denoising methods for EEG and MEG

`mne-denoise` brings together complementary spatial, spectral, adaptive, and source-informed denoising methods, with NumPy support and integration with MNE-Python objects.

Choose a method based on the artifact and the information available in your recording, and evaluate both what was removed and what was preserved.

:::::{div} mdn-hero-controls
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

::::{div} mdn-install
```{code-block} bash
pip install mne-denoise
```
::::
:::::
::::::

```{note}
**mne-denoise is under active development.** Until version 1.0, the public API
may evolve between releases. For reproducible analyses, record the mne-denoise
version used in your work.
```

## What do you need to clean?

A useful place to start is the type of artifact or structure you want to
address.

::::{grid} 1 1 2 3
:gutter: 3
:class-container: mdn-method-grid

:::{grid-item-card} Transient high-amplitude artifacts
:class-card: mdn-method-card
:shadow: none

Reconstruct high-variance periods using a relatively clean calibration reference.

**Methods:** {doc}`ASR <asr>`
:::
:::{grid-item-card} Sensor-specific noise
:class-card: mdn-method-card
:shadow: none

Suppress sensor-specific noise using spatial redundancy or forward-model information.

**Methods:** {doc}`SNS <sns>` · {doc}`SOUND <sound>`
:::
:::{grid-item-card} Power-line contamination
:class-card: mdn-method-card
:shadow: none

Address narrowband line contamination with spectral interpolation or DSS-based spatial separation.

**Methods:** {doc}`Spectrum interpolation <spectrum_interpolation>` · {doc}`ZapLine <zapline>`
:::
:::{grid-item-card} Reproducible or structured components
:class-card: mdn-method-card
:shadow: none

Extract or suppress components defined by reproducibility, temporal or spectral structure, or repeated trials.

**Methods:** {doc}`DSS <dss>` · {doc}`SSA <ssa>`
:::
:::{grid-item-card} CCA and reference-informed cleaning
:class-card: mdn-method-card
:shadow: none

Use shared structure between signals, lagged copies, or reference channels to identify components for cleaning.

**Methods:** {doc}`BSS-CCA <bss_cca>` · {doc}`iCanClean <icanclean>`
:::
:::{grid-item-card} TMS-evoked muscle artifact
:class-card: mdn-method-card
:shadow: none

Suppress a fitted artifact subspace and reconstruct the signal with source-informed geometry.

**Method:** {doc}`SSP-SIR <sspsir>`
:::

::::

## Start here

Use the documentation in this order when you are new to the package:

* {doc}`Getting started <getting-started>` for installation and first NumPy/MNE
  workflows.
* {doc}`Choose a method <methods>` for assumptions, inputs, and method guides.
* {doc}`Evaluation <evaluation>` for assessing cleaning and signal
  preservation.

## Citation

Using mne-denoise in research? Please cite the software version you used and the primary scientific paper for each method.

{doc}`Citation <citing>` · [GitHub](https://github.com/mne-tools/mne-denoise)

```{toctree}
:hidden:
:maxdepth: 2

Get started <getting-started>
Methods <methods>
Examples <auto_examples/index>
API <api>
Contribute <development>
```
