---
html_theme.sidebar_primary.remove: true
html_theme.sidebar_secondary.remove: true
---

# mne-denoise

::::::{div} mdn-hero

<h2>Denoising methods for EEG and MEG</h2>

`mne-denoise` brings together complementary spatial, spectral, adaptive, and source-informed denoising methods, with NumPy support and integration with MNE-Python objects.

Choose a method based on the artifact and the information available in your recording, and evaluate both what was removed and what was preserved.

:::::{div} mdn-hero-controls
::::{div} mdn-hero-actions
:::{button-ref} getting-started
:ref-type: doc
:class: sd-btn mdn-btn-primary
Get started
:::
:::{button-ref} methods
:ref-type: doc
:class: sd-btn mdn-btn-secondary
Choose a method
:::
:::{button-ref} auto_examples/index
:ref-type: doc
:class: sd-btn mdn-btn-secondary
Browse examples
:::
::::

::::{div} mdn-install
::::{tab-set}
:class: mdn-install-tabs

:::{tab-item} pip
```{code-block} bash
pip install mne-denoise
```
:::

:::{tab-item} uv
```{code-block} bash
uv pip install mne-denoise
```
:::

:::{tab-item} conda
```{code-block} bash
conda install -c conda-forge mne-denoise
```
:::
::::
::::
:::::
::::::

```{admonition} Development status
:class: note

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
:class-card: mdn-method-card mdn-method-card-terracotta
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
:class-card: mdn-method-card mdn-method-card-terracotta
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
:class-card: mdn-method-card mdn-method-card-terracotta
:shadow: none

Suppress a fitted artifact subspace and reconstruct the signal with source-informed geometry.

**Method:** {doc}`SSP-SIR <sspsir>`
:::

::::

::::{div} mdn-citation-note

<h3>Using mne-denoise in research?</h3>

Please cite the software version you used and the primary scientific paper for each method.

{doc}`Citation <citing>` · [GitHub](https://github.com/mne-tools/mne-denoise)

::::

```{toctree}
:hidden:
:maxdepth: 2

Get started <getting-started>
Methods <methods>
Examples <auto_examples/index>
API reference <api>
Development <development>
```
