# Choose a method

mne-denoise includes methods that rely on different kinds of information in
the recording. Some use clean calibration periods, some spatial or spectral
structure, some reference channels or repeated trials, and others a forward
model.

No denoising method is appropriate for every recording. Match the method to
the artifact, the available structure, and the signal you need to preserve.

The sections below are intended as a starting point. The individual method
pages describe the assumptions and behavior in more detail.

## Transient high-amplitude artifacts

### ASR

Detect and reconstruct windows whose component variance departs from a clean
reference. **Main assumption/input:** a relatively clean calibration period or
calibration data. See the {doc}`ASR guide <asr>`; adaptive ASR variants remain
part of this family.

## Sensor-specific noise

### SNS

Reconstruct sensors from spatially redundant neighboring sensors.
**Main assumption/input:** noise is specific to individual sensors while the
signal of interest is spatially shared. See the {doc}`SNS guide <sns>`.

### SOUND

Estimate channel-specific noise from forward-model geometry and reconstruct a
cleaned sensor-space signal. **Main assumption/input:** a compatible EEG
montage or an explicit lead field. See the {doc}`SOUND guide <sound>`.

## Power-line contamination

### Spectrum interpolation

Replace amplitudes around target line-noise frequencies using neighboring
spectral bins. **Main assumption/input:** a known line frequency and enough
neighboring frequency bins. See the {doc}`spectrum interpolation guide
<spectrum_interpolation>`.

### ZapLine

Use line-locked DSS components to suppress power-line noise and harmonics.
**Main assumption/input:** a line frequency and data with suitable spectral
resolution. See the {doc}`ZapLine guide <zapline>`.

## Reproducible or structured components

### DSS

Extract, retain, or subtract components defined by a user-supplied bias.
**Main assumption/input:** a baseline covariance and a reproducibility,
spectral, temporal, or lagged-trial bias. See the {doc}`DSS guide <dss>`, which
also covers TimeShiftDSS and the other DSS variants.

### SSA

Decompose each channel in a delay-coordinate space and reconstruct selected
components. **Main assumption/input:** an embedding window and a frequency or
local-subspace selection rule. See the {doc}`SSA guide <ssa>`.

## CCA and reference-informed cleaning

### BSS-CCA

Use canonical correlation with a lagged copy of the signal to identify
components for cleaning. **Main assumption/input:** a lag choice and a rule
for selecting the canonical-correlation end to remove. See the
{doc}`BSS-CCA guide <bss_cca>`.

### iCanClean

Remove shared variance between primary channels and a physical or derived
reference. **Main assumption/input:** matched observations and a suitable
reference block. See the {doc}`iCanClean guide <icanclean>`.

## TMS-evoked muscle artifact

### SSP-SIR

Suppress a fitted artifact subspace and reconstruct the signal with
source-informed geometry. **Main assumption/input:** a TMS-evoked artifact
window and a forward model or compatible EEG montage. See the
{doc}`SSP-SIR guide <sspsir>`.

## Compact comparison

| Method | Main target | Main structure used | Additional requirement |
| ------ | ----------- | ------------------- | ---------------------- |
| {doc}`ASR <asr>` | Transient high-variance artifacts | Clean calibration covariance | Calibration period/data |
| {doc}`SNS <sns>` | Sensor-specific noise | Spatial sensor redundancy | None beyond channel data |
| {doc}`SOUND <sound>` | Sensor-specific noise | Forward-model geometry | Montage or forward model |
| {doc}`Spectrum interpolation <spectrum_interpolation>` | Power-line contamination | Neighboring spectral amplitudes | Line frequency and sampling rate |
| {doc}`ZapLine <zapline>` | Power-line noise and harmonics | Line-locked DSS components | Line frequency and DSS settings |
| {doc}`DSS <dss>` | Reproducible or structured components | Bias covariance relative to baseline | Bias or segment definition |
| {doc}`SSA <ssa>` | Structured channel-wise components | Delay-coordinate decomposition | Embedding and selection settings |
| {doc}`BSS-CCA <bss_cca>` | Components separated by lagged CCA | Lagged temporal correlation | Lag and component rule |
| {doc}`iCanClean <icanclean>` | Shared reference variance | CCA with reference channels | Reference block or pseudo-reference |
| {doc}`SSP-SIR <sspsir>` | TMS-evoked muscle artifact | Artifact subspace and lead field | Forward model or EEG montage |

```{admonition} Evaluate the result
:class: mdn-evaluation-note

Whichever method you choose, evaluate both artifact attenuation and
preservation of the signal of interest. See the {doc}`evaluation` guide.
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Methods

ASR <asr>
SNS <sns>
SOUND <sound>
Spectrum interpolation <spectrum_interpolation>
ZapLine <zapline>
DSS <dss>
SSA <ssa>
BSS-CCA <bss_cca>
iCanClean <icanclean>
SSP-SIR <sspsir>
Evaluating denoising <evaluation>
```
