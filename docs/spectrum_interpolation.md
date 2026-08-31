(spectrum-interpolation)=
# Spectrum interpolation

## Overview

Spectrum interpolation attenuates narrow-band power-line noise by replacing
Fourier amplitudes around target frequencies with amplitudes estimated from
neighbouring frequency bands. It keeps the original Fourier phase for the
modified bins, following Leske and Dalal {footcite:p}`leske_dalal2019_spectrum`.

This is an amplitude-domain operation, not a notch filter and not a proof that
the neural signal is preserved. A neural rhythm or other signal inside an
interpolated band is changed along with the line-noise component.

## Minimal API

For a continuous NumPy array, use the functional interface with explicit target
frequencies:

```python
import numpy as np
from mne_denoise.spectrum_interpolation import interpolate_spectrum

targets = 60.0 * np.arange(1, 4)
clean = interpolate_spectrum(
    data,
    sfreq=500.0,
    freqs=targets,
    bandwidth=1.0,
    neighbour_width=2.0,
)
```

The estimator resolves the fundamental and harmonics for you and integrates
with MNE containers:

```python
from mne_denoise.spectrum_interpolation import SpectrumInterpolation

cleaner = SpectrumInterpolation(
    line_freq=60.0,
    n_harmonics=3,
    bandwidth=1.0,
)
clean_raw = cleaner.fit_transform(raw)
```

``fit`` records the sampling frequency and target frequencies; ``transform``
applies the same spectral operation without learning from the transform batch.
MNE ``Raw``, ``Epochs``, and ``Evoked`` inputs are copied. Non-data channels are
left unchanged. The functional function accepts only a 2D array with shape
``(n_channels, n_times)``; the estimator also accepts 3D arrays with shape
``(n_epochs, n_channels, n_times)`` and processes each leading record
independently.

## What is interpolated

For each target frequency ``f``:

1. an FFT is computed along the time axis;
2. bins in ``[f - bandwidth, f + bandwidth]`` are selected;
3. bins in the left and right neighbouring bands, each of width
   ``neighbour_width``, provide a replacement amplitude;
4. the mean neighbouring amplitude replaces the selected amplitude for each
   channel; and
5. the original phase is combined with the new amplitude before the inverse
   real FFT.

If a target band contains no FFT bin, the nearest bin is selected. If no
neighbouring bins are available, that target is left untouched. Target
frequencies at or above Nyquist are not used by the functional routine and are
rejected when the estimator resolves its target list.

``line_freq`` may be a scalar fundamental frequency or an explicit sequence of
frequencies. With a scalar, ``n_harmonics`` counts targets beginning with the
fundamental; ``None`` resolves all harmonics below Nyquist. The actual FFT-bin
spacing is ``sfreq / n_times``, so the same bandwidth can affect different
numbers of bins for different record lengths.

## Continuous and epoched data

The operation is record-based. A continuous recording is transformed as one
FFT segment. Estimator input with shape ``(n_epochs, n_channels, n_times)`` is
flattened over the leading dimensions so each epoch/channel time series is
processed independently; spectra are not pooled across epoch boundaries. Short
epochs, non-integer numbers of line cycles, and edge transients can change the
neighbour estimates and should be inspected. Splitting a continuous recording
into epochs is therefore a scientific choice, not a layout-only conversion.

## Assumptions and limitations

The method assumes that the target line-noise peaks can be estimated from
adjacent spectral content and that replacing their amplitudes is preferable to
retaining them. It does not estimate a spatial artifact subspace, infer neural
source identity, or protect a neural rhythm that overlaps a target band. The
phase-preservation property concerns the Fourier representation; it is not
equivalent to preservation of neural information.

Report the sampling frequency, target frequencies or harmonic count, bandwidth,
neighbour width, record/epoch length, and any prior filtering. Compare both
line-noise attenuation and changes to signal-of-interest controls.

## Published method versus package behavior

Amplitude interpolation with phase reuse is the published method
{footcite:p}`leske_dalal2019_spectrum`. The functional/estimator interface,
explicit harmonic resolution, nearest-bin fallback, no-neighbour fallback, and
MNE container handling are mne-denoise implementation details. They should not
be read as additional claims from the publication.

## References

```{footbibliography}
```
