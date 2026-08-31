# Spectrum interpolation

Spectrum interpolation replaces amplitudes near target line-noise frequencies
with amplitudes estimated from neighboring frequencies while retaining the
original phase {footcite:p}`leske_dalal2019_spectrum`.

## Usage

```python
from mne_denoise.spectrum_interpolation import SpectrumInterpolation

model = SpectrumInterpolation(line_freq=60.0, n_harmonics=2)
clean = model.fit_transform(raw)
```

## Key points

- line_freq can be a scalar fundamental or an explicit target-frequency
  sequence. With a scalar, n_harmonics resolves targets below Nyquist.
- bandwidth defines the target interval and neighbour_width defines the
  neighboring amplitude bands.
- The functional API accepts (n_channels, n_times) arrays. The estimator
  also accepts (n_epochs, n_channels, n_times) and processes each record
  independently.
- MNE Raw, Epochs, and Evoked inputs are copied; non-selected channels are
  preserved.
- FFT resolution depends on segment length (sfreq / n_times), so short
  segments may provide few neighboring bins.

The operation changes amplitudes in the target bins, including any neural
activity there; it is not a spatial artifact-subspace method.

## References

```{footbibliography}
```
