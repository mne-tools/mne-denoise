(evaluation)=
# Evaluating denoising

## Attenuation is not preservation

Every denoiser can reduce an artifact by also reducing signal that shares its
time, frequency, spatial direction, or forward-model mismatch. Evaluate two
questions separately:

1. Did the target artifact become smaller?
2. Did the signal of interest remain acceptably unchanged?

Use controls suited to the experiment: held-out recordings, simulated mixtures
with known sources, unaffected events or channels, surrogate data, and a
forward model where one is scientifically justified. A single lower PSD,
variance, or residual amplitude is not sufficient evidence of successful
denoising.

## Spectral QA functions

The public functions in ``mne_denoise.qa`` operate on precomputed PSD arrays.
The expected PSD layout is ``(n_freqs,)`` for one aggregate spectrum or
``(n_channels, n_freqs)`` for channel-wise spectra. They return a scalar for
1D input or a per-channel array where noted.

| Function | Definition and interpretation |
| --- | --- |
| ``peak_attenuation_db`` | ``10 log10(max(before) / max(after))`` in the target band. Positive dB means the maximum target peak was reduced; 1D is scalar and 2D is per channel. |
| ``suppression_ratio`` | ``10 log10(mean(before) / mean(after))`` in the target band, after averaging channels for 2D input. Positive dB means lower mean target power; an empty band is ``NaN`` and non-positive after-power gives ``+inf``. |
| ``noise_surround_ratio`` | Mean power in the peak window divided by mean power in the left/right surrounding windows. Values above 1 indicate a residual peak; output is scalar or per channel. |
| ``underclean_proportion`` | Fraction of channels whose ``noise_surround_ratio`` exceeds ``threshold_ratio``. It is a proportion in ``[0, 1]``; a 1D input returns ``0.0`` or ``1.0``. |
| ``overclean_proportion`` | Fraction of channels whose surrounding spectral floor is attenuated by more than ``threshold_db``. It is a proportion in ``[0, 1]`` and does not measure every frequency. |
| ``below_noise_distortion_db`` | Mean absolute ``10 log10(after / before)`` outside excluded target bands. Lower dB means less broadband spectral change; output is scalar or per channel. |
| ``spectral_distortion`` | RMS dB log-ratio over the package's 2--160 Hz evaluation range after excluding target harmonics. Positive and negative changes contribute equally; this is package-defined. |
| ``geometric_mean_psd_ratio`` | Geometric mean of ``after / before`` over a selected frequency range. 1 means no net change, below 1 net attenuation, and output is scalar or per channel. |

The target-band widths, harmonic exclusions, frequency limits, and PSD floors
are parameters of the individual functions. Report them with the result.
Empty masks have documented sentinel behavior: for example, peak attenuation
returns ``NaN``, broadband distortion returns zero, and the geometric ratio
returns one. A missing surround band is floored at ``1e-30`` in the peak-to-
surround calculation, so an actual nonzero peak can produce a very large
underclean ratio.

## Data-change utilities

The remaining public QA helpers operate on before/after arrays with matching
shapes:

* ``variance_removed`` returns ``100 * (1 - var(after) / var(before))``. It is
  dimensionless and expressed as a percentage; positive means lower overall
  variance and negative means increased variance. A zero before-variance
  returns ``0.0``.
* ``rms_change`` returns the RMS of ``before - after`` in the input data units.
  Zero means identical arrays.
* ``max_abs_change`` returns the largest absolute sample-wise change in the
  input units; empty input raises because no maximum exists.
* ``channel_variance_ratio`` returns ``var(after) / var(before)`` per channel.
  It accepts ``(n_channels, n_times)`` or ``(n_epochs, n_channels, n_times)``;
  the latter pools over epochs and time. Values below one mean lower variance,
  not necessarily better signal preservation.

``compute_all_qa_metrics`` is a package-defined convenience wrapper for two
MNE ``Raw`` objects. It computes PSDs through MNE, reports channel medians,
and returns fundamental plus per-harmonic attenuation and residual-ratio
diagnostics. It is not a published validation protocol; choose the PSD
settings, line frequency, harmonics, and comparison windows explicitly.

## Forward-model overcorrection

``quantify_overcorrection(operator, leadfield)`` evaluates a fitted linear
sensor operator without requiring a neural ground truth. For each lead-field
column ``l``, it compares the original topography with ``l' = operator @ l``:

```text
amplitude_change = (||l'|| - ||l||) / ||l||
correlation      = (l' . l) / (||l'|| ||l||)
relative_error   = ||l' - l|| / ||l||
goodness_of_fit  = 1 - relative_error**2
```

``amplitude_change`` is dimensionless: zero is unchanged magnitude and ``-1``
is complete deletion. ``correlation`` describes topographic shape, with 1
identical direction and ``NaN`` when either topography has zero norm.
``relative_error`` is zero for an identical topography and grows with combined
magnitude/shape distortion. ``goodness_of_fit`` is one for an identical
topography and zero for deletion; it can be negative when the filtered
topography is farther from the original than zero would be. Zero-norm sources
produce ``NaN`` for the undefined relative quantities.

The forward-model rationale is discussed by Mutanen et al.
{footcite:p}`mutanen2022_source_artifact`, but these four exact formulas are
mne-denoise utility definitions, not metrics attributed to that paper. Use the
same channel reference, channel order, and lead field for the operator and
comparison. Values are comparative within a sensor geometry; they are not
universal cross-dataset scores.

## Evidence boundary

Published methods provide scientific motivation and, where applicable,
source-specific evaluation procedures. Package tests establish software
contracts and numerical behavior; they do not establish that a method is
validated for every artifact, montage, preprocessing pipeline, or neural
endpoint. Document the method, fitted settings, preprocessing, evaluation
windows, and controls used for each result.

## References

```{footbibliography}
```
