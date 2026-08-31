(evaluation)=
# Evaluating denoising

Artifact attenuation and preservation of the signal of interest are separate
questions. A lower peak, PSD, variance, or residual amplitude is not by itself
evidence that the desired neural signal was preserved.

Use controls suited to the experiment, such as held-out data, simulated
mixtures, unaffected events or channels, surrogate data, or a forward model
when appropriate. Compare both artifact attenuation and changes to the signal
of interest.

The public mne_denoise.qa module provides spectral and data-change metrics.
mne_denoise.quantify_overcorrection compares a fitted sensor operator with a
lead field using amplitude, correlation, relative-error, and goodness-of-fit
definitions owned by this package. The individual APIs document formulas,
units, and sentinel values.

Useful evaluation categories include:

- target-artifact attenuation;
- broadband or signal-of-interest change; and
- forward-model distortion when a source model is available.

Choose and report the frequency bands, windows, references, and preprocessing
used for each comparison.
