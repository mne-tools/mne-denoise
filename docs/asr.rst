Artifact Subspace Reconstruction
================================

Overview
--------

Artifact Subspace Reconstruction (ASR) estimates a reference covariance from
relatively clean data, detects windows whose channel covariance exceeds that
reference, and reconstructs the affected subspace. The standard method follows
the signal-reconstruction formulation in the Kothe and Jung patent and the
evaluation/implementation literature by Chang et al.
:footcite:p:`kothe_jung2016_asr,chang2018_asr,chang2020_asr`.

ASR is intended for transient, high-variance, spatially structured artifacts in
continuous EEG. It is not a general-purpose filter, and artifact attenuation
does not by itself establish preservation of neural activity.

Standard estimator
------------------

For an MNE ``Raw`` object, the estimator can select EEG channels by default and
returns a copied container with non-selected channels untouched:

.. code-block:: python

   from mne_denoise.asr import ASR

   asr = ASR(cutoff=20.0, calibration="auto", picks="eeg")
   asr.fit(raw)
   raw_clean = asr.transform(raw)

For a NumPy array, pass ``sfreq`` explicitly. The array layout is
``(n_channels, n_times)``:

.. code-block:: python

   asr = ASR(sfreq=250.0, cutoff=20.0)
   clean = asr.fit_transform(data)

``fit`` calibrates the reference state. ``transform`` applies that fitted state
to a recording and does not mutate the input. ``fit_transform`` performs both
operations. The estimator exposes calibration and reconstruction diagnostics,
including the clean-window mask, the last repair mask, reconstructed-component
counts, and window-level thresholds.

Calibration and reconstruction
-----------------------------

The calibration data should represent the clean covariance of the channel type
being processed. ``calibration="auto"`` selects windows using robust RMS/
covariance criteria; an explicit mask or calibration data can be used when the
application has a trusted clean period. The cutoff is a threshold in the
whitened component space, not a universal percentage of variance or a
transferable artifact-amplitude scale.

ASR repairs affected windows using a spatial reconstruction and blends the
result over the processing window. The package also exposes final clean-window
rejection diagnostics and ``to_annotations`` so that changed or rejected spans
can be inspected without deleting samples. These annotation and diagnostics
APIs are software contracts; they are not additional scientific validation.

ASR should be applied with the surrounding preprocessing made explicit. In
particular, review high-pass filtering, referencing, bad-channel handling,
rank-reducing projectors, and whether the calibration period contains the
signal and artifact regimes relevant to the analysis. Filtering used only for
statistics in an ASR configuration does not replace the user's preprocessing.

Riemannian and adaptive variants
-------------------------------

The package exposes several research variants. Their names should be reported
alongside the standard parameters because they alter calibration or state
updates.

``method="riemannian_windowed"`` uses a robust Riemannian calibration
covariance and then a windowed reconstruction path. Its cutoff and reconstruction
behavior are package-specific and should be checked on representative data.
``method="riemannian"`` is retained for compatibility with the
``rASRMatlab``-style path and requires the explicit experimental opt-in. The
Riemannian ASR formulation is related to the modification of Blum et al.
:footcite:p:`blum2019_riemannian_asr`.

``AdaptiveASR`` updates the calibration state between processing chunks. Its
PSP/PSW and moving-window variants are related to the adaptive ASR work of Tsai
et al. :footcite:p:`tsai2024_adaptive_asr`. The package's chunking, state-reset,
and final-state semantics are implementation conventions. Use complete update
segments, document the variant and update settings, and validate the effect on
both artifacts and neural controls.

``JugglerASR`` changes the calibration-reference selection strategy while
retaining the standard reconstruction stage. Its ``dbscan`` and ``gev``
strategies are based on the Juggler's ASR description by Kim et al.
:footcite:p:`kim2025_juggler_asr`. The local package implementation is covered
by software tests and the cited method description; this is not a universal
replacement for standard ASR.

``GuidedASR`` and ``process_guided_asr`` are unpublished, unvalidated
experimental research APIs. They add artifact/preserve bias information to
soft reconstruction. Do not present them as established preprocessing methods;
validate signal preservation and artifact attenuation independently before any
scientific use.

Choosing parameters
-------------------

There is no source-backed universal cutoff, calibration duration, variant, or
preprocessing sequence for all EEG/MEG recordings. A sensible starting point
depends on sampling rate, channel type, expected artifact duration, clean-data
availability, and whether the analysis is offline or streaming. Freeze those
choices before held-out evaluation, and report the calibration rule, cutoff,
window/step settings, channel picks, rank, and variant.

For streaming use, ``AdaptiveASR`` exposes ``partial_fit`` and state-reset
operations. These make stateful processing possible; they do not remove the
need to define chunk boundaries and validate state carry-over.

Evaluation
----------

Inspect at least the clean-window mask, repaired/rejected spans, component
reconstruction counts, and a before/after signal measure. A lower artifact
metric can coexist with distortion of the signal of interest. Use held-out
events, simulated controls, or independent recordings where possible. See
:doc:`evaluation` for package QA utilities and forward-model overcorrection
metrics.

References
----------

.. footbibliography::
