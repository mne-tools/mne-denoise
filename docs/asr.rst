Artifact Subspace Reconstruction
================================

Artifact Subspace Reconstruction (ASR) repairs short, high-amplitude,
spatially structured EEG artifacts by calibrating a clean covariance model and
reconstructing burst-contaminated subspaces with the standard clean_rawdata
lookahead, moving-covariance, and raised-cosine blending procedure.

The production target is standard Euclidean ASR. The module also exposes
specialized Juggler-style calibration variants, MATLAB-fixture-backed adaptive
ASR variants, and an experimental Riemannian backend; these variants are useful
for research and comparison workflows but should be reported explicitly when
used.

Basic usage
-----------

For MNE ``Raw`` objects, ASR defaults to EEG channels and leaves other channel
types unchanged.

.. code-block:: python

   from mne_denoise.asr import ASR

   asr = ASR(
       cutoff=20.0,
       calibration="auto",
       picks="eeg",
   )

   asr.fit(raw)
   raw_clean = asr.transform(raw)

For NumPy arrays, pass the sampling frequency explicitly. Arrays use shape
``(n_channels, n_times)``.

.. code-block:: python

   asr = ASR(sfreq=250.0, cutoff=20.0)
   clean = asr.fit_transform(data)

Typical preprocessing pipeline
------------------------------

ASR sits between high-pass filtering and ICA in a standard EEG pipeline. It is
applied to continuous data *after* filtering and *before* epoching/ICA, so that
high-amplitude bursts do not bias the ICA decomposition:

.. code-block:: python

   import mne
   from mne_denoise.asr import ASR

   raw = mne.io.read_raw_fif("sub-01_raw.fif", preload=True)
   raw.set_eeg_reference("average")

   # 1. High-pass filter (ASR assumes high-pass-filtered data; see below).
   raw.filter(l_freq=1.0, h_freq=None)

   # 2. ASR: calibrate on the clean parts of the recording, then repair bursts.
   asr = ASR(cutoff=20.0, picks="eeg")
   raw_clean = asr.fit_transform(raw)

   # 3. ICA on the ASR-cleaned data (now free of high-variance bursts).
   ica = mne.preprocessing.ICA(
       n_components=0.99, method="infomax",
       fit_params=dict(extended=True), random_state=97,
   )
   ica.fit(raw_clean)

The repaired and rejected spans are also available as annotations via
``asr.to_annotations(...)`` so you can review what ASR changed without deleting
samples. The ``examples/asr`` gallery has runnable, synthetic versions of these
workflows.

Riemannian ASR
--------------

Two Riemannian backends are available.

``method="riemannian_windowed"`` is the **recommended** Riemannian backend and
is first-class (no ``experimental`` flag). It keeps the Riemannian
(geometric-median) robust calibration covariance but applies a standard
per-window eigendecomposition at processing time, so its ``cutoff`` knob works
the same monotone way as standard ASR:

.. code-block:: python

   asr = ASR(sfreq=250.0, cutoff=20.0, method="riemannian_windowed")
   clean = asr.fit_transform(data)

Its processing is byte-identical to standard ASR given the same calibration
state, and a direct ``clean_rawdata/asr_process`` MATLAB cross-check matches it
to ``relerr < 1e-13`` (see ``tests/parity/test_riemannian_windowed_parity.py``).

``method="riemannian"`` is the MATLAB-``rASRMatlab``-faithful backend and stays
behind an explicit experimental opt-in. It computes one covariance and one
reconstruction matrix for the whole stream, which makes it **cutoff-invariant
on real EEG** — use it only for MATLAB parity, not for cutoff tuning:

.. code-block:: python

   asr = ASR(sfreq=250.0, cutoff=20.0, method="riemannian", experimental=True)
   clean = asr.fit_transform(data)

Reference cross-checks
----------------------

MATLAB parity fixtures under ``tests/parity`` remain the authoritative
pass/fail validation for the standard and experimental backends.

For additional local source comparison, ``scripts/run_asr_reference_benchmark.py``
can benchmark against optional checkouts that are not shipped with this
repository:

- standard ASR against an optional ``refs/asr/repos/python-meegkit`` checkout
- experimental Riemannian ASR against an optional
  ``refs/asr/repos/timeflux_rasr`` checkout, as a qualitative comparison only

The ``timeflux_rasr`` comparison is intentionally not a parity test because
that implementation is epoched/trial-based and currently depends on older
``pyriemann`` internals. Current local source comparisons can diverge
substantially for the experimental Riemannian backend on synthetic continuous
data, so the MATLAB-backed parity fixtures remain the only supported oracle for
that path.

Adaptive ASR
------------

The local AASR MATLAB reference is exposed as :class:`mne_denoise.asr.AdaptiveASR`.
This variant keeps standard ASR burst reconstruction but updates the
calibration subspace between chunks using the Hebbian/anti-Hebbian PSP/PSW
rules from the AASR repository.

.. code-block:: python

   from mne_denoise.asr import AdaptiveASR

   aasr = AdaptiveASR(
       sfreq=250.0,
       cutoff=20.0,
       variant="psw",
   )

   aasr.fit(chunk_1)
   aasr.partial_fit(chunk_2)
   clean = aasr.transform(full_data)

The public API follows the package (sklearn-style) conventions:

- ``fit()`` for initial calibration; ``partial_fit()`` for adaptive updates
- ``transform()`` for burst repair with the current adaptive state
- ``reset_process_state()`` to replay the reconstruction path deterministically

A moving-window variant is available via ``variant="mw"``. Its
``mw_mode="sliding"`` option (calibrate-and-clean per window) is the
recommended MW configuration; the default ``mw_mode="final_state"`` mirrors the
MATLAB ``AASR_demo`` Cell 4 semantics.

The adaptive variants are specialized research paths validated against the
MATLAB fixture files stored under ``tests/parity``. Those fixtures were
generated from an AASR reference checkout for both:

- ``variant="psp"``
- ``variant="psw"``

across first-update and repeated-update cases. The reference checkout itself is
not required to use or test the package.

JugglerASR
----------

Juggler's ASR keeps the standard ASR burst-repair stage and replaces only the
reference-data selector used during calibration. Two source-backed strategies
from Kim et al. (2025) are exposed:

- ``strategy="dbscan"``: top-five amplitude features clustered with
  Chebyshev-distance DBSCAN
- ``strategy="gev"``: a fitted generalized extreme-value model on the maximum
  per-sample amplitude

.. code-block:: python

   from mne_denoise.asr import JugglerASR

   jasr = JugglerASR(
       cutoff=20.0,
       strategy="dbscan",
   )

   jasr.fit(raw)
   raw_clean = jasr.transform(raw)
   reference_mask = jasr.get_calibration_mask()  # sample-based for Juggler

The paper specifies the sample-selection logic but does not provide a local
MATLAB oracle in this repository, so this implementation is validated through
unit tests and the published algorithm description rather than a parity
fixture. Treat it as a specialized calibration strategy for high-motion MoBI
data, not as a universal replacement for standard ASR.

Choosing a variant
------------------

A quick decision guide:

- **Most EEG** — ``ASR(method="standard")`` at ``cutoff=20`` (Chang 2020
  recommends 20-30 for adult EEG). The default, and the right default.
- **Need Riemannian-robust calibration with a working cutoff** —
  ``ASR(method="riemannian_windowed")``.
- **Online / streaming BCI** — ``AdaptiveASR(variant="psw")`` (strongest SNR
  gain) or ``variant="psp"`` (best ground-truth correlation).
- **Per-segment cleaning** — ``AdaptiveASR(variant="mw", mw_mode="sliding")``
  (avoid the coarse-window ``final_state`` default, which can over-clean).
- **Extreme MoBI / high motion** where the clean-windows criterion collapses —
  ``JugglerASR(strategy="gev")`` (tight selector) or ``strategy="dbscan"``
  (more permissive). These survive contamination levels where standard ASR
  refuses to calibrate.

Visualizing results
-------------------

``mne_denoise.viz`` ships three ASR-specific diagnostics that have no generic
equivalent --- :func:`~mne_denoise.viz.plot_asr_repair_timeline`,
:func:`~mne_denoise.viz.plot_asr_component_reconstruction`, and
:func:`~mne_denoise.viz.plot_asr_calibration_fraction`. They take the fitted
estimator and honour ``ax=`` / ``show=`` / ``fname=``:

.. code-block:: python

   from mne_denoise.viz import plot_asr_repair_timeline, plot_signal_overlay

   asr = ASR(sfreq=250.0, cutoff=20.0).fit(raw)
   clean = asr.transform(raw)
   plot_signal_overlay(raw, clean, raw.times, pick="Fp1")  # generic before/after
   plot_asr_repair_timeline(asr)                            # repaired windows

For before/after overlays, PSD comparison, and per-channel variance
topographies, reuse the generic helpers
:func:`~mne_denoise.viz.plot_signal_overlay`,
:func:`~mne_denoise.viz.plot_psd_comparison`, and
:func:`~mne_denoise.viz.plot_power_ratio_map` (they accept MNE objects or NumPy
arrays). The ``plot_05_asr_visualization.py`` gallery example exercises the
full set end-to-end.

Real-data validation
--------------------

Use ``scripts/run_asr_real_data_validation.py`` for local, reproducible
smoke validation on cached real EEG data. The script discovers files under
``.cache/asr_datasets`` and ``data`` by default, injects known burst artifacts
into a copy of the recording, runs the requested ASR variants, and writes JSON,
CSV, and Markdown reports with:

- wall time, process CPU time, sampled RSS peak, and Python allocation peak
- ASR calibration and processing memory modes
- shape, finite-output, channel-order, sampling-frequency, bad-channel, and
  annotation preservation checks
- injected-burst attenuation and non-burst distortion metrics

Example:

.. code-block:: console

   py -3.12 scripts/run_asr_real_data_validation.py \
       --max-duration 120 \
       --max-mem-mb 512 \
       --low-mem-mb 0.1 \
       --output reports/asr_real_data_validation.json \
       --fail-on-error

Important assumptions
---------------------

ASR should be applied after bad channels are removed or excluded and after the
data have been high-pass filtered in the surrounding MNE pipeline. The
statistics-only filter in ``ASR`` does not replace user-visible preprocessing;
it only affects covariance estimates. Average-reference projectors and other
rank-reducing projectors should be reviewed before calibration.

Diagnostics
-----------

After ``transform``, the estimator stores audit fields:

``clean_window_mask_``
   Calibration windows retained as clean.

``sample_mask_``
   Samples repaired during the last transform.

``n_components_reconstructed_``
   Number of reconstructed components per processing window.

``diagnostics_``
   Window starts, stops, variance estimates, thresholds, and summary fractions.
   Long-recording memory fields include ``memory_mode``, ``max_mem_mb``,
   ``estimated_full_cov_bytes``, ``peak_cov_buffer_bytes``, ``chunk_samples``,
   and ``used_memory_bound``.

``to_annotations(kind=...)``
   Unified annotation export. ``kind="repair"`` (default) annotates repaired
   windows from the last transform; ``kind="rejection"`` annotates samples
   removed by the final window-rejection pass; ``kind="calibration"`` annotates
   the reference samples chosen by :class:`JugglerASR` (sample-based backends
   only). All return ``mne.Annotations``.

``get_calibration_mask()``
   Returns the boolean calibration mask — window-based for standard / Riemannian
   / adaptive backends, sample-based for :class:`JugglerASR` (see
   ``calibration_mask_kind_``).

``get_rejection_mask()``
   Returns the retained-sample mask from optional clean_windows-style final
   window rejection.

``variance_removed()``
   Computes ASR-specific variance-change and repair-extent metrics from
   before/after data and an optional fitted ``ASR`` instance.

``compute_clean_window_mask()``
   Exposes clean_windows-style retained-sample masking as a standalone helper
   for continuous arrays.

Threshold fitting
-----------------

ASR calibration uses ``fit_rms_distribution()`` to estimate robust clean RMS
statistics for each calibration component. The fitter follows the
clean_rawdata truncated generalized-Gaussian grid search and stores
per-component ``threshold_mu``, ``threshold_sigma``, ``threshold_beta``, and
``threshold_fit_error`` in ``calibration_info_``.

Final window rejection
----------------------

``ASR`` can optionally apply a non-destructive clean_windows-style final pass
after burst repair by setting ``window_criterion`` and
``window_criterion_tolerances``. This mirrors clean_rawdata's distinction
between burst repair and later segment rejection:

.. code-block:: python

   asr = ASR(
       cutoff=20.0,
       calibration="auto",
       window_criterion=0.25,
       window_criterion_tolerances=(-np.inf, 7.0),
   )

   raw_clean = asr.fit_transform(raw)
   keep_mask = asr.get_rejection_mask()
   reject_annotations = asr.to_annotations("rejection")

This step does not delete samples from the returned object. It records the
retained/rejected mask and exposes it for downstream QC, annotation, or
manual trimming.
