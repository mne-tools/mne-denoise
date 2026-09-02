r"""
Removing 60-Hz line noise from real MEG with ZapLine
=====================================================

Can standard ZapLine reduce the 60-Hz line-noise peak in a real multichannel
MEG recording while limiting broadband spectral distortion? The MNE Sample
gradiometers provide a compact real-data example with no known clean neural
reference, so attenuation and spectral preservation are reported as separate
quantities.

ZapLine combines spectral and spatial filtering for multichannel recordings
:footcite:p:`decheveigne2020_zapline`. A reduced line peak demonstrates artifact
attenuation in this recording; an off-line spectral change is only a
preservation control and does not establish that neural structure was preserved.

References
----------
.. footbibliography::
"""

# %%
# Load a compact MNE Sample MEG recording
# ---------------------------------------
import mne
import numpy as np

from mne_denoise.qa import below_noise_distortion_db, suppression_ratio
from mne_denoise.viz import plot_psd_comparison
from mne_denoise.zapline import ZapLine

sample_path = mne.datasets.sample.data_path()
raw = mne.io.read_raw_fif(
    sample_path / "MEG" / "sample" / "sample_audvis_raw.fif",
    preload=False,
    verbose="ERROR",
)
raw.pick("grad", exclude="bads").crop(0.0, 30.0).load_data()
raw.resample(300.0, verbose="ERROR")

# %%
# Fit standard ZapLine and evaluate two real-data endpoints
# ----------------------------------------------------------
model = ZapLine(
    sfreq=raw.info["sfreq"],
    line_freq=60.0,
    n_select="auto",
    verbose=False,
)
cleaned = model.fit_transform(raw)

# Use the same MNE PSD settings before and after cleaning. The public QA
# functions require the resulting arrays, while the Raw objects remain the
# inputs to the estimator and visualization.
n_fft = int(round(raw.info["sfreq"] * 4.0))
spectrum_before = raw.compute_psd(
    fmin=1.0,
    fmax=140.0,
    n_fft=n_fft,
    verbose="ERROR",
)
spectrum_after = cleaned.compute_psd(
    fmin=1.0,
    fmax=140.0,
    n_fft=n_fft,
    verbose="ERROR",
)
freqs = spectrum_before.freqs
psd_before = spectrum_before.get_data(return_freqs=False)
psd_after = spectrum_after.get_data(return_freqs=False)

line_suppression = suppression_ratio(
    freqs,
    psd_before,
    psd_after,
    target_freq=60.0,
    bandwidth=2.0,
)
off_band_distortion = below_noise_distortion_db(
    freqs,
    psd_before,
    psd_after,
    exclude_freq=60.0,
    exclude_bw=5.0,
    fmin=2.0,
    fmax=140.0,
    n_harmonics=1,
)

print(f"MEG gradiometer channels: {len(raw.ch_names)}")
print(f"Recording: {raw.times[-1]:.1f} s at {raw.info['sfreq']:.1f} Hz")
print(f"Components removed: {model.n_removed_}")
print(f"60-Hz suppression: {line_suppression:.2f} dB")
print(
    "Median off-line-band spectral distortion "
    f"(preservation control): {np.median(off_band_distortion):.2f} dB"
)
print("The spectral control is not a clean neural ground truth.")

# %%
# Inspect the before/after spectrum
# ---------------------------------
plot_psd_comparison(
    raw,
    cleaned,
    fmin=1.0,
    fmax=140.0,
    line_freq=60.0,
    show=False,
)
