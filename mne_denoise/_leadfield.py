"""Shared spherical lead-field construction for forward-model-based denoisers.

SOUND and SSP-SIR both require an EEG lead-field matrix. When the user does not
supply an individualised forward model, both methods (following Mutanen et al.,
2016, 2018) fall back to a simple three-layer spherical head model built from the
electrode montage. This module centralises that construction by reusing
MNE-Python's tested forward machinery (``make_sphere_model`` →
``setup_volume_source_space`` → ``make_forward_solution``), and returns the
lead field in the average reference both methods operate in.

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

from __future__ import annotations

import warnings

import numpy as np

# Three-layer spherical defaults from the SOUND/SSP-SIR reference implementation
# (Mutanen et al. 2016/2018): inner-skull / outer-skull / scalp radii 81/85/88 mm
# and brain / skull / scalp conductivities 0.33 / 0.33-over-50 / 0.33 S/m.
_RELATIVE_RADII = (81.0 / 88.0, 85.0 / 88.0, 1.0)
_SIGMAS = (0.33, 0.33 / 50.0, 0.33)


def _average_reference(leadfield: np.ndarray) -> np.ndarray:
    """Re-reference a lead field to the average reference (over channels)."""
    return leadfield - leadfield.mean(axis=0, keepdims=True)


def _leadfield_from_forward(forward, info) -> np.ndarray:
    """Extract an average-referenced lead field from a user ``mne.Forward``."""
    gain = np.asarray(forward["sol"]["data"], dtype=float)
    row_names = list(forward["sol"]["row_names"])
    wanted = list(info["ch_names"])
    missing = [ch for ch in wanted if ch not in row_names]
    if missing:
        raise ValueError(
            "The supplied forward model is missing channels present in the data: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}."
        )
    idx = [row_names.index(ch) for ch in wanted]
    return _average_reference(gain[idx])


def make_spherical_leadfield(
    info,
    *,
    forward=None,
    pos: float = 15.0,
    relative_radii: tuple[float, ...] = _RELATIVE_RADII,
    sigmas: tuple[float, ...] = _SIGMAS,
    verbose: bool = False,
) -> np.ndarray:
    """Build (or extract) an average-referenced EEG lead field.

    Parameters
    ----------
    info : mne.Info
        Measurement info with EEG channel positions (a montage must be set).
    forward : mne.Forward | None
        A pre-computed forward solution. If given, its gain matrix is used
        (channels aligned to ``info``) instead of building a spherical model.
    pos : float
        Grid spacing in mm for the volume source space when building the
        spherical model. Smaller values give more sources. Default 15.0.
    relative_radii : tuple of float
        Relative radii of the spherical shells (inner skull / outer skull /
        scalp), defaulting to the SOUND/SSP-SIR values (81/85/88 mm).
    sigmas : tuple of float
        Conductivities of the shells (brain / skull / scalp), defaulting to
        0.33 / 0.0066 / 0.33 S/m.
    verbose : bool
        Passed through to the MNE forward routines.

    Returns
    -------
    leadfield : ndarray, shape (n_channels, n_sources)
        Average-referenced lead-field matrix (free-orientation gain).

    Notes
    -----
    The spherical model reuses MNE-Python's forward solver and is equivalent to
    the spherical lead field of the original MATLAB implementation in the sense
    that it spans the same plausible-topography subspace; for best results,
    supply an anatomically accurate ``forward`` when available.
    """
    if forward is not None:
        return _leadfield_from_forward(forward, info)

    import mne

    with warnings.catch_warnings():
        # The best-fit sphere centre can sit >20 mm from the head-frame origin
        # for partial/idealised montages; harmless for spanning the topography
        # subspace that SOUND/SSP-SIR rely on.
        warnings.filterwarnings("ignore", message=".*from head frame origin.*")
        sphere = mne.make_sphere_model(
            r0="auto",
            head_radius="auto",
            info=info,
            relative_radii=relative_radii,
            sigmas=sigmas,
            verbose=verbose,
        )
        src = mne.setup_volume_source_space(
            sphere=sphere, pos=pos, sphere_units="mm", verbose=verbose
        )
        fwd = mne.make_forward_solution(
            info, trans=None, src=src, bem=sphere, eeg=True, meg=False, verbose=verbose
        )
    return _average_reference(np.asarray(fwd["sol"]["data"], dtype=float))

