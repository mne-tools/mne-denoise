"""Lead-field construction shared by the forward-model denoisers.

SOUND and SSP-SIR are both forward-model methods: they need a lead-field matrix
``L`` whose columns are the scalp topographies that cortical current sources
produce, and they use it as a prior for what a plausible brain signal looks
like.  Whatever a sensor records that ``L`` cannot explain is treated as noise
or artifact, so the lead field is what separates "brain" from "everything
else" in both algorithms.

An individualised forward model computed from the participant's anatomy is
always the better input.  When none is available, both methods fall back to a
three-layer spherical head model derived from the electrode montage alone
(Mutanen et al., 2016, 2018).  That is workable because neither algorithm uses
``L`` directly — only ``L @ L.T``, the lead-field covariance describing the
typical cross-correlations between channels — so a head model needs to capture
those correlations, not the anatomy that produced them.

This module centralises the construction so the two estimators resolve their
lead field identically:

- :class:`SphericalHeadModel` — the shell radii and conductivities of the
  fallback head model, with the published values as :data:`REFERENCE_HEAD`.
- :func:`fibonacci_sphere` — deterministic, quasi-uniform directions on a
  sphere, used to place the source dipoles.
- :func:`make_spherical_leadfield` — build the fallback lead field from a
  montage.
- :func:`resolve_leadfield` — choose between a user-supplied forward model and
  that fallback; this is the entry point the estimators call.

Every lead field returned here is average referenced, the reference both
algorithms operate in.

References
----------
Mutanen, T. P., Kukkonen, M., Nieminen, J. O., Stenroos, M., Sarvas, J., &
Ilmoniemi, R. J. (2016). Recovering TMS-evoked EEG responses masked by muscle
artifacts. NeuroImage, 139, 157-166.

Mutanen, T. P., Metsomaa, J., Liljander, S., & Ilmoniemi, R. J. (2018).
Automatic and robust noise suppression in EEG and MEG: The SOUND algorithm.
NeuroImage, 166, 135-151.

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np

from . import _mne

if TYPE_CHECKING:
    import mne

__all__ = [
    "REFERENCE_HEAD",
    "SphericalHeadModel",
    "fibonacci_sphere",
    "make_spherical_leadfield",
    "resolve_leadfield",
]


@dataclass(frozen=True)
class SphericalHeadModel:
    """Three-layer spherical head model for the fallback lead field.

    Radii are millimetres, as published, and the derived properties convert
    them to the relative form MNE-Python's sphere model expects.  Keeping
    millimetres as the single source of truth means each published value
    appears exactly once, the shells cannot drift out of proportion with one
    another, and the whole geometry rescales coherently when a montage's
    best-fit head radius differs from ``scalp``.

    Parameters
    ----------
    inner_skull : float
        Inner-skull radius in mm. Default 81.
    outer_skull : float
        Outer-skull radius in mm. Default 85.
    scalp : float
        Scalp radius in mm, the radius the others are expressed relative to.
        Default 88.
    dipole_shell : float
        Radius in mm of the shell on which the source dipoles are placed.
        Default 76.
    brain_conductivity : float
        Conductivity of the brain compartment in S/m. Default 0.33.
    scalp_conductivity : float
        Conductivity of the scalp compartment in S/m. Default 0.33.
    skull_conductivity_ratio : float
        Skull conductivity as a fraction of the brain's. Default 1/50, the
        contrast used in the publications this fallback follows.

    See Also
    --------
    REFERENCE_HEAD : The published defaults, pre-instantiated.

    Examples
    --------
    >>> from mne_denoise._leadfield import SphericalHeadModel
    >>> head = SphericalHeadModel()
    >>> [round(r, 4) for r in head.relative_radii]
    [0.9205, 0.9659, 1.0]
    >>> round(head.conductivities[1], 4)  # skull, 1/50 of brain
    0.0066
    """

    inner_skull: float = 81.0
    outer_skull: float = 85.0
    scalp: float = 88.0
    dipole_shell: float = 76.0
    brain_conductivity: float = 0.33
    scalp_conductivity: float = 0.33
    skull_conductivity_ratio: float = 1.0 / 50.0

    @property
    def relative_radii(self) -> tuple[float, float, float]:
        """Shell radii relative to the scalp, as ``make_sphere_model`` expects."""
        return (self.inner_skull / self.scalp, self.outer_skull / self.scalp, 1.0)

    @property
    def conductivities(self) -> tuple[float, float, float]:
        """Brain / skull / scalp conductivities in S/m."""
        return (
            self.brain_conductivity,
            self.brain_conductivity * self.skull_conductivity_ratio,
            self.scalp_conductivity,
        )

    @property
    def dipole_relative_radius(self) -> float:
        """Radius of the source shell relative to the scalp."""
        return self.dipole_shell / self.scalp


#: Head model of the published SOUND / SSP-SIR fallback: shells at 81 / 85 /
#: 88 mm, sources at 76 mm, and a skull 50 times less conductive than brain.
REFERENCE_HEAD = SphericalHeadModel()


def _average_reference(leadfield: np.ndarray) -> np.ndarray:
    """Re-reference a lead field to the average over channels."""
    return leadfield - leadfield.mean(axis=0, keepdims=True)


def _validate_leadfield(
    leadfield: np.ndarray, *, what: str = "leadfield"
) -> np.ndarray:
    """Return a finite, non-empty two-dimensional leadfield matrix."""
    leadfield = np.asarray(leadfield, dtype=float)
    if leadfield.ndim != 2:
        raise ValueError(f"{what} must be 2D, got shape {leadfield.shape}.")
    if 0 in leadfield.shape:
        raise ValueError(
            f"{what} must contain at least one channel and one source column."
        )
    if not np.isfinite(leadfield).all():
        raise ValueError(f"{what} must contain only finite values.")
    return leadfield


def _forward_gain(forward: mne.Forward) -> np.ndarray:
    """Extract and validate the gain matrix of a forward solution."""
    return _validate_leadfield(
        forward["sol"]["data"], what="The supplied forward gain matrix"
    )


def _leadfield_from_forward(forward: mne.Forward, info: mne.Info) -> np.ndarray:
    """Extract an average-referenced lead field from a user forward solution.

    The forward's rows are reordered to match ``info``'s channel order, so a
    forward computed elsewhere (with its own channel ordering) lines up with
    the data being cleaned.
    """
    gain = _forward_gain(forward)
    row_names = list(forward["sol"]["row_names"])
    if len(row_names) != gain.shape[0]:
        raise ValueError(
            "The supplied forward has a different number of row names and "
            "gain-matrix rows."
        )
    wanted = list(info["ch_names"])
    missing = [ch for ch in wanted if ch not in row_names]
    if missing:
        raise ValueError(
            "The supplied forward model is missing channels present in the data: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}."
        )
    idx = [row_names.index(ch) for ch in wanted]
    return _average_reference(gain[idx])


def fibonacci_sphere(n_points: int) -> np.ndarray:
    """Generate unit vectors quasi-uniformly covering the sphere.

    Points are placed on a Fibonacci lattice, which spreads them more evenly
    than random sampling and, being deterministic, makes any lead field built
    from them reproducible run to run.

    Parameters
    ----------
    n_points : int
        Number of directions to generate. Must be at least 1.

    Returns
    -------
    directions : ndarray, shape (n_points, 3)
        Unit vectors. The same ``n_points`` always yields the same set.

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise._leadfield import fibonacci_sphere
    >>> directions = fibonacci_sphere(100)
    >>> bool(np.allclose(np.linalg.norm(directions, axis=1), 1.0))
    True
    """
    if n_points < 1:
        raise ValueError(f"n_points must be >= 1, got {n_points}.")
    idx = np.arange(n_points) + 0.5
    polar = np.arccos(1.0 - 2.0 * idx / n_points)
    azimuth = np.pi * (1.0 + 5.0**0.5) * idx
    return np.column_stack(
        [
            np.cos(azimuth) * np.sin(polar),
            np.sin(azimuth) * np.sin(polar),
            np.cos(polar),
        ]
    )


def make_spherical_leadfield(
    info: mne.Info,
    *,
    n_dipoles: int = 5000,
    head_model: SphericalHeadModel = REFERENCE_HEAD,
    verbose: bool = False,
) -> np.ndarray:
    """Build an average-referenced lead field from a spherical head model.

    Radially oriented dipoles are placed on a shell inside the inner skull and
    their topographies computed with MNE-Python's forward solver.  The sphere
    is fitted to the montage, so the geometry follows the participant's head
    size even though no anatomy is available.

    Parameters
    ----------
    info : mne.Info
        Measurement info carrying EEG channel positions; a montage must be set.
    n_dipoles : int
        Number of source dipoles on the shell. Default 5000, at which
        ``L @ L.T`` is converged for practical purposes — halving it changes
        the lead-field covariance by only a few percent.
    head_model : SphericalHeadModel
        Shell radii and conductivities. Defaults to :data:`REFERENCE_HEAD`.
    verbose : bool
        Passed through to the MNE-Python forward routines.

    Returns
    -------
    leadfield : ndarray, shape (n_channels, n_dipoles)
        Average-referenced lead field with fixed (radial) source orientations.

    See Also
    --------
    resolve_leadfield : Choose between this and a user-supplied forward model.

    Notes
    -----
    Only ``leadfield @ leadfield.T`` enters SOUND and SSP-SIR, where it acts as
    the source-covariance prior: it sets the minimum-norm weighting in SOUND
    and the truncation scale of the source-informed reconstruction in SSP-SIR.
    Its *spectrum* therefore matters, not merely its span, which is why the
    published shell-of-radial-dipoles geometry is followed rather than a volume
    grid.  A volume grid with free orientations spans a comparable subspace
    (mean principal-angle cosine ~0.94 over the leading topographies for a
    32-channel montage) but has a visibly faster-decaying spectrum, changing
    the effective regularisation.

    The published construction draws the dipoles at random; a deterministic
    lattice is used here instead.  Two random 5000-dipole draws differ from
    each other in ``L @ L.T`` by roughly 13%, and the lattice sits about 10%
    from such a draw — inside that sampling spread, while being exactly
    repeatable.
    """
    if (
        isinstance(n_dipoles, (bool, np.bool_))
        or not isinstance(n_dipoles, Integral)
        or n_dipoles < 1
    ):
        raise ValueError(f"n_dipoles must be a positive integer, got {n_dipoles!r}.")
    _mne.require_mne("automatic spherical lead-field construction")
    eeg_picks = _mne.mne.pick_types(info, meg=False, eeg=True, exclude=())
    if len(eeg_picks) != len(info["ch_names"]):
        raise ValueError(
            "Automatic spherical lead-field construction supports EEG channels "
            "only; provide an explicit forward model for MEG or mixed channel types."
        )

    with warnings.catch_warnings():
        # The best-fit sphere centre can sit >20 mm from the head-frame origin
        # for partial or idealised montages; harmless for spanning the
        # topography subspace that SOUND and SSP-SIR rely on.
        warnings.filterwarnings("ignore", message=".*from head frame origin.*")
        sphere = _mne.mne.make_sphere_model(
            r0="auto",
            head_radius="auto",
            info=info,
            relative_radii=head_model.relative_radii,
            sigmas=head_model.conductivities,
            verbose=verbose,
        )
        # ``sphere["layers"][-1]["rad"]`` is the fitted scalp radius in metres;
        # the source shell tracks it so the geometry stays proportional.
        head_radius = float(sphere["layers"][-1]["rad"])
        directions = fibonacci_sphere(n_dipoles)
        positions = (
            directions * (head_model.dipole_relative_radius * head_radius)
            + sphere["r0"]
        )
        src = _mne.mne.setup_volume_source_space(
            pos={"rr": positions, "nn": directions}, sphere_units="m", verbose=verbose
        )
        fwd = _mne.mne.make_forward_solution(
            info, trans=None, src=src, bem=sphere, eeg=True, meg=False, verbose=verbose
        )
        fwd = _mne.mne.convert_forward_solution(
            fwd, force_fixed=True, use_cps=False, verbose=verbose
        )
    return _average_reference(np.asarray(fwd["sol"]["data"], dtype=float))


def resolve_leadfield(
    *,
    inst: mne.io.BaseRaw | mne.BaseEpochs | mne.Evoked | None,
    ch_names: list[str] | None,
    n_channels: int,
    method: str,
    forward: mne.Forward | None = None,
    n_dipoles: int = 5000,
    head_model: SphericalHeadModel = REFERENCE_HEAD,
) -> np.ndarray:
    """Resolve the lead field an estimator should use.

    Both :class:`~mne_denoise.sound.SOUND` and
    :class:`~mne_denoise.sspsir.SSPSIR` accept the same three kinds of input,
    and this function is where that choice is made once for both:

    - **MNE object, no forward** — build the spherical fallback from its
      montage.
    - **MNE object with a forward** — use the forward, reordering its rows to
      the object's channel order so the two line up by name.
    - **Plain array with a forward** — use the forward as given; with no
      channel names to match on, only the channel *count* can be checked.

    A plain array without a forward carries no channel positions at all and
    raises.

    All arguments are keyword-only, because ``ch_names`` matters only on the
    MNE-object paths and ``n_channels`` only on the array path; naming them at
    the call site keeps which-applies-when visible.

    Parameters
    ----------
    inst : mne.io.BaseRaw | mne.Epochs | mne.Evoked | None
        The MNE object being cleaned, or None for plain-array input.
    ch_names : list of str | None
        Names of the channels being processed. Used to pick the montage and to
        align ``forward``; ignored when ``inst`` is None.
    n_channels : int
        Number of channels in the data. Used to validate ``forward`` on the
        array path, where no names are available.
    method : str
        Name of the calling method, used in the error raised when no channel
        positions can be found.
    forward : mne.Forward | None
        Pre-computed forward solution. Preferred over the spherical fallback
        whenever an anatomically accurate model is available.
    n_dipoles : int
        Number of dipoles for the spherical fallback. See
        :func:`make_spherical_leadfield`.
    head_model : SphericalHeadModel
        Head model for the spherical fallback. Defaults to
        :data:`REFERENCE_HEAD`.

    Returns
    -------
    leadfield : ndarray, shape (n_channels, n_sources)
        Average-referenced lead field.

    Raises
    ------
    ValueError
        If no channel positions are available, if ``forward`` is missing
        channels present in the data, or if its channel count disagrees with
        the data on the array path.

    See Also
    --------
    make_spherical_leadfield : The fallback this dispatches to.
    """
    if inst is not None:
        _mne.require_mne("MNE lead-field resolution")
        info = inst.copy().pick(ch_names).info
        if forward is not None:
            return _leadfield_from_forward(forward, info)
        return make_spherical_leadfield(
            info, n_dipoles=n_dipoles, head_model=head_model
        )
    if forward is not None:
        gain = _forward_gain(forward)
        if gain.shape[0] != n_channels:
            raise ValueError(
                "For array input, the forward must have the same number of "
                f"channels as the data ({gain.shape[0]} vs {n_channels})."
            )
        return _average_reference(gain)
    raise ValueError(
        f"{method} needs channel positions: pass an MNE object with a montage, "
        "or provide a `forward` for array input."
    )
