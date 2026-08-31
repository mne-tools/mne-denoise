"""Lead-field construction helpers."""

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
    """Three-layer spherical head model used by the fallback lead field.

    Parameters
    ----------
    inner_skull, outer_skull, scalp : float, default=81, 85, 88
        Shell radii in millimetres.
    dipole_shell : float, default=76
        Radius of the source shell in millimetres.
    brain_conductivity, scalp_conductivity : float, default=0.33
        Conductivities in S/m.
    skull_conductivity_ratio : float, default=1 / 50
        Skull-to-brain conductivity ratio.

    See Also
    --------
    REFERENCE_HEAD : Default model instance.
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


def fibonacci_sphere(n_points: int) -> np.ndarray:
    """Generate deterministic unit vectors on a Fibonacci sphere.

    Parameters
    ----------
    n_points : int
        Number of directions; must be positive.

    Returns
    -------
    directions : ndarray, shape (n_points, 3)
        Unit vectors.
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
    """Build an average-referenced spherical lead field with MNE-Python.

    Parameters
    ----------
    info : mne.Info
        EEG measurement info with a montage.
    n_dipoles : int, default=5000
        Number of radial source dipoles.
    head_model : SphericalHeadModel, default=REFERENCE_HEAD
        Shell geometry and conductivities.
    verbose : bool, default=False
        Verbosity passed to MNE forward routines.

    Returns
    -------
    leadfield : ndarray, shape (n_channels, n_dipoles)
        Average-referenced lead field.

    Notes
    -----
    The source shell follows the fitted scalp radius. Dipoles use the deterministic
    Fibonacci construction above.
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
    """Resolve an average-referenced lead field for SOUND or SSP-SIR.

    Parameters
    ----------
    inst : mne.io.BaseRaw | mne.BaseEpochs | mne.Evoked | None
        MNE input, or None for array input.
    ch_names : list of str | None
        Processed MNE channel names, or None for arrays.
    n_channels : int
        Number of array-input channels.
    method : str
        Calling method name used in errors.
    forward : mne.Forward | None, default=None
        Forward solution; its rows are reordered to MNE channel order.
    n_dipoles : int, default=5000
        Number of dipoles for the spherical fallback.
    head_model : SphericalHeadModel, default=REFERENCE_HEAD
        Model for the spherical fallback.

    Returns
    -------
    leadfield : ndarray, shape (n_channels, n_sources)
        Average-referenced lead field.

    Raises
    ------
    ValueError
        If channel positions or a compatible forward solution are unavailable.
    """
    if inst is not None:
        _mne.require_mne("MNE lead-field resolution")
        info = inst.copy().pick(ch_names).info
        if forward is not None:
            _validate_leadfield(
                forward["sol"]["data"], what="The supplied forward gain matrix"
            )
            picked_forward = _mne.mne.pick_channels_forward(
                forward,
                include=info["ch_names"],
                ordered=True,
                copy=True,
            )
            gain = _validate_leadfield(
                picked_forward["sol"]["data"],
                what="The supplied forward gain matrix",
            )
            return _average_reference(gain)
        return make_spherical_leadfield(
            info, n_dipoles=n_dipoles, head_model=head_model
        )
    if forward is not None:
        gain = _validate_leadfield(
            forward["sol"]["data"], what="The supplied forward gain matrix"
        )
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
