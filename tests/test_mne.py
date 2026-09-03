"""Tests for optional MNE-Python integration helpers."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import mne_denoise._mne as mne_compat


def test_mne_availability_and_require_when_available():
    """The availability flag and requirement helper handle available MNE."""
    assert mne_compat.HAS_MNE is (mne_compat.mne is not None)
    if mne_compat.mne is None:
        pytest.skip("MNE-Python is not installed")

    assert mne_compat.require_mne("test feature") is None


def test_require_mne_when_unavailable(monkeypatch):
    """Requiring MNE reports the requested feature when unavailable."""
    monkeypatch.setattr(mne_compat, "mne", None)

    with pytest.raises(ImportError, match="SSP-SIR.*MNE-Python") as caught:
        mne_compat.require_mne("SSP-SIR")
    assert "mne-denoise[mne]" in str(caught.value)


def test_no_mne_imports_and_pure_numerical_cores_remain_usable():
    """The package and numerical cores work when imports of MNE are blocked."""
    probe = textwrap.dedent(
        """
        import builtins

        real_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            if name == "mne" or name.startswith("mne."):
                raise ModuleNotFoundError("No module named 'mne'", name="mne")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = blocked_import

        import numpy as np

        import mne_denoise
        import mne_denoise.viz
        from mne_denoise._cca import canonical_correlation
        from mne_denoise.dss import compute_dss, iterative_dss
        from mne_denoise.sns import compute_sns
        from mne_denoise.sound import compute_sound, compute_sound_ref_best
        from mne_denoise.spectrum_interpolation import interpolate_spectrum
        from mne_denoise.ssa import compute_basic_ssa

        rng = np.random.default_rng(0)
        data = rng.standard_normal((4, 128))
        leadfield = rng.standard_normal((4, 6))
        leadfield -= leadfield.mean(axis=0, keepdims=True)

        cleaned = interpolate_spectrum(data, 256.0, [60.0])
        assert cleaned.shape == data.shape
        _, _, correlations, _, _ = canonical_correlation(data.T, data[:2].T)
        assert correlations.size == 2

        compute_sound(data, leadfield, n_iter=1, random_state=0)
        compute_sound_ref_best(data, leadfield, n_iter=1, random_state=0)
        compute_dss(np.eye(4), np.diag([4.0, 3.0, 2.0, 1.0]), n_components=2)
        iterative_dss(data, lambda source: source**3, 1, max_iter=2, random_state=0)
        compute_sns(data, n_neighbors=2)
        compute_basic_ssa(data, 256.0, window_length=8)

        from mne_denoise._mne import HAS_MNE, mne

        assert not HAS_MNE
        assert mne is None
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_mne_specific_leadfield_path_uses_canonical_gate(monkeypatch):
    """MNE-only spherical construction fails explicitly at its feature boundary."""
    mne = pytest.importorskip("mne")
    from mne_denoise._leadfield import make_spherical_leadfield

    info = mne.create_info(3, 100.0, "eeg")
    monkeypatch.setattr(mne_compat, "mne", None)
    with pytest.raises(
        ImportError, match="automatic spherical lead-field construction.*MNE-Python"
    ):
        make_spherical_leadfield(info, n_dipoles=3)
