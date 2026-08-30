"""Internal helpers for optional MNE-Python integration."""

from __future__ import annotations

try:
    import mne
except ModuleNotFoundError as error:  # pragma: no cover - no-MNE environments
    if error.name != "mne":
        raise
    mne = None

HAS_MNE = mne is not None


def require_mne(feature: str) -> None:
    """Require MNE-Python for an MNE-specific feature."""
    if mne is None:
        raise ImportError(
            f"{feature} requires the optional MNE-Python integration.\n"
            "Install it with:\n\n"
            '    pip install "mne-denoise[mne]"'
        )
