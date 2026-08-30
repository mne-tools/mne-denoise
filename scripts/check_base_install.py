"""Validate the base installation without optional runtime dependencies."""

from importlib.util import find_spec


def main() -> None:
    """Check the base installation contract."""
    for package in ("mne", "matplotlib", "tqdm"):
        assert find_spec(package) is None, f"{package} leaked into the base environment"

    import numpy as np

    import mne_denoise
    from mne_denoise import compute_covariance

    covariance = compute_covariance(np.eye(3))
    assert np.isfinite(covariance).all()

    try:
        import mne_denoise.viz  # noqa: F401
    except ImportError as error:
        assert "mne-denoise[viz]" in str(error)
    else:
        raise AssertionError("mne_denoise.viz unexpectedly imported without Matplotlib")


if __name__ == "__main__":
    main()
