"""Regression tests for the real-data SSVEP benchmark loader."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
from scipy.io import savemat


def _runner_module():
    path = Path(__file__).parents[2] / "scripts" / "run_ssvep_arm.py"
    spec = spec_from_file_location("run_ssvep_arm", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_beta_structured_mat(tmp_path):
    """BETA's structured ``data`` variable must not enter the Tsinghua path."""
    epoch = np.arange(2 * 20 * 3 * 4, dtype=float).reshape(2, 20, 3, 4)
    eeg = np.empty((1, 1), dtype=[("Epoch", "O")])
    eeg["Epoch"][0, 0] = epoch
    suppl = np.empty(
        (1, 1),
        dtype=[("Frequency", "O"), ("Srate", "O")],
    )
    suppl["Frequency"][0, 0] = np.array([[8.0, 8.2, 8.4]])
    suppl["Srate"][0, 0] = np.array([[250]])
    payload = np.empty(
        (1, 1),
        dtype=[("EEG", "O"), ("Suppl_info", "O")],
    )
    payload["EEG"][0, 0] = eeg
    payload["Suppl_info"][0, 0] = suppl
    savemat(tmp_path / "S1.mat", {"data": payload})

    data, freqs, sfreq = _runner_module()._load_tsinghua(tmp_path, "S1")

    np.testing.assert_array_equal(data, epoch)
    np.testing.assert_array_equal(freqs, [8.0, 8.2, 8.4])
    assert sfreq == 250.0
