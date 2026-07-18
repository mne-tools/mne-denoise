"""Regression tests for the generalized Klados AASR runner."""

from __future__ import annotations

import numpy as np
import pytest

import scripts.run_aasr_klados_validation as klados_runner


class _RecordingAdaptiveASR:
    instances: list[_RecordingAdaptiveASR] = []
    fail_updates = False

    def __init__(self, *, variant: str, **kwargs):
        self.variant = variant
        self.blocksize = 1
        self.fit_shapes: list[tuple[int, ...]] = []
        self.update_shapes: list[tuple[int, ...]] = []
        type(self).instances.append(self)

    def fit(self, data: np.ndarray):
        self.fit_shapes.append(data.shape)
        return self

    def partial_fit(self, data: np.ndarray):
        self.update_shapes.append(data.shape)
        if self.fail_updates:
            raise RuntimeError("adaptive update failed")
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        return data.copy()


@pytest.fixture(autouse=True)
def _reset_fake():
    _RecordingAdaptiveASR.instances = []
    _RecordingAdaptiveASR.fail_updates = False


def test_init_uses_psw_state_from_only_the_initial_window(monkeypatch):
    monkeypatch.setattr(klados_runner, "AdaptiveASR", _RecordingAdaptiveASR)
    data = np.zeros((3, 40))

    cleaned, model, _ = klados_runner._run_variant(
        "init",
        data,
        sfreq=2.0,
        cutoff=20.0,
        init_window_s=3.0,
        psw_window_s=5.0,
        mw_window_s=5.0,
    )

    assert model.variant == "psw"
    assert model.fit_shapes == [(3, 6)]
    assert model.update_shapes == []
    np.testing.assert_array_equal(cleaned, data)


def test_paired_trial_discovery_uses_intersection_and_numeric_order():
    pure = {"sim10_resampled", "sim2_resampled", "sim3_resampled", "metadata"}
    contaminated = {"sim2_con", "sim10_con", "sim11_con"}

    assert klados_runner._paired_trial_indices(pure, contaminated) == [2, 10]


@pytest.mark.parametrize("variant", ["psp", "psw"])
def test_adaptive_update_failure_is_recorded_by_outer_runner(monkeypatch, variant):
    monkeypatch.setattr(klados_runner, "AdaptiveASR", _RecordingAdaptiveASR)
    _RecordingAdaptiveASR.fail_updates = True
    data = np.zeros((3, 40))

    with pytest.raises(RuntimeError, match="adaptive update failed"):
        klados_runner._run_variant(
            variant,
            data,
            sfreq=2.0,
            cutoff=20.0,
            init_window_s=3.0,
            psw_window_s=5.0,
            mw_window_s=5.0,
        )
