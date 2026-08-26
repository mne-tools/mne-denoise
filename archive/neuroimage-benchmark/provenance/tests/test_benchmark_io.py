"""Tests for mne_denoise.benchmarks.io module."""

import json

import numpy as np
import pytest

from mne_denoise.benchmarks.io import (
    LineNoiseGroupData,
    _NumpyEncoder,
    aggregate_benchmark_results,
    load_subject_benchmark_results,
    save_subject_benchmark_results,
)

pd = pytest.importorskip("pandas")


class TestNumpyEncoder:
    """Tests for _NumpyEncoder JSON serialization."""

    def test_numpy_int(self):
        result = json.dumps({"v": np.int64(42)}, cls=_NumpyEncoder)
        assert json.loads(result)["v"] == 42

    def test_numpy_float(self):
        result = json.dumps({"v": np.float32(3.14)}, cls=_NumpyEncoder)
        assert abs(json.loads(result)["v"] - 3.14) < 0.01

    def test_numpy_array(self):
        arr = np.array([1, 2, 3])
        result = json.dumps({"v": arr}, cls=_NumpyEncoder)
        assert json.loads(result)["v"] == [1, 2, 3]

    def test_regular_types(self):
        result = json.dumps({"s": "hello", "i": 5}, cls=_NumpyEncoder)
        d = json.loads(result)
        assert d["s"] == "hello"
        assert d["i"] == 5

    def test_unsupported_type(self):
        with pytest.raises(TypeError):
            json.dumps({"v": object()}, cls=_NumpyEncoder)


class TestSaveSubjectBenchmarkResults:
    """Tests for save_subject_benchmark_results."""

    def test_save_metrics(self, tmp_path):
        out_dir = tmp_path / "sub-01" / "line_noise" / "zapline"
        save_subject_benchmark_results(
            out_dir,
            subject="sub-01",
            method="zapline",
            metrics={"peak_attenuation_db": 12.3, "R_f0": 0.85},
        )
        tsv = out_dir / "metrics.tsv"
        assert tsv.exists()
        df = pd.read_csv(tsv, sep="\t")
        assert df.iloc[0]["subject"] == "sub-01"
        assert df.iloc[0]["method"] == "zapline"
        np.testing.assert_allclose(df.iloc[0]["peak_attenuation_db"], 12.3)

    def test_save_condition_metrics(self, tmp_path):
        out_dir = tmp_path / "sub-01" / "line_noise" / "dss"
        cond = [
            {"condition": "lab", "peak_attenuation_db": 11.0},
            {"condition": "campus", "peak_attenuation_db": 13.5},
        ]
        save_subject_benchmark_results(
            out_dir,
            subject="sub-01",
            method="dss",
            condition_metrics=cond,
        )
        tsv = out_dir / "condition_metrics.tsv"
        assert tsv.exists()
        df = pd.read_csv(tsv, sep="\t")
        assert len(df) == 2
        assert "subject" in df.columns
        assert "method" in df.columns

    def test_save_condition_metrics_dataframe(self, tmp_path):
        out_dir = tmp_path / "sub-02" / "line_noise" / "dss"
        df_cond = pd.DataFrame(
            [
                {"condition": "lab", "R_f0": 0.9},
                {"condition": "park", "R_f0": 1.1},
            ]
        )
        save_subject_benchmark_results(
            out_dir,
            subject="sub-02",
            method="dss",
            condition_metrics=df_cond,
        )
        tsv = out_dir / "condition_metrics.tsv"
        assert tsv.exists()
        loaded = pd.read_csv(tsv, sep="\t")
        assert len(loaded) == 2

    def test_save_model_info(self, tmp_path):
        out_dir = tmp_path / "sub-01" / "line_noise" / "zapline"
        save_subject_benchmark_results(
            out_dir,
            subject="sub-01",
            method="zapline",
            model_info={"n_components": np.int64(4), "line_freq": 50.0},
        )
        jpath = out_dir / "model.json"
        assert jpath.exists()
        with open(jpath) as f:
            d = json.load(f)
        assert d["n_components"] == 4
        assert d["subject"] == "sub-01"

    def test_save_all_parts(self, tmp_path):
        out_dir = tmp_path / "sub-01" / "line_noise" / "zapline"
        save_subject_benchmark_results(
            out_dir,
            subject="sub-01",
            method="zapline",
            metrics={"R_f0": 0.9},
            condition_metrics=[{"condition": "a", "R_f0": 0.8}],
            model_info={"param": "val"},
        )
        assert (out_dir / "metrics.tsv").exists()
        assert (out_dir / "condition_metrics.tsv").exists()
        assert (out_dir / "model.json").exists()

    def test_save_none_parts(self, tmp_path):
        """Saving with all None doesn't crash, just creates directory."""
        out_dir = tmp_path / "sub-01" / "line_noise" / "method"
        save_subject_benchmark_results(out_dir, subject="sub-01", method="method")
        assert out_dir.exists()


class TestLoadSubjectBenchmarkResults:
    """Tests for load_subject_benchmark_results."""

    def _create_subject(self, base, sub, methods):
        """Helper to create benchmark files for a subject."""
        for method, metrics in methods.items():
            out_dir = base / sub / "line_noise" / method
            save_subject_benchmark_results(
                out_dir, subject=sub, method=method, metrics=metrics
            )

    def test_load_basic(self, tmp_path):
        self._create_subject(tmp_path, "sub-01", {"zapline": {"R_f0": 0.85}})
        result = load_subject_benchmark_results(tmp_path / "sub-01" / "line_noise")
        assert result["subject"] == "sub-01"
        assert not result["df"].empty
        assert result["df"].iloc[0]["method"] == "zapline"

    def test_load_multiple_methods(self, tmp_path):
        self._create_subject(
            tmp_path,
            "sub-01",
            {"zapline": {"R_f0": 0.85}, "dss": {"R_f0": 0.90}},
        )
        result = load_subject_benchmark_results(tmp_path / "sub-01" / "line_noise")
        assert len(result["df"]) == 2

    def test_load_filter_methods(self, tmp_path):
        self._create_subject(
            tmp_path,
            "sub-01",
            {"zapline": {"R_f0": 0.85}, "dss": {"R_f0": 0.90}},
        )
        result = load_subject_benchmark_results(
            tmp_path / "sub-01" / "line_noise", methods=["zapline"]
        )
        assert len(result["df"]) == 1

    def test_load_with_condition_metrics(self, tmp_path):
        out_dir = tmp_path / "sub-01" / "line_noise" / "zapline"
        save_subject_benchmark_results(
            out_dir,
            subject="sub-01",
            method="zapline",
            metrics={"R_f0": 0.85},
            condition_metrics=[{"condition": "lab", "R_f0": 0.8}],
            model_info={"n_components": 4},
        )
        result = load_subject_benchmark_results(tmp_path / "sub-01" / "line_noise")
        assert not result["df_cond"].empty
        assert "zapline" in result["model_info"]

    def test_load_empty_dir(self, tmp_path):
        sub_dir = tmp_path / "sub-01" / "line_noise"
        sub_dir.mkdir(parents=True)
        result = load_subject_benchmark_results(sub_dir)
        assert result["df"].empty


class TestAggregateBenchmarkResults:
    """Tests for aggregate_benchmark_results."""

    def _populate(self, root, subjects, methods):
        for sub in subjects:
            for method, metrics in methods.items():
                out_dir = root / sub / "line_noise" / method
                save_subject_benchmark_results(
                    out_dir,
                    subject=sub,
                    method=method,
                    metrics=metrics,
                )

    def test_aggregate_basic(self, tmp_path):
        self._populate(
            tmp_path,
            ["sub-01", "sub-02"],
            {"zapline": {"R_f0": 0.85, "peak_attenuation_db": 10.0}},
        )
        grp = aggregate_benchmark_results(tmp_path)
        assert isinstance(grp, LineNoiseGroupData)
        assert len(grp.df_all) == 2
        assert "condition" in grp.df_all.columns
        assert (grp.df_all["condition"] == "all").all()

    def test_aggregate_with_conditions(self, tmp_path):
        for sub in ["sub-01", "sub-02"]:
            out_dir = tmp_path / sub / "line_noise" / "zapline"
            save_subject_benchmark_results(
                out_dir,
                subject=sub,
                method="zapline",
                metrics={"R_f0": 0.85},
                condition_metrics=[{"condition": "lab", "R_f0": 0.8}],
            )
        grp = aggregate_benchmark_results(tmp_path)
        assert not grp.df_cond.empty
        assert len(grp.df_metrics) == len(grp.df_all) + len(grp.df_cond)

    def test_aggregate_filter_subjects(self, tmp_path):
        self._populate(
            tmp_path,
            ["sub-01", "sub-02", "sub-03"],
            {"zapline": {"R_f0": 0.85}},
        )
        grp = aggregate_benchmark_results(tmp_path, subjects=["sub-01", "sub-03"])
        assert len(grp.df_all) == 2

    def test_aggregate_filter_methods(self, tmp_path):
        for sub in ["sub-01"]:
            for method in ["zapline", "dss"]:
                out_dir = tmp_path / sub / "line_noise" / method
                save_subject_benchmark_results(
                    out_dir,
                    subject=sub,
                    method=method,
                    metrics={"R_f0": 0.85},
                )
        grp = aggregate_benchmark_results(tmp_path, methods=["dss"])
        assert len(grp.df_all) == 1
        assert grp.df_all.iloc[0]["method"] == "dss"

    def test_aggregate_empty(self, tmp_path):
        grp = aggregate_benchmark_results(tmp_path)
        assert grp.df_all.empty
        assert grp.df_cond.empty
        assert grp.df_metrics.empty

    def test_aggregate_missing_subject_dir(self, tmp_path):
        """Subject specified but line_noise dir missing → skip."""
        (tmp_path / "sub-01").mkdir()
        grp = aggregate_benchmark_results(tmp_path, subjects=["sub-01"])
        assert grp.df_all.empty
