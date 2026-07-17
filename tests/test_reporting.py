from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from optical_iscai.experiment import (
    ExperimentResult,
    ExperimentRun,
    MonteCarloConfiguration,
)
from optical_iscai.reporting import write_experiment_report


def _result() -> ExperimentResult:
    runs = (
        ExperimentRun(
            condition_index=0,
            repetition_index=0,
            run_seed=11,
            parameters={"snr_db": -5.0},
            ground_truth_count=2,
            track_count=2,
            matched_count=1,
            missed_count=1,
            false_track_count=1,
            precision=0.5,
            recall=0.5,
            f1_score=0.5,
            range_rmse_m=1.25,
            velocity_rmse_m_s=0.4,
        ),
        ExperimentRun(
            condition_index=0,
            repetition_index=1,
            run_seed=12,
            parameters={"snr_db": -5.0},
            ground_truth_count=2,
            track_count=0,
            matched_count=0,
            missed_count=2,
            false_track_count=0,
            precision=float("nan"),
            recall=0.0,
            f1_score=float("nan"),
            range_rmse_m=float("nan"),
            velocity_rmse_m_s=float("nan"),
        ),
    )
    return ExperimentResult(runs=runs, parameter_names=("snr_db",))


def test_write_experiment_report_creates_expected_files(tmp_path) -> None:
    config = MonteCarloConfiguration(repetitions=2, seed=7)
    paths = write_experiment_report(
        _result(),
        tmp_path / "experiment-001",
        experiment_name="smoke-test",
        monte_carlo_config=config,
        metadata={"paper": "PC-FMCW headlamp", "numpy_scalar": np.int64(3)},
    )

    assert paths.runs_csv.exists()
    assert paths.runs_parquet.exists()
    assert paths.summary_csv.exists()
    assert paths.summary_parquet.exists()
    assert paths.manifest_json.exists()

    csv_runs = pd.read_csv(paths.runs_csv)
    parquet_runs = pd.read_parquet(paths.runs_parquet)
    assert len(csv_runs) == 2
    assert len(parquet_runs) == 2
    assert list(csv_runs["run_seed"]) == [11, 12]

    manifest = json.loads(paths.manifest_json.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["experiment_name"] == "smoke-test"
    assert manifest["run_count"] == 2
    assert manifest["condition_count"] == 1
    assert manifest["parameter_names"] == ["snr_db"]
    assert manifest["monte_carlo"] == {"repetitions": 2, "seed": 7}
    assert manifest["metadata"]["numpy_scalar"] == 3


def test_write_experiment_report_refuses_overwrite_by_default(tmp_path) -> None:
    output = tmp_path / "report"
    write_experiment_report(_result(), output)

    with pytest.raises(FileExistsError, match="report files already exist"):
        write_experiment_report(_result(), output)


def test_write_experiment_report_allows_explicit_overwrite(tmp_path) -> None:
    output = tmp_path / "report"
    write_experiment_report(_result(), output)
    paths = write_experiment_report(
        _result(),
        output,
        experiment_name="replacement",
        overwrite=True,
    )

    manifest = json.loads(paths.manifest_json.read_text(encoding="utf-8"))
    assert manifest["experiment_name"] == "replacement"


def test_write_experiment_report_validates_result_type(tmp_path) -> None:
    with pytest.raises(TypeError, match="ExperimentResult"):
        write_experiment_report(object(), tmp_path / "bad")  # type: ignore[arg-type]
