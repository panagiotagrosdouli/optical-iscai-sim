import numpy as np
import pytest

from optical_iscai.experiment import (
    MonteCarloConfiguration,
    parameter_grid,
    run_monte_carlo,
)
from optical_iscai.pipeline import PipelineResult
from optical_iscai.sequence import SequenceFrameResult, SequenceResult
from optical_iscai.tracking import TrackEstimate


def _sequence_with_exact_track(parameters, rng):
    jitter = float(rng.normal(0.0, 0.0))
    truth_range = float(parameters.get("range_m", 20.0))
    truth_velocity = float(parameters.get("velocity_m_s", 2.0))
    track = TrackEstimate(
        track_id=1,
        range_m=truth_range + jitter,
        velocity_m_s=truth_velocity,
        covariance=np.eye(2),
        age=2,
        hits=2,
        misses=0,
        confirmed=True,
    )
    pipeline = PipelineResult(
        range_doppler=None,
        cfar=None,
        cell_detections=(),
        clusters=(),
        tracks=(track,),
    )
    frame = SequenceFrameResult(
        frame_index=0,
        start_time_s=0.0,
        frame=None,
        pipeline=pipeline,
        target_ranges_m=np.asarray([truth_range]),
        target_velocities_m_s=np.asarray([truth_velocity]),
    )
    return SequenceResult(frames=(frame,))


def test_parameter_grid_expands_cartesian_product():
    grid = parameter_grid({"snr_db": [-5, 0], "target_count": [1, 2]})

    assert grid == (
        {"snr_db": -5, "target_count": 1},
        {"snr_db": -5, "target_count": 2},
        {"snr_db": 0, "target_count": 1},
        {"snr_db": 0, "target_count": 2},
    )


def test_parameter_grid_rejects_empty_axis():
    with pytest.raises(ValueError, match="at least one value"):
        parameter_grid({"snr_db": []})


def test_monte_carlo_runs_every_condition_and_repetition():
    result = run_monte_carlo(
        _sequence_with_exact_track,
        sweep={"range_m": [10.0, 20.0], "velocity_m_s": [1.0]},
        config=MonteCarloConfiguration(repetitions=3, seed=7),
    )

    assert len(result.runs) == 6
    assert result.parameter_names == ("range_m", "velocity_m_s")
    assert all(run.precision == pytest.approx(1.0) for run in result.runs)
    assert all(run.recall == pytest.approx(1.0) for run in result.runs)
    assert all(run.f1_score == pytest.approx(1.0) for run in result.runs)
    assert len({run.run_seed for run in result.runs}) == 6

    frame = result.to_dataframe()
    assert len(frame) == 6
    assert set(frame["range_m"]) == {10.0, 20.0}

    summary = result.summarize()
    assert len(summary) == 2
    assert np.allclose(summary["precision_mean"], 1.0)
    assert np.all(summary["precision_count"] == 3)


def test_monte_carlo_is_reproducible():
    config = MonteCarloConfiguration(repetitions=2, seed=123)
    first = run_monte_carlo(_sequence_with_exact_track, config=config)
    second = run_monte_carlo(_sequence_with_exact_track, config=config)

    assert [run.run_seed for run in first.runs] == [run.run_seed for run in second.runs]
    assert first.to_dataframe().equals(second.to_dataframe())


def test_configuration_validation():
    with pytest.raises(ValueError, match="at least one"):
        run_monte_carlo(
            _sequence_with_exact_track,
            config=MonteCarloConfiguration(repetitions=0),
        )


def test_simulator_must_return_sequence_result():
    def invalid_simulator(parameters, rng):
        return object()

    with pytest.raises(TypeError, match="SequenceResult"):
        run_monte_carlo(invalid_simulator)
