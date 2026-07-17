import numpy as np
import pytest

from optical_iscai.evaluation import EvaluationConfiguration, evaluate_sequence
from optical_iscai.pipeline import PipelineResult
from optical_iscai.sequence import SequenceFrameResult, SequenceResult
from optical_iscai.tracking import TrackEstimate


def make_track(track_id, range_m, velocity_m_s, *, confirmed=True):
    return TrackEstimate(
        track_id=track_id,
        range_m=range_m,
        velocity_m_s=velocity_m_s,
        covariance=np.eye(2),
        age=3,
        hits=3 if confirmed else 1,
        misses=0,
        confirmed=confirmed,
    )


def make_frame(index, truth, tracks):
    pipeline = PipelineResult(
        range_doppler=None,
        cfar=None,
        cell_detections=(),
        clusters=(),
        tracks=tuple(tracks),
    )
    truth_array = np.asarray(truth, dtype=float)
    return SequenceFrameResult(
        frame_index=index,
        start_time_s=0.1 * index,
        frame=None,
        pipeline=pipeline,
        target_ranges_m=truth_array[:, 0],
        target_velocities_m_s=truth_array[:, 1],
    )


def test_perfect_sequence_scores_one_and_zero_rmse():
    sequence = SequenceResult(
        frames=(
            make_frame(0, [(20.0, 2.0)], [make_track(1, 20.0, 2.0)]),
            make_frame(1, [(19.8, 2.0)], [make_track(1, 19.8, 2.0)]),
        )
    )
    result = evaluate_sequence(sequence)
    assert result.matched_count == 2
    assert result.recall == pytest.approx(1.0)
    assert result.precision == pytest.approx(1.0)
    assert result.range_rmse_m == pytest.approx(0.0)
    assert result.velocity_rmse_m_s == pytest.approx(0.0)


def test_misses_false_tracks_and_signed_errors_are_reported():
    sequence = SequenceResult(
        frames=(
            make_frame(
                0,
                [(10.0, 1.0), (30.0, -2.0)],
                [make_track(1, 10.5, 0.5), make_track(2, 80.0, 0.0)],
            ),
        )
    )
    result = evaluate_sequence(sequence)
    frame = result.frames[0]
    assert frame.matched_count == 1
    assert frame.missed_count == 1
    assert frame.false_track_count == 1
    assert frame.range_errors_m.tolist() == pytest.approx([0.5])
    assert frame.velocity_errors_m_s.tolist() == pytest.approx([-0.5])
    assert result.recall == pytest.approx(0.5)
    assert result.precision == pytest.approx(0.5)


def test_unconfirmed_tracks_are_excluded_by_default():
    sequence = SequenceResult(
        frames=(make_frame(0, [(10.0, 1.0)], [make_track(1, 10.0, 1.0, confirmed=False)]),)
    )
    default_result = evaluate_sequence(sequence)
    all_tracks_result = evaluate_sequence(
        sequence, EvaluationConfiguration(confirmed_only=False)
    )
    assert default_result.track_count == 0
    assert default_result.matched_count == 0
    assert all_tracks_result.track_count == 1
    assert all_tracks_result.matched_count == 1


def test_gates_reject_distant_tracks():
    sequence = SequenceResult(
        frames=(make_frame(0, [(10.0, 1.0)], [make_track(1, 12.0, 1.0)]),)
    )
    result = evaluate_sequence(
        sequence,
        EvaluationConfiguration(range_gate_m=1.0, velocity_gate_m_s=1.0),
    )
    assert result.matched_count == 0
    assert np.isnan(result.range_rmse_m)
    assert np.isnan(result.velocity_rmse_m_s)


def test_invalid_configuration_and_input_are_rejected():
    with pytest.raises(ValueError):
        EvaluationConfiguration(range_gate_m=0.0).validate()
    with pytest.raises(ValueError):
        EvaluationConfiguration(velocity_gate_m_s=np.inf).validate()
    with pytest.raises(TypeError):
        evaluate_sequence(object())
