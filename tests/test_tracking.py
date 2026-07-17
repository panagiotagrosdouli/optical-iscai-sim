import numpy as np
import pytest

from optical_iscai.clustering import DetectionCluster
from optical_iscai.tracking import MultiTargetTracker, TrackerConfiguration


def detection(range_m: float, velocity_m_s: float, power: float = 1.0) -> DetectionCluster:
    return DetectionCluster(
        range_m=range_m,
        velocity_m_s=velocity_m_s,
        total_power=power,
        peak_power=power,
        peak_range_m=range_m,
        peak_velocity_m_s=velocity_m_s,
        peak_range_index=0,
        peak_velocity_index=0,
        cell_count=1,
        cell_indices=((0, 0),),
    )


def test_new_detection_starts_track():
    tracker = MultiTargetTracker()
    tracks = tracker.update([detection(50.0, 4.0)], dt_s=0.1)
    assert len(tracks) == 1
    assert tracks[0].track_id == 1
    assert tracks[0].range_m == pytest.approx(50.0)
    assert tracks[0].velocity_m_s == pytest.approx(4.0)
    assert tracks[0].hits == 1
    assert not tracks[0].confirmed


def test_positive_velocity_predicts_decreasing_range():
    tracker = MultiTargetTracker(
        TrackerConfiguration(confirmation_hits=1, maximum_misses=2)
    )
    tracker.update([detection(50.0, 4.0)], dt_s=0.1)
    tracks = tracker.update([], dt_s=0.5)
    assert tracks[0].range_m == pytest.approx(48.0)
    assert tracks[0].velocity_m_s == pytest.approx(4.0)
    assert tracks[0].misses == 1


def test_nearest_detection_keeps_track_identity():
    tracker = MultiTargetTracker(
        TrackerConfiguration(
            range_gate_m=3.0,
            velocity_gate_m_s=2.0,
            confirmation_hits=2,
        )
    )
    tracker.update(
        [detection(30.0, 2.0), detection(80.0, -1.0)],
        dt_s=0.1,
    )
    tracks = tracker.update(
        [detection(79.9, -1.1), detection(29.8, 2.1)],
        dt_s=0.1,
    )
    assert [track.track_id for track in tracks] == [1, 2]
    assert all(track.confirmed for track in tracks)
    assert tracks[0].range_m < 31.0
    assert tracks[1].range_m > 79.0


def test_out_of_gate_detection_starts_new_track():
    tracker = MultiTargetTracker(
        TrackerConfiguration(range_gate_m=1.0, velocity_gate_m_s=1.0)
    )
    tracker.update([detection(20.0, 0.0)], dt_s=0.1)
    tracks = tracker.update([detection(40.0, 0.0)], dt_s=0.1)
    assert len(tracks) == 2
    assert tracks[0].misses == 1
    assert tracks[1].track_id == 2


def test_track_is_deleted_after_maximum_misses():
    tracker = MultiTargetTracker(
        TrackerConfiguration(maximum_misses=1, confirmation_hits=1)
    )
    tracker.update([detection(10.0, 0.0)], dt_s=0.1)
    assert len(tracker.update([], dt_s=0.1)) == 1
    assert tracker.update([], dt_s=0.1) == ()


def test_kalman_update_reduces_measurement_error():
    tracker = MultiTargetTracker(
        TrackerConfiguration(
            confirmation_hits=1,
            initial_range_std_m=5.0,
            initial_velocity_std_m_s=5.0,
            measurement_range_std_m=0.2,
            measurement_velocity_std_m_s=0.2,
        )
    )
    tracker.update([detection(100.0, 5.0)], dt_s=0.1)
    predicted_range = 99.5
    measured_range = 99.0
    track = tracker.update([detection(measured_range, 5.0)], dt_s=0.1)[0]
    assert abs(track.range_m - measured_range) < abs(predicted_range - measured_range)
    assert np.allclose(track.covariance, track.covariance.T)
    assert np.all(np.linalg.eigvalsh(track.covariance) >= -1e-12)


def test_reset_restarts_identifiers():
    tracker = MultiTargetTracker()
    tracker.update([detection(10.0, 0.0)], dt_s=0.1)
    tracker.reset()
    tracks = tracker.update([detection(20.0, 0.0)], dt_s=0.1)
    assert tracks[0].track_id == 1


@pytest.mark.parametrize("dt_s", [0.0, -1.0, np.nan, np.inf])
def test_invalid_time_step_is_rejected(dt_s):
    tracker = MultiTargetTracker()
    with pytest.raises(ValueError):
        tracker.update([], dt_s=dt_s)


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError):
        MultiTargetTracker(TrackerConfiguration(range_gate_m=0.0))
    with pytest.raises(ValueError):
        MultiTargetTracker(TrackerConfiguration(confirmation_hits=0))
    with pytest.raises(ValueError):
        MultiTargetTracker(TrackerConfiguration(maximum_misses=-1))
