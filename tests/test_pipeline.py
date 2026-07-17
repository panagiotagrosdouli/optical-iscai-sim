import numpy as np
import pytest

from optical_iscai.cfar import CFARConfiguration
from optical_iscai.clustering import ClusterConfiguration
from optical_iscai.pipeline import PipelineConfiguration, SensingPipeline
from optical_iscai.range_doppler import RangeDopplerMap
from optical_iscai.tracking import TrackerConfiguration


def make_config(*, confirmation_hits=2):
    return PipelineConfiguration(
        cfar=CFARConfiguration(
            training_velocity=2,
            training_range=3,
            guard_velocity=1,
            guard_range=1,
            probability_false_alarm=1e-3,
        ),
        clustering=ClusterConfiguration(
            connectivity=8,
            minimum_cells=1,
            minimum_total_power=0.0,
        ),
        tracking=TrackerConfiguration(
            range_gate_m=3.0,
            velocity_gate_m_s=2.0,
            confirmation_hits=confirmation_hits,
            maximum_misses=1,
        ),
    )


def make_map(range_index=30, velocity_index=15):
    ranges = np.arange(64, dtype=float)
    velocities = np.arange(32, dtype=float) - 16.0
    power = np.ones((velocities.size, ranges.size), dtype=float)
    power[velocity_index, range_index] = 100.0
    power[velocity_index, range_index + 1] = 50.0
    return RangeDopplerMap(range_m=ranges, velocity_m_s=velocities, power=power)


def test_pipeline_links_cfar_clustering_and_tracking():
    pipeline = SensingPipeline(make_config())
    result = pipeline.process_range_doppler(make_map(), frame_interval_s=0.1)

    assert len(result.cell_detections) == 2
    assert len(result.clusters) == 1
    assert result.clusters[0].cell_count == 2
    assert result.clusters[0].range_m == pytest.approx((30 * 100 + 31 * 50) / 150)
    assert len(result.tracks) == 1
    assert result.tracks[0].track_id == 1
    assert not result.tracks[0].confirmed
    assert result.confirmed_tracks == ()


def test_track_is_confirmed_across_successive_frames():
    pipeline = SensingPipeline(make_config(confirmation_hits=2))
    first = pipeline.process_range_doppler(make_map(30, 15), frame_interval_s=0.1)
    second = pipeline.process_range_doppler(make_map(30, 15), frame_interval_s=0.1)

    assert first.tracks[0].track_id == second.tracks[0].track_id
    assert second.tracks[0].hits == 2
    assert second.tracks[0].confirmed
    assert len(second.confirmed_tracks) == 1


def test_reset_restarts_track_identifiers():
    pipeline = SensingPipeline(make_config(confirmation_hits=1))
    first = pipeline.process_range_doppler(make_map(), frame_interval_s=0.1)
    assert first.tracks[0].track_id == 1

    pipeline.reset()
    second = pipeline.process_range_doppler(make_map(), frame_interval_s=0.1)
    assert second.tracks[0].track_id == 1
    assert second.tracks[0].age == 1


def test_invalid_frame_interval_is_rejected():
    pipeline = SensingPipeline(make_config())
    with pytest.raises(ValueError):
        pipeline.process_range_doppler(make_map(), frame_interval_s=0.0)


def test_configuration_is_validated():
    config = make_config()
    invalid = PipelineConfiguration(
        cfar=config.cfar,
        clustering=ClusterConfiguration(connectivity=6),
        tracking=config.tracking,
    )
    with pytest.raises(ValueError):
        SensingPipeline(invalid)
