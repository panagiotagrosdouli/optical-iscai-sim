from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from optical_iscai.cfar import Detection
from optical_iscai.clustering import DetectionCluster
from optical_iscai.dataset import sequence_to_tables
from optical_iscai.sequence import SequenceResult
from optical_iscai.tracking import TrackEstimate


def make_sequence() -> SequenceResult:
    detection = Detection(
        velocity_index=3,
        range_index=7,
        velocity_m_s=2.0,
        range_m=25.0,
        power=12.0,
        threshold=4.0,
    )
    cluster = DetectionCluster(
        range_m=25.1,
        velocity_m_s=1.9,
        total_power=20.0,
        peak_power=12.0,
        peak_range_m=25.0,
        peak_velocity_m_s=2.0,
        peak_range_index=7,
        peak_velocity_index=3,
        cell_count=2,
        cell_indices=((3, 7), (3, 8)),
    )
    track = TrackEstimate(
        track_id=4,
        range_m=25.2,
        velocity_m_s=1.8,
        covariance=np.array([[0.25, 0.02], [0.02, 0.16]]),
        age=3,
        hits=3,
        misses=0,
        confirmed=True,
    )
    pipeline = SimpleNamespace(
        cell_detections=(detection,),
        clusters=(cluster,),
        tracks=(track,),
        confirmed_tracks=(track,),
    )
    frame = SimpleNamespace(
        frame_index=2,
        start_time_s=0.2,
        frame=None,
        pipeline=pipeline,
        target_ranges_m=np.array([24.8, 40.0]),
        target_velocities_m_s=np.array([2.0, -1.0]),
    )
    return SequenceResult(frames=(frame,))


def test_sequence_to_tables_contains_all_entity_levels():
    tables = sequence_to_tables(make_sequence(), sequence_id="run-a")

    assert tables.frames.iloc[0].to_dict() == {
        "sequence_id": "run-a",
        "frame_index": 2,
        "start_time_s": 0.2,
        "target_count": 2,
        "cell_detection_count": 1,
        "cluster_count": 1,
        "track_count": 1,
        "confirmed_track_count": 1,
    }
    assert list(tables.targets["target_index"]) == [0, 1]
    assert tables.detections.loc[0, "range_index"] == 7
    assert tables.clusters.loc[0, "cell_count"] == 2
    assert tables.tracks.loc[0, "track_id"] == 4
    assert bool(tables.tracks.loc[0, "confirmed"])
    assert tables.tracks.loc[0, "velocity_variance"] == pytest.approx(0.16)


def test_sequence_id_validation():
    sequence = SequenceResult(frames=())
    with pytest.raises(ValueError):
        sequence_to_tables(sequence, sequence_id=" ")
    with pytest.raises(TypeError):
        sequence_to_tables(object())


def test_write_parquet_round_trip(tmp_path):
    tables = sequence_to_tables(make_sequence())
    paths = tables.write_parquet(tmp_path / "dataset")

    assert set(paths) == {"frames", "targets", "detections", "clusters", "tracks"}
    assert all(path.exists() for path in paths.values())
    restored = pd.read_parquet(paths["tracks"])
    assert restored.loc[0, "track_id"] == 4
    assert restored.loc[0, "range_m"] == pytest.approx(25.2)
