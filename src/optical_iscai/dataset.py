"""Convert simulated sensing sequences into normalized machine-learning tables.

The exporter keeps frame-level metadata, target ground truth, detections, clusters,
and tracker estimates in separate tables.  This avoids variable-length array cells
and makes the resulting Parquet files straightforward to query and join.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from optical_iscai.sequence import SequenceResult


@dataclass(frozen=True, slots=True)
class SequenceTables:
    """Normalized tabular representation of one simulated sequence."""

    frames: pd.DataFrame
    targets: pd.DataFrame
    detections: pd.DataFrame
    clusters: pd.DataFrame
    tracks: pd.DataFrame

    def write_parquet(self, output_directory: str | Path) -> dict[str, Path]:
        """Write all tables to one directory and return their paths."""
        directory = Path(output_directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths = {
            "frames": directory / "frames.parquet",
            "targets": directory / "targets.parquet",
            "detections": directory / "detections.parquet",
            "clusters": directory / "clusters.parquet",
            "tracks": directory / "tracks.parquet",
        }
        for name, path in paths.items():
            getattr(self, name).to_parquet(path, index=False)
        return paths


def sequence_to_tables(sequence: SequenceResult, *, sequence_id: str = "sequence-0") -> SequenceTables:
    """Flatten a :class:`SequenceResult` into relational data frames.

    ``sequence_id`` is copied into every table so outputs from multiple simulation
    runs can be concatenated without losing provenance.
    """
    if not isinstance(sequence, SequenceResult):
        raise TypeError("sequence must be a SequenceResult")
    if not isinstance(sequence_id, str) or not sequence_id.strip():
        raise ValueError("sequence_id must be a non-empty string")

    frame_rows: list[dict[str, object]] = []
    target_rows: list[dict[str, object]] = []
    detection_rows: list[dict[str, object]] = []
    cluster_rows: list[dict[str, object]] = []
    track_rows: list[dict[str, object]] = []

    for item in sequence.frames:
        frame_index = int(item.frame_index)
        common = {
            "sequence_id": sequence_id,
            "frame_index": frame_index,
            "start_time_s": float(item.start_time_s),
        }
        frame_rows.append(
            {
                **common,
                "target_count": int(item.target_ranges_m.size),
                "cell_detection_count": len(item.pipeline.cell_detections),
                "cluster_count": len(item.pipeline.clusters),
                "track_count": len(item.pipeline.tracks),
                "confirmed_track_count": len(item.pipeline.confirmed_tracks),
            }
        )

        for target_index, (range_m, velocity_m_s) in enumerate(
            zip(item.target_ranges_m, item.target_velocities_m_s, strict=True)
        ):
            target_rows.append(
                {
                    **common,
                    "target_index": target_index,
                    "range_m": float(range_m),
                    "velocity_m_s": float(velocity_m_s),
                }
            )

        for detection_index, detection in enumerate(item.pipeline.cell_detections):
            detection_rows.append(
                {
                    **common,
                    "detection_index": detection_index,
                    "range_index": int(detection.range_index),
                    "velocity_index": int(detection.velocity_index),
                    "range_m": float(detection.range_m),
                    "velocity_m_s": float(detection.velocity_m_s),
                    "power": float(detection.power),
                    "threshold": float(detection.threshold),
                }
            )

        for cluster_index, cluster in enumerate(item.pipeline.clusters):
            cluster_rows.append(
                {
                    **common,
                    "cluster_index": cluster_index,
                    "range_m": float(cluster.range_m),
                    "velocity_m_s": float(cluster.velocity_m_s),
                    "total_power": float(cluster.total_power),
                    "peak_power": float(cluster.peak_power),
                    "peak_range_m": float(cluster.peak_range_m),
                    "peak_velocity_m_s": float(cluster.peak_velocity_m_s),
                    "peak_range_index": int(cluster.peak_range_index),
                    "peak_velocity_index": int(cluster.peak_velocity_index),
                    "cell_count": int(cluster.cell_count),
                }
            )

        for track in item.pipeline.tracks:
            track_rows.append(
                {
                    **common,
                    "track_id": int(track.track_id),
                    "range_m": float(track.range_m),
                    "velocity_m_s": float(track.velocity_m_s),
                    "range_variance": float(track.covariance[0, 0]),
                    "range_velocity_covariance": float(track.covariance[0, 1]),
                    "velocity_variance": float(track.covariance[1, 1]),
                    "age": int(track.age),
                    "hits": int(track.hits),
                    "misses": int(track.misses),
                    "confirmed": bool(track.confirmed),
                }
            )

    return SequenceTables(
        frames=pd.DataFrame.from_records(frame_rows),
        targets=pd.DataFrame.from_records(target_rows),
        detections=pd.DataFrame.from_records(detection_rows),
        clusters=pd.DataFrame.from_records(cluster_rows),
        tracks=pd.DataFrame.from_records(track_rows),
    )
