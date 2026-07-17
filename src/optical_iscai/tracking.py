"""Multi-frame target tracking for clustered range-Doppler detections.

The tracker uses a constant radial-velocity Kalman model and greedy global
nearest-neighbour association.  The project sign convention is preserved:
positive velocity means approaching, so predicted range evolves as
``range_next = range_now - velocity * dt``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from optical_iscai.clustering import DetectionCluster


@dataclass(frozen=True, slots=True)
class TrackerConfiguration:
    """Configuration for nearest-neighbour multi-target tracking."""

    range_gate_m: float = 5.0
    velocity_gate_m_s: float = 3.0
    process_acceleration_std_m_s2: float = 2.0
    measurement_range_std_m: float = 0.5
    measurement_velocity_std_m_s: float = 0.5
    initial_range_std_m: float = 2.0
    initial_velocity_std_m_s: float = 2.0
    confirmation_hits: int = 2
    maximum_misses: int = 2

    def validate(self) -> None:
        positive = {
            "range_gate_m": self.range_gate_m,
            "velocity_gate_m_s": self.velocity_gate_m_s,
            "process_acceleration_std_m_s2": self.process_acceleration_std_m_s2,
            "measurement_range_std_m": self.measurement_range_std_m,
            "measurement_velocity_std_m_s": self.measurement_velocity_std_m_s,
            "initial_range_std_m": self.initial_range_std_m,
            "initial_velocity_std_m_s": self.initial_velocity_std_m_s,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and greater than zero")
        for name, value in {
            "confirmation_hits": self.confirmation_hits,
            "maximum_misses": self.maximum_misses,
        }.items():
            if isinstance(value, bool) or int(value) != value or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.confirmation_hits < 1:
            raise ValueError("confirmation_hits must be at least one")


@dataclass(frozen=True, slots=True)
class TrackEstimate:
    """Public immutable snapshot of one target track."""

    track_id: int
    range_m: float
    velocity_m_s: float
    covariance: NDArray[np.float64]
    age: int
    hits: int
    misses: int
    confirmed: bool


@dataclass(slots=True)
class _Track:
    track_id: int
    state: NDArray[np.float64]
    covariance: NDArray[np.float64]
    age: int = 1
    hits: int = 1
    misses: int = 0


class MultiTargetTracker:
    """Track clustered detections across successive frames."""

    def __init__(self, config: TrackerConfiguration | None = None) -> None:
        self.config = TrackerConfiguration() if config is None else config
        self.config.validate()
        self._tracks: list[_Track] = []
        self._next_track_id = 1

    @property
    def tracks(self) -> tuple[TrackEstimate, ...]:
        """Return immutable snapshots sorted by track identifier."""
        return tuple(self._snapshot(track) for track in sorted(self._tracks, key=lambda t: t.track_id))

    def reset(self) -> None:
        """Remove all tracks and restart identifier allocation."""
        self._tracks.clear()
        self._next_track_id = 1

    def update(
        self,
        detections: tuple[DetectionCluster, ...] | list[DetectionCluster],
        dt_s: float,
    ) -> tuple[TrackEstimate, ...]:
        """Advance the tracker by one frame and ingest target detections."""
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("dt_s must be finite and greater than zero")

        measurements = np.asarray(
            [[detection.range_m, detection.velocity_m_s] for detection in detections],
            dtype=np.float64,
        )
        if measurements.size == 0:
            measurements = np.empty((0, 2), dtype=np.float64)
        if measurements.ndim != 2 or measurements.shape[1] != 2:
            raise ValueError("detections must provide range and velocity measurements")
        if not np.all(np.isfinite(measurements)):
            raise ValueError("detection measurements must be finite")

        for track in self._tracks:
            self._predict(track, dt_s)

        assignments = self._associate(measurements)
        assigned_tracks = {track_index for track_index, _ in assignments}
        assigned_measurements = {measurement_index for _, measurement_index in assignments}

        for track_index, measurement_index in assignments:
            self._correct(self._tracks[track_index], measurements[measurement_index])

        for index, track in enumerate(self._tracks):
            track.age += 1
            if index in assigned_tracks:
                track.hits += 1
                track.misses = 0
            else:
                track.misses += 1

        self._tracks = [
            track for track in self._tracks if track.misses <= self.config.maximum_misses
        ]

        for measurement_index, measurement in enumerate(measurements):
            if measurement_index not in assigned_measurements:
                self._start_track(measurement)

        return self.tracks

    def _predict(self, track: _Track, dt_s: float) -> None:
        transition = np.array([[1.0, -dt_s], [0.0, 1.0]], dtype=np.float64)
        acceleration_input = np.array([[-0.5 * dt_s**2], [dt_s]], dtype=np.float64)
        process_noise = (
            acceleration_input @ acceleration_input.T
            * self.config.process_acceleration_std_m_s2**2
        )
        track.state = transition @ track.state
        track.covariance = transition @ track.covariance @ transition.T + process_noise

    def _correct(self, track: _Track, measurement: NDArray[np.float64]) -> None:
        measurement_noise = np.diag(
            [
                self.config.measurement_range_std_m**2,
                self.config.measurement_velocity_std_m_s**2,
            ]
        )
        innovation = measurement - track.state
        innovation_covariance = track.covariance + measurement_noise
        kalman_gain = np.linalg.solve(
            innovation_covariance.T, track.covariance.T
        ).T
        track.state = track.state + kalman_gain @ innovation
        identity = np.eye(2, dtype=np.float64)
        # Joseph form preserves symmetry and positive semidefiniteness better.
        residual = identity - kalman_gain
        track.covariance = (
            residual @ track.covariance @ residual.T
            + kalman_gain @ measurement_noise @ kalman_gain.T
        )

    def _associate(self, measurements: NDArray[np.float64]) -> list[tuple[int, int]]:
        candidates: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(self._tracks):
            for measurement_index, measurement in enumerate(measurements):
                range_error = abs(float(measurement[0] - track.state[0]))
                velocity_error = abs(float(measurement[1] - track.state[1]))
                if (
                    range_error <= self.config.range_gate_m
                    and velocity_error <= self.config.velocity_gate_m_s
                ):
                    normalized_distance = np.hypot(
                        range_error / self.config.range_gate_m,
                        velocity_error / self.config.velocity_gate_m_s,
                    )
                    candidates.append(
                        (float(normalized_distance), track_index, measurement_index)
                    )

        assignments: list[tuple[int, int]] = []
        used_tracks: set[int] = set()
        used_measurements: set[int] = set()
        for _, track_index, measurement_index in sorted(candidates):
            if track_index in used_tracks or measurement_index in used_measurements:
                continue
            assignments.append((track_index, measurement_index))
            used_tracks.add(track_index)
            used_measurements.add(measurement_index)
        return assignments

    def _start_track(self, measurement: NDArray[np.float64]) -> None:
        covariance = np.diag(
            [
                self.config.initial_range_std_m**2,
                self.config.initial_velocity_std_m_s**2,
            ]
        ).astype(np.float64)
        self._tracks.append(
            _Track(
                track_id=self._next_track_id,
                state=measurement.copy(),
                covariance=covariance,
            )
        )
        self._next_track_id += 1

    def _snapshot(self, track: _Track) -> TrackEstimate:
        return TrackEstimate(
            track_id=track.track_id,
            range_m=float(track.state[0]),
            velocity_m_s=float(track.state[1]),
            covariance=track.covariance.copy(),
            age=track.age,
            hits=track.hits,
            misses=track.misses,
            confirmed=track.hits >= self.config.confirmation_hits,
        )
