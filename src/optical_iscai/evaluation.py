"""Evaluation utilities for synthetic sensing and tracking sequences.

The functions in this module compare confirmed tracker outputs against the known
point-target ground truth produced by :mod:`optical_iscai.sequence`. Matching is
performed independently in every frame with gated nearest-neighbour assignment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from optical_iscai.sequence import SequenceResult
from optical_iscai.tracking import TrackEstimate


@dataclass(frozen=True, slots=True)
class EvaluationConfiguration:
    """Gates used to associate confirmed tracks with ground-truth targets."""

    range_gate_m: float = 5.0
    velocity_gate_m_s: float = 3.0
    confirmed_only: bool = True

    def validate(self) -> None:
        for name, value in {
            "range_gate_m": self.range_gate_m,
            "velocity_gate_m_s": self.velocity_gate_m_s,
        }.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and greater than zero")


@dataclass(frozen=True, slots=True)
class FrameEvaluation:
    """Association and error statistics for one sequence frame."""

    frame_index: int
    start_time_s: float
    ground_truth_count: int
    track_count: int
    matched_count: int
    missed_count: int
    false_track_count: int
    range_errors_m: NDArray[np.float64]
    velocity_errors_m_s: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SequenceEvaluation:
    """Aggregate evaluation results for a complete sequence."""

    frames: tuple[FrameEvaluation, ...]

    @property
    def ground_truth_count(self) -> int:
        return int(sum(frame.ground_truth_count for frame in self.frames))

    @property
    def track_count(self) -> int:
        return int(sum(frame.track_count for frame in self.frames))

    @property
    def matched_count(self) -> int:
        return int(sum(frame.matched_count for frame in self.frames))

    @property
    def missed_count(self) -> int:
        return int(sum(frame.missed_count for frame in self.frames))

    @property
    def false_track_count(self) -> int:
        return int(sum(frame.false_track_count for frame in self.frames))

    @property
    def recall(self) -> float:
        if self.ground_truth_count == 0:
            return float("nan")
        return self.matched_count / self.ground_truth_count

    @property
    def precision(self) -> float:
        if self.track_count == 0:
            return float("nan")
        return self.matched_count / self.track_count

    @property
    def range_rmse_m(self) -> float:
        errors = self._concatenate("range_errors_m")
        if errors.size == 0:
            return float("nan")
        return float(np.sqrt(np.mean(errors**2)))

    @property
    def velocity_rmse_m_s(self) -> float:
        errors = self._concatenate("velocity_errors_m_s")
        if errors.size == 0:
            return float("nan")
        return float(np.sqrt(np.mean(errors**2)))

    def _concatenate(self, field_name: str) -> NDArray[np.float64]:
        arrays = [getattr(frame, field_name) for frame in self.frames]
        non_empty = [array for array in arrays if array.size]
        if not non_empty:
            return np.empty(0, dtype=np.float64)
        return np.concatenate(non_empty).astype(np.float64, copy=False)


def _selected_tracks(
    tracks: tuple[TrackEstimate, ...],
    confirmed_only: bool,
) -> tuple[TrackEstimate, ...]:
    if not confirmed_only:
        return tracks
    return tuple(track for track in tracks if track.confirmed)


def _associate(
    truth: NDArray[np.float64],
    tracks: tuple[TrackEstimate, ...],
    config: EvaluationConfiguration,
) -> list[tuple[int, int]]:
    candidates: list[tuple[float, int, int]] = []
    for truth_index, target in enumerate(truth):
        for track_index, track in enumerate(tracks):
            range_error = abs(float(track.range_m - target[0]))
            velocity_error = abs(float(track.velocity_m_s - target[1]))
            if (
                range_error <= config.range_gate_m
                and velocity_error <= config.velocity_gate_m_s
            ):
                distance = np.hypot(
                    range_error / config.range_gate_m,
                    velocity_error / config.velocity_gate_m_s,
                )
                candidates.append((float(distance), truth_index, track_index))

    matches: list[tuple[int, int]] = []
    used_truth: set[int] = set()
    used_tracks: set[int] = set()
    for _, truth_index, track_index in sorted(candidates):
        if truth_index in used_truth or track_index in used_tracks:
            continue
        matches.append((truth_index, track_index))
        used_truth.add(truth_index)
        used_tracks.add(track_index)
    return matches


def evaluate_sequence(
    sequence: SequenceResult,
    config: EvaluationConfiguration | None = None,
) -> SequenceEvaluation:
    """Compare sequence tracks with per-frame target ground truth.

    Errors are signed as ``estimate - truth``. Association is one-to-one and
    independently recomputed for each frame.
    """

    cfg = EvaluationConfiguration() if config is None else config
    cfg.validate()
    if not isinstance(sequence, SequenceResult):
        raise TypeError("sequence must be a SequenceResult")

    frame_results: list[FrameEvaluation] = []
    for sequence_frame in sequence.frames:
        truth_ranges = np.asarray(sequence_frame.target_ranges_m, dtype=np.float64)
        truth_velocities = np.asarray(
            sequence_frame.target_velocities_m_s, dtype=np.float64
        )
        if truth_ranges.ndim != 1 or truth_velocities.ndim != 1:
            raise ValueError("ground-truth range and velocity arrays must be one-dimensional")
        if truth_ranges.shape != truth_velocities.shape:
            raise ValueError("ground-truth range and velocity arrays must have equal shape")
        if not np.all(np.isfinite(truth_ranges)) or not np.all(
            np.isfinite(truth_velocities)
        ):
            raise ValueError("ground-truth values must be finite")

        truth = np.column_stack((truth_ranges, truth_velocities))
        tracks = _selected_tracks(sequence_frame.pipeline.tracks, cfg.confirmed_only)
        matches = _associate(truth, tracks, cfg)
        range_errors = np.asarray(
            [tracks[j].range_m - truth[i, 0] for i, j in matches],
            dtype=np.float64,
        )
        velocity_errors = np.asarray(
            [tracks[j].velocity_m_s - truth[i, 1] for i, j in matches],
            dtype=np.float64,
        )
        frame_results.append(
            FrameEvaluation(
                frame_index=sequence_frame.frame_index,
                start_time_s=sequence_frame.start_time_s,
                ground_truth_count=int(truth.shape[0]),
                track_count=len(tracks),
                matched_count=len(matches),
                missed_count=int(truth.shape[0] - len(matches)),
                false_track_count=len(tracks) - len(matches),
                range_errors_m=range_errors,
                velocity_errors_m_s=velocity_errors,
            )
        )

    return SequenceEvaluation(frames=tuple(frame_results))
