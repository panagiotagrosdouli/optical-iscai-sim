"""Multi-frame optical PC-FMCW simulation and sensing pipeline execution.

This module advances point targets through absolute time, simulates one coherent
processing interval per frame, and feeds every range-Doppler map through the
stateful CFAR, clustering, and tracking pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from optical_iscai.frame import FrameConfiguration, FrameResult, simulate_frame
from optical_iscai.pipeline import (
    PipelineConfiguration,
    PipelineResult,
    SensingPipeline,
)
from optical_iscai.propagation import PointTarget, monostatic_doppler_hz
from optical_iscai.waveform import PCFMCWParameters


@dataclass(frozen=True, slots=True)
class SequenceConfiguration:
    """Configuration for a uniformly sampled sequence of sensing frames."""

    frame_count: int = 10
    frame_interval_s: float = 0.1
    reset_pipeline: bool = True

    def validate(
        self,
        params: PCFMCWParameters,
        frame_config: FrameConfiguration,
    ) -> None:
        if isinstance(self.frame_count, bool) or int(self.frame_count) != self.frame_count:
            raise ValueError("frame_count must be an integer")
        if self.frame_count < 1:
            raise ValueError("frame_count must be at least one")
        if not np.isfinite(self.frame_interval_s) or self.frame_interval_s <= 0.0:
            raise ValueError("frame_interval_s must be finite and greater than zero")

        repetition_interval = frame_config.repetition_interval_s(params)
        frame_duration = (
            (int(frame_config.chirp_count) - 1) * repetition_interval
            + params.chirp_duration_s
        )
        if self.frame_interval_s < frame_duration:
            raise ValueError(
                "frame_interval_s must not be shorter than the coherent frame duration"
            )


@dataclass(frozen=True, slots=True)
class SequenceFrameResult:
    """Simulation, processing, and ground truth for one frame."""

    frame_index: int
    start_time_s: float
    frame: FrameResult
    pipeline: PipelineResult
    target_ranges_m: NDArray[np.float64]
    target_velocities_m_s: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SequenceResult:
    """Complete ordered output of a multi-frame sensing sequence."""

    frames: tuple[SequenceFrameResult, ...]

    @property
    def frame_times_s(self) -> NDArray[np.float64]:
        return np.asarray([frame.start_time_s for frame in self.frames], dtype=np.float64)

    @property
    def confirmed_track_counts(self) -> NDArray[np.int64]:
        return np.asarray(
            [len(frame.pipeline.confirmed_tracks) for frame in self.frames],
            dtype=np.int64,
        )


def _targets_at_time(
    targets: tuple[PointTarget, ...],
    time_s: float,
    wavelength_m: float,
) -> tuple[PointTarget, ...]:
    snapshots: list[PointTarget] = []
    for target in targets:
        current_range = target.range_m - target.radial_velocity_m_s * time_s
        if current_range < 0.0:
            raise ValueError("a target crosses zero range during the simulated sequence")
        doppler_hz = monostatic_doppler_hz(
            target.radial_velocity_m_s,
            wavelength_m,
        )
        snapshots.append(
            PointTarget(
                range_m=current_range,
                radial_velocity_m_s=target.radial_velocity_m_s,
                amplitude=target.amplitude,
                phase_rad=target.phase_rad + 2.0 * np.pi * doppler_hz * time_s,
            )
        )
    return tuple(snapshots)


def simulate_sequence(
    params: PCFMCWParameters,
    targets: list[PointTarget] | tuple[PointTarget, ...],
    frame_config: FrameConfiguration | None = None,
    pipeline_config: PipelineConfiguration | None = None,
    sequence_config: SequenceConfiguration | None = None,
    *,
    bits: NDArray[np.integer] | None = None,
    pipeline: SensingPipeline | None = None,
) -> SequenceResult:
    """Simulate and process a uniformly spaced sequence of sensing frames.

    A supplied ``pipeline`` may be reused across calls. By default its tracker is
    reset at the start according to ``SequenceConfiguration.reset_pipeline``.
    When frame noise has a base random seed, each frame receives ``seed + index``
    so that the complete sequence is reproducible without repeating noise samples.
    """

    frame_cfg = FrameConfiguration() if frame_config is None else frame_config
    frame_cfg.validate(params)
    sequence_cfg = SequenceConfiguration() if sequence_config is None else sequence_config
    sequence_cfg.validate(params, frame_cfg)

    target_tuple = tuple(targets)
    if not target_tuple:
        raise ValueError("targets must contain at least one PointTarget")
    if not all(isinstance(target, PointTarget) for target in target_tuple):
        raise TypeError("targets must contain only PointTarget instances")

    sensing_pipeline = (
        SensingPipeline(pipeline_config) if pipeline is None else pipeline
    )
    if pipeline is not None and pipeline_config is not None:
        raise ValueError("pipeline_config cannot be supplied together with pipeline")
    if sequence_cfg.reset_pipeline:
        sensing_pipeline.reset()

    outputs: list[SequenceFrameResult] = []
    for frame_index in range(int(sequence_cfg.frame_count)):
        start_time = frame_index * sequence_cfg.frame_interval_s
        frame_targets = _targets_at_time(
            target_tuple,
            start_time,
            params.wavelength_m,
        )
        current_frame_config = frame_cfg
        if frame_cfg.random_seed is not None:
            current_frame_config = replace(
                frame_cfg,
                random_seed=int(frame_cfg.random_seed) + frame_index,
            )

        frame_result = simulate_frame(
            params,
            frame_targets,
            current_frame_config,
            bits=bits,
        )
        pipeline_result = sensing_pipeline.process_frame(
            frame_result,
            frame_interval_s=sequence_cfg.frame_interval_s,
        )
        outputs.append(
            SequenceFrameResult(
                frame_index=frame_index,
                start_time_s=float(start_time),
                frame=frame_result,
                pipeline=pipeline_result,
                target_ranges_m=np.asarray(
                    [target.range_m for target in frame_targets],
                    dtype=np.float64,
                ),
                target_velocities_m_s=np.asarray(
                    [target.radial_velocity_m_s for target in frame_targets],
                    dtype=np.float64,
                ),
            )
        )

    return SequenceResult(frames=tuple(outputs))
