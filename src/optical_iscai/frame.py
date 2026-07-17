"""Multi-chirp frame simulation for optical PC-FMCW sensing.

The frame simulator connects waveform generation, point-target propagation,
coherent dechirping, and two-dimensional range-Doppler processing.  Target
range is updated at every chirp start, while Doppler phase remains continuous
across slow time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from optical_iscai.propagation import (
    PointTarget,
    generate_multi_target_echo,
    monostatic_doppler_hz,
)
from optical_iscai.range_doppler import RangeDopplerMap, range_doppler_map
from optical_iscai.receiver import dechirp
from optical_iscai.waveform import PCFMCWParameters, generate_chirp


@dataclass(frozen=True, slots=True)
class FrameConfiguration:
    """Numerical configuration for one coherent processing interval."""

    chirp_count: int = 64
    chirp_repetition_interval_s: float | None = None
    n_range_fft: int | None = None
    n_doppler_fft: int | None = None
    range_window: str | None = "hann"
    doppler_window: str | None = "hann"

    def repetition_interval_s(self, params: PCFMCWParameters) -> float:
        interval = (
            params.chirp_duration_s
            if self.chirp_repetition_interval_s is None
            else float(self.chirp_repetition_interval_s)
        )
        if not np.isfinite(interval) or interval < params.chirp_duration_s:
            raise ValueError(
                "chirp_repetition_interval_s must be finite and not shorter "
                "than chirp_duration_s"
            )
        return interval

    def validate(self, params: PCFMCWParameters) -> None:
        if isinstance(self.chirp_count, bool) or int(self.chirp_count) != self.chirp_count:
            raise ValueError("chirp_count must be an integer")
        if self.chirp_count < 2:
            raise ValueError("chirp_count must be at least two")
        self.repetition_interval_s(params)


@dataclass(frozen=True, slots=True)
class FrameResult:
    """Signals, processing result, and per-chirp target ground truth."""

    time_s: NDArray[np.float64]
    transmitted: NDArray[np.complex128]
    received: NDArray[np.complex128]
    beat: NDArray[np.complex128]
    range_doppler: RangeDopplerMap
    target_ranges_m: NDArray[np.float64]
    target_velocities_m_s: NDArray[np.float64]


def _targets_at_chirp_start(
    targets: tuple[PointTarget, ...],
    start_time_s: float,
    wavelength_m: float,
) -> tuple[PointTarget, ...]:
    snapshots: list[PointTarget] = []
    for target in targets:
        # Positive velocity means closing motion, so one-way range decreases.
        current_range = target.range_m - target.radial_velocity_m_s * start_time_s
        if current_range < 0.0:
            raise ValueError("a target crosses zero range during the simulated frame")

        doppler_hz = monostatic_doppler_hz(
            target.radial_velocity_m_s,
            wavelength_m,
        )
        snapshots.append(
            PointTarget(
                range_m=current_range,
                radial_velocity_m_s=target.radial_velocity_m_s,
                amplitude=target.amplitude,
                phase_rad=target.phase_rad + 2.0 * np.pi * doppler_hz * start_time_s,
            )
        )
    return tuple(snapshots)


def simulate_frame(
    params: PCFMCWParameters,
    targets: list[PointTarget] | tuple[PointTarget, ...],
    config: FrameConfiguration | None = None,
    *,
    bits: NDArray[np.integer] | None = None,
) -> FrameResult:
    """Simulate a coherent multi-chirp frame and its range-Doppler map.

    The same phase-coded bit sequence is transmitted on every chirp.  This is a
    deterministic baseline; later dataset generators may provide one sequence
    per chirp and store those communication symbols as explicit metadata.
    """

    frame_config = FrameConfiguration() if config is None else config
    frame_config.validate(params)

    target_tuple = tuple(targets)
    if not target_tuple:
        raise ValueError("targets must contain at least one PointTarget")
    if not all(isinstance(target, PointTarget) for target in target_tuple):
        raise TypeError("targets must contain only PointTarget instances")

    local_time, chirp = generate_chirp(params, bits)
    chirp_count = int(frame_config.chirp_count)
    sample_count = chirp.size
    repetition_interval = frame_config.repetition_interval_s(params)

    transmitted = np.tile(chirp, (chirp_count, 1)).astype(np.complex128, copy=False)
    received = np.zeros((chirp_count, sample_count), dtype=np.complex128)
    beat = np.zeros_like(received)
    ranges = np.zeros((chirp_count, len(target_tuple)), dtype=np.float64)
    velocities = np.zeros_like(ranges)

    for chirp_index in range(chirp_count):
        start_time = chirp_index * repetition_interval
        snapshots = _targets_at_chirp_start(
            target_tuple,
            start_time,
            params.wavelength_m,
        )
        received[chirp_index] = generate_multi_target_echo(
            local_time,
            chirp,
            snapshots,
            params.wavelength_m,
        )
        beat[chirp_index] = dechirp(chirp, received[chirp_index])
        ranges[chirp_index] = [target.range_m for target in snapshots]
        velocities[chirp_index] = [
            target.radial_velocity_m_s for target in snapshots
        ]

    processing = range_doppler_map(
        beat,
        params,
        chirp_repetition_interval_s=repetition_interval,
        n_range_fft=frame_config.n_range_fft,
        n_doppler_fft=frame_config.n_doppler_fft,
        range_window=frame_config.range_window,
        doppler_window=frame_config.doppler_window,
    )

    absolute_time = (
        np.arange(chirp_count, dtype=np.float64)[:, None] * repetition_interval
        + local_time[None, :]
    )
    return FrameResult(
        time_s=absolute_time,
        transmitted=transmitted,
        received=received,
        beat=beat,
        range_doppler=processing,
        target_ranges_m=ranges,
        target_velocities_m_s=velocities,
    )
