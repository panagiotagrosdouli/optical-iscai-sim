"""Scenario adapters for YAML-driven Monte Carlo experiments.

The adapters in this module translate flat experiment parameters into the existing
waveform, frame, pipeline, target, and sequence configuration objects.  They are
intended as reproducible baseline scenarios rather than calibrated optical link
budgets; target amplitude remains an explicit complex-envelope simulation input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from optical_iscai.frame import FrameConfiguration
from optical_iscai.pipeline import PipelineConfiguration
from optical_iscai.propagation import PointTarget
from optical_iscai.sequence import SequenceConfiguration, SequenceResult, simulate_sequence
from optical_iscai.waveform import PCFMCWParameters

Parameter = int | float | str | bool


@dataclass(frozen=True, slots=True)
class HighwayScenario:
    """Fully resolved inputs for one highway sequence simulation."""

    waveform: PCFMCWParameters
    frame: FrameConfiguration
    pipeline: PipelineConfiguration
    sequence: SequenceConfiguration
    targets: tuple[PointTarget, ...]

    def simulate(self) -> SequenceResult:
        """Run the resolved scenario through the complete sensing pipeline."""
        return simulate_sequence(
            self.waveform,
            self.targets,
            frame_config=self.frame,
            pipeline_config=self.pipeline,
            sequence_config=self.sequence,
        )


def _number(parameters: Mapping[str, Parameter], name: str, default: float) -> float:
    value = parameters.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _integer(parameters: Mapping[str, Parameter], name: str, default: int) -> int:
    value = parameters.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be an integer")
    integer = int(value)
    if integer != value:
        raise ValueError(f"{name} must be an integer")
    return integer


def build_highway_scenario(
    parameters: Mapping[str, Parameter],
    rng: np.random.Generator,
) -> HighwayScenario:
    """Resolve a randomized multi-target highway scenario.

    Supported flat parameters include ``snr_db``, ``target_count``, range and
    velocity bounds, target amplitude bounds, waveform sampling values, coherent
    chirp count, and sequence timing.  Positive radial velocity denotes closing
    motion, matching :class:`~optical_iscai.propagation.PointTarget`.
    """
    if not isinstance(parameters, Mapping):
        raise TypeError("parameters must be a mapping")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")

    target_count = _integer(parameters, "target_count", 3)
    if target_count < 1:
        raise ValueError("target_count must be at least one")

    range_min = _number(parameters, "range_min_m", 20.0)
    range_max = _number(parameters, "range_max_m", 120.0)
    velocity_min = _number(parameters, "velocity_min_m_s", -0.02)
    velocity_max = _number(parameters, "velocity_max_m_s", 0.02)
    amplitude_min = _number(parameters, "amplitude_min", 0.6)
    amplitude_max = _number(parameters, "amplitude_max", 1.0)
    if not 0.0 <= range_min < range_max:
        raise ValueError("range bounds must satisfy 0 <= range_min_m < range_max_m")
    if velocity_min > velocity_max:
        raise ValueError("velocity_min_m_s must not exceed velocity_max_m_s")
    if not 0.0 <= amplitude_min <= amplitude_max:
        raise ValueError("amplitude bounds must satisfy 0 <= amplitude_min <= amplitude_max")

    waveform = PCFMCWParameters(
        carrier_frequency_hz=_number(parameters, "carrier_frequency_hz", 193.4e12),
        bandwidth_hz=_number(parameters, "bandwidth_hz", 20e6),
        chirp_duration_s=_number(parameters, "chirp_duration_s", 20e-6),
        data_rate_bps=_number(parameters, "data_rate_bps", 1e6),
        sample_rate_hz=_number(parameters, "sample_rate_hz", 40e6),
        amplitude=_number(parameters, "transmit_amplitude", 1.0),
    )

    frame = FrameConfiguration(
        chirp_count=_integer(parameters, "chirp_count", 16),
        receiver_snr_db=_number(parameters, "snr_db", 10.0),
        random_seed=int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32)),
    )
    sequence = SequenceConfiguration(
        frame_count=_integer(parameters, "frame_count", 8),
        frame_interval_s=_number(parameters, "frame_interval_s", 1e-3),
        reset_pipeline=True,
    )

    ranges = rng.uniform(range_min, range_max, size=target_count)
    velocities = rng.uniform(velocity_min, velocity_max, size=target_count)
    amplitudes = rng.uniform(amplitude_min, amplitude_max, size=target_count)
    phases = rng.uniform(-np.pi, np.pi, size=target_count)
    targets = tuple(
        PointTarget(
            range_m=float(target_range),
            radial_velocity_m_s=float(velocity),
            amplitude=float(amplitude),
            phase_rad=float(phase),
        )
        for target_range, velocity, amplitude, phase in zip(
            ranges, velocities, amplitudes, phases, strict=True
        )
    )

    scenario = HighwayScenario(
        waveform=waveform,
        frame=frame,
        pipeline=PipelineConfiguration(),
        sequence=sequence,
        targets=targets,
    )
    frame.validate(waveform)
    sequence.validate(waveform, frame)
    return scenario


def simulate_highway(
    parameters: dict[str, Parameter],
    rng: np.random.Generator,
) -> SequenceResult:
    """Monte Carlo adapter used by ``optical_iscai.cli`` highway YAML files."""
    return build_highway_scenario(parameters, rng).simulate()
