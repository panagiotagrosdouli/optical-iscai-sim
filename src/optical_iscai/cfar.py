"""Two-dimensional cell-averaging CFAR detection.

The detector operates on a linear-power range-Doppler map.  For each cell under
test (CUT), it estimates the local noise floor from a rectangular ring of
training cells surrounding a rectangular guard region.  Edge cells for which a
complete window is unavailable are left untested.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from optical_iscai.range_doppler import RangeDopplerMap


@dataclass(frozen=True, slots=True)
class CFARConfiguration:
    """Configuration of a rectangular 2D CA-CFAR detector."""

    training_velocity: int = 4
    training_range: int = 8
    guard_velocity: int = 1
    guard_range: int = 2
    probability_false_alarm: float = 1e-6

    def validate(self) -> None:
        values = {
            "training_velocity": self.training_velocity,
            "training_range": self.training_range,
            "guard_velocity": self.guard_velocity,
            "guard_range": self.guard_range,
        }
        for name, value in values.items():
            if isinstance(value, bool) or int(value) != value or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.training_velocity == 0 and self.training_range == 0:
            raise ValueError("at least one training extent must be greater than zero")
        if (
            not np.isfinite(self.probability_false_alarm)
            or not 0.0 < self.probability_false_alarm < 1.0
        ):
            raise ValueError("probability_false_alarm must be between zero and one")


@dataclass(frozen=True, slots=True)
class CFARResult:
    """Threshold, local noise estimate, and binary detection mask."""

    threshold: NDArray[np.float64]
    noise_power: NDArray[np.float64]
    detections: NDArray[np.bool_]
    training_cell_count: int
    scale_factor: float


@dataclass(frozen=True, slots=True)
class Detection:
    """Physical coordinates and power of one detected map cell."""

    velocity_index: int
    range_index: int
    velocity_m_s: float
    range_m: float
    power: float
    threshold: float


def _training_mask(config: CFARConfiguration) -> NDArray[np.bool_]:
    outer_v = config.training_velocity + config.guard_velocity
    outer_r = config.training_range + config.guard_range
    mask = np.ones((2 * outer_v + 1, 2 * outer_r + 1), dtype=bool)

    center_v = outer_v
    center_r = outer_r
    mask[
        center_v - config.guard_velocity : center_v + config.guard_velocity + 1,
        center_r - config.guard_range : center_r + config.guard_range + 1,
    ] = False
    return mask


def ca_cfar_2d(
    power: NDArray[np.floating],
    config: CFARConfiguration | None = None,
) -> CFARResult:
    """Apply rectangular two-dimensional CA-CFAR to a linear-power array.

    The threshold multiplier assumes independent exponentially distributed
    noise power samples:

        alpha = N * (P_fa ** (-1 / N) - 1)

    where ``N`` is the number of training cells.
    """

    cfg = CFARConfiguration() if config is None else config
    cfg.validate()

    values = np.asarray(power, dtype=np.float64)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("power must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("power must contain finite non-negative values")

    mask = _training_mask(cfg)
    training_count = int(np.count_nonzero(mask))
    if training_count == 0:
        raise ValueError("configuration produces no training cells")

    outer_v = cfg.training_velocity + cfg.guard_velocity
    outer_r = cfg.training_range + cfg.guard_range
    window_rows, window_cols = mask.shape
    if values.shape[0] < window_rows or values.shape[1] < window_cols:
        raise ValueError("power map is smaller than the configured CFAR window")

    alpha = training_count * (
        cfg.probability_false_alarm ** (-1.0 / training_count) - 1.0
    )
    threshold = np.full(values.shape, np.nan, dtype=np.float64)
    noise = np.full(values.shape, np.nan, dtype=np.float64)
    detections = np.zeros(values.shape, dtype=bool)

    for velocity_index in range(outer_v, values.shape[0] - outer_v):
        for range_index in range(outer_r, values.shape[1] - outer_r):
            window = values[
                velocity_index - outer_v : velocity_index + outer_v + 1,
                range_index - outer_r : range_index + outer_r + 1,
            ]
            noise_estimate = float(np.mean(window[mask]))
            cut_threshold = alpha * noise_estimate
            noise[velocity_index, range_index] = noise_estimate
            threshold[velocity_index, range_index] = cut_threshold
            detections[velocity_index, range_index] = (
                values[velocity_index, range_index] > cut_threshold
            )

    return CFARResult(
        threshold=threshold,
        noise_power=noise,
        detections=detections,
        training_cell_count=training_count,
        scale_factor=float(alpha),
    )


def detect_range_doppler(
    result: RangeDopplerMap,
    config: CFARConfiguration | None = None,
) -> tuple[CFARResult, tuple[Detection, ...]]:
    """Run CA-CFAR and return detections with physical coordinates."""

    if result.power.shape != (result.velocity_m_s.size, result.range_m.size):
        raise ValueError("range-Doppler power shape must match its physical axes")

    cfar = ca_cfar_2d(result.power, config)
    coordinates = np.argwhere(cfar.detections)
    detections = tuple(
        Detection(
            velocity_index=int(velocity_index),
            range_index=int(range_index),
            velocity_m_s=float(result.velocity_m_s[velocity_index]),
            range_m=float(result.range_m[range_index]),
            power=float(result.power[velocity_index, range_index]),
            threshold=float(cfar.threshold[velocity_index, range_index]),
        )
        for velocity_index, range_index in coordinates
    )
    return cfar, detections
