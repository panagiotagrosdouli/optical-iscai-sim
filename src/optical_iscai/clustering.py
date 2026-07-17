"""Cluster neighbouring CFAR detections into target-level measurements.

The CA-CFAR stage may mark several adjacent range-Doppler cells for one physical
target.  This module groups connected cells and represents every cluster using a
power-weighted centroid together with its strongest member cell.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from optical_iscai.cfar import CFARResult
from optical_iscai.range_doppler import RangeDopplerMap


@dataclass(frozen=True, slots=True)
class ClusterConfiguration:
    """Configuration for connected-component detection clustering."""

    connectivity: int = 8
    minimum_cells: int = 1
    minimum_total_power: float = 0.0

    def validate(self) -> None:
        if self.connectivity not in (4, 8):
            raise ValueError("connectivity must be either 4 or 8")
        if (
            isinstance(self.minimum_cells, bool)
            or int(self.minimum_cells) != self.minimum_cells
            or self.minimum_cells < 1
        ):
            raise ValueError("minimum_cells must be a positive integer")
        if not np.isfinite(self.minimum_total_power) or self.minimum_total_power < 0.0:
            raise ValueError("minimum_total_power must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class DetectionCluster:
    """One target-level measurement formed from adjacent detected cells."""

    range_m: float
    velocity_m_s: float
    total_power: float
    peak_power: float
    peak_range_m: float
    peak_velocity_m_s: float
    peak_range_index: int
    peak_velocity_index: int
    cell_count: int
    cell_indices: tuple[tuple[int, int], ...]


def _neighbour_offsets(connectivity: int) -> tuple[tuple[int, int], ...]:
    if connectivity == 4:
        return ((-1, 0), (1, 0), (0, -1), (0, 1))
    return tuple(
        (dv, dr)
        for dv in (-1, 0, 1)
        for dr in (-1, 0, 1)
        if not (dv == 0 and dr == 0)
    )


def _connected_components(
    mask: NDArray[np.bool_],
    connectivity: int,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    visited = np.zeros(mask.shape, dtype=bool)
    offsets = _neighbour_offsets(connectivity)
    components: list[tuple[tuple[int, int], ...]] = []

    for start_v, start_r in np.argwhere(mask):
        start = (int(start_v), int(start_r))
        if visited[start]:
            continue

        stack = [start]
        visited[start] = True
        cells: list[tuple[int, int]] = []
        while stack:
            velocity_index, range_index = stack.pop()
            cells.append((velocity_index, range_index))
            for delta_v, delta_r in offsets:
                neighbour_v = velocity_index + delta_v
                neighbour_r = range_index + delta_r
                if not (
                    0 <= neighbour_v < mask.shape[0]
                    and 0 <= neighbour_r < mask.shape[1]
                ):
                    continue
                neighbour = (neighbour_v, neighbour_r)
                if mask[neighbour] and not visited[neighbour]:
                    visited[neighbour] = True
                    stack.append(neighbour)

        components.append(tuple(sorted(cells)))

    return tuple(components)


def cluster_cfar_detections(
    range_doppler: RangeDopplerMap,
    cfar: CFARResult,
    config: ClusterConfiguration | None = None,
) -> tuple[DetectionCluster, ...]:
    """Group adjacent CFAR cells and return one measurement per component."""

    cfg = ClusterConfiguration() if config is None else config
    cfg.validate()

    expected_shape = (range_doppler.velocity_m_s.size, range_doppler.range_m.size)
    if range_doppler.power.shape != expected_shape:
        raise ValueError("range-Doppler power shape must match its physical axes")
    if cfar.detections.shape != expected_shape:
        raise ValueError("CFAR detection mask shape must match the range-Doppler map")

    clusters: list[DetectionCluster] = []
    for cells in _connected_components(cfar.detections, cfg.connectivity):
        if len(cells) < cfg.minimum_cells:
            continue

        velocity_indices = np.asarray([cell[0] for cell in cells], dtype=np.int64)
        range_indices = np.asarray([cell[1] for cell in cells], dtype=np.int64)
        powers = range_doppler.power[velocity_indices, range_indices].astype(
            np.float64, copy=False
        )
        total_power = float(np.sum(powers))
        if total_power < cfg.minimum_total_power:
            continue

        if total_power > 0.0:
            centroid_range = float(
                np.sum(powers * range_doppler.range_m[range_indices]) / total_power
            )
            centroid_velocity = float(
                np.sum(powers * range_doppler.velocity_m_s[velocity_indices])
                / total_power
            )
        else:
            centroid_range = float(np.mean(range_doppler.range_m[range_indices]))
            centroid_velocity = float(
                np.mean(range_doppler.velocity_m_s[velocity_indices])
            )

        peak_relative_index = int(np.argmax(powers))
        peak_velocity_index = int(velocity_indices[peak_relative_index])
        peak_range_index = int(range_indices[peak_relative_index])
        clusters.append(
            DetectionCluster(
                range_m=centroid_range,
                velocity_m_s=centroid_velocity,
                total_power=total_power,
                peak_power=float(powers[peak_relative_index]),
                peak_range_m=float(range_doppler.range_m[peak_range_index]),
                peak_velocity_m_s=float(
                    range_doppler.velocity_m_s[peak_velocity_index]
                ),
                peak_range_index=peak_range_index,
                peak_velocity_index=peak_velocity_index,
                cell_count=len(cells),
                cell_indices=cells,
            )
        )

    return tuple(sorted(clusters, key=lambda cluster: cluster.total_power, reverse=True))
