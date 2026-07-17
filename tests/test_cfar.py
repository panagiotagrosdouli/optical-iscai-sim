import numpy as np
import pytest

from optical_iscai.cfar import (
    CFARConfiguration,
    ca_cfar_2d,
    detect_range_doppler,
)
from optical_iscai.range_doppler import RangeDopplerMap


def test_cfar_detects_isolated_target() -> None:
    power = np.ones((21, 31), dtype=float)
    power[10, 15] = 100.0
    config = CFARConfiguration(
        training_velocity=3,
        training_range=4,
        guard_velocity=1,
        guard_range=1,
        probability_false_alarm=1e-3,
    )

    result = ca_cfar_2d(power, config)

    assert result.detections[10, 15]
    assert np.count_nonzero(result.detections) == 1
    assert result.noise_power[10, 15] == pytest.approx(1.0)
    assert result.threshold[10, 15] < power[10, 15]


def test_guard_cells_exclude_target_leakage() -> None:
    power = np.ones((17, 21), dtype=float)
    power[8, 10] = 80.0
    power[8, 11] = 30.0
    config = CFARConfiguration(
        training_velocity=2,
        training_range=3,
        guard_velocity=1,
        guard_range=1,
        probability_false_alarm=1e-2,
    )

    result = ca_cfar_2d(power, config)

    assert result.noise_power[8, 10] == pytest.approx(1.0)
    assert result.detections[8, 10]


def test_edge_cells_are_not_tested() -> None:
    power = np.ones((15, 19), dtype=float)
    result = ca_cfar_2d(
        power,
        CFARConfiguration(
            training_velocity=2,
            training_range=3,
            guard_velocity=1,
            guard_range=1,
        ),
    )

    assert np.isnan(result.threshold[0, 0])
    assert np.isnan(result.noise_power[0, 0])
    assert not result.detections[0, 0]
    assert np.isfinite(result.threshold[7, 9])


def test_constant_noise_produces_no_detection() -> None:
    power = np.full((15, 19), 5.0)
    result = ca_cfar_2d(
        power,
        CFARConfiguration(
            training_velocity=2,
            training_range=3,
            guard_velocity=1,
            guard_range=1,
            probability_false_alarm=1e-3,
        ),
    )

    assert not np.any(result.detections)


def test_physical_detection_coordinates() -> None:
    ranges = np.linspace(0.0, 100.0, 31)
    velocities = np.linspace(-10.0, 10.0, 21)
    power = np.ones((velocities.size, ranges.size))
    power[12, 18] = 100.0
    rd_map = RangeDopplerMap(ranges, velocities, power)

    cfar, detections = detect_range_doppler(
        rd_map,
        CFARConfiguration(
            training_velocity=3,
            training_range=4,
            guard_velocity=1,
            guard_range=1,
            probability_false_alarm=1e-3,
        ),
    )

    assert cfar.detections[12, 18]
    assert len(detections) == 1
    detection = detections[0]
    assert detection.range_m == pytest.approx(ranges[18])
    assert detection.velocity_m_s == pytest.approx(velocities[12])
    assert detection.power == pytest.approx(100.0)


@pytest.mark.parametrize(
    "config",
    [
        CFARConfiguration(training_velocity=-1),
        CFARConfiguration(training_range=0, training_velocity=0),
        CFARConfiguration(probability_false_alarm=0.0),
        CFARConfiguration(probability_false_alarm=1.0),
    ],
)
def test_invalid_configuration_is_rejected(config: CFARConfiguration) -> None:
    with pytest.raises(ValueError):
        ca_cfar_2d(np.ones((25, 25)), config)


def test_invalid_power_maps_are_rejected() -> None:
    config = CFARConfiguration(
        training_velocity=1,
        training_range=1,
        guard_velocity=0,
        guard_range=0,
    )

    with pytest.raises(ValueError, match="two-dimensional"):
        ca_cfar_2d(np.ones(10), config)
    with pytest.raises(ValueError, match="non-negative"):
        ca_cfar_2d(-np.ones((7, 7)), config)
    with pytest.raises(ValueError, match="smaller"):
        ca_cfar_2d(np.ones((2, 2)), config)
