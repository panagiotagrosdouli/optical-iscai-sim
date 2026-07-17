import numpy as np
import pytest

from optical_iscai.scenarios import build_highway_scenario


def test_build_highway_scenario_is_reproducible() -> None:
    parameters = {
        "target_count": 2,
        "frame_count": 2,
        "chirp_count": 4,
        "frame_interval_s": 1e-3,
        "snr_db": 5.0,
    }

    first = build_highway_scenario(parameters, np.random.default_rng(7))
    second = build_highway_scenario(parameters, np.random.default_rng(7))

    assert first.waveform == second.waveform
    assert first.frame == second.frame
    assert first.sequence == second.sequence
    assert first.targets == second.targets
    assert len(first.targets) == 2


def test_build_highway_scenario_applies_parameter_overrides() -> None:
    scenario = build_highway_scenario(
        {
            "target_count": 1,
            "range_min_m": 50.0,
            "range_max_m": 51.0,
            "velocity_min_m_s": 0.01,
            "velocity_max_m_s": 0.01,
            "amplitude_min": 0.8,
            "amplitude_max": 0.8,
            "frame_count": 3,
            "chirp_count": 8,
            "snr_db": -4.0,
        },
        np.random.default_rng(3),
    )

    target = scenario.targets[0]
    assert 50.0 <= target.range_m <= 51.0
    assert target.radial_velocity_m_s == pytest.approx(0.01)
    assert target.amplitude == pytest.approx(0.8)
    assert scenario.sequence.frame_count == 3
    assert scenario.frame.chirp_count == 8
    assert scenario.frame.receiver_snr_db == pytest.approx(-4.0)


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"target_count": 0}, "target_count"),
        ({"range_min_m": 10.0, "range_max_m": 10.0}, "range bounds"),
        ({"velocity_min_m_s": 1.0, "velocity_max_m_s": -1.0}, "velocity_min"),
        ({"amplitude_min": -0.1}, "amplitude bounds"),
        ({"frame_count": 1.5}, "frame_count"),
    ],
)
def test_build_highway_scenario_rejects_invalid_parameters(
    parameters: dict[str, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_highway_scenario(parameters, np.random.default_rng(0))
