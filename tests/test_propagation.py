import numpy as np
import pytest

from optical_iscai.propagation import (
    PointTarget,
    generate_echo,
    generate_multi_target_echo,
    monostatic_doppler_hz,
    round_trip_delay_s,
)
from optical_iscai.waveform import PCFMCWParameters, SPEED_OF_LIGHT_M_S


def test_round_trip_delay() -> None:
    assert round_trip_delay_s(150.0) == pytest.approx(2.0 * 150.0 / SPEED_OF_LIGHT_M_S)


def test_monostatic_doppler_sign_and_value() -> None:
    wavelength = 1.55e-6
    assert monostatic_doppler_hz(30.0, wavelength) == pytest.approx(2.0 * 30.0 / wavelength)
    assert monostatic_doppler_hz(-30.0, wavelength) < 0.0


def test_integer_sample_delay_has_no_wraparound() -> None:
    sample_rate = 1_000.0
    time_s = np.arange(10, dtype=np.float64) / sample_rate
    transmitted = np.arange(10, dtype=np.float64).astype(np.complex128)
    delay_samples = 2
    target_range = delay_samples / sample_rate * SPEED_OF_LIGHT_M_S / 2.0

    echo = generate_echo(
        time_s,
        transmitted,
        PointTarget(range_m=target_range),
        wavelength_m=1.55e-6,
    )

    np.testing.assert_allclose(echo[:delay_samples], 0.0)
    np.testing.assert_allclose(echo[delay_samples:], transmitted[:-delay_samples])


def test_stationary_zero_range_target_scales_signal() -> None:
    time_s = np.linspace(0.0, 1e-6, 16, endpoint=False)
    transmitted = np.exp(1j * 2.0 * np.pi * 1e6 * time_s)
    target = PointTarget(range_m=0.0, amplitude=0.25, phase_rad=np.pi / 2.0)

    echo = generate_echo(time_s, transmitted, target, wavelength_m=1.55e-6)
    expected = 0.25 * transmitted * np.exp(1j * np.pi / 2.0)
    np.testing.assert_allclose(echo, expected)


def test_multi_target_echo_is_coherent_sum() -> None:
    params = PCFMCWParameters(
        bandwidth_hz=1e6,
        chirp_duration_s=1e-3,
        data_rate_bps=1e3,
        sample_rate_hz=2e6,
    )
    time_s = np.arange(params.samples_per_chirp) / params.sample_rate_hz
    transmitted = np.ones(params.samples_per_chirp, dtype=np.complex128)
    targets = [
        PointTarget(range_m=0.0, amplitude=0.3),
        PointTarget(range_m=0.0, amplitude=0.2, phase_rad=np.pi),
    ]

    echo = generate_multi_target_echo(time_s, transmitted, targets, params.wavelength_m)
    np.testing.assert_allclose(echo, 0.1 + 0.0j, atol=1e-12)


@pytest.mark.parametrize(
    "target",
    [
        PointTarget(range_m=0.0),
        PointTarget(range_m=10.0, radial_velocity_m_s=5.0),
    ],
)
def test_echo_shape_and_finiteness(target: PointTarget) -> None:
    time_s = np.arange(100, dtype=np.float64) / 1e6
    transmitted = np.ones(100, dtype=np.complex128)
    echo = generate_echo(time_s, transmitted, target, wavelength_m=1.55e-6)
    assert echo.shape == transmitted.shape
    assert np.all(np.isfinite(echo))


def test_invalid_target_parameters() -> None:
    with pytest.raises(ValueError):
        PointTarget(range_m=-1.0)
    with pytest.raises(ValueError):
        PointTarget(range_m=1.0, amplitude=-0.1)
