import numpy as np
import pytest

from optical_iscai.noise import (
    BOLTZMANN_CONSTANT_J_K,
    ELEMENTARY_CHARGE_C,
    add_complex_awgn,
    shot_noise_variance_a2,
    thermal_noise_variance_a2,
)


def test_shot_noise_variance_matches_equation() -> None:
    current = 2.0e-3
    dark_current = 1.0e-6
    bandwidth = 10.0e6
    expected = 2.0 * ELEMENTARY_CHARGE_C * (current + dark_current) * bandwidth
    assert shot_noise_variance_a2(
        current,
        bandwidth,
        dark_current_a=dark_current,
    ) == pytest.approx(expected)


def test_thermal_noise_variance_matches_equation() -> None:
    temperature = 300.0
    resistance = 50.0
    bandwidth = 20.0e6
    expected = 4.0 * BOLTZMANN_CONSTANT_J_K * temperature * bandwidth / resistance
    assert thermal_noise_variance_a2(temperature, resistance, bandwidth) == pytest.approx(expected)


def test_awgn_is_reproducible_with_seeded_generator() -> None:
    signal = np.ones(4096, dtype=np.complex128)
    first = add_complex_awgn(signal, 15.0, rng=np.random.default_rng(7))
    second = add_complex_awgn(signal, 15.0, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(first, second)


def test_awgn_achieves_requested_snr_statistically() -> None:
    signal = np.ones(200_000, dtype=np.complex128)
    noisy = add_complex_awgn(signal, 12.0, rng=np.random.default_rng(123))
    noise = noisy - signal
    measured_snr_db = 10.0 * np.log10(
        np.mean(np.abs(signal) ** 2) / np.mean(np.abs(noise) ** 2)
    )
    assert measured_snr_db == pytest.approx(12.0, abs=0.1)


@pytest.mark.parametrize(
    "args",
    [
        (-1.0, 1.0),
        (1.0, -1.0),
    ],
)
def test_shot_noise_rejects_negative_inputs(args: tuple[float, float]) -> None:
    with pytest.raises(ValueError):
        shot_noise_variance_a2(*args)


def test_thermal_noise_rejects_invalid_resistance() -> None:
    with pytest.raises(ValueError):
        thermal_noise_variance_a2(300.0, 0.0, 1.0e6)


def test_awgn_rejects_zero_power_signal() -> None:
    with pytest.raises(ValueError):
        add_complex_awgn(np.zeros(16, dtype=np.complex128), 10.0)
