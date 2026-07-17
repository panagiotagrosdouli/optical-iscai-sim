"""Noise utilities for coherent optical receiver simulation.

The first dataset baseline uses circular complex additive white Gaussian noise
(AWGN) specified by a measured complex-envelope SNR.  Separate helper
functions expose shot-noise and thermal-noise current variances so a later
link-budget model can derive the equivalent baseband noise power from physical
receiver parameters.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

BOLTZMANN_CONSTANT_J_K = 1.380649e-23
ELEMENTARY_CHARGE_C = 1.602176634e-19


def shot_noise_variance_a2(
    photocurrent_a: float,
    bandwidth_hz: float,
    *,
    dark_current_a: float = 0.0,
) -> float:
    """Return photodetector shot-noise current variance ``2 q I B``."""

    values = {
        "photocurrent_a": photocurrent_a,
        "bandwidth_hz": bandwidth_hz,
        "dark_current_a": dark_current_a,
    }
    for name, value in values.items():
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if bandwidth_hz == 0.0:
        return 0.0
    return float(
        2.0
        * ELEMENTARY_CHARGE_C
        * (photocurrent_a + dark_current_a)
        * bandwidth_hz
    )


def thermal_noise_variance_a2(
    temperature_k: float,
    resistance_ohm: float,
    bandwidth_hz: float,
) -> float:
    """Return resistor thermal-noise current variance ``4 k T B / R``."""

    if not np.isfinite(temperature_k) or temperature_k <= 0.0:
        raise ValueError("temperature_k must be finite and greater than zero")
    if not np.isfinite(resistance_ohm) or resistance_ohm <= 0.0:
        raise ValueError("resistance_ohm must be finite and greater than zero")
    if not np.isfinite(bandwidth_hz) or bandwidth_hz < 0.0:
        raise ValueError("bandwidth_hz must be finite and non-negative")
    return float(
        4.0
        * BOLTZMANN_CONSTANT_J_K
        * temperature_k
        * bandwidth_hz
        / resistance_ohm
    )


def add_complex_awgn(
    signal: NDArray[np.complexfloating],
    snr_db: float,
    *,
    rng: np.random.Generator | None = None,
) -> NDArray[np.complex128]:
    """Add circular complex AWGN at the requested measured signal SNR.

    SNR is defined using the mean complex-envelope power ``mean(abs(x)**2)``.
    The total complex noise power is divided equally between the in-phase and
    quadrature components.  A caller-supplied NumPy generator enables exact
    dataset reproducibility.
    """

    samples = np.asarray(signal, dtype=np.complex128)
    if samples.size == 0:
        raise ValueError("signal must contain at least one sample")
    if not np.all(np.isfinite(samples)):
        raise ValueError("signal must contain finite values")
    if not np.isfinite(snr_db):
        raise ValueError("snr_db must be finite")

    signal_power = float(np.mean(np.abs(samples) ** 2))
    if signal_power <= 0.0:
        raise ValueError("signal must have non-zero mean power")

    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    component_std = np.sqrt(noise_power / 2.0)
    generator = np.random.default_rng() if rng is None else rng
    noise = component_std * (
        generator.standard_normal(samples.shape)
        + 1j * generator.standard_normal(samples.shape)
    )
    return (samples + noise).astype(np.complex128, copy=False)
