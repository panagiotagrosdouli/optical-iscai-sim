"""Target propagation and coherent echo generation for optical PC-FMCW.

This module models the deterministic delay and Doppler terms used by the
paper's received-signal equation.  Link-budget attenuation, atmospheric loss,
receiver noise, and target reflectivity calibration are intentionally kept
separate so that every dataset column can retain explicit provenance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from optical_iscai.waveform import SPEED_OF_LIGHT_M_S


@dataclass(frozen=True, slots=True)
class PointTarget:
    """A point target used by the first-order coherent echo model.

    Parameters
    ----------
    range_m:
        One-way distance from the ego transmitter to the target.
    radial_velocity_m_s:
        Closing radial velocity. Positive values indicate an approaching
        target and therefore a positive Doppler shift under this convention.
    amplitude:
        Complex-envelope amplitude scale after propagation and reflection.
        This is a controlled simulation input, not yet a physical link budget.
    phase_rad:
        Constant target/channel phase offset.
    """

    range_m: float
    radial_velocity_m_s: float = 0.0
    amplitude: float = 1.0
    phase_rad: float = 0.0

    def __post_init__(self) -> None:
        values = {
            "range_m": self.range_m,
            "radial_velocity_m_s": self.radial_velocity_m_s,
            "amplitude": self.amplitude,
            "phase_rad": self.phase_rad,
        }
        for name, value in values.items():
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.range_m < 0:
            raise ValueError("range_m must be greater than or equal to zero")
        if self.amplitude < 0:
            raise ValueError("amplitude must be greater than or equal to zero")


def round_trip_delay_s(range_m: float) -> float:
    """Return monostatic optical round-trip delay ``2R/c``."""

    if not np.isfinite(range_m) or range_m < 0:
        raise ValueError("range_m must be finite and greater than or equal to zero")
    return 2.0 * range_m / SPEED_OF_LIGHT_M_S


def monostatic_doppler_hz(radial_velocity_m_s: float, wavelength_m: float) -> float:
    """Return monostatic Doppler shift ``2v/lambda``.

    Positive radial velocity denotes closing motion in this project.
    """

    if not np.isfinite(radial_velocity_m_s):
        raise ValueError("radial_velocity_m_s must be finite")
    if not np.isfinite(wavelength_m) or wavelength_m <= 0:
        raise ValueError("wavelength_m must be finite and greater than zero")
    return 2.0 * radial_velocity_m_s / wavelength_m


def _fractional_delay(
    time_s: NDArray[np.float64],
    signal: NDArray[np.complex128],
    delay_s: float,
) -> NDArray[np.complex128]:
    """Apply a non-circular fractional delay using linear interpolation."""

    query_time = time_s - delay_s
    real = np.interp(query_time, time_s, signal.real, left=0.0, right=0.0)
    imag = np.interp(query_time, time_s, signal.imag, left=0.0, right=0.0)
    return (real + 1j * imag).astype(np.complex128, copy=False)


def generate_echo(
    time_s: NDArray[np.floating],
    transmitted: NDArray[np.complexfloating],
    target: PointTarget,
    wavelength_m: float,
) -> NDArray[np.complex128]:
    """Generate one delayed and Doppler-shifted coherent target echo.

    The returned complex envelope is

    ``a s(t - tau) exp(j (2 pi f_D t + phi))``.

    Samples whose delayed time lies outside the supplied transmitted waveform
    are zero. This avoids the physically incorrect circular wrap-around that a
    plain array roll would introduce.
    """

    t = np.asarray(time_s, dtype=np.float64)
    tx = np.asarray(transmitted, dtype=np.complex128)
    if t.ndim != 1 or tx.ndim != 1 or t.shape != tx.shape:
        raise ValueError("time_s and transmitted must be one-dimensional arrays of equal shape")
    if t.size < 2:
        raise ValueError("at least two samples are required")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(tx)):
        raise ValueError("time_s and transmitted must contain finite values")
    if not np.all(np.diff(t) > 0):
        raise ValueError("time_s must be strictly increasing")

    delayed = _fractional_delay(t, tx, round_trip_delay_s(target.range_m))
    doppler_hz = monostatic_doppler_hz(target.radial_velocity_m_s, wavelength_m)
    phase = 2.0 * np.pi * doppler_hz * t + target.phase_rad
    return target.amplitude * delayed * np.exp(1j * phase)


def generate_multi_target_echo(
    time_s: NDArray[np.floating],
    transmitted: NDArray[np.complexfloating],
    targets: list[PointTarget] | tuple[PointTarget, ...],
    wavelength_m: float,
) -> NDArray[np.complex128]:
    """Coherently sum echoes from multiple point targets."""

    tx = np.asarray(transmitted, dtype=np.complex128)
    result = np.zeros_like(tx, dtype=np.complex128)
    for target in targets:
        result += generate_echo(time_s, tx, target, wavelength_m)
    return result
