"""Phase-coded FMCW waveform utilities.

The optical carrier in the reference paper is 193.4 THz. Directly sampling that
carrier is neither necessary nor practical for system-level simulation, so this
module generates the complex baseband envelope

    s(t) = A exp(j [pi * mu * t^2 + phi_DPSK(t)])

where ``mu = bandwidth / chirp_duration``. The omitted carrier term can be
reintroduced analytically when required.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

SPEED_OF_LIGHT_M_S = 299_792_458.0


@dataclass(frozen=True, slots=True)
class PCFMCWParameters:
    """Physical and numerical parameters for one PC-FMCW chirp."""

    carrier_frequency_hz: float = 193.4e12
    bandwidth_hz: float = 10e9
    chirp_duration_s: float = 10e-6
    data_rate_bps: float = 1e9
    sample_rate_hz: float = 20e9
    amplitude: float = 1.0

    def __post_init__(self) -> None:
        positive = {
            "carrier_frequency_hz": self.carrier_frequency_hz,
            "bandwidth_hz": self.bandwidth_hz,
            "chirp_duration_s": self.chirp_duration_s,
            "data_rate_bps": self.data_rate_bps,
            "sample_rate_hz": self.sample_rate_hz,
            "amplitude": self.amplitude,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and greater than zero")

        if self.sample_rate_hz < 2.0 * self.bandwidth_hz:
            raise ValueError(
                "sample_rate_hz must be at least twice bandwidth_hz for the "
                "default complex-envelope discretization"
            )

    @property
    def chirp_slope_hz_per_s(self) -> float:
        return self.bandwidth_hz / self.chirp_duration_s

    @property
    def wavelength_m(self) -> float:
        return SPEED_OF_LIGHT_M_S / self.carrier_frequency_hz

    @property
    def ideal_range_resolution_m(self) -> float:
        return SPEED_OF_LIGHT_M_S / (2.0 * self.bandwidth_hz)

    @property
    def symbols_per_chirp(self) -> int:
        value = self.data_rate_bps * self.chirp_duration_s
        rounded = int(round(value))
        if not np.isclose(value, rounded, rtol=0.0, atol=1e-9):
            raise ValueError("data_rate_bps * chirp_duration_s must be an integer")
        return rounded

    @property
    def samples_per_chirp(self) -> int:
        value = self.sample_rate_hz * self.chirp_duration_s
        rounded = int(round(value))
        if not np.isclose(value, rounded, rtol=0.0, atol=1e-9):
            raise ValueError("sample_rate_hz * chirp_duration_s must be an integer")
        return rounded


def differential_encode(bits: NDArray[np.integer]) -> NDArray[np.float64]:
    """Map binary information bits to cumulative DPSK phases in {0, pi}.

    A bit value of one produces a phase transition of pi, while zero preserves
    the previous phase. The returned array contains one phase per symbol.
    """

    bit_array = np.asarray(bits)
    if bit_array.ndim != 1:
        raise ValueError("bits must be a one-dimensional array")
    if not np.all((bit_array == 0) | (bit_array == 1)):
        raise ValueError("bits must contain only zeros and ones")

    phase_state = np.mod(np.cumsum(bit_array.astype(np.int64)), 2)
    return phase_state.astype(np.float64) * np.pi


def generate_chirp(
    params: PCFMCWParameters,
    bits: NDArray[np.integer] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """Generate one phase-coded FMCW complex-envelope chirp.

    Parameters
    ----------
    params:
        Waveform and sampling configuration.
    bits:
        Optional binary vector with exactly ``params.symbols_per_chirp``
        elements. When omitted, an all-zero DPSK sequence is used.
    """

    symbol_count = params.symbols_per_chirp
    if bits is None:
        bit_array = np.zeros(symbol_count, dtype=np.int8)
    else:
        bit_array = np.asarray(bits)
        if bit_array.shape != (symbol_count,):
            raise ValueError(f"bits must have shape ({symbol_count},)")

    symbol_phases = differential_encode(bit_array)
    sample_count = params.samples_per_chirp
    time_s = np.arange(sample_count, dtype=np.float64) / params.sample_rate_hz

    symbol_indices = np.floor(time_s * params.data_rate_bps).astype(np.int64)
    symbol_indices = np.minimum(symbol_indices, symbol_count - 1)
    data_phase = symbol_phases[symbol_indices]

    chirp_phase = np.pi * params.chirp_slope_hz_per_s * time_s**2
    envelope = params.amplitude * np.exp(1j * (chirp_phase + data_phase))
    return time_s, envelope.astype(np.complex128, copy=False)
