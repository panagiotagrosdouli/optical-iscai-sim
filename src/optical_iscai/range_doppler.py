"""Two-dimensional FMCW range-Doppler processing.

The input is a matrix of complex dechirped samples with shape
``(number_of_chirps, samples_per_chirp)``.  A fast-time FFT estimates beat
frequency (and therefore uncorrected range), while a slow-time FFT estimates
Doppler across successive chirps.

With the repository convention ``beat = tx * conj(rx)`` and positive target
velocity denoting closing motion, a positive physical Doppler shift appears at
a negative slow-time FFT frequency.  The velocity axis therefore uses

    v = -f_slow * wavelength / 2.

The fast-time range axis remains Doppler-coupled.  A later estimator may correct
it using the jointly estimated Doppler frequency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from optical_iscai.receiver import beat_frequency_to_range_m
from optical_iscai.waveform import PCFMCWParameters


@dataclass(frozen=True, slots=True)
class RangeDopplerMap:
    """Power map and physical axes produced by the two-dimensional FFT."""

    range_m: NDArray[np.float64]
    velocity_m_s: NDArray[np.float64]
    power: NDArray[np.float64]


def _window(length: int, name: str | None) -> NDArray[np.float64]:
    if name is None:
        return np.ones(length, dtype=np.float64)
    if name == "hann":
        return np.hanning(length)
    raise ValueError("window must be 'hann' or None")


def range_doppler_map(
    beat_matrix: NDArray[np.complexfloating],
    params: PCFMCWParameters,
    *,
    chirp_repetition_interval_s: float | None = None,
    n_range_fft: int | None = None,
    n_doppler_fft: int | None = None,
    range_window: str | None = "hann",
    doppler_window: str | None = "hann",
) -> RangeDopplerMap:
    """Compute a positive-range, FFT-shifted range-Doppler power map.

    Parameters
    ----------
    beat_matrix:
        Dechirped data arranged as ``(chirps, fast_time_samples)``.
    params:
        PC-FMCW waveform parameters.
    chirp_repetition_interval_s:
        Time between chirp starts. Defaults to ``params.chirp_duration_s``.
    n_range_fft, n_doppler_fft:
        Optional FFT lengths. They may zero-pad but may not truncate the input.
    range_window, doppler_window:
        ``"hann"`` or ``None`` for each processing dimension.
    """

    beat = np.asarray(beat_matrix, dtype=np.complex128)
    if beat.ndim != 2 or beat.shape[0] < 2 or beat.shape[1] < 2:
        raise ValueError("beat_matrix must be two-dimensional with at least two chirps and samples")
    if not np.all(np.isfinite(beat)):
        raise ValueError("beat_matrix must contain finite values")

    chirp_count, sample_count = beat.shape
    range_fft_length = sample_count if n_range_fft is None else int(n_range_fft)
    doppler_fft_length = chirp_count if n_doppler_fft is None else int(n_doppler_fft)
    if range_fft_length < sample_count:
        raise ValueError("n_range_fft must be greater than or equal to the sample count")
    if doppler_fft_length < chirp_count:
        raise ValueError("n_doppler_fft must be greater than or equal to the chirp count")

    repetition_interval = (
        params.chirp_duration_s
        if chirp_repetition_interval_s is None
        else float(chirp_repetition_interval_s)
    )
    if not np.isfinite(repetition_interval) or repetition_interval <= 0:
        raise ValueError("chirp_repetition_interval_s must be finite and greater than zero")
    if repetition_interval < params.chirp_duration_s:
        raise ValueError("chirp_repetition_interval_s cannot be shorter than chirp_duration_s")

    fast_weights = _window(sample_count, range_window)
    slow_weights = _window(chirp_count, doppler_window)
    weighted = beat * slow_weights[:, None] * fast_weights[None, :]

    range_fft = np.fft.fft(weighted, n=range_fft_length, axis=1)
    range_frequency = np.fft.fftfreq(range_fft_length, d=1.0 / params.sample_rate_hz)
    positive_range = range_frequency >= 0.0
    range_fft = range_fft[:, positive_range]
    range_frequency = range_frequency[positive_range]

    doppler_fft = np.fft.fftshift(
        np.fft.fft(range_fft, n=doppler_fft_length, axis=0),
        axes=0,
    )
    slow_frequency = np.fft.fftshift(
        np.fft.fftfreq(doppler_fft_length, d=repetition_interval)
    )

    ranges = beat_frequency_to_range_m(range_frequency, params.chirp_slope_hz_per_s)
    velocities = -slow_frequency * params.wavelength_m / 2.0
    power = np.abs(doppler_fft) ** 2

    return RangeDopplerMap(
        np.asarray(ranges, dtype=np.float64),
        velocities.astype(np.float64, copy=False),
        power.astype(np.float64, copy=False),
    )


def estimate_peak_range_velocity(
    result: RangeDopplerMap,
    *,
    ignore_dc_range: bool = True,
) -> tuple[float, float]:
    """Return the range and velocity coordinates of the strongest map cell."""

    if result.power.ndim != 2 or result.power.size == 0:
        raise ValueError("power must be a non-empty two-dimensional array")
    if result.power.shape != (result.velocity_m_s.size, result.range_m.size):
        raise ValueError("power shape must match velocity and range axes")

    range_start = 1 if ignore_dc_range else 0
    if result.range_m.size <= range_start:
        raise ValueError("range axis does not contain a searchable bin")

    searchable = result.power[:, range_start:]
    velocity_index, relative_range_index = np.unravel_index(
        int(np.argmax(searchable)), searchable.shape
    )
    range_index = range_start + relative_range_index
    return float(result.range_m[range_index]), float(result.velocity_m_s[velocity_index])
