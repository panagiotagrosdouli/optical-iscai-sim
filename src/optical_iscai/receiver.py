"""Coherent dechirp and one-dimensional FMCW range processing.

The reference paper mixes the received PC-FMCW echo with a local oscillator,
low-pass filters the result, and then applies FFT processing.  This module
implements the first reproducible receiver baseline for a single chirp.

For the project sign convention, the dechirped signal is

    y(t) = s_tx(t) * conj(s_rx(t))

so a stationary delayed up-chirp produces a positive beat frequency.  Doppler
couples into that frequency; therefore ``beat_frequency_to_range`` is an
uncorrected range estimate unless Doppler is known or estimated separately.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from optical_iscai.waveform import PCFMCWParameters, SPEED_OF_LIGHT_M_S


@dataclass(frozen=True, slots=True)
class RangeSpectrum:
    """Positive-frequency range spectrum for one dechirped chirp."""

    frequency_hz: NDArray[np.float64]
    range_m: NDArray[np.float64]
    power: NDArray[np.float64]


def dechirp(
    transmitted: NDArray[np.complexfloating],
    received: NDArray[np.complexfloating],
) -> NDArray[np.complex128]:
    """Mix a received echo with the conjugated optical-envelope reference.

    The chosen product ``transmitted * conj(received)`` places the stationary
    target beat of an up-chirp at positive frequency under this repository's
    delay convention.
    """

    tx = np.asarray(transmitted, dtype=np.complex128)
    rx = np.asarray(received, dtype=np.complex128)
    if tx.ndim != 1 or rx.ndim != 1 or tx.shape != rx.shape:
        raise ValueError("transmitted and received must be one-dimensional arrays of equal shape")
    if tx.size < 2:
        raise ValueError("at least two samples are required")
    if not np.all(np.isfinite(tx)) or not np.all(np.isfinite(rx)):
        raise ValueError("transmitted and received must contain finite values")
    return tx * np.conjugate(rx)


def beat_frequency_to_range_m(
    beat_frequency_hz: float | NDArray[np.floating],
    chirp_slope_hz_per_s: float,
) -> float | NDArray[np.float64]:
    """Convert beat frequency to uncorrected monostatic range.

    Uses ``R = c f_b / (2 mu)``.  The equation assumes a stationary target or
    a beat frequency from which Doppler has already been removed.
    """

    if not np.isfinite(chirp_slope_hz_per_s) or chirp_slope_hz_per_s <= 0:
        raise ValueError("chirp_slope_hz_per_s must be finite and greater than zero")

    frequency = np.asarray(beat_frequency_hz, dtype=np.float64)
    if not np.all(np.isfinite(frequency)) or np.any(frequency < 0):
        raise ValueError("beat_frequency_hz must contain finite non-negative values")

    result = SPEED_OF_LIGHT_M_S * frequency / (2.0 * chirp_slope_hz_per_s)
    if result.ndim == 0:
        return float(result)
    return result.astype(np.float64, copy=False)


def range_spectrum(
    beat_signal: NDArray[np.complexfloating],
    params: PCFMCWParameters,
    *,
    n_fft: int | None = None,
    window: str | None = "hann",
) -> RangeSpectrum:
    """Compute a positive-frequency power spectrum and its range axis.

    Parameters
    ----------
    beat_signal:
        Complex dechirped samples for one chirp.
    params:
        Waveform parameters defining sampling rate and chirp slope.
    n_fft:
        FFT length. Defaults to the number of beat samples. Zero-padding is
        allowed, but truncating the input is rejected.
    window:
        ``"hann"`` or ``None``.
    """

    beat = np.asarray(beat_signal, dtype=np.complex128)
    if beat.ndim != 1 or beat.size < 2:
        raise ValueError("beat_signal must be a one-dimensional array with at least two samples")
    if not np.all(np.isfinite(beat)):
        raise ValueError("beat_signal must contain finite values")

    fft_length = beat.size if n_fft is None else int(n_fft)
    if fft_length < beat.size:
        raise ValueError("n_fft must be greater than or equal to the number of samples")

    if window is None:
        weights = np.ones(beat.size, dtype=np.float64)
    elif window == "hann":
        weights = np.hanning(beat.size)
    else:
        raise ValueError("window must be 'hann' or None")

    spectrum = np.fft.fft(beat * weights, n=fft_length)
    frequency = np.fft.fftfreq(fft_length, d=1.0 / params.sample_rate_hz)
    positive = frequency >= 0.0
    frequency_positive = frequency[positive].astype(np.float64, copy=False)
    power = np.abs(spectrum[positive]) ** 2
    ranges = beat_frequency_to_range_m(
        frequency_positive,
        params.chirp_slope_hz_per_s,
    )
    return RangeSpectrum(frequency_positive, ranges, power.astype(np.float64, copy=False))


def estimate_peak_range_m(spectrum: RangeSpectrum, *, ignore_dc: bool = True) -> float:
    """Return the range bin with maximum spectral power."""

    if spectrum.power.ndim != 1 or spectrum.power.size == 0:
        raise ValueError("spectrum power must be a non-empty one-dimensional array")
    if spectrum.frequency_hz.shape != spectrum.power.shape or spectrum.range_m.shape != spectrum.power.shape:
        raise ValueError("frequency, range, and power arrays must have equal shape")

    start = 1 if ignore_dc else 0
    if spectrum.power.size <= start:
        raise ValueError("spectrum does not contain a searchable bin")
    peak_index = start + int(np.argmax(spectrum.power[start:]))
    return float(spectrum.range_m[peak_index])
