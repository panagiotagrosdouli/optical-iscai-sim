import numpy as np
import pytest

from optical_iscai.propagation import PointTarget, generate_echo
from optical_iscai.receiver import (
    beat_frequency_to_range_m,
    dechirp,
    estimate_peak_range_m,
    range_spectrum,
)
from optical_iscai.waveform import PCFMCWParameters, SPEED_OF_LIGHT_M_S, generate_chirp


def compact_params() -> PCFMCWParameters:
    return PCFMCWParameters(
        carrier_frequency_hz=193.4e12,
        bandwidth_hz=2.0e6,
        chirp_duration_s=1.0e-3,
        data_rate_bps=1.0e3,
        sample_rate_hz=4.0e6,
    )


def test_dechirp_is_elementwise_tx_times_conjugate_rx() -> None:
    tx = np.array([1 + 1j, 2 - 1j], dtype=np.complex128)
    rx = np.array([0.5 - 1j, -1 + 2j], dtype=np.complex128)
    assert np.allclose(dechirp(tx, rx), tx * np.conjugate(rx))


def test_beat_frequency_to_range_round_trip() -> None:
    slope = 2.0e9
    expected_range = 75.0
    beat_hz = 2.0 * slope * expected_range / SPEED_OF_LIGHT_M_S
    assert beat_frequency_to_range_m(beat_hz, slope) == pytest.approx(expected_range)


def test_stationary_target_peak_is_at_expected_range_bin() -> None:
    params = compact_params()
    time_s, transmitted = generate_chirp(params)

    # Choose a target whose beat falls exactly on an FFT bin (1 kHz spacing).
    beat_hz = 1_000.0
    target_range_m = beat_frequency_to_range_m(beat_hz, params.chirp_slope_hz_per_s)
    target = PointTarget(range_m=target_range_m, amplitude=1.0)
    received = generate_echo(time_s, transmitted, target, params.wavelength_m)

    beat = dechirp(transmitted, received)
    spectrum = range_spectrum(beat, params, window="hann")
    estimate = estimate_peak_range_m(spectrum)

    assert estimate == pytest.approx(target_range_m, abs=params.ideal_range_resolution_m)


def test_range_spectrum_rejects_fft_shorter_than_signal() -> None:
    params = compact_params()
    beat = np.ones(params.samples_per_chirp, dtype=np.complex128)
    with pytest.raises(ValueError, match="n_fft"):
        range_spectrum(beat, params, n_fft=beat.size - 1)


def test_range_spectrum_rejects_unknown_window() -> None:
    params = compact_params()
    beat = np.ones(params.samples_per_chirp, dtype=np.complex128)
    with pytest.raises(ValueError, match="window"):
        range_spectrum(beat, params, window="blackman")


def test_dechirp_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="equal shape"):
        dechirp(np.ones(4, dtype=complex), np.ones(3, dtype=complex))
