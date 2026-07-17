import numpy as np
import pytest

from optical_iscai.range_doppler import estimate_peak_range_velocity, range_doppler_map
from optical_iscai.waveform import PCFMCWParameters, SPEED_OF_LIGHT_M_S


def make_params():
    return PCFMCWParameters(
        carrier_frequency_hz=193.4e12,
        bandwidth_hz=20e6,
        chirp_duration_s=100e-6,
        data_rate_bps=10e3,
        sample_rate_hz=2e6,
    )


def test_peak_matches_fft_bin():
    params = make_params()
    chirps = 32
    samples = params.samples_per_chirp
    tri = params.chirp_duration_s
    range_bin = 12
    doppler_bin = 5

    beat_hz = range_bin * params.sample_rate_hz / samples
    slow_hz = doppler_bin / (chirps * tri)
    fast_t = np.arange(samples) / params.sample_rate_hz
    slow_t = np.arange(chirps) * tri
    matrix = np.exp(1j * 2 * np.pi * slow_hz * slow_t)[:, None]
    matrix = matrix * np.exp(1j * 2 * np.pi * beat_hz * fast_t)[None, :]

    result = range_doppler_map(
        matrix,
        params,
        chirp_repetition_interval_s=tri,
        range_window=None,
        doppler_window=None,
    )
    found_range, found_velocity = estimate_peak_range_velocity(result)
    expected_range = SPEED_OF_LIGHT_M_S * beat_hz / (2 * params.chirp_slope_hz_per_s)
    expected_velocity = -slow_hz * params.wavelength_m / 2
    assert found_range == pytest.approx(expected_range)
    assert found_velocity == pytest.approx(expected_velocity)


def test_zero_padding_shape():
    params = make_params()
    matrix = np.ones((8, params.samples_per_chirp), dtype=complex)
    result = range_doppler_map(matrix, params, n_range_fft=512, n_doppler_fft=16)
    assert result.power.shape == (16, 256)
    assert result.range_m.shape == (256,)
    assert result.velocity_m_s.shape == (16,)


def test_invalid_inputs_are_rejected():
    params = make_params()
    with pytest.raises(ValueError):
        range_doppler_map(np.ones(10), params)

    matrix = np.ones((8, params.samples_per_chirp), dtype=complex)
    with pytest.raises(ValueError):
        range_doppler_map(matrix, params, n_range_fft=64)
    with pytest.raises(ValueError):
        range_doppler_map(matrix, params, n_doppler_fft=4)
    with pytest.raises(ValueError):
        range_doppler_map(
            matrix,
            params,
            chirp_repetition_interval_s=params.chirp_duration_s / 2,
        )
