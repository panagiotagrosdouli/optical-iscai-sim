import numpy as np
import pytest

from optical_iscai.frame import FrameConfiguration, simulate_frame
from optical_iscai.propagation import PointTarget
from optical_iscai.range_doppler import estimate_peak_range_velocity
from optical_iscai.waveform import PCFMCWParameters


def compact_parameters() -> PCFMCWParameters:
    return PCFMCWParameters(
        carrier_frequency_hz=10e9,
        bandwidth_hz=20e6,
        chirp_duration_s=100e-6,
        data_rate_bps=10e3,
        sample_rate_hz=40e6,
    )


def test_frame_shapes_and_ground_truth() -> None:
    params = compact_parameters()
    config = FrameConfiguration(
        chirp_count=8,
        chirp_repetition_interval_s=120e-6,
        n_range_fft=4096,
        n_doppler_fft=8,
    )
    target = PointTarget(range_m=25.0, radial_velocity_m_s=2.0, amplitude=0.8)

    result = simulate_frame(params, [target], config)

    expected_shape = (config.chirp_count, params.samples_per_chirp)
    assert result.time_s.shape == expected_shape
    assert result.transmitted.shape == expected_shape
    assert result.received.shape == expected_shape
    assert result.beat.shape == expected_shape
    assert result.target_ranges_m.shape == (config.chirp_count, 1)
    assert result.target_velocities_m_s.shape == (config.chirp_count, 1)
    assert np.all(result.target_velocities_m_s == 2.0)
    assert np.all(np.diff(result.target_ranges_m[:, 0]) < 0.0)


def test_stationary_target_has_constant_range_ground_truth() -> None:
    params = compact_parameters()
    result = simulate_frame(
        params,
        [PointTarget(range_m=40.0)],
        FrameConfiguration(chirp_count=4),
    )

    assert np.allclose(result.target_ranges_m[:, 0], 40.0)
    assert np.allclose(result.target_velocities_m_s[:, 0], 0.0)


def test_frame_range_doppler_peak_is_near_target() -> None:
    params = compact_parameters()
    chirp_count = 64
    repetition_interval = params.chirp_duration_s
    velocity_bin = params.wavelength_m / (
        2.0 * chirp_count * repetition_interval
    )
    target = PointTarget(
        range_m=30.0,
        radial_velocity_m_s=velocity_bin,
        amplitude=1.0,
    )
    config = FrameConfiguration(
        chirp_count=chirp_count,
        n_range_fft=8192,
        n_doppler_fft=chirp_count,
        range_window="hann",
        doppler_window=None,
    )

    result = simulate_frame(params, [target], config)
    estimated_range, estimated_velocity = estimate_peak_range_velocity(
        result.range_doppler
    )

    range_bin = (
        299_792_458.0
        * params.sample_rate_hz
        / (2.0 * params.chirp_slope_hz_per_s * config.n_range_fft)
    )
    assert estimated_range == pytest.approx(target.range_m, abs=2.0 * range_bin)
    assert estimated_velocity == pytest.approx(target.radial_velocity_m_s, abs=0.6 * velocity_bin)


def test_frame_rejects_invalid_configuration_and_targets() -> None:
    params = compact_parameters()

    with pytest.raises(ValueError, match="at least two"):
        simulate_frame(
            params,
            [PointTarget(range_m=10.0)],
            FrameConfiguration(chirp_count=1),
        )

    with pytest.raises(ValueError, match="at least one"):
        simulate_frame(params, [], FrameConfiguration(chirp_count=2))

    with pytest.raises(ValueError, match="not shorter"):
        simulate_frame(
            params,
            [PointTarget(range_m=10.0)],
            FrameConfiguration(
                chirp_count=2,
                chirp_repetition_interval_s=params.chirp_duration_s / 2.0,
            ),
        )


def test_frame_rejects_target_crossing_zero_range() -> None:
    params = compact_parameters()
    with pytest.raises(ValueError, match="crosses zero range"):
        simulate_frame(
            params,
            [PointTarget(range_m=0.001, radial_velocity_m_s=100.0)],
            FrameConfiguration(chirp_count=3, chirp_repetition_interval_s=100e-6),
        )
