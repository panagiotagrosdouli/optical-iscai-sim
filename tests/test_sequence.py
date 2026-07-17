import numpy as np
import pytest

from optical_iscai.cfar import CFARConfiguration
from optical_iscai.frame import FrameConfiguration
from optical_iscai.pipeline import PipelineConfiguration
from optical_iscai.propagation import PointTarget
from optical_iscai.sequence import SequenceConfiguration, simulate_sequence
from optical_iscai.tracking import TrackerConfiguration
from optical_iscai.waveform import PCFMCWParameters


def make_params():
    return PCFMCWParameters(
        carrier_frequency_hz=193.4e12,
        bandwidth_hz=1e6,
        chirp_duration_s=20e-6,
        data_rate_bps=50e3,
        sample_rate_hz=2e6,
    )


def make_frame_config(seed=None):
    return FrameConfiguration(
        chirp_count=8,
        n_range_fft=64,
        n_doppler_fft=16,
        range_window=None,
        doppler_window=None,
        receiver_snr_db=30.0 if seed is not None else None,
        random_seed=seed,
    )


def make_pipeline_config():
    return PipelineConfiguration(
        cfar=CFARConfiguration(
            training_velocity=1,
            training_range=2,
            guard_velocity=0,
            guard_range=0,
            probability_false_alarm=1e-3,
        ),
        tracking=TrackerConfiguration(confirmation_hits=1),
    )


def test_sequence_advances_target_ground_truth():
    params = make_params()
    result = simulate_sequence(
        params,
        [PointTarget(range_m=40.0, radial_velocity_m_s=5.0)],
        make_frame_config(),
        make_pipeline_config(),
        SequenceConfiguration(frame_count=3, frame_interval_s=0.01),
    )

    assert result.frame_times_s == pytest.approx([0.0, 0.01, 0.02])
    ranges = [frame.target_ranges_m[0] for frame in result.frames]
    assert ranges == pytest.approx([40.0, 39.95, 39.90])
    assert all(frame.target_velocities_m_s[0] == pytest.approx(5.0) for frame in result.frames)
    assert [frame.frame_index for frame in result.frames] == [0, 1, 2]


def test_noisy_sequence_is_reproducible():
    params = make_params()
    kwargs = dict(
        params=params,
        targets=[PointTarget(range_m=30.0, radial_velocity_m_s=0.0)],
        frame_config=make_frame_config(seed=17),
        pipeline_config=make_pipeline_config(),
        sequence_config=SequenceConfiguration(frame_count=2, frame_interval_s=0.01),
    )
    first = simulate_sequence(**kwargs)
    second = simulate_sequence(**kwargs)

    for first_frame, second_frame in zip(first.frames, second.frames, strict=True):
        assert np.array_equal(first_frame.frame.received, second_frame.frame.received)
        assert np.array_equal(
            first_frame.frame.range_doppler.power,
            second_frame.frame.range_doppler.power,
        )

    assert not np.array_equal(
        first.frames[0].frame.received,
        first.frames[1].frame.received,
    )


def test_invalid_sequence_configuration_is_rejected():
    params = make_params()
    frame_config = make_frame_config()

    with pytest.raises(ValueError):
        SequenceConfiguration(frame_count=0).validate(params, frame_config)
    with pytest.raises(ValueError):
        SequenceConfiguration(frame_interval_s=0.0).validate(params, frame_config)
    with pytest.raises(ValueError):
        SequenceConfiguration(frame_interval_s=1e-6).validate(params, frame_config)


def test_target_cannot_cross_zero_range():
    params = make_params()
    with pytest.raises(ValueError, match="crosses zero range"):
        simulate_sequence(
            params,
            [PointTarget(range_m=0.01, radial_velocity_m_s=10.0)],
            make_frame_config(),
            make_pipeline_config(),
            SequenceConfiguration(frame_count=2, frame_interval_s=0.01),
        )


def test_pipeline_and_pipeline_config_are_mutually_exclusive():
    from optical_iscai.pipeline import SensingPipeline

    params = make_params()
    config = make_pipeline_config()
    with pytest.raises(ValueError):
        simulate_sequence(
            params,
            [PointTarget(range_m=20.0)],
            make_frame_config(),
            config,
            SequenceConfiguration(frame_count=1, frame_interval_s=0.01),
            pipeline=SensingPipeline(config),
        )
