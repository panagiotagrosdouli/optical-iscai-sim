import numpy as np
import pytest

from optical_iscai.waveform import (
    PCFMCWParameters,
    differential_encode,
    generate_chirp,
)


def test_paper_baseline_derived_parameters() -> None:
    params = PCFMCWParameters()

    assert params.chirp_slope_hz_per_s == pytest.approx(1.0e15)
    assert params.wavelength_m == pytest.approx(1.550116122e-6, rel=1e-9)
    assert params.ideal_range_resolution_m == pytest.approx(0.0149896229)
    assert params.symbols_per_chirp == 10_000
    assert params.samples_per_chirp == 200_000


def test_differential_encoding() -> None:
    bits = np.array([0, 1, 0, 1, 1], dtype=np.int8)
    phases = differential_encode(bits)

    np.testing.assert_allclose(phases, [0.0, np.pi, np.pi, 0.0, np.pi])


def test_generate_unmodulated_chirp_has_constant_amplitude() -> None:
    params = PCFMCWParameters(
        bandwidth_hz=1e6,
        chirp_duration_s=10e-6,
        data_rate_bps=1e6,
        sample_rate_hz=2e6,
        amplitude=2.5,
    )
    time_s, signal = generate_chirp(params)

    assert len(time_s) == params.samples_per_chirp
    assert signal.shape == time_s.shape
    np.testing.assert_allclose(np.abs(signal), params.amplitude, atol=1e-12)
    assert time_s[0] == 0.0
    assert time_s[-1] < params.chirp_duration_s


def test_generate_chirp_rejects_wrong_bit_count() -> None:
    params = PCFMCWParameters(
        bandwidth_hz=1e6,
        chirp_duration_s=10e-6,
        data_rate_bps=1e6,
        sample_rate_hz=2e6,
    )

    with pytest.raises(ValueError, match="bits must have shape"):
        generate_chirp(params, np.zeros(2, dtype=np.int8))


def test_invalid_sample_rate_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least twice"):
        PCFMCWParameters(bandwidth_hz=10e9, sample_rate_hz=10e9)
