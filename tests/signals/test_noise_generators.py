import numpy as np
import pandas as pd
import pytest

from indsl.signals.generator import sine_wave
from indsl.signals.noise import RedNoise, Sinusoidal, TimeSampler, TimeSeries, white_noise
from indsl.exceptions import UserValueError


@pytest.mark.core
def test_white_noise():
    wave = sine_wave(
        start_date=pd.Timestamp("1975-05-09"),
        end_date=pd.Timestamp("1975-05-10"),
        wave_amplitude=2,
        wave_mean=10,
        wave_period=pd.Timedelta("0.25 D"),
        wave_phase=np.pi,
    )
    wave_plus_noise = white_noise(wave)
    assert np.round(wave_plus_noise.mean()) == 10


# Parameterized test for various snr_db values
# Anything above 3083 causes overflow.
@pytest.mark.core
def test_white_noise_snr_values():
    snr_db = 3083
    wave = sine_wave(
        start_date=pd.Timestamp("1975-05-09"),
        end_date=pd.Timestamp("1975-05-10"),
        wave_amplitude=2,
        wave_mean=10,
        wave_period=pd.Timedelta("0.25 D"),
        wave_phase=np.pi,
    )
    with pytest.raises(UserValueError) as excinfo:
        white_noise(wave, snr_db=snr_db)

    expected = f"snr_db value of {snr_db} is too large and causes overflow."
    assert expected in str(excinfo.value)


def test_sample_irregular_time():
    """
    Test that the algorithm can return fewer points by using percentage parameter
    """
    duration = 1000
    resolution = 1
    percentage = 80
    time_sampler = TimeSampler(stop_time=duration)
    time_vector = time_sampler.sample_irregular_time(resolution=resolution, keep_percentage=percentage)

    assert len(time_vector) == 800


@pytest.mark.core
def test_sample_regular_time():
    duration = 1000
    num_points = 100
    time_sampler = TimeSampler(stop_time=duration)
    time_vector = time_sampler.sample_regular_time(num_points=num_points)

    assert len(time_vector) == num_points
    assert time_vector[0] == 0
    assert time_vector[-1] == duration


@pytest.mark.core
def test_sample_irregular_time_num_points():
    duration = 1000
    num_points = 100
    time_sampler = TimeSampler(stop_time=duration)
    time_vector = time_sampler.sample_irregular_time(num_points=num_points)

    assert len(time_vector) == num_points


@pytest.mark.core
def test_sinusoidal():
    frequency = 1
    amplitude = 2
    time_vector = np.arange(0, 10, 0.01)
    sinusoidal = Sinusoidal(amplitude=amplitude, frequency=frequency)
    signal = sinusoidal.sample_vectorized(time_vector)

    assert signal.max() == amplitude
    assert signal.shape == time_vector.shape


@pytest.mark.core
def test_time_series():
    frequency = 1
    amplitude = 2
    sinusoidal = Sinusoidal(amplitude=amplitude, frequency=frequency)
    red_noise = RedNoise()
    time_series = TimeSeries(signal_generator=sinusoidal, noise_generator=red_noise)
    time_vector = np.arange(0, 10, 0.01)
    samples, signals, errors = time_series.sample(time_vector)

    assert signals.max() == amplitude
    assert errors.std() > 0  # Check that there is noise
