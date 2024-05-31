# Copyright 2021 Cognite AS
import math

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from numpy.random import default_rng

from indsl.exceptions import UserValueError
from indsl.smooth.butterworth import butterworth
from indsl.smooth.chebyshev import chebyshev
from indsl.smooth.savitzky_golay import sg


def generate_wave_with_uniform_sampling(
    waves: int = 3,
    max_amplitude: float = 50.0,
    max_frequency: float = 0.002,
    signal_duration_hours: float = 1.0,
    signal_frequency_hz: float = 0.1,
    seed: int = 12345,
):
    """Generate a pandas series with a timestamp and wave signal with one or
    multiple frequency and amplitude components using sine waves: sin(2*pi*f*t + phi).
    """
    rg = default_rng(seed)
    dt = np.arange(0, signal_duration_hours * 3600, 1 / signal_frequency_hz)
    amplitude = max_amplitude * rg.random(waves)
    phase = np.pi * rg.random(waves)
    freq = max_frequency * rg.random(waves)
    time_array = datetime.now() - timedelta(seconds=dt[-1]) + dt * timedelta(seconds=1)

    wave = 0
    for ind in range(waves):
        wave = wave + amplitude[ind] * np.sin(2 * np.pi * freq[ind] * dt + phase[ind])
    return pd.Series(wave, index=time_array)


@pytest.mark.core
def test_sg_smoother_no_noise():
    """The smoothed data should be almost the same as the original data give
    that it has no noise, and the filter is using a small window and polyorder
    1."""
    wavy_data = generate_wave_with_uniform_sampling()
    smooth_wave = sg(wavy_data, window_length=5)
    assert math.isclose(wavy_data.mean() - smooth_wave.mean(), 0, abs_tol=0.005)
    with pytest.raises(UserValueError, match="The window length must be a positive odd integer."):
        _ = sg(wavy_data, window_length=-50, polyorder=3)
    with pytest.raises(
        UserValueError,
        match="The window length must be less than or equal to the number of data points in your time series.",
    ):
        _ = sg(wavy_data, window_length=len(wavy_data) + 20)
    with pytest.raises(UserValueError, match="The polynomial order must be less than the window length."):
        _ = sg(wavy_data, window_length=5, polyorder=6)

    with pytest.raises(UserValueError):
        sg(wavy_data, window_length=0)
    with pytest.raises(UserValueError):
        sg(wavy_data, window_length=-2)
    with pytest.raises(UserValueError):
        sg(wavy_data, window_length=len(wavy_data) + 1)
    with pytest.raises(UserValueError):
        win_len = 5
        sg(wavy_data, window_length=win_len, polyorder=win_len + 1)


@pytest.mark.parametrize("filter_function", [butterworth, chebyshev])
def test_smoothers_lowpass(filter_function):
    """Signal is comprised of 2 sine waves at different frequencies; use a low
    pass filterto keep only the lowest frequency.

    Test for sine wave of a certain period by checking if values are
    repeating.
    """

    # Inputs
    low_freq = 25
    high_freq = 60
    num = int(1e6)
    t = np.linspace(0, 1, num)  # discretising 1000 times in 1 sec

    # Create signals
    sig_low = np.sin(2 * np.pi * low_freq * t)
    sig_high = np.sin(2 * np.pi * high_freq * t)
    sig = sig_low + sig_high
    input_df = pd.Series(sig, index=t)

    # Perfrom a low pass filter to retrieve lowest frequency
    Wn = 30 * 2 / num
    filtered = filter_function(input_df, Wn=Wn)

    # Grab the last 2 cycles and check if its in line with expected frequency (should have full periods)
    lookback = int(2 * num / low_freq)
    last_cycles = filtered.iloc[-lookback - 1 :].tolist()

    # Check that the expected frequency is correct by asserting when we expect cycles to repeat
    np.testing.assert_almost_equal(last_cycles[0], last_cycles[-1], decimal=2)
    assert max(last_cycles) > 0.95
    assert min(last_cycles) < -0.95


@pytest.mark.parametrize("filter_function", [butterworth, chebyshev])
def test_smoothers_highpass(filter_function):
    """Signal is comprised of 2 sine waves at different frequencies; use a high
    pass filter to keep only the highest frequency.

    Test for sine wave of a certain period by checking if values are
    repeating.
    """

    # Inputs
    low_freq = 25
    high_freq = 60
    num = int(1e6)
    t = np.linspace(0, 1, num)  # discretising 1000 times in 1 sec

    # Create signals
    sig_low = np.sin(2 * np.pi * low_freq * t)
    sig_high = np.sin(2 * np.pi * high_freq * t)
    sig = sig_low + sig_high
    input_df = pd.Series(sig, index=t)

    # Perform a low pass filter to retrieve lowest frequency
    Wn = 30 * 2 / num
    filtered = filter_function(input_df, Wn=Wn, btype="highpass")

    # Grab the last 2 cycles and check if its in line with expected frequency (should have full periods)
    lookback = int(2 * num / high_freq)
    last_cycles = filtered.iloc[-lookback - 1 :].tolist()

    # Check that the expected frequency is correct by asserting when we expect cycles to repeat
    np.testing.assert_almost_equal(last_cycles[0], last_cycles[-1], decimal=2)
    assert max(last_cycles) > 0.95
    assert min(last_cycles) < -0.95


@pytest.mark.parametrize("length", range(3))
@pytest.mark.parametrize("filter_function", [butterworth, chebyshev])
def test_smoothers_little_data(filter_function, length):
    data = pd.Series(np.random.random(length), index=pd.date_range(start=0, freq="1s", periods=length))
    filter_function(data)


@pytest.mark.parametrize("filter_function", [butterworth, chebyshev])
def test_smoothers_nan_inf_data(filter_function):
    data = pd.Series([np.inf, np.nan], index=pd.date_range(start=0, freq="1s", periods=2))
    filter_function(data)


@pytest.mark.core
def test_butterworth_validation():
    data = pd.Series(np.random.randn(10))
    with pytest.raises(UserValueError):
        butterworth(data, Wn=10)
