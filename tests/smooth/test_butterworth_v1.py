# Copyright 2021 Cognite AS
import numpy as np
import pandas as pd
import pytest

from indsl.exceptions import UserValueError
from indsl.smooth.butterworth_v1 import butterworth


@pytest.mark.core
def test_butterworth_lowpass():
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
    filtered = butterworth(input_df, Wn=Wn)

    # Grab the last 2 cycles and check if its in line with expected frequency (should have full periods)
    lookback = int(2 * num / low_freq)
    last_cycles = filtered.iloc[-lookback - 1 :].tolist()

    # Check that the expected frequency is correct by asserting when we expect cycles to repeat
    np.testing.assert_almost_equal(last_cycles[0], last_cycles[-1], decimal=2)
    assert max(last_cycles) > 0.95
    assert min(last_cycles) < -0.95


@pytest.mark.core
def test_butterworth_highpass():
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
    filtered = butterworth(input_df, Wn=Wn, btype="highpass")

    # Grab the last 2 cycles and check if its in line with expected frequency (should have full periods)
    lookback = int(2 * num / high_freq)
    last_cycles = filtered.iloc[-lookback - 1 :].tolist()

    # Check that the expected frequency is correct by asserting when we expect cycles to repeat
    np.testing.assert_almost_equal(last_cycles[0], last_cycles[-1], decimal=2)
    assert max(last_cycles) > 0.95
    assert min(last_cycles) < -0.95

    with pytest.raises(UserValueError) as excinfo:
        _ = butterworth(input_df, Wn=Wn, output="Gustavo", btype="highpass")
    assert "output argument is not 'sos', 'ba' or 'zpk'." == str(excinfo.value)


@pytest.mark.parametrize("output_str", ["ba", "zpk"])
@pytest.mark.core
def test_butterworth_ba_zpk(output_str):
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

    # High pass filter to remove all wavy data (all the data)
    filtered = butterworth(input_df, Wn=0.50, output=output_str, btype="highpass")

    assert np.round(filtered.mean()) == 0
