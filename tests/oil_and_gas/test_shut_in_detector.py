# Copyright 2021 Cognite AS
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from pandas.testing import assert_series_equal

from indsl.exceptions import UserValueError
from indsl.oil_and_gas.shut_in_detector import calculate_shutin_interval, calculate_threshold


@pytest.mark.core
def test_shutin_calc_one_shutin_before():
    # test shut-in calculations when there is shut-in that is coming from the past
    # Define input ts
    date_range = pd.date_range(start=datetime(2000, 10, 1), end=datetime(2000, 10, 10), periods=6)
    valve_ts = pd.Series([1, 1, 1, 1, 5, 5], index=date_range)

    # Expected res
    exp_res = pd.Series([0, 0, 0, 0, 1], index=date_range[1:])

    # Check result
    res = calculate_shutin_interval(valve_ts)
    assert_series_equal(res, exp_res)


@pytest.mark.core
def test_shutin_calc_one_shutin_after():
    # test shut-in calculations when there is shut-in that is going to the future
    # Define input ts
    date_range = pd.date_range(start=datetime(2000, 10, 1), end=datetime(2000, 10, 10), periods=6)
    valve_ts = pd.Series([5, 5, 1, 1, 1, 1], index=date_range)

    # Expected res
    exp_res = pd.Series([1, 0, 0, 0, 0], index=date_range[1:])

    # Check result
    res = calculate_shutin_interval(valve_ts)
    assert_series_equal(res, exp_res)


@pytest.mark.core
def test_shutin_calc_one_shutin():
    # test shut-in calculations when there is full shut-in in the considered period
    # Define input ts
    date_range = pd.date_range(start=datetime(2000, 10, 1), end=datetime(2000, 10, 10), periods=6)
    valve_ts = pd.Series([5, 5, 1, 1, 5, 5], index=date_range)

    # Expected res
    exp_res = pd.Series([1, 0, 0, 0, 1], index=date_range[1:])

    # Check result
    res = calculate_shutin_interval(valve_ts)
    assert_series_equal(res, exp_res)


@pytest.mark.core
def test_threshold_calculation_error():
    # Define input ts
    date_range = pd.date_range(start=datetime(2000, 10, 1), end=datetime(2000, 10, 10), periods=6)
    valve_ts = pd.Series([1, 1, 1, 1, 1, 1], index=date_range)

    # Check if exception is raised
    with pytest.raises(UserValueError, match="Not enough data to detect the threshold"):
        _ = calculate_threshold(valve_ts)


@pytest.mark.core
def test_shutin_below_threshold_false():
    # test shut-in calculations when shutin_state_below_threshold is set to False
    # Define input ts
    date_range = pd.date_range(start=datetime(2000, 10, 1), end=datetime(2000, 10, 10), periods=6)
    valve_ts = pd.Series([5, 5, 1, 1, 5, 5], index=date_range)

    # Expected res
    exp_res = pd.Series([0, 0, 1, 0, 0], index=date_range[1:])

    # Check result
    res = calculate_shutin_interval(valve_ts, shutin_state_below_threshold=False)
    assert_series_equal(res, exp_res)


@pytest.mark.parametrize(
    "valve_data, shutin_threshold, expected_output",
    [
        ([1, 1, 1, 1, 1, 1], 0.5, [1, 1, 1, 1, 1, 1]),  # All 1's
        ([0, 0, 0, 0, 0, 0], 0.5, [0, 0, 0, 0, 0, 0]),  # All 0's
    ],
)
def test_calculate_shutin_no_state_change(valve_data, shutin_threshold, expected_output):
    """Test if the function correctly handles cases where the valve state doesn't change."""

    # Define common date range
    date_range = pd.date_range(start="2021-01-01", periods=6, freq="H")

    # Cases where all valve states are 1 and 0
    valve_ts = pd.Series(valve_data, index=date_range)

    result = calculate_shutin_interval(valve_ts, shutin_threshold=shutin_threshold).astype(np.int64)

    expected = pd.Series(expected_output, index=date_range)

    pd.testing.assert_series_equal(result, expected)  # Check if the result is as expected
