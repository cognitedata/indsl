import pandas as pd
import pytest

from pandas.testing import assert_series_equal

from indsl.exceptions import UserValueError
from indsl.oil_and_gas.well_prod_status import calculate_well_prod_status


@pytest.mark.core
def test_all_valve_combos():
    # first datapoint is when all valves are open, second is when all
    # valves are closed and last is when only choke is open

    # Inputs
    master = pd.Series([100, 0, 0])
    wing = pd.Series([100, 0, 0])
    choke = pd.Series([100, 0, 100])
    threshold_master = 1
    threshold_wing = 1
    threshold_choke = 5

    res = calculate_well_prod_status(master, wing, choke, threshold_master, threshold_wing, threshold_choke)
    exp_res = pd.Series([1, 0, 0]).astype(int)

    assert_series_equal(res.astype(int), exp_res)


@pytest.mark.core
def test_high_choke_threshold():
    # test the condition when the choke threshold is very high
    # first test is when choke opening is below threshold

    # Inputs
    master = pd.Series([100])
    wing = pd.Series([100])
    choke = pd.Series([20])
    threshold_master = 1
    threshold_wing = 1
    threshold_choke = 50

    res = calculate_well_prod_status(master, wing, choke, threshold_master, threshold_wing, threshold_choke)
    exp_res = pd.Series([0], dtype=int)

    assert_series_equal(res, exp_res)

    # check choke opening above crazy threshold
    choke = pd.Series([100])

    res = calculate_well_prod_status(master, wing, choke, threshold_master, threshold_wing, threshold_choke)
    exp_res = pd.Series([1], dtype=int)

    assert_series_equal(res, exp_res)


@pytest.mark.core
def test_zero2one_valve_range():
    # test the condition when the choke threshold is very high
    # first test is when choke opening is below threshold

    # Inputs
    master = pd.Series([100, 100, 100, 100, 100])
    wing = pd.Series([0.1, 0.2, 0.3, 0.005, 0.1])
    choke = pd.Series([20, 20, 20, 20, 20])
    threshold_master = 1
    threshold_wing = 1
    threshold_choke = 5

    res = calculate_well_prod_status(master, wing, choke, threshold_master, threshold_wing, threshold_choke)
    exp_res = pd.Series([1, 1, 1, 0, 1], dtype=int)

    assert_series_equal(res.astype(int), exp_res)


@pytest.mark.core
def test_raise_error_threshold():
    # threshold values should always be greater than or equal to 0

    # Inputs
    master = pd.Series([100])
    wing = pd.Series([100])
    choke = pd.Series([20])
    threshold_master = -1
    threshold_wing = 1
    threshold_choke = 50

    # check if condition statement triggers ValueError
    with pytest.raises(ValueError, match="Threshold value has to be greater than or equal to 0"):
        _ = calculate_well_prod_status(master, wing, choke, threshold_master, threshold_wing, threshold_choke)

    # threshold values should always be greater than or equal to 100

    # Inputs
    master = pd.Series([100])
    wing = pd.Series([100])
    choke = pd.Series([20])
    threshold_master = 101
    threshold_wing = 1
    threshold_choke = 50

    # check if condition statement triggers ValueError
    with pytest.raises(ValueError, match="Threshold value has to be less than or equal to 100"):
        _ = calculate_well_prod_status(master, wing, choke, threshold_master, threshold_wing, threshold_choke)


@pytest.mark.core
def test_empty_valve():
    # check if empty valve raises error

    # Inputs
    master = pd.Series([])
    wing = pd.Series([100])
    choke = pd.Series([20])
    threshold_master = 1
    threshold_wing = 1
    threshold_choke = 50

    # check if condition statement triggers ValueError
    with pytest.raises(UserValueError, match="Empty Series are not allowed for valve inputs"):
        _ = calculate_well_prod_status(master, wing, choke, threshold_master, threshold_wing, threshold_choke)
