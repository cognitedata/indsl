from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from pandas.testing import assert_series_equal

from indsl.oil_and_gas.shut_in_variables import calculate_shutin_variable


DT_RANGE = pd.date_range(start=datetime(2000, 10, 1), end=datetime(2000, 10, 3), periods=49)
TEST_SIGNAL = pd.DataFrame(index=DT_RANGE, columns=["signal"], data=range(1, len(DT_RANGE) + 1), dtype=np.int64)
DEFAULT_HRS_AFTER_SHUTIN = 1


@pytest.mark.core
def test_empty_signal():
    # test variable shut-in calculations when the shut-ins are inside the time period and none is valid
    # because the first one is coming from the past and second one is too short
    # define shutin signal
    shutin_ser = pd.Series(dtype=int)

    # expected result
    exp_res = pd.Series([], index=pd.to_datetime([]), dtype=float)

    # actual result
    res = calculate_shutin_variable(TEST_SIGNAL["signal"], shutin_ser, DEFAULT_HRS_AFTER_SHUTIN)

    assert_series_equal(res, exp_res, check_freq=False)


@pytest.mark.core
def test_raise_error_shutin_dtype():
    # test raise ValueError if the shut-in detector signal has non-int numbers
    # define shutin signal
    shutin_ser = pd.Series([1.1] * 49, index=DT_RANGE)

    # Check if "if" statement triggers ValueError
    with pytest.raises(ValueError, match="The results from shut-in detector contain non-integer numbers"):
        _ = calculate_shutin_variable(TEST_SIGNAL["signal"], shutin_ser, DEFAULT_HRS_AFTER_SHUTIN)


@pytest.mark.core
def test_two_shutins_all_included():
    # test variable shut-in calculations when the shut-ins are inside the time period and both are valid
    # define shutin signal
    shutin_ser = pd.Series([1] * 10 + [0] * 10 + [1] * 25 + [0] * 3 + [1] * 1, index=DT_RANGE)

    # expected result
    exp_res = pd.Series([12.0, 47.0], index=[datetime(2000, 10, 1, 11), datetime(2000, 10, 2, 22)])

    # actual result
    res = calculate_shutin_variable(TEST_SIGNAL["signal"], shutin_ser, DEFAULT_HRS_AFTER_SHUTIN)

    assert_series_equal(res, exp_res, check_freq=False)


@pytest.mark.core
def test_two_shutins_one_included():
    # test variable shut-in calculations when the shut-ins are inside the time period and only the first one is valid
    # and the second one is invalid because it is too short
    # this test tests "else branch"
    # define shutin signal
    shutin_ser = pd.Series([1] * 10 + [0] * 10 + [1] * 25 + [0] * 3 + [0] * 1, index=DT_RANGE)

    hrs_after_shutin = 5

    # expected result
    exp_res = pd.Series([16.0], index=[datetime(2000, 10, 1, 15)])

    # actual result
    res = calculate_shutin_variable(TEST_SIGNAL["signal"], shutin_ser, hrs_after_shutin)

    assert_series_equal(res, exp_res, check_freq=False)


@pytest.mark.core
def test_two_shutins_two_included():
    # test variable shut-in calculations when the shut-ins are inside the time period and the second one is not finished
    # this test tests "if branch"
    # define shutin signal
    shutin_ser = pd.Series([1] * 10 + [0] * 10 + [1] * 25 + [0] * 3 + [0] * 1, index=DT_RANGE)

    hrs_after_shutin = 3

    # expected result
    exp_res = pd.Series([14.0, 49.0], index=[datetime(2000, 10, 1, 13), datetime(2000, 10, 3, 0)])

    # actual result
    res = calculate_shutin_variable(TEST_SIGNAL["signal"], shutin_ser, hrs_after_shutin)

    assert_series_equal(res, exp_res, check_freq=False)


@pytest.mark.core
def test_two_shutins_zero_included():
    # test variable shut-in calculations when the shut-ins are inside the time period and none is valid
    # because the first one is coming from the past and second one is too short
    # define shutin signal
    shutin_ser = pd.Series([0] * 10 + [0] * 10 + [1] * 25 + [0] * 3 + [0] * 1, index=DT_RANGE)

    hrs_after_shutin = 5

    # expected result
    exp_res = pd.Series([], index=pd.to_datetime([]), dtype=float)

    # actual result
    res = calculate_shutin_variable(TEST_SIGNAL["signal"], shutin_ser, hrs_after_shutin)

    assert_series_equal(res, exp_res, check_freq=False)


@pytest.mark.core
def test_raise_error_flowing_data():
    # test raise ValueError if no flowing data is present
    # define shutin signal
    shutin_ser = pd.Series([0] * 49, index=DT_RANGE, dtype=np.int64)

    # Check if "if" statement triggers ValueError
    with pytest.raises(ValueError, match="The signal does not contain flowing data"):
        _ = calculate_shutin_variable(TEST_SIGNAL["signal"], shutin_ser, DEFAULT_HRS_AFTER_SHUTIN)
