# Copyright 2022 Cognite AS

import numpy as np
import pandas as pd
import pytest

from indsl.data_quality.value_decrease_indication import _prolong_indicator, value_decrease_check
from indsl.exceptions import UserTypeError, UserValueError


@pytest.mark.parametrize(
    "values, indicator, expectation",
    [
        (np.array([1, 2, 3, 4, 5]), np.array([0, 0, 0, 0, 0]), np.array([0, 0, 0, 0, 0])),
        (np.array([1, 2, 1, 2, 2]), np.array([0, 0, 1, 0, 0]), np.array([0, 0, 1, 0, 0])),
        (np.array([1, 5, 1, 3, 6]), np.array([0, 0, 1, 0, 0]), np.array([0, 0, 1, 1, 0])),
        (np.array([1, 2, 1, 2, 1]), np.array([0, 0, 1, 0, 1]), np.array([0, 0, 1, 0, 1])),
        (np.array([1, 7, 1, 5, 2]), np.array([0, 0, 1, 0, 1]), np.array([0, 0, 1, 1, 1])),
        (np.array([1, 7, 3, 1, 8]), np.array([0, 0, 1, 1, 0]), np.array([0, 0, 1, 1, 0])),
    ],
)
def test_prolong_indicator(values, indicator, expectation):
    result = _prolong_indicator(values, indicator)
    assert np.array_equal(result, expectation)


# test function checking decrease in time series values with default threshold (0.0)
@pytest.mark.parametrize(
    "input, expectation",
    [
        (
            pd.Series([1], index=[pd.to_datetime(1490195805, unit="s")]),
            pd.Series([0], index=[pd.to_datetime(1490195805, unit="s")]),
        ),
        (
            pd.Series([1, 2, 3, 4, 5], index=pd.date_range("2018-01-01", periods=5, freq="H")),
            pd.Series([0, 0, 0, 0, 0], index=pd.date_range("2018-01-01", periods=5, freq="H")),
        ),
        (
            pd.Series([1, 2, 1, 3, 5], index=pd.date_range("2018-01-01", periods=5, freq="H")),
            pd.Series([0, 0, 1, 0, 0], index=pd.date_range("2018-01-01", periods=5, freq="H")),
        ),
        (
            pd.Series([1, 0, 1, 0, 1], index=pd.date_range("2018-01-01", periods=5, freq="H")),
            pd.Series([0, 1, 0, 1, 0], index=pd.date_range("2018-01-01", periods=5, freq="H")),
        ),
        (
            pd.Series([1, 2, 3, 3, 2], index=pd.date_range("2018-01-01", periods=5, freq="H")),
            pd.Series([0, 0, 0, 0, 1], index=pd.date_range("2018-01-01", periods=5, freq="H")),
        ),
        (
            pd.Series([1, 2, 3, 2, 2], index=pd.date_range("2018-01-01", periods=5, freq="H")),
            pd.Series([0, 0, 0, 1, 1], index=pd.date_range("2018-01-01", periods=5, freq="H")),
        ),
        (
            pd.Series([1, 2, 1, 1, 5, 6, 0, 1, 6, 2], index=pd.date_range("2018-01-01", periods=10, freq="H")),
            pd.Series([0, 0, 1, 1, 0, 0, 1, 1, 0, 1], index=pd.date_range("2018-01-01", periods=10, freq="H")),
        ),
        (
            pd.Series([10, 20, 1, 1, 5, 6, 0, 10, 6, 7], index=pd.date_range("2018-01-01", periods=10, freq="H")),
            pd.Series([0, 0, 1, 1, 1, 1, 1, 1, 1, 1], index=pd.date_range("2018-01-01", periods=10, freq="H")),
        ),
    ],
)
def test_value_decrease_default_threshold(input, expectation):
    result = value_decrease_check(input)
    assert result.equals(expectation)


@pytest.mark.parametrize(
    "input, threshold, expectation",
    [
        (
            pd.Series([1], index=[pd.to_datetime(1490195805, unit="s")]),
            3,
            pd.Series([0], index=[pd.to_datetime(1490195805, unit="s")]),
        ),
        (
            pd.Series([1, 2, 3, 4, 5], index=pd.date_range("2018-01-01", periods=5, freq="H")),
            0.55,
            pd.Series([0, 0, 0, 0, 0], index=pd.date_range("2018-01-01", periods=5, freq="H")),
        ),
        (
            pd.Series([1, 2, 1, 1, 5], index=pd.date_range("2018-01-01", periods=5, freq="H")),
            1.3,
            pd.Series([0, 0, 0, 0, 0], index=pd.date_range("2018-01-01", periods=5, freq="H")),
        ),
        (
            pd.Series([1, 0, 1, 0, 1], index=pd.date_range("2018-01-01", periods=5, freq="H")),
            0.3,
            pd.Series([0, 1, 0, 1, 0], index=pd.date_range("2018-01-01", periods=5, freq="H")),
        ),
        (
            pd.Series([1, 0, 1, 0, 1], index=pd.date_range("2018-01-01", periods=5, freq="H")),
            2,
            pd.Series([0, 0, 0, 0, 0], index=pd.date_range("2018-01-01", periods=5, freq="H")),
        ),
        (
            pd.Series([1, 2, 3, 3, 2], index=pd.date_range("2018-01-01", periods=5, freq="H")),
            0.4,
            pd.Series([0, 0, 0, 0, 1], index=pd.date_range("2018-01-01", periods=5, freq="H")),
        ),
        (
            pd.Series([4, 2, 5, 2, 3], index=pd.date_range("2018-01-01", periods=5, freq="H")),
            2.7,
            pd.Series([0, 0, 0, 1, 1], index=pd.date_range("2018-01-01", periods=5, freq="H")),
        ),
        (
            pd.Series([1, 2, 1, 1, 5, 6, 0, 1, 6, 2], index=pd.date_range("2018-01-01", periods=10, freq="H")),
            10.377,
            pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], index=pd.date_range("2018-01-01", periods=10, freq="H")),
        ),
        (
            pd.Series([10, 20, 1, 1, 5, 26, 20, 4, 6, 20], index=pd.date_range("2018-01-01", periods=10, freq="H")),
            15,
            pd.Series([0, 0, 1, 1, 1, 0, 0, 1, 1, 0], index=pd.date_range("2018-01-01", periods=10, freq="H")),
        ),
    ],
)
def test_value_decrease_added_threshold(input, threshold, expectation):
    """test function checking decrease in time series values with user-added threshold"""
    result = value_decrease_check(input, threshold)
    assert result.equals(expectation)


@pytest.mark.core
def test_value_decrease_threshold_negative_error():
    """test errors for threshold (Threshold should be a non-negative float)"""
    with pytest.raises(UserValueError):
        value_decrease_check(
            pd.Series([1, 2, 3, 4, 5], index=pd.date_range("2018-01-01", periods=5, freq="H")), -3
        )  # threshold is negative float


@pytest.mark.core
def test_value_decrease_threshold_negative_error_message():
    with pytest.raises(UserValueError) as exc:
        value_decrease_check(
            pd.Series([1, 2, 3, 4, 5], index=pd.date_range("2018-01-01", periods=5, freq="H")), -7.1
        )  # testing the error message
    assert "Threshold should be a non-negative float." in str(exc.value)


@pytest.mark.core
def test_value_decrease_threshold_type_error():
    with pytest.raises(UserTypeError):
        value_decrease_check(
            pd.Series([1, 2, 3, 4, 5], index=pd.date_range("2018-01-01", periods=5, freq="H")), "7.1"
        )  # threshold is not a float


@pytest.mark.core
def test_value_decrease_threshold_no_errors():
    try:
        value_decrease_check(
            pd.Series([1, 2, 3, 4, 5], index=pd.date_range("2018-01-01", periods=5, freq="H")), 1.0
        )  # no error raised
    except pytest.raises(UserValueError) as exc:
        assert False, f"'value_decrease_check' raised an exception {exc}"


@pytest.mark.core
def test_value_decrease_x_type_error():
    """test errors for x (x should be a non-empty time series with increasing datetime index)"""
    with pytest.raises(UserTypeError):
        value_decrease_check("not a time series", 7.1)  # x is not a time series


@pytest.mark.core
def test_value_decrease_x_empty_error():
    with pytest.raises(UserValueError):
        value_decrease_check(
            pd.Series([np.nan, np.nan], index=pd.date_range("2018-01-01", periods=2, freq="H")).dropna()
        )  # x is empty


@pytest.mark.core
def test_value_decrease_x_index_datetime_error():
    with pytest.raises(UserTypeError):
        value_decrease_check(pd.Series([1, 2, 3, 4, 5], index=[1, 2, 3, 4, 5]), 7.1)  # index is not datetime


@pytest.mark.core
def test_value_decrease_x_index_increasing_error():
    with pytest.raises(UserValueError):
        value_decrease_check(
            pd.Series([1, 2, 3], index=pd.to_datetime([1490195805, 1490195705, 1490195905], unit="s")), 7.1
        )  # index not increasing
