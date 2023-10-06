# Copyright 2022 Cognite AS

import numpy as np
import pandas as pd
import pytest

from indsl.data_quality.value_decrease_indication_v1 import _prolong_indicator, negative_running_hours_check
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
def test_negative_running_hours_default_threshold(input, expectation):
    """
    test negative running hours function with default threshold (0.0)
    """
    result = negative_running_hours_check(input)
    assert result.equals(expectation)


# test negative running hours function with user-added threshold
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
def test_negative_running_hours_added_threshold(input, threshold, expectation):
    result = negative_running_hours_check(input, threshold)
    assert result.equals(expectation)


@pytest.mark.core
def test_negative_running_hours_threshold_negative_error():
    with pytest.raises(UserValueError):
        negative_running_hours_check(
            pd.Series([1, 2, 3, 4, 5], index=pd.date_range("2018-01-01", periods=5, freq="H")), -3
        )  # threshold is negative float


@pytest.mark.core
def test_negative_running_hours_threshold_negative_error_message():
    with pytest.raises(UserValueError) as exc:
        negative_running_hours_check(
            pd.Series([1, 2, 3, 4, 5], index=pd.date_range("2018-01-01", periods=5, freq="H")), -7.1
        )
    assert "Threshold should be a non-negative float." in str(exc.value)


@pytest.mark.core
def test_negative_running_hours_threshold_type_error():
    with pytest.raises(UserTypeError):
        negative_running_hours_check(
            pd.Series([1, 2, 3, 4, 5], index=pd.date_range("2018-01-01", periods=5, freq="H")), "7.1"
        )  # threshold is not a float


@pytest.mark.core
def test_negative_running_hours_threshold_no_errors():
    try:
        negative_running_hours_check(
            pd.Series([1, 2, 3, 4, 5], index=pd.date_range("2018-01-01", periods=5, freq="H")), 1.0
        )  # no error raised
    except pytest.raises(UserValueError) as exc:
        assert False, f"'negative_running_hours_check' raised an exception {exc}"


@pytest.mark.core
def test_negative_running_hours_x_type_error():
    """test errors for x (x should be a non-empty time series with increasing datetime index)"""
    with pytest.raises(UserTypeError):
        negative_running_hours_check("Not a time series", 7.1)  # x not time series


@pytest.mark.core
def test_negative_running_hours_x_empty_error():
    with pytest.raises(UserValueError):
        negative_running_hours_check(
            pd.Series([np.nan, np.nan], index=pd.date_range("2018-01-01", periods=2, freq="H")).dropna()
        )  # x empty


@pytest.mark.core
def test_negative_running_hours_x_index_datetime_error():
    with pytest.raises(UserTypeError):
        negative_running_hours_check(pd.Series([1, 2, 3, 4, 5], index=[1, 2, 3, 4, 5]), 7.1)  # index not datetime


@pytest.mark.core
def test_negative_running_hours_x_index_increasing_error():
    with pytest.raises(UserValueError):
        negative_running_hours_check(
            pd.Series([1, 2, 3], index=pd.to_datetime([1490195805, 1490195705, 1490195905], unit="s")), 7.1
        )  # index not increasing
