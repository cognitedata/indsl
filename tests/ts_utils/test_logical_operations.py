# Copyright 2022 Cognite AS
import pandas as pd
import pytest

from pandas import testing as tm

from indsl.ts_utils import logical_check


# Some examples of input types
time_index = pd.date_range(start="2022-07-01 00:00:00", end="2022-07-01 01:00:00", periods=8)
A = pd.Series(index=time_index, data=[0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7])
B = pd.Series(index=time_index, data=[8.8, 7.7, 6.6, 5.5, 4.4, 3.3, 2.2, 1.1])


@pytest.mark.core
def test_logical_check_eq():
    # A == B
    res = logical_check(value_1=A, value_2=B, value_true=1, value_false=0, operation="Equality")
    expected = pd.Series(index=time_index, data=[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], name="result")
    tm.assert_series_equal(res, expected)


@pytest.mark.core
def test_logical_check_ne():
    # A != B
    res = logical_check(value_1=A, value_2=B, value_true=1, value_false=0, operation="Inequality")
    expected = pd.Series(index=time_index, data=[1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0], name="result")
    tm.assert_series_equal(res, expected)


@pytest.mark.core
def test_logical_check_gt():
    # A > B
    res = logical_check(value_1=A, value_2=B, value_true=1, value_false=0, operation="Greater than")
    expected = pd.Series(index=time_index, data=[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0], name="result")
    tm.assert_series_equal(res, expected)


@pytest.mark.core
def test_logical_check_ge():
    # A >= B
    res = logical_check(value_1=A, value_2=B, value_true=1, value_false=0, operation="Greater or equal than")
    expected = pd.Series(index=time_index, data=[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], name="result")
    tm.assert_series_equal(res, expected)


@pytest.mark.core
def test_logical_check_lt():
    # A < B
    res = logical_check(value_1=A, value_2=B, value_true=1, value_false=0, operation="Smaller than")
    expected = pd.Series(index=time_index, data=[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], name="result")
    tm.assert_series_equal(res, expected)


@pytest.mark.core
def test_logical_check_le():
    # A >= B
    res = logical_check(value_1=A, value_2=B, value_true=1, value_false=0, operation="Smaller or equal than")
    expected = pd.Series(index=time_index, data=[1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], name="result")
    tm.assert_series_equal(res, expected)


@pytest.mark.core
def test_logical_check_constants():
    res = logical_check(
        value_1=5.0, value_2=3.0, value_true=pd.Series([1, 2, 3]), value_false=4.0, operation="Equality"
    )
    assert res == 4.0

    res = logical_check(
        value_1=5.0, value_2=3.0, value_true=pd.Series([1, 2, 3]), value_false=4.0, operation="Greater than"
    )
    tm.assert_series_equal(res, pd.Series([1, 2, 3]))
