# Copyright 2021 Cognite AS
import numpy as np
import pandas as pd
import pytest
import pandas.testing as tm

from indsl.ts_utils import absolute, add, arithmetic_mean, div, inv, mod, mul, neg, power, sqrt, sub
from indsl.ts_utils.operators import arithmetic_mean_many, sample_average


@pytest.mark.core
def test_add():
    assert add(5, 70) == 75

    # Add arrays
    num1 = np.array([5, 70])
    num2 = np.array([70, 5])
    assert add(num1, num2).mean() == 75

    # Add pandas series
    arr1 = pd.Series(data=np.array([5, 6, 7]))
    arr2 = pd.Series(data=np.array([70, 69, 68]))
    assert add(arr1, arr2).mean() == 75

    # Add pandas data frame
    df1 = pd.DataFrame(data={"a": np.array([5, 6, 7]), "b": np.array([70, 69, 68])})
    df2 = pd.DataFrame(data={"a": np.array([70, 69, 68]), "b": np.array([5, 6, 7])})
    assert add(df1.a, df1.b).mean() == 75
    assert add(df1, df2).mean().mean() == 75


@pytest.mark.core
def test_sub():
    assert sub(70, 5) == 65

    # Subtract arrays
    num1 = np.array([5, 70])
    num2 = np.array([70, 5])
    assert sub(num1, num2).mean() == 0

    # Subtract pandas series
    arr1 = pd.Series(data=np.array([70, 71, 72]))
    arr2 = pd.Series(data=np.array([5, 6, 7]))
    assert sub(arr1, arr2).mean() == 65

    # Subtract pandas data frame
    df1 = pd.DataFrame(data={"a": np.array([70, 71, 72]), "b": np.array([5, 6, 7])})
    df2 = pd.DataFrame(data={"a": np.array([5, 6, 7]), "b": np.array([70, 71, 72])})
    assert sub(df1.a, df1.b).mean() == 65
    assert sub(df1, df2).mean().mean() == 0


@pytest.mark.core
def test_mul():
    assert mul(70, 5) == 350

    # Multiply arrays
    num1 = np.array([5, 70])
    num2 = np.array([70, 5])
    assert mul(num1, num2).mean() == 350
    assert mul(num1, 2).mean() == 75

    # Multiply pandas series
    arr1 = pd.Series(data=np.array([70, 100, 50]))
    arr2 = pd.Series(data=np.array([5, 2, 7]))
    assert mul(arr1, arr2).mean() == 300
    assert mul(arr2, 3).mean() == 14

    # Multiply pandas data frame
    df1 = pd.DataFrame(data={"a": np.array([70, 100, 50]), "b": np.array([5, 2, 7])})
    df2 = pd.DataFrame(data={"a": np.array([5, 2, 7]), "b": np.array([70, 100, 50])})
    assert mul(df1.a, df1.b).mean() == 300
    assert mul(df1, df2).mean().mean() == 300

    assert mul(df1.b, 3).mean() == 14
    assert mul(df1, 3).mean().mean() == 117


@pytest.mark.core
def test_div():
    assert div(70, 5) == 14

    # Divide arrays
    num1 = np.array([35, 42])
    num2 = np.array([5, 6])
    assert div(num1, num2).mean() == 7
    assert div(num1, 7).mean() == 5.5

    # Divide pandas series
    arr1 = pd.Series(data=np.array([35, 42, 49]))
    arr2 = pd.Series(data=np.array([5, 6, 7]))
    assert div(arr1, arr2).mean() == 7
    assert div(arr1, 7).mean() == 6

    # Divide pandas series with zeros in denominator
    arr1 = pd.Series(data=np.array([35, 42, 49]))
    arr2 = pd.Series(data=np.array([5, 6, 0]))
    assert div(arr1, arr2).mean() == 7

    # Divide pandas data frame
    df1 = pd.DataFrame(data={"a": np.array([70, 84, 98]), "b": np.array([35, 42, 49])})
    df2 = pd.DataFrame(data={"a": np.array([10, 12, 14]), "b": np.array([5, 6, 7])})
    assert div(df1.a, df1.b).mean() == 2
    assert div(df1, df2).mean().mean() == 7

    assert div(df1.a, 7).mean() == 12
    assert div(df1, 7).mean().mean() == 9


@pytest.mark.core
def test_power():
    assert power(2, 5) == 32

    # Power for arrays
    num1 = np.array([2, 2])
    num2 = np.array([2, 3])
    assert power(num1, num2).mean() == 6
    assert power(num2, 2).mean() == 6.5

    # Power for pandas series
    arr1 = pd.Series(data=np.array([3, 3, 3]))
    arr2 = pd.Series(data=np.array([1, 2, 3]))
    assert power(arr1, arr2).mean() == 13
    assert power(arr2, 3).mean() == 12

    # Power for pandas data frame
    df1 = pd.DataFrame(data={"a": np.array([9, 6, 3]), "b": np.array([1, 2, 3])})
    df2 = pd.DataFrame(data={"a": np.array([1, 2, 3]), "b": np.array([2, 3, 4])})
    assert power(df1.a, df1.b).mean() == 24
    assert power(df1, df2).mean().mean() == 27

    assert power(df1.a, 2).mean() == 42
    assert power(df1, 3).mean().mean() == 168


@pytest.mark.core
def test_inv():
    assert inv(5) == 1 / 5

    # Inverse of arrays
    num1 = np.array([2, 4])
    assert inv(num1).mean() == 0.375

    # Inverse of pandas series
    arr1 = pd.Series(data=np.array([3, 4, 6]))
    assert np.all(inv(arr1) == np.array([1 / 3, 1 / 4, 1 / 6]))

    # Add pandas data frame
    df1 = pd.DataFrame(data={"a": np.array([3, 4, 6]), "b": np.array([70, 69, 68])})
    assert np.all(inv(df1.a) == np.array([1 / 3, 1 / 4, 1 / 6]))
    assert np.all(
        inv(df1) == pd.DataFrame(data={"a": np.array([1 / 3, 1 / 4, 1 / 6]), "b": np.array([1 / 70, 1 / 69, 1 / 68])})
    )


@pytest.mark.core
def test_sqrt():
    assert sqrt(4) == 2

    # Square root of array
    num1 = np.array([4, 16])
    assert sqrt(num1).mean() == 3

    # Square root of pandas series
    arr1 = pd.Series(data=np.array([9, 36, 81]))
    assert sqrt(arr1).mean() == 6

    # Square root of pandas data frame
    df1 = pd.DataFrame(data={"a": np.array([9, 36, 81]), "b": np.array([225, 144, 324])})
    assert sqrt(df1.a).mean() == 6
    assert sqrt(df1).mean().mean() == 10.5


@pytest.mark.core
def test_neg():
    assert neg(4) == -4

    # Negation of array
    num1 = np.array([4, -16])
    assert neg(num1).mean() == 6

    # Negation of pandas series
    arr1 = pd.Series(data=np.array([9, 36, -81]))
    assert neg(arr1).mean() == 12

    # Negation of pandas data frame
    df1 = pd.DataFrame(data={"a": np.array([9, 36, -81]), "b": np.array([-9, -36, 81])})
    assert neg(df1.a).mean() == 12
    assert neg(df1).mean().mean() == 0


@pytest.mark.core
def test_absolute():
    assert absolute(-4) == 4

    # Absolute of array
    num1 = np.array([4, -16])
    assert absolute(num1).mean() == 10

    # Absolute of pandas series
    arr1 = pd.Series(data=np.array([9, 36, -81]))
    assert absolute(arr1).mean() == 42

    # Absolute of pandas data frame
    df1 = pd.DataFrame(data={"a": np.array([9, 36, -81]), "b": np.array([-9, -36, 81])})
    assert absolute(df1.a).mean() == 42
    assert absolute(df1).mean().mean() == 42


@pytest.mark.core
def test_mod():
    assert mod(5, 2) == 1

    # Modulo of arrays
    num1 = np.array([5, 19])
    num2 = np.array([2, 3])
    assert mod(num1, num2).mean() == 1

    # Modulo of pandas series
    arr1 = pd.Series(data=np.array([5, 19, 21]))
    arr2 = pd.Series(data=np.array([2, 3, 4]))
    assert mod(arr1, arr2).mean() == 1

    # Modulo of pandas data frame
    df1 = pd.DataFrame(data={"a": np.array([5, 19, 21]), "b": np.array([2, 3, 4])})
    df2 = pd.DataFrame(data={"a": np.array([70, 69, 68]), "b": np.array([5, 6, 7])})
    assert mod(df1.a, df1.b).mean() == 1
    assert mod(df1, df2).mean().mean() == 9


@pytest.mark.core
def test_arithmetic_mean():
    assert arithmetic_mean(4, 2) == 3

    # Mean of series
    arr1 = pd.Series(data=[5, 10, 10])
    arr2 = pd.Series(data=[3, 3, 5])
    assert arithmetic_mean(arr1, arr2).mean() == 6

    # Mean of pandas data frame
    df1 = pd.DataFrame(data={"a": np.array([5, 19, 21]), "b": np.array([2, 3, 4])})
    df2 = pd.DataFrame(data={"a": np.array([70, 69, 68]), "b": np.array([5, 6, 7])})
    assert arithmetic_mean(df1.a, df1.b).mean() == 9.0
    assert arithmetic_mean(df1, df2).mean().mean() == 23.25


@pytest.mark.core
def test_arithmetic_mean_many():
    # Mean of 3 series
    s1 = pd.Series(data=[1, 2, 3], index=[0, 1, 2])
    s2 = pd.Series(data=[2, 3, 4], index=[0, 1, 2])
    s3 = pd.Series(data=[3, 4, 5], index=[0, 1, 2])
    assert arithmetic_mean_many([s1, s2, s3]).sum() == 9
    assert arithmetic_mean_many([s1, s2, s3]).mean() == 3

    # Mean of 4 series
    s1 = pd.Series(data=[2, 4, 6], index=[0, 1, 2])
    s2 = pd.Series(data=[4, 6, 8], index=[0, 1, 2])
    s3 = pd.Series(data=[6, 8, 10], index=[0, 1, 2])
    s4 = pd.Series(data=[8, 10, 12], index=[0, 1, 2])
    assert arithmetic_mean_many([s1, s2, s3, s4]).sum() == 21.0
    assert arithmetic_mean_many([s1, s2, s3, s4]).mean() == 7.0

    # Mean of numbers
    assert arithmetic_mean_many([3, 6, 9]) == 6

    # Mean of 2 series and one float
    s1 = pd.Series(data=[1, 2, 3], index=[0, 1, 2])
    s2 = pd.Series(data=[2, 4, 6], index=[0, 1, 2])
    assert arithmetic_mean_many([s1, s2, 3]).sum() == 9


@pytest.mark.core
def test_average_no_threshold():

    test_data = pd.Series(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=pd.date_range("1975-05-09 00:00:00", "1975-05-09 09:00:00", freq="1h")
    )

    expected = pd.Series(
        [5.5] * 10, index=pd.date_range("1975-05-09 00:00:00", "1975-05-09 09:00:00", freq="1h"), name="constant_ts"
    )
    test_timeseries = sample_average(test_data)

    tm.assert_series_equal(test_timeseries, expected)


@pytest.mark.core
def test_average_with_threshold():

    test_data = pd.Series(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=pd.date_range("1975-05-09 00:00:00", "1975-05-09 09:00:00", freq="1h")
    )

    expected = pd.Series(
        [7.0] * 7, index=pd.date_range("1975-05-09 03:00:00", "1975-05-09 09:00:00", freq="1h"), name="constant_ts"
    )
    test_timeseries = sample_average(test_data, threshold=4.0, condition="Above")

    tm.assert_series_equal(test_timeseries, expected)


@pytest.mark.core
def test_average_uneven_sampling():

    test_data = pd.Series(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        index=pd.to_datetime(
            [
                "1975-05-09 00:00:00",
                "1975-05-09 00:30:00",
                "1975-05-09 02:30:00",
                "1975-05-09 05:30:00",
                "1975-05-09 07:30:00",
                "1975-05-09 08:00:00",
                "1975-05-09 10:00:00",
                "1975-05-09 13:00:00",
                "1975-05-09 16:00:00",
                "1975-05-09 20:00:00",
            ]
        ),
        name="constant_ts",
    )

    expected = pd.Series(
        [5.5] * 10,
        index=pd.to_datetime(
            [
                "1975-05-09 00:00:00",
                "1975-05-09 00:30:00",
                "1975-05-09 02:30:00",
                "1975-05-09 05:30:00",
                "1975-05-09 07:30:00",
                "1975-05-09 08:00:00",
                "1975-05-09 10:00:00",
                "1975-05-09 13:00:00",
                "1975-05-09 16:00:00",
                "1975-05-09 20:00:00",
            ]
        ),
        name="constant_ts",
    )

    test_timeseries = sample_average(test_data)

    tm.assert_series_equal(test_timeseries, expected)
