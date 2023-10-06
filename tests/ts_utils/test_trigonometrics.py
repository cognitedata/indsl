# Copyright 2021 Cognite AS
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from pandas.testing import assert_frame_equal, assert_series_equal

from indsl.ts_utils import (
    arccos,
    arccosh,
    arcsin,
    arcsinh,
    arctan,
    arctan2,
    arctanh,
    cos,
    cosh,
    deg2rad,
    rad2deg,
    sin,
    sinh,
    tan,
    tanh,
)


@pytest.mark.core
def test_sin():
    assert sin(np.pi / 2) == 1

    # Trigonometric sine of array
    num1 = np.array([np.pi / 2, np.pi])
    assert np.all(sin(num1) == pytest.approx(np.array([1, 0])))

    # Trigonometric sine of pandas series
    arr1 = pd.Series(data=np.array([0, np.pi / 2, np.pi]))
    assert sin(arr1).mean() == pytest.approx(1 / 3)

    # Trigonometric sine of pandas data frame
    df1 = pd.DataFrame(data={"a": ([0, np.pi / 2, np.pi]), "b": np.array([np.pi, 3 / 2 * np.pi, 2 * np.pi])})
    assert sin(df1.a).mean() == pytest.approx(1 / 3)
    assert sin(df1).mean().mean() == pytest.approx(0)


@pytest.mark.core
def test_cos():
    assert cos(np.pi) == -1

    # Trigonometric cosine of array
    num1 = np.array([np.pi, 0])
    assert np.all(cos(num1) == pytest.approx(np.array([-1, 1])))

    # Trigonometric cosine of pandas series
    arr1 = pd.Series(data=np.array([0, np.pi / 2, np.pi]))
    assert cos(arr1).mean() == pytest.approx(0)

    # Trigonometric cosine of pandas data frame
    df1 = pd.DataFrame(data={"a": ([0, np.pi / 2, np.pi]), "b": np.array([np.pi, 3 / 2 * np.pi, 2 * np.pi])})

    assert cos(df1.a).mean() == pytest.approx(0)
    assert cos(df1).mean().mean() == pytest.approx(0)


@pytest.mark.core
def test_tan():
    assert tan(np.pi) == pytest.approx(0)

    # Trigonometric tangent of array
    num1 = np.array([np.pi, np.pi / 4])
    assert np.all(tan(num1) == pytest.approx(np.array([0, 1])))

    # Trigonometric tangent of pandas series
    arr1 = pd.Series(data=np.array([-np.pi / 4, 0, np.pi / 4]))
    assert tan(arr1).mean() == pytest.approx(0)

    # Trigonometric tangent of pandas data frame
    df1 = pd.DataFrame(data={"a": ([-np.pi / 4, 0, np.pi / 4]), "b": np.array([3 / 4 * np.pi, np.pi, 5 / 4 * np.pi])})

    assert tan(df1.a).mean() == pytest.approx(0)
    assert tan(df1).mean().mean() == pytest.approx(0)


@pytest.mark.core
def test_arcsin():
    assert arcsin(1) == 1 / 2 * np.pi

    # Trigonometric arcsine of array
    num1 = np.array([1, 0])
    assert np.all(arcsin(num1) == pytest.approx(np.array([np.pi / 2, 0])))

    # Trigonometric arcsine of pandas series
    arr1 = pd.Series(data=np.array([0, 1, -1]))
    assert arcsin(arr1).mean() == pytest.approx(0)

    # Trigonometric arcsine of pandas data frame
    df1 = pd.DataFrame(data={"a": ([0, 1, -1]), "b": np.array([-0.5, 0, 0.5])})
    assert arcsin(df1.a).mean() == pytest.approx(0)
    assert sin(df1).mean().mean() == pytest.approx(0)


@pytest.mark.core
def test_arccos():
    assert arccos(-1) == np.pi

    # Trigonometric arccosine of array
    num1 = np.array([-1, 1])
    assert np.all(arccos(num1) == pytest.approx(np.array([np.pi, 0])))

    # Trigonometric arccosine of pandas series
    arr1 = pd.Series(data=np.array([-1, 0, 1]))
    assert arccos(arr1).mean() == pytest.approx(1 / 2 * np.pi)

    # Trigonometric arccosine of pandas data frame
    df1 = pd.DataFrame(data={"a": ([-1, 0, 1]), "b": np.array([-0.75, 0, 0.75])})

    assert arccos(df1.a).mean() == pytest.approx(1 / 2 * np.pi)
    assert arccos(df1).mean().mean() == pytest.approx(1 / 2 * np.pi)


@pytest.mark.core
def test_arctan():
    assert arctan(0) == pytest.approx(0)

    # Trigonometric arctangent of array
    num1 = np.array([-0.75, 0.75])
    assert arctan(num1).mean() == pytest.approx(0)

    # Trigonometric arctangent of pandas series
    arr1 = pd.Series(data=np.array([-0.5, 0, 0.5]))
    assert arctan(arr1).mean() == pytest.approx(0)

    # Trigonometric arctangent of pandas data frame
    df1 = pd.DataFrame(data={"a": ([-0.5, 0, 0.5]), "b": np.array([-1, 0, 1])})

    assert tan(df1.a).mean() == pytest.approx(0)
    assert tan(df1).mean().mean() == pytest.approx(0)


@pytest.mark.parametrize("align_timesteps", [True, False])
@pytest.mark.parametrize(
    "x1, x2, expected_result",
    [
        (1, 0, np.pi / 2),
        (-1, 0, -np.pi / 2),
        (1, 1, np.pi / 4),
        (-1, 1, -np.pi / 4),
        (-1, -1, -3 * np.pi / 4),
        (1, -1, 3 * np.pi / 4),
        (0, -1, np.pi),
        (0, 0, 0.0),
    ],
)
def test_arctan2_numbers(x1, x2, expected_result, align_timesteps):
    assert arctan2(x1, x2, align_timesteps=align_timesteps) == expected_result


@pytest.mark.core
def test_arctan2_array():
    x1_list = np.array([1, -1, 1, -1, -1, 1, 0, 0])
    x2_list = np.array([0, 0, 1, 1, -1, -1, -1, 0])
    expected_result_list = np.array(
        [np.pi / 2, -np.pi / 2, np.pi / 4, -np.pi / 4, -3 * np.pi / 4, 3 * np.pi / 4, np.pi, 0.0]
    )

    assert (arctan2(x1_list, x2_list) == expected_result_list).all()


@pytest.mark.core
def test_arctan2_pandas_series():
    # Trigonometric arctan2 of pandas series
    index_series1 = pd.DatetimeIndex([datetime(2020, 7, 13, 1, 0, i) for i in range(1, 10) if i != 8])
    n1 = pd.Series(data=[1, -1, 1, -1, -1, 1, 0, 0], index=index_series1)

    index_series2 = pd.date_range("2020-07-13 01:00:01", periods=8, freq="1s")
    n2 = pd.Series(data=[0, 0, 1, 1, -1, -1, -1, 0], index=index_series2)

    res_align_timesteps = arctan2(n1, n2, align_timesteps=True)
    exp_res_align_timesteps = pd.Series(
        [np.pi / 2, -np.pi / 2, np.pi / 4, -np.pi / 4, -3 * np.pi / 4, 3 * np.pi / 4, np.pi, 0.0],
        index=pd.date_range("2020-07-13 01:00:01", periods=8, freq="1s"),
    )
    assert_series_equal(exp_res_align_timesteps, res_align_timesteps)

    res_not_align_timesteps = arctan2(n1, n2, align_timesteps=False)
    exp_res_not_align_timesteps = pd.Series(
        [np.pi / 2, -np.pi / 2, np.pi / 4, -np.pi / 4, -3 * np.pi / 4, 3 * np.pi / 4, np.pi, np.NaN, np.NaN],
        index=pd.date_range("2020-07-13 01:00:01", periods=9, freq="1s"),
    )
    assert_series_equal(exp_res_not_align_timesteps, res_not_align_timesteps)


@pytest.mark.core
def test_rad2deg():
    assert rad2deg(np.pi) == 180

    # Convert array
    num1 = np.array([np.pi, np.pi / 2])
    assert rad2deg(num1).mean() == 135

    # Convert pandas series
    arr1 = pd.Series(data=np.array([np.pi, np.pi / 2, np.pi / 4]))
    assert rad2deg(arr1).mean() == 105

    # Convert pandas data frame
    df1 = pd.DataFrame(
        data={"a": np.array([np.pi, np.pi / 2, np.pi / 4]), "b": np.array([1 / 3 * np.pi, 1 / 6 * np.pi, 0])}
    )
    assert rad2deg(df1.a).mean() == 105
    assert rad2deg(df1).mean().mean() == 67.5


@pytest.mark.core
def test_deg2rad():
    assert deg2rad(180) == np.pi

    # Convert array
    num1 = np.array([180, 90])
    assert deg2rad(num1).mean() == 3 / 4 * np.pi

    # Convert pandas series
    arr1 = pd.Series(data=np.array([180, 90, 45]))
    assert deg2rad(arr1).mean() == pytest.approx(7 / 12 * np.pi)

    # Convert pandas data frame
    df1 = pd.DataFrame(data={"a": np.array([180, 90, 45]), "b": np.array([60, 30, 0])})
    assert deg2rad(df1.a).mean() == pytest.approx(7 / 12 * np.pi)
    assert deg2rad(df1).mean().mean() == pytest.approx(9 / 24 * np.pi)


@pytest.mark.core
def test_sinh():
    def calc_sinh(x):
        return (np.e**x - np.e ** (-x)) / 2

    assert sinh(2) == pytest.approx(calc_sinh(2))

    # sinh of array
    num1 = np.array([3, 5])
    assert np.all(sinh(num1) == pytest.approx(calc_sinh(num1)))

    # sinh of pandas series
    arr1 = pd.Series(data=np.array([1, 2, 3]))
    assert_series_equal(sinh(arr1), calc_sinh(arr1))

    # sinh of pandas data frame
    df1 = pd.DataFrame(data={"a": ([4, 5, 7])})
    assert_frame_equal(sinh(df1), calc_sinh(df1))


@pytest.mark.core
def test_cosh():
    def calc_cosh(x):
        return (np.e**x + np.e ** (-x)) / 2

    assert cosh(2) == pytest.approx(calc_cosh(2))

    # cosh of array
    num1 = np.array([3, 5])
    assert np.all(cosh(num1) == pytest.approx(calc_cosh(num1)))

    # cosh of pandas series
    arr1 = pd.Series(data=np.array([1, 2, 3]))
    assert_series_equal(cosh(arr1), calc_cosh(arr1))

    # cosh of pandas data frame
    df1 = pd.DataFrame(data={"a": ([4, 5, 7])})
    assert_frame_equal(cosh(df1), calc_cosh(df1))


@pytest.mark.core
def test_tanh():
    def calc_tanh(x):
        return (np.e**x - np.e ** (-x)) / (np.e**x + np.e ** (-x))

    assert tanh(2) == pytest.approx(calc_tanh(2))

    # tanh of array
    num1 = np.array([3, 5])
    assert np.all(tanh(num1) == pytest.approx(calc_tanh(num1)))

    # tanh of pandas series
    arr1 = pd.Series(data=np.array([1, 2, 3]))
    assert_series_equal(tanh(arr1), calc_tanh(arr1))

    # tanh of pandas dataframe
    df1 = pd.DataFrame(data={"a": ([4, 5, 7])})
    assert_frame_equal(tanh(df1), calc_tanh(df1))


@pytest.mark.core
def test_arcsinh():
    def calc_arcsinh(x):
        return np.log(x + np.sqrt(1 + x**2))

    assert arcsinh(2) == pytest.approx(calc_arcsinh(2))

    # arcsinh of array
    num1 = np.array([3, 5])
    assert np.all(arcsinh(num1) == pytest.approx(np.array(calc_arcsinh(num1))))

    # arcsinh of pandas series
    arr1 = pd.Series(data=np.array([1, 2, 3]))
    assert_series_equal(arcsinh(arr1), calc_arcsinh(arr1))

    # arcsinh of pandas dataframe
    df1 = pd.DataFrame(data={"a": ([4, 5, 7])})
    assert_frame_equal(arcsinh(df1), calc_arcsinh(df1))


@pytest.mark.core
def test_arccosh():
    def calc_arccosh(x):
        return np.log(x + np.sqrt(x**2 - 1))

    assert arccosh(2) == pytest.approx(calc_arccosh(2))

    # arccosh of array
    num1 = np.array([3, 5])
    assert np.all(arccosh(num1) == pytest.approx(arccosh(num1)))

    # arccosh of pandas series
    arr1 = pd.Series(data=np.array([2, 3, 4]))
    assert_series_equal(arccosh(arr1), calc_arccosh(arr1))

    # arccosh of pandas dataframe
    df1 = pd.DataFrame(data={"a": ([4, 5, 7])})
    assert_frame_equal(arccosh(df1), calc_arccosh(df1))


@pytest.mark.core
def test_arctanh():
    def calc_arctanh(x):
        return 0.5 * np.log((1 + x) / (1 - x))

    assert arctanh(0.3) == pytest.approx(calc_arctanh(0.3))

    # arctanh of array
    num1 = np.array([0.4, 0.5])
    assert np.all(arctanh(num1) == pytest.approx(calc_arctanh(num1)))

    # arctanh of pandas series
    arr1 = pd.Series(data=np.array([0.2, 0.3, 0.4]))
    assert_series_equal(arctanh(arr1), calc_arctanh(arr1))

    # arctanh of pandas dataframe
    df1 = pd.DataFrame(data={"a": ([0.4, 0.5, 0.7])})
    assert_frame_equal(arctanh(df1), calc_arctanh(df1))
