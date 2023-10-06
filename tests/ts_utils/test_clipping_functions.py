# Copyright 2021 Cognite AS
import numpy as np
import pandas as pd
import pytest

from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal, assert_series_equal

from indsl.ts_utils import bin_map, clip, maximum, minimum, threshold


@pytest.mark.core
def test_clip():
    assert clip(100, 1, 50) == 50
    # Clipping of array
    num1 = np.array([1, 2, 3, 4, 5])
    assert_array_equal(clip(num1, 2, 4), np.array([2, 2, 3, 4, 4]))

    # Clipping of pandas series
    series1 = pd.Series(data=np.array([1, 2, 3, 4, 5]))
    assert_series_equal(clip(series1, 2, 4), pd.Series(data=np.array([2, 2, 3, 4, 4])))
    assert_series_equal(clip(series1, low=2), pd.Series(data=np.array([2, 2, 3, 4, 5])))
    assert_series_equal(clip(series1, high=4), pd.Series(data=np.array([1, 2, 3, 4, 4])))

    # Clipping of pandas data frame
    df1 = pd.DataFrame(data={"a": np.array([1, 2, 3, 4, 5]), "b": np.array([1, 2, 3, 4, 5])})
    assert_series_equal(clip(df1.a, 2, 4), pd.Series(data=np.array([2, 2, 3, 4, 4]), name="a"))
    assert_frame_equal(
        clip(df1, 2, 4), pd.DataFrame(data={"a": np.array([2, 2, 3, 4, 4]), "b": np.array([2, 2, 3, 4, 4])})
    )


@pytest.mark.core
def test_maximum():
    assert maximum(300, 150) == 300

    # Maximum value of array
    num1 = np.array([1, 22, 3, 44, 5])
    num2 = np.array([11, 2, 33, 4, 55])
    assert_array_equal(maximum(num1, num2), np.array([11, 22, 33, 44, 55]))

    # Maximum value of pandas series
    arr1 = pd.Series(data=np.array([1, 22, 3, 44, 5]))
    arr2 = pd.Series(data=np.array([11, 2, 33, 4, 55]))
    assert_series_equal(maximum(arr1, arr2), pd.Series(data=np.array([11, 22, 33, 44, 55])))

    # Maximum value of pandas data frame
    df1 = pd.DataFrame(data={"a": np.array([1, 22, 3, 44, 5]), "b": np.array([66, 7, 88, 9, 1010])})
    df2 = pd.DataFrame(data={"a": np.array([11, 2, 33, 4, 55]), "b": np.array([6, 77, 8, 99, 10])})
    assert_series_equal(maximum(df1.a, df1.b), pd.Series(data=np.array([66, 22, 88, 44, 1010])))
    assert_frame_equal(
        maximum(df1, df2),
        pd.DataFrame(data={"a": np.array([11, 22, 33, 44, 55]), "b": np.array([66, 77, 88, 99, 1010])}),
    )


@pytest.mark.core
def test_minimum():
    assert minimum(300, 150) == 150

    # Minimum value of array
    num1 = np.array([1, 22, 3, 44, 5])
    num2 = np.array([11, 2, 33, 4, 55])
    assert_array_equal(minimum(num1, num2), np.array([1, 2, 3, 4, 5]))

    # Minimum value of pandas series
    arr1 = pd.Series(data=np.array([1, 22, 3, 44, 5]))
    arr2 = pd.Series(data=np.array([11, 2, 33, 4, 55]))
    assert_series_equal(minimum(arr1, arr2), pd.Series(np.array([1, 2, 3, 4, 5])))

    # Minimum value of pandas data frame
    df1 = pd.DataFrame(data={"a": np.array([1, 22, 3, 44, 5]), "b": np.array([66, 7, 88, 9, 1010])}, dtype=np.int64)
    df2 = pd.DataFrame(data={"a": np.array([11, 2, 33, 4, 55]), "b": np.array([6, 77, 8, 99, 10])}, dtype=np.int64)
    assert_series_equal(minimum(df1.a, df1.b), pd.Series(np.array([1, 7, 3, 9, 5]), dtype=np.int64))
    assert_frame_equal(
        minimum(df1, df2),
        pd.DataFrame(data={"a": np.array([1, 2, 3, 4, 5]), "b": np.array([6, 7, 8, 9, 10])}, dtype=np.int64),
    )


@pytest.mark.core
def test_bin_map():
    assert bin_map(300, 150) == 1

    # Minimum value of array
    num1 = np.array([1, 22, 3, 44, 5])
    num2 = np.array([11, 2, 33, 4, 55])
    assert_array_equal(bin_map(num1, num2), np.array([0, 1, 0, 1, 0]))

    # Minimum value of pandas series
    series1 = pd.Series(data=np.array([1, 22, 3, 44, 5]))
    series2 = pd.Series(data=np.array([11, 2, 33, 4, 55]))
    assert_series_equal(bin_map(series1, series2), pd.Series(data=np.array([0, 1, 0, 1, 0]), dtype=np.int64))

    # Minimum value of pandas data frame
    df1 = pd.DataFrame(data={"a": np.array([1, 22, 3, 44, 5]), "b": np.array([66, 7, 88, 9, 1010])})
    df2 = pd.DataFrame(data={"a": np.array([11, 2, 33, 4, 55]), "b": np.array([6, 77, 8, 99, 10])})
    assert_series_equal(bin_map(df1.a, df1.b), pd.Series(data=np.array([0, 1, 0, 1, 0]), dtype=np.int64))
    assert_frame_equal(
        bin_map(df1, df2),
        pd.DataFrame(data={"a": np.array([0, 1, 0, 1, 0]), "b": np.array([1, 0, 1, 0, 1])}, dtype=np.int64),
    )


@pytest.mark.core
def test_threshold():
    series = pd.Series(data=np.array([-np.inf, 22, 3, 44, 5, 11, 2, np.inf, 4, 55]))
    assert_series_equal(threshold(series, 4, 11), pd.Series(np.array([0, 0, 0, 0, 1, 1, 0, 0, 1, 0]), dtype=np.int64))

    # Test default values
    assert_series_equal(threshold(series), pd.Series([1] * len(series), dtype=np.int64))
