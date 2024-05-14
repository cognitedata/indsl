# Copyright 2021 Cognite AS
import random

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from pandas.testing import assert_series_equal

from indsl.exceptions import UserTypeError, UserValueError
from indsl.resample.resample import resample, resample_to_granularity

from ..generate_data import create_uniform_data


# Test for empty data
@pytest.mark.core
def test_empty_data():
    with pytest.raises(UserTypeError) as e:
        resample(pd.Series([1], dtype="float64"))
    assert "Expected a time series, got index type int64" in str(e.value)

    with pytest.raises(UserValueError) as e:
        resample(pd.Series([], index=pd.to_datetime([]), dtype="float64"), granularity_current=pd.Timedelta("1s"))
    assert "Expected data to be of length > 0, got length 0"

    with pytest.raises(UserTypeError) as e:
        resample_to_granularity(pd.Series([1], dtype="float64"))
    assert "Expected a time series, got index type int64" in str(e.value)

    with pytest.raises(UserValueError) as e:
        resample_to_granularity(pd.Series([], index=pd.to_datetime([]), dtype="float64"))

    assert "Time series is empty" in str(e.value)


# Test for all NaN data
@pytest.mark.core
def test_all_nan_data():
    with pytest.raises(UserTypeError):
        test_data = create_uniform_data(np.ones(10) * np.nan)
        imputed_data = resample(test_data, method="fourier")
        assert len(imputed_data) == len(test_data)
        assert imputed_data.isna().all()


@pytest.mark.core
def test_positive_granularity():
    with pytest.raises(UserValueError) as e:
        test_data = pd.Series(
            [random.random() * i for i in range(24)], index=pd.date_range("2020-02-03", periods=24, freq="1h")
        )
        granularity = pd.Timedelta(0, "m")
        resample_to_granularity(test_data, granularity=granularity)
        assert "Timedelta must be strictly positive. The smallest possible value is '1ns'" in str(e.value)


# test correct output size
@pytest.mark.parametrize("test_length", [(200), (10)])
def test_resample_length(test_length):
    test_data = create_uniform_data(np.ones(10))
    resampled_data = resample(test_data, num=test_length)
    assert len(resampled_data) == test_length


# length of output should exceed input
@pytest.mark.parametrize("method", [("fourier"), ("polyphase"), ("interpolate")])
def test_if_upsampled(method):
    data = pd.Series([random.random() * i for i in range(24)], index=pd.date_range("2020-02-03", periods=24, freq="1h"))

    # upsampled to 30 min
    upsampled = resample(data, granularity_next=pd.Timedelta("30min"), method=method)

    assert len(data) < len(upsampled)


# length of input should exceed output
@pytest.mark.parametrize("method", [("fourier"), ("polyphase"), ("interpolate")])
def test_if_downsampled(method):
    data = pd.Series([random.random() * i for i in range(24)], index=pd.date_range("2020-02-03", periods=24, freq="1h"))

    # upsampled to 30 min
    upsampled = resample(data, granularity_next=pd.Timedelta("3120min"), method=method)

    assert len(data) > len(upsampled)


# test distribution of ramsapled when upsampling should be smaller
@pytest.mark.parametrize("method", [("polyphase"), ("interpolate")])
def test_test_resample_distribution(method):
    periods = 4000
    data = pd.Series(
        [random.random() * i for i in range(periods)], index=pd.date_range("2020-02-03", periods=periods, freq="1h")
    )

    # data[1:3] = np.nan
    # baseline
    std_d = np.std(data)

    # upsampled to 30 min
    upsampled = resample(data, granularity_next=pd.Timedelta("6min"), method=method)
    std_up = np.std(upsampled)

    assert std_d >= std_up


@pytest.mark.core
def test_no_value_returned_is_nan():
    """Should be no missing data in returned data."""
    data = pd.Series([random.random() * i for i in range(24)], index=pd.date_range("2020-02-03", periods=24, freq="1h"))

    data[1:5] = np.nan

    resampled = resample(data, granularity_next=pd.Timedelta("30min"), method="fourier")

    assert not resampled.isna().any()


def test_resample_to_granularity_count():
    data = pd.Series([random.random() * i for i in range(24)], index=pd.date_range("2020-02-03", periods=24, freq="1h"))
    resampled_data = resample_to_granularity(data, granularity=pd.Timedelta("2h"), aggregate="count")
    assert len(resampled_data) == len(data) // 2
    assert all(resampled_data == 2)


@pytest.mark.parametrize(
    "aggregate, expected_resampled_data",
    [
        ("interpolation", np.array(range(24 * 2 - 1)) / 2),
        ("stepInterpolation", np.array(range(24 * 2 - 1)) // 2),
    ],
)
def test_resample_to_granularity_interpolate(aggregate, expected_resampled_data):
    data = pd.Series(list(range(24)), index=pd.date_range("2020-02-03", periods=24, freq="1h"))
    resampled_data = resample_to_granularity(data, granularity=pd.Timedelta("30m"), aggregate=aggregate)
    assert len(resampled_data) == len(data) * 2 - 1
    assert all(resampled_data.values == expected_resampled_data)


@pytest.mark.core
def test_resample_downsample_method():
    not_uniform_data = pd.Series(
        [1, 2, 3, 4, 5],
        index=pd.DatetimeIndex(
            [
                datetime(2020, 7, 13, 1, 0, 1),
                datetime(2020, 7, 14, 1, 0, 3),
                datetime(2020, 7, 15, 1, 0, 5),
                datetime(2020, 7, 15, 1, 0, 8),
                datetime(2020, 7, 14, 1, 0, 9),
            ]
        ),
    )

    with pytest.raises(UserTypeError) as excinfo:
        resample(data=not_uniform_data, granularity_next=None, num=-1)
    expected = "Either num or granularity_next has to be set."
    assert expected in str(excinfo.value)

    # Not uniform data
    res_data = resample(data=not_uniform_data, granularity_current=None)
    assert_series_equal(not_uniform_data, res_data)

    # uniform
    uniform_data = pd.Series([1, np.NaN, 3, 4], index=pd.date_range("2020-07-14 01:00:01", periods=4, freq="1s"))
    res_data = resample(data=uniform_data, method="mean", granularity_next=pd.Timedelta("2s"))
    expected_res = pd.Series([1.0, 2.0, 4.0], index=pd.date_range("2020-07-14 01:00:00", periods=3, freq="2s"))
    expected_res = expected_res.asfreq(freq="2s")
    assert_series_equal(res_data, expected_res)
