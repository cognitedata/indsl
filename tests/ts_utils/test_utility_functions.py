# Copyright 2021 Cognite AS
from datetime import datetime, timedelta
from typing import get_args

import numpy as np
import pandas as pd
import pytest

from pandas.testing import assert_series_equal

from indsl.exceptions import UserValueError
from indsl.ts_utils import get_timestamps, set_timestamps, time_shift, union
from indsl.ts_utils.utility_functions import (
    TimeUnits,
    create_series_from_timesteps,
    generate_step_series,
    normality_assumption_test,
)

from ..generate_data import create_uniform_data


supported_time_units = get_args(TimeUnits)


@pytest.mark.core
@pytest.mark.parametrize("time_unit", supported_time_units)
def test_get_timestamps(time_unit):
    data = create_uniform_data(np.ones(3))
    timestamps = get_timestamps(data, time_unit).to_numpy()
    index = pd.to_datetime(timestamps, unit=time_unit)

    # Precision loss due to conversion to float64 in get_timestamps
    assert np.all(np.abs(index - data.index) < pd.Timedelta(500))


@pytest.mark.core
@pytest.mark.parametrize("time_unit", supported_time_units)
def test_set_timestamps(time_unit):
    timestamps = np.array([0.0, 1637611275, 3275222550]) * 1e9 / pd.Timedelta(1, unit=time_unit).value
    index = pd.to_datetime(timestamps, unit=time_unit)
    data = create_uniform_data(np.ones(3))

    assert not all(index == data.index)

    timestamp_ts = pd.Series(timestamps)
    out = set_timestamps(timestamp_ts, data, unit=time_unit)

    assert all(index == out.index)


@pytest.mark.core
@pytest.mark.parametrize("time_unit", supported_time_units)
def test_time_shift(time_unit):
    data = create_uniform_data(np.ones(3))
    data_shift = time_shift(data, 10, time_unit)

    assert all(data_shift.index - data.index == pd.Timedelta(10, unit=time_unit))


@pytest.mark.core
def test_union():
    series1 = pd.Series([0, 1, 2], index=pd.date_range("2021-01-01 00:00:00", periods=3, freq="2s"))
    series2 = pd.Series([10, 11, 12], index=pd.date_range("2021-01-01 00:00:00", periods=3, freq="s"))

    out = union(series1, series2)
    assert all(out.to_numpy() == [0, 11, 1, 2])


@pytest.mark.core
@pytest.mark.parametrize(
    "input, expected_result",
    [
        (
            pd.Series([0, 0, 1, 1, 0, 1], index=pd.date_range("2020-01-01", periods=6, freq="10min")),
            pd.Series(
                [0, 0, 1, 1, 0, 0],
                index=pd.DatetimeIndex(
                    [
                        "2020-01-01 00:00:00.000",
                        "2020-01-01 00:19:59.999",
                        "2020-01-01 00:20:00.000",
                        "2020-01-01 00:39:59.999",
                        "2020-01-01 00:40:00.000",
                        "2020-01-01 00:49:59.999",
                    ]
                ),
            ),
        ),
        (
            pd.Series([1, 0, 0, 0, 1, 1, 1], index=pd.date_range("2020-01-01", periods=7, freq="10min")),
            pd.Series(
                [1, 1, 0, 0, 1, 1],
                index=pd.DatetimeIndex(
                    [
                        "2020-01-01 00:00:00.000",
                        "2020-01-01 00:09:59.999",
                        "2020-01-01 00:10:00.000",
                        "2020-01-01 00:39:59.999",
                        "2020-01-01 00:40:00.000",
                        "2020-01-01 01:00:00.000",
                    ]
                ),
            ),
        ),
        (
            pd.Series([1, 0, 0, 0, 1], index=pd.date_range("2020-01-01", periods=5, freq="10min")),
            pd.Series(
                [1, 1, 0, 0],
                index=pd.DatetimeIndex(
                    [
                        "2020-01-01 00:00:00.000",
                        "2020-01-01 00:09:59.999",
                        "2020-01-01 00:10:00.000",
                        "2020-01-01 00:39:59.999",
                    ]
                ),
            ),
        ),
        (
            pd.Series([0, 0, 1, 0], index=pd.date_range("2020-01-01", periods=4, freq="10min")),
            pd.Series(
                [0, 0, 1, 1, 0],
                index=pd.DatetimeIndex(
                    [
                        "2020-01-01 00:00:00.000",
                        "2020-01-01 00:19:59.999 ",
                        "2020-01-01 00:20:00.000",
                        "2020-01-01 00:29:59.999",
                        "2020-01-01 00:30:00.000",
                    ]
                ),
            ),
        ),
        (
            pd.Series([1, 1], index=pd.date_range("2020-01-01", periods=2, freq="10min")),
            pd.Series(
                [1, 1],
                index=pd.DatetimeIndex(
                    [
                        "2020-01-01 00:00:00.000",
                        "2020-01-01 00:10:00.000",
                    ]
                ),
            ),
        ),
    ],
)
def test_generate_step_series(input, expected_result):
    step_series = generate_step_series(input)
    assert_series_equal(step_series, expected_result, check_freq=False)


@pytest.mark.core
def test_create_series_from_timesteps():
    expected_timestamps = [
        datetime.fromisoformat("2021-01-01 00:00:00.000000"),
        datetime.fromisoformat("2021-01-01 00:00:01.000000"),
        datetime.fromisoformat("2021-01-01 00:00:01.000278"),
        datetime.fromisoformat("2021-01-01 00:00:02.000278"),
    ]
    expected_series = pd.Series(np.zeros(len(expected_timestamps)), index=expected_timestamps)

    second = timedelta(seconds=1)
    timesteps = 1 * [second] + [1.0 / 60 / 60 * second] + 1 * [second]
    x = create_series_from_timesteps(timesteps)
    assert_series_equal(x, expected_series)


@pytest.mark.core
def test_normality_assumption_validation():
    data = pd.Series(np.random.randn(2))
    with pytest.raises(UserValueError):
        normality_assumption_test(data)
