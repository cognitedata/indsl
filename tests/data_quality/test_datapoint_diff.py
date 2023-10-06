# Copyright 2022 Cognite AS
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from indsl.data_quality.datapoint_diff import datapoint_diff_over_time_period
from indsl.exceptions import UserTypeError, UserValueError


@pytest.mark.parametrize(
    "data, time_period, difference_threshold, tolerance, expected",
    [
        (
            pd.Series(
                data=np.array([20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0]),
                index=np.array(
                    [
                        datetime(2021, 10, 1, 10, 0, 0),
                        datetime(2021, 10, 1, 11, 0, 0),
                        datetime(2021, 10, 1, 12, 0, 0),
                        datetime(2021, 10, 1, 13, 0, 0),
                        datetime(2021, 10, 1, 14, 0, 0),
                        datetime(2021, 10, 1, 15, 0, 0),
                        datetime(2021, 10, 1, 16, 0, 0),
                        datetime(2021, 10, 1, 17, 0, 0),
                    ]
                ),
            ),
            pd.Timedelta("4H"),
            4,
            pd.Timedelta("1H"),
            pd.Series(
                data=np.array([0.0, 0.0]),
                index=np.array(
                    [
                        datetime(2021, 10, 1, 10, 0, 0),
                        datetime(2021, 10, 1, 17, 0, 0),
                    ]
                ),
            ),
        ),
        (
            pd.Series(
                data=np.array(
                    [
                        20.0,
                        10.0,
                        22.0,
                        23.0,
                        24.0,
                        25.0,
                        26.0,
                        27.0,
                        28.0,
                        29.0,
                        30.0,
                        31.0,
                        32.0,
                        33.0,
                        34.0,
                        35.0,
                        36.0,
                        37.0,
                        38.0,
                        39.0,
                        40.0,
                        41.0,
                        42.0,
                        42.0,
                        43.0,
                        45.0,
                        46.0,
                    ]
                ),
                index=np.array(
                    [
                        datetime(2021, 10, 1, 10, 0, 0),
                        datetime(2021, 10, 1, 11, 0, 0),
                        datetime(2021, 10, 1, 12, 0, 0),
                        datetime(2021, 10, 1, 13, 0, 0),
                        datetime(2021, 10, 1, 14, 0, 0),
                        datetime(2021, 10, 1, 15, 0, 0),
                        datetime(2021, 10, 1, 16, 0, 0),
                        datetime(2021, 10, 1, 17, 0, 0),
                        datetime(2021, 10, 1, 18, 0, 0),
                        datetime(2021, 10, 1, 19, 0, 0),
                        datetime(2021, 10, 1, 20, 0, 0),
                        datetime(2021, 10, 1, 21, 0, 0),
                        datetime(2021, 10, 1, 22, 0, 0),
                        datetime(2021, 10, 1, 23, 0, 0),
                        datetime(2021, 10, 2, 00, 0, 0),
                        datetime(2021, 10, 2, 1, 0, 0),
                        datetime(2021, 10, 2, 2, 0, 0),
                        datetime(2021, 10, 2, 3, 0, 0),
                        datetime(2021, 10, 2, 4, 0, 0),
                        datetime(2021, 10, 2, 5, 0, 0),
                        datetime(2021, 10, 2, 6, 0, 0),
                        datetime(2021, 10, 2, 7, 0, 0),
                        datetime(2021, 10, 2, 8, 0, 0),
                        datetime(2021, 10, 2, 9, 0, 0),
                        datetime(2021, 10, 2, 10, 0, 0),
                        datetime(2021, 10, 2, 11, 0, 0),
                        datetime(2021, 10, 2, 12, 0, 0),
                    ]
                ),
            ),
            pd.Timedelta("1D"),
            24,
            pd.Timedelta("1H"),
            pd.Series(
                data=np.array([0.0, 0.0, 1.0, 1.0, 0.0]),
                index=np.array(
                    [
                        datetime(2021, 10, 1, 10, 0, 0),
                        datetime(2021, 10, 1, 10, 59, 59),
                        datetime(2021, 10, 1, 11, 0, 0),
                        datetime(2021, 10, 1, 11, 59, 59),
                        datetime(2021, 10, 1, 12, 0, 0),
                    ]
                ),
            ),
        ),
        (
            pd.Series(
                data=np.array(
                    [35.0, 10.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 36.0, 31.0, 32.0, 33.0, 34.0]
                ),
                index=np.array(
                    [
                        datetime(2021, 10, 1, 10, 0, 0),
                        datetime(2021, 10, 1, 11, 0, 0),
                        datetime(2021, 10, 1, 12, 0, 0),
                        datetime(2021, 10, 1, 13, 0, 0),
                        datetime(2021, 10, 1, 14, 0, 0),
                        datetime(2021, 10, 1, 15, 0, 0),
                        datetime(2021, 10, 1, 16, 0, 0),
                        datetime(2021, 10, 1, 17, 0, 0),
                        datetime(2021, 10, 1, 18, 0, 0),
                        datetime(2021, 10, 1, 19, 0, 0),
                        datetime(2021, 10, 1, 20, 0, 0),
                        datetime(2021, 10, 1, 21, 0, 0),
                        datetime(2021, 10, 1, 22, 0, 0),
                        datetime(2021, 10, 1, 23, 0, 0),
                        datetime(2021, 10, 2, 00, 0, 0),
                    ]
                ),
            ),
            pd.Timedelta("5H"),
            5,
            pd.Timedelta("1H"),
            pd.Series(
                data=np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]),
                index=np.array(
                    [
                        datetime(2021, 10, 1, 10, 0, 0),
                        datetime(2021, 10, 1, 15, 59, 59),
                        datetime(2021, 10, 1, 16, 0, 0),
                        datetime(2021, 10, 1, 16, 59, 59),
                        datetime(2021, 10, 1, 17, 0, 0),
                        datetime(2021, 10, 1, 19, 59, 59),
                        datetime(2021, 10, 1, 20, 0, 0),
                        datetime(2021, 10, 1, 20, 59, 59),
                        datetime(2021, 10, 1, 21, 0, 0),
                        datetime(2021, 10, 1, 0, 0, 0),
                    ]
                ),
            ),
        ),
        (
            pd.Series(
                data=np.array(
                    [35.0, 10.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 36.0, 31.0, 32.0, 33.0, 34.0]
                ),
                index=np.array(
                    [
                        datetime(2021, 10, 1, 10, 0, 0),
                        datetime(2021, 10, 1, 11, 0, 0),
                        datetime(2021, 10, 1, 12, 0, 0),
                        datetime(2021, 10, 1, 13, 0, 0),
                        datetime(2021, 10, 1, 14, 0, 0),
                        datetime(2021, 10, 1, 15, 0, 0),
                        datetime(2021, 10, 1, 16, 0, 0),
                        datetime(2021, 10, 1, 17, 0, 0),
                        datetime(2021, 10, 1, 18, 0, 0),
                        datetime(2021, 10, 1, 19, 0, 0),
                        datetime(2021, 10, 1, 20, 0, 0),
                        datetime(2021, 10, 1, 21, 0, 0),
                        datetime(2021, 10, 1, 22, 0, 0),
                        datetime(2021, 10, 1, 23, 0, 0),
                        datetime(2021, 10, 2, 00, 0, 0),
                    ]
                ),
            ),
            pd.Timedelta("5H"),
            10,
            pd.Timedelta("1H"),
            pd.Series(
                data=np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]),
                index=np.array(
                    [
                        datetime(2021, 10, 1, 10, 0, 0),
                        datetime(2021, 10, 1, 15, 59, 59),
                        datetime(2021, 10, 1, 16, 0, 0),
                        datetime(2021, 10, 1, 16, 59, 59),
                        datetime(2021, 10, 1, 17, 0, 0),
                        datetime(2021, 10, 1, 19, 59, 59),
                        datetime(2021, 10, 1, 20, 0, 0),
                        datetime(2021, 10, 1, 20, 59, 59),
                        datetime(2021, 10, 1, 21, 0, 0),
                        datetime(2021, 10, 1, 0, 0, 0),
                    ]
                ),
            ),
        ),
        (
            pd.Series(
                data=np.array([35.0, 36.0]),
                index=np.array(
                    [
                        datetime(2021, 10, 1, 10, 0, 0),
                        datetime(2021, 10, 1, 11, 0, 0),
                    ]
                ),
            ),
            pd.Timedelta("2H"),
            2,
            pd.Timedelta("1H"),
            pd.Series(
                data=np.array([0.0, 0.0]),
                index=np.array(
                    [
                        datetime(2021, 10, 1, 10, 0, 0),
                        datetime(2021, 10, 1, 11, 0, 0),
                    ]
                ),
            ),
        ),
    ],
)
def test_datapoint_diff_last_x_hours(data, time_period, difference_threshold, tolerance, expected):
    hour_count_check_ts = datapoint_diff_over_time_period(
        data=data, time_period=time_period, difference_threshold=difference_threshold, tolerance=tolerance
    )

    np.testing.assert_array_equal(hour_count_check_ts, expected)


def test_datapoint_diff_no_time_index():
    with pytest.raises(UserTypeError, match="Expected a time series, got index type int64"):
        datapoint_diff_over_time_period(pd.Series(dtype=np.float64), pd.Timedelta("10H"), 10, pd.Timedelta("1H"))


def test_datapoint_diff_no_data():
    with pytest.raises(UserValueError, match="Time series is empty."):
        datapoint_diff_over_time_period(
            pd.Series(index=pd.date_range("2022-01-01 10:00:00", "2022-01-01 10:29:00", periods=0), dtype=np.float64),
            pd.Timedelta("5H"),
            5,
            pd.Timedelta("1H"),
        )


def test_datapoint_diff_min_length():
    with pytest.raises(UserValueError, match="Expected series with length >= 2, got length 1"):
        datapoint_diff_over_time_period(
            pd.Series(index=pd.date_range("2022-01-01 10:00:00", "2022-01-01 10:29:00", periods=1), dtype=np.float64),
            pd.Timedelta("1H"),
            1,
            pd.Timedelta("1H"),
        )


def test_datapoint_diff_timedelta_unit():
    with pytest.raises(UserValueError, match="Unit of timedelta should be in days, hours, minutes or seconds"):
        datapoint_diff_over_time_period(
            pd.Series(index=pd.date_range("2022-01-01 10:00:00", "2022-01-01 11:00:00", periods=6), dtype=np.float64),
            pd.Timedelta("0.1s"),
            1,
            pd.Timedelta("1H"),
        )
