# Copyright 2024 Cognite AS
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal
from indsl.signals.user_specified_timeseries import user_specified_timeseries


@pytest.mark.core
def test_user_specified_timeseries():
    """Timestamp given as Unix time in sec"""
    import pickle

    input_time = np.arange(1722470401, 1722643201, 60)  # every minute for two days
    input_y = np.sin((input_time - input_time[0]) / (input_time[-1] - input_time[0]) * 2 * np.pi)
    ts_series = user_specified_timeseries(input_time.tolist(), input_y.tolist())

    # Manually create the series
    # Convert input timestamps to datetime
    timestamps = []
    for time_ in input_time:
        timestamps.append(pd.Timestamp(time_, unit="s"))
    ts_compare = pd.Series(input_y, index=timestamps)

    assert_series_equal(ts_series, ts_compare)


@pytest.mark.core
def test_user_specified_timeseries_2():
    """Timestamp given as a string in datetime format"""
    import pickle

    input_time = np.arange(1722470401, 1722643201, 60)  # every minute for two days
    input_y = np.sin((input_time - input_time[0]) / (input_time[-1] - input_time[0]) * 2 * np.pi)

    input_time_str = [pd.Timestamp(time_, unit="s").strftime("%Y-%m-%d %X") for time_ in input_time]
    # print(tmp)

    ts_series = user_specified_timeseries(input_time_str, input_y.tolist())

    # Manually create the series
    # Convert input timestamps to datetime
    timestamps = []
    for time_ in input_time_str:
        timestamps.append(pd.Timestamp(time_))
    ts_compare = pd.Series(input_y, index=timestamps)

    assert_series_equal(ts_series, ts_compare)


@pytest.mark.core
def test_user_specified_timeseries_3():
    """Timestamp given as a string in datetime different format"""
    import pickle

    input_time_str = [
        "8-1-24 0:00",
        "8-1-24 0:10",
        "8-1-24 0:20",
        "8-1-24 0:30",
        "8-1-24 0:40",
        "8-1-24 0:50",
        "8-1-24 1:00",
        "8-1-24 1:10",
        "8-1-24 1:20",
        "8-1-24 1:30",
        "8-1-24 1:40",
        "8-1-24 1:50",
        "8-1-24 2:00",
        "8-1-24 2:10",
        "8-1-24 2:20",
        "8-1-24 2:30",
        "8-1-24 2:40",
        "8-1-24 2:50",
    ]
    input_time_pd = [pd.Timestamp(time_) for time_ in input_time_str]  #
    # print(input_time_pd)
    # Convert to timestamps
    input_time = np.array([time_.timestamp() for time_ in input_time_pd])
    input_y = np.sin((input_time - input_time[0]) / (input_time[-1] - input_time[0]) * 2 * np.pi)

    ts_series = user_specified_timeseries(input_time_str, input_y.tolist())

    # Manually create the series
    # Convert input timestamps to datetime
    timestamps = []
    for time_ in input_time_str:
        timestamps.append(pd.Timestamp(time_))
    ts_compare = pd.Series(input_y, index=input_time_pd)

    assert_series_equal(ts_series, ts_compare)


# test_user_specified_timeseries_3()
