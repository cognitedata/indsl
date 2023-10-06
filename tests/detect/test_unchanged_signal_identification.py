# Copyright 2021 Cognite AS
import random

import numpy as np
import numpy.random
import pandas as pd
import pytest

from indsl.detect.unchanged_signal_detector import unchanged_signal_detector
from indsl.exceptions import UserTypeError, UserValueError
from indsl.ts_utils.ts_utils import number_of_events


def get_data(start_date, end_date, size):
    # generate random time series
    ts_values = np.random.uniform(1, 80, size=size)
    data = pd.Series(ts_values, index=pd.date_range(start_date, end_date, periods=size), name="value")

    data = data.sort_index()

    return data


@pytest.mark.core
@pytest.mark.parametrize(
    "data, duration, min_nr_data_points, nr_consecutive_data_points, expected_nr_of_events",
    [
        (
            get_data("2022-01-01 10:00:00", "2022-01-01 11:59:00", 60),
            pd.Timedelta(minutes=30),
            3,
            8,
            0,
        ),
        (
            get_data("2022-01-01 10:00:00", "2022-01-01 11:59:00", 120),
            pd.Timedelta(minutes=10),
            3,
            12,
            1,
        ),
        (
            get_data("2022-01-01 10:00:00", "2022-01-01 11:59:00", 120),
            pd.Timedelta(minutes=10),
            5,
            10,
            0,
        ),
        (
            get_data("2022-01-01 10:00:00", "2022-01-01 11:59:00", 120),
            pd.Timedelta(minutes=9),
            5,
            10,
            1,
        ),
        (
            get_data("2022-01-01 10:00:00", "2022-01-01 10:59:00", 60),
            pd.Timedelta(minutes=1),
            4,
            4,
            1,
        ),
        (
            get_data("2022-01-01 10:00:00", "2022-01-01 10:29:00", 30),
            pd.Timedelta(minutes=1),
            2,
            3,
            1,
        ),
        (
            get_data("2022-01-01 10:00:00", "2022-01-01 10:29:00", 30),
            pd.Timedelta(minutes=1),
            3,
            2,
            0,
        ),
    ],
)
def test_unchanged_signal(data, duration, min_nr_data_points, nr_consecutive_data_points, expected_nr_of_events):
    # generate a random signal value and set the same value for consecutive data points starting at one random position
    random_signal_value = np.random.uniform(1, 80)

    start_position = random.randint(0, int(len(data) - nr_consecutive_data_points))
    data.values[start_position : start_position + nr_consecutive_data_points] = random_signal_value

    # calculate if the signal value has stayed constant for longer than the given duration.
    calculated_ts = unchanged_signal_detector(data, duration, min_nr_data_points)

    assert number_of_events(calculated_ts) == expected_nr_of_events


def test_unchanged_signal_no_time_index():
    pytest.raises(UserTypeError, unchanged_signal_detector, pd.Series(dtype=np.float64), pd.Timedelta(minutes=10), 4)


def test_unchanged_signal_no_data():
    pytest.raises(
        UserValueError,
        unchanged_signal_detector,
        pd.Series(index=pd.date_range("2022-01-01 10:00:00", "2022-01-01 10:29:00", periods=0), dtype=np.float64),
        pd.Timedelta(minutes=10),
        3,
    )
