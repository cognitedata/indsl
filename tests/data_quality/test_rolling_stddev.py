# Copyright 2022 Cognite AS

import numpy as np
import pandas as pd
import pytest

import indsl.data_quality.rolling_stddev

from indsl.exceptions import UserValueError


@pytest.mark.core
def test_rolling_stddev_time_delta():
    data = pd.Series(1, index=pd.date_range("2022-01-01 10:00:00", "2022-01-01 11:00:00", freq="15s")).sample(
        n=60, random_state=1
    )

    data = data.sort_index()

    stddev_expected = np.array(
        [
            0.00000,
            42.426407,
            52.678269,
            44.158804,
            40.527768,
            39.591667,
            45.355737,
            44.641429,
            43.588989,
            41.503012,
            39.68627,
            38.817756,
            37.93973,
            36.472178,
            35.767104,
            35.707142,
            37.873145,
            39.370039,
            37.534298,
            37.894979,
            37.187293,
            34.943564,
            47.101658,
            44.641429,
            44.570871,
            45.332108,
            45.332108,
            59.600947,
            59.638333,
            57.774666,
            57.226416,
            56.920998,
            57.258811,
            56.449646,
            56.524331,
            62.672619,
            74.595061,
            79.475554,
            78.193379,
            71.079341,
            73.824115,
            63.608035,
            66.11678,
            69.865943,
            71.344236,
            71.652571,
            70.443401,
            67.4242,
            67.510683,
            65.562863,
            46.179594,
            45.42655,
            44.809119,
            44.791182,
            44.6123,
            43.521462,
            43.362001,
            53.576155,
            52.775886,
            52.97058,
        ]
    )

    stddev_calculated = indsl.data_quality.rolling_stddev.rolling_stddev_timedelta(
        data, time_window=pd.Timedelta(minutes=15), min_periods=1
    ).values

    # Convert to seconds
    stddev_calculated_sec = stddev_calculated / 1000

    return np.testing.assert_array_almost_equal(
        stddev_calculated_sec,
        stddev_expected,
        decimal=5,
        err_msg="Calculated rolling standard deviation values do not match with the expected",
        verbose=True,
    )


@pytest.mark.core
def test_rolling_stddev_time_delta_without_time_unit():
    data = pd.Series(1, index=pd.date_range("2022-01-01 10:00:00", "2022-01-01 11:00:00", freq="15s")).sample(
        n=60, random_state=1
    )
    data.index = pd.to_datetime(data.index)

    data = data.sort_index()

    time_window = pd.Timedelta(15)

    return np.testing.assert_raises(
        UserValueError, indsl.data_quality.rolling_stddev.rolling_stddev_timedelta, data, time_window, 1
    )
