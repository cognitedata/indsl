# Copyright 2021 Cognite AS
import random

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from indsl.detect.utils import resample_timeseries


input_series_1 = pd.Series(
    data=np.array([1.0, 2.0, 5.0, 6.0, 8.0]),
    index=np.array(
        [
            datetime(2021, 10, 1, 10, 1, 0),
            datetime(2021, 10, 1, 10, 2, 0),
            datetime(2021, 10, 1, 10, 5, 0),
            datetime(2021, 10, 1, 10, 6, 0),
            datetime(2021, 10, 1, 10, 8, 0),
        ]
    ),
)
resampled_series_1_index = pd.date_range(
    start=datetime(2021, 10, 1, 10, 1, 0),
    end=datetime(2021, 10, 1, 10, 8, 0),
    freq=pd.DateOffset(seconds=60.0),
)
input_series_2 = pd.Series(
    data=np.array([1.0, 1.0, 2.0, 3.0, 6.0, 5.0, 5.0, 3.5, 3.2, 3.1, 1.0]),
    index=np.array(
        [
            datetime(2021, 10, 1, 10, 2, 0),
            datetime(2021, 10, 1, 10, 4, 0),
            datetime(2021, 10, 1, 10, 5, 0),
            datetime(2021, 10, 1, 10, 8, 0),
            datetime(2021, 10, 1, 10, 10, 0),
            datetime(2021, 10, 1, 10, 11, 0),
            datetime(2021, 10, 1, 10, 12, 0),
            datetime(2021, 10, 1, 10, 14, 0),
            datetime(2021, 10, 1, 10, 15, 0),
            datetime(2021, 10, 1, 10, 17, 0),
            datetime(2021, 10, 1, 10, 18, 0),
        ]
    ),
)
resampled_series_2_index = pd.date_range(
    start=datetime(2021, 10, 1, 10, 2, 0),
    end=datetime(2021, 10, 1, 10, 18, 0),
    freq=pd.DateOffset(seconds=60.0),
)


class RNGContext:
    """Context manager that sets random seed and then returns to original state
    when exiting."""

    def __init__(self, seed: int = 20):
        self.seed = seed

    def __enter__(self):
        self.start_state_numpy = np.random.get_state()
        self.start_state_random_module = random.getstate()
        np.random.seed(self.seed)
        random.seed(self.seed)

    def __exit__(self, exc_type, exc_value, traceback):
        np.random.set_state(self.start_state_numpy)
        random.setstate(self.start_state_random_module)


@pytest.mark.parametrize(
    "data, expected, is_step",
    [
        (
            input_series_1,
            pd.Series(data=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]), index=resampled_series_1_index),
            False,
        ),
        (
            input_series_2,
            pd.Series(
                data=np.array(
                    [
                        1.0,
                        1.0,
                        1.0,
                        2.0,
                        2.333333333333333333333333,
                        2.666666666666666666666667,
                        3.0,
                        4.5,
                        6.0,
                        5.0,
                        5.0,
                        4.25,
                        3.5,
                        3.2,
                        3.15,
                        3.1,
                        1.0,
                    ]
                ),
                index=resampled_series_2_index,
            ),
            False,
        ),
        (
            input_series_1,
            pd.Series(data=np.array([1.0, 2.0, 2.0, 2.0, 5.0, 6.0, 6.0, 8.0]), index=resampled_series_1_index),
            True,
        ),
        (
            input_series_2,
            pd.Series(
                data=np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 6.0, 5.0, 5.0, 5.0, 3.5, 3.2, 3.2, 3.1, 1.0]),
                index=resampled_series_2_index,
            ),
            True,
        ),
    ],
)
def test_resample_timeseries(data, expected, is_step):
    """Unit test for the utility method resample_timeseries."""
    # call method
    resampled_timeseries = resample_timeseries(data=data, is_step=is_step)

    # assertions
    np.testing.assert_array_almost_equal(resampled_timeseries, expected)
