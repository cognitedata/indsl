# Copyright 2024 Cognite AS
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal
from indsl.signals.user_specified_timeseries import user_specified_timeseries


@pytest.mark.core
def test_user_specified_timeseries():
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
