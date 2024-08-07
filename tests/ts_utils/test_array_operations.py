# Copyright 2024 Cognite AS
import numpy as np
import pandas as pd
import math
import pytest

from indsl.ts_utils import time_weighted_mean, time_weighted_std, timeseries_min, timeseries_max


@pytest.mark.core
def test_time_weighted_mean():
    WHP_series = pd.read_pickle("./tests/ts_utils/pd_series_WHP.pkl")

    from scipy.integrate import trapezoid

    # We need to convert the datetime to timestamp
    timestamps = np.array([val.timestamp() for val in WHP_series.index])
    # integrate over the time series to get the sum
    timeseries_sum = trapezoid(WHP_series.values, x=timestamps)
    # scale by the time range for the integration, giving the time weighted average
    timeseries_average = timeseries_sum / (timestamps[-1] - timestamps[0])

    time_weighted_mean_indsl = time_weighted_mean(WHP_series)
    assert math.isclose(time_weighted_mean_indsl.values[0] - timeseries_average, 0, abs_tol=1e-8)


@pytest.mark.core
def test_time_weighted_std():
    WHP_series = pd.read_pickle("./tests/ts_utils/pd_series_WHP.pkl")

    from scipy.integrate import trapezoid

    # We need to convert the datetime to timestamp
    timestamps = np.array([val.timestamp() for val in WHP_series.index])

    # first get the mean value
    # integrate over the time series to get the sum
    timeseries_sum = trapezoid(WHP_series.values, x=timestamps)
    # scale by the time range for the integration, giving the time weighted average
    timeseries_average = timeseries_sum / (timestamps[-1] - timestamps[0])

    # Then calculate the stadard deviation
    timeseries_std = np.sqrt(
        trapezoid((WHP_series.values - timeseries_average) ** 2, x=timestamps) / (timestamps[-1] - timestamps[0])
    )

    time_weighted_std_indsl = time_weighted_std(WHP_series)
    assert math.isclose(time_weighted_std_indsl.values[0] - timeseries_std, 0, abs_tol=1e-8)


@pytest.mark.core
def test_timeseries_min():
    WHP_series = pd.read_pickle("./tests/ts_utils/pd_series_WHP.pkl")

    timeseries_min_test = WHP_series.min()

    timeseries_min_indsl = timeseries_min(WHP_series)
    assert math.isclose(timeseries_min_indsl.values[0] - timeseries_min_test, 0, abs_tol=1e-8)


@pytest.mark.core
def test_timeseries_max():
    WHP_series = pd.read_pickle("./tests/ts_utils/pd_series_WHP.pkl")

    timeseries_max_test = WHP_series.max()

    timeseries_max_indsl = timeseries_max(WHP_series)
    assert math.isclose(timeseries_max_indsl.values[0] - timeseries_max_test, 0, abs_tol=1e-8)
