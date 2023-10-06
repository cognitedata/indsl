# Copyright 2021 Cognite AS
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from indsl.exceptions import UserValueError
from indsl.ts_utils.numerical_calculus import differentiate, sliding_window_integration, trapezoidal_integration


@pytest.mark.core
@pytest.mark.parametrize(
    "data_type,granularity,expected_result",
    [
        ("constant", "1h", [0.0, 0.0, 0.0]),
        ("constant", "12h", [0.0, 0.0, 0.0]),
        ("linear", "1h", [10.0 / 12 / 2, 10.0 / 12 / 2, 10.0 / 12 / 2]),
        ("linear", "12h", [5.0, 5.0, 5.0]),
    ],
)
def test_differentiate(create_data, data_type, granularity, expected_result):
    data = create_data[data_type]
    res = differentiate(data, pd.Timedelta(granularity))
    assert (res == expected_result).all()


@pytest.mark.core
@pytest.mark.parametrize("periods", [0, 1])
def test_differentiate_fails(periods):
    index = pd.date_range(start="01-01-1970 00:00:00", periods=periods, end="01-02-1970 00:0:00")
    data = pd.Series([0] * periods, index=index, dtype=np.float64)

    with pytest.raises(UserValueError):
        differentiate(data, pd.Timedelta("1h"))


@pytest.mark.core
@pytest.mark.parametrize(
    "data_type,granularity,expected_result",
    [
        ("constant", "1h", [0.0, 12.0, 24.0]),
        ("constant", "12h", [0.0, 1.0, 2.0]),
        ("linear", "1h", [0.0, 30, 120.0]),
        ("linear", "12h", [0.0, 2.5, 10.0]),
    ],
)
def test_trapezoidal_integration(create_data, data_type, granularity, expected_result):
    data = create_data[data_type]
    res = trapezoidal_integration(data, pd.Timedelta(granularity))
    assert (res == expected_result).all()


@pytest.mark.core
def test_trapezoidal_integration_fails():
    data = pd.Series([], dtype=np.float64)

    with pytest.raises(UserValueError):
        trapezoidal_integration(data, pd.Timedelta("1h"))


@pytest.mark.extras
@pytest.mark.parametrize(
    "datapoints,datapoint_freq,window_length,integrand_rate,result",
    [
        (2 * 60 * 60, "1s", pd.Timedelta("1 h"), pd.Timedelta("1 h"), 1.0),
        (2 * 60 * 6, "10s", pd.Timedelta("1 h"), pd.Timedelta("1 h"), 1.0),
        (2 * 60, "60s", pd.Timedelta("1 h"), pd.Timedelta("1 h"), 1.0),
    ],
)
def test_sliding_window_integration(datapoints, datapoint_freq, window_length, integrand_rate, result):
    y = np.ones(datapoints)
    series = pd.Series(y, index=pd.date_range(start=datetime(2000, 1, 1), periods=datapoints, freq=datapoint_freq))
    res = sliding_window_integration(series, window_length, integrand_rate)  #
    assert res[len(res) - 1] == pytest.approx(result)


@pytest.mark.extras
@pytest.mark.parametrize(
    "num_datapoints,window_length,integrand_rate,freq,error",
    [
        (0, pd.Timedelta("1h"), pd.Timedelta("1s"), "1s", UserValueError),
        (60, pd.Timedelta("1h"), pd.Timedelta("1s"), "1s", UserValueError),
        (600, pd.Timedelta("-1s"), pd.Timedelta("-1h"), "1s", UserValueError),
        (600, pd.Timedelta("10s"), pd.Timedelta("-1h"), "1s", UserValueError),
        (60, pd.Timedelta("1s"), pd.Timedelta("1h"), "10s", UserValueError),
    ],
)
def test_failures(num_datapoints, window_length, integrand_rate, freq, error):
    y = np.ones(num_datapoints)
    series = pd.Series(y, index=pd.date_range(start=datetime(2000, 1, 1), periods=num_datapoints, freq=freq))
    with pytest.raises(error):
        sliding_window_integration(series, window_length, integrand_rate)
