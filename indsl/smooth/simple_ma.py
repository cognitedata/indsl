# Copyright 2023 Cognite AS
import pandas as pd

from indsl import versioning
from indsl.type_check import check_types

from . import simple_ma_v1  # noqa


@versioning.register(version="2.0", changelog="Update parameter type")
@check_types
def sma(data: pd.Series, time_window: pd.Timedelta = pd.Timedelta("60min"), min_periods: int = 1) -> pd.Series:
    """Simple moving average (SMA).

    Plain simple average that computes the sum of the values of the observations in a time_window divided by the number of observations in the time_window.
    SMA time series are much less noisy than the original time series. However, SMA time series lag the original time series, which means that changes in the trend are only seen with a delay (lag) of time_window/2.

    Args:
        data: Time series.
        time_window: Window.
            Length of the time period to compute the average. Defaults to '60min'.
            Accepted string format: '3w', '10d', '5h', '30min', '10s'.
        min_periods: Minimum samples.
            Minimum number of observations in window required to have a value (otherwise result is NA). Defaults  to 1.

    Returns:
        pandas.Series: Smoothed time series
    """
    return data.rolling(time_window, min_periods=min_periods).mean()
