# Copyright 2023 Cognite AS
from typing import Optional

import numpy as np
import pandas as pd

from indsl import versioning
from indsl.decorators import jit
from indsl.ts_utils.ts_utils import time_parse
from indsl.type_check import check_types


@versioning.register(version="1.0", deprecated=True)
@check_types
def lwma(
    data: pd.Series,
    time_window: str = "60min",
    min_periods: int = 1,
    resample: bool = False,
    resample_window: str = "60min",
) -> pd.Series:
    """Linear weighted moving average.

    The linear weighted moving average gives more weight to the more recent observations and gradually less to the older
    ones.

    Args:
        data: Time series.
        time_window: Time window.
            Length of the time period to compute the rolling mean. Defaults to '60min'.
            If the user gives a number without unit (such as '60'), it will be considered as the number of minutes.
            Accepted string format: '3w', '10d', '5h', '30min', '10s'.
        min_periods: Minimum samples.
            Minimum number of observations in the time window required to estimate a value (otherwise result is NA).
            Defaults to 1.
        resample: Resample.
            Resamples the calculated linear weighted moving average series. Defaults to False
        resample_window: Resampling window.
            Time window used to resample the calculated linear weighted moving average series. Defaults to '60min'.

    Returns:
        pandas.Series: Smoothed time series.
    """
    time_window_ = time_parse(time_window)
    lwma = np.zeros_like(data)
    win_pts_lst = np.array(data.rolling(time_window_).count().values.astype(int))

    for i in range(len(data)):
        lwma[i] = get_lwma_val(data.values, win_pts_lst, i, min_periods=min_periods)
    lwma_series = pd.Series(lwma, index=data.index)
    if resample:
        resample_window_ = time_parse(resample_window)
        lwma_series = lwma_series.resample(resample_window_).mean()
    return lwma_series


@jit(nopython=True)
def get_lwma_val(data: np.ndarray, win_pts_lst: np.ndarray, i: int, min_periods: int) -> Optional[float]:
    """Calculates the Linear Weighted Moving Average for the current datapoint.

    Args:
        data (numpy.ndarray): Data values of the timeseries.
        win_pts_lst (numpy.ndarray): Array with the number of points for each time window.
        i (int): Index of the current value.
        min_periods (int): Minimum number of observations in window required to have a value (otherwise result is NA).

    Returns:
        None: Length of data was smaller than minimum number of observations in window required to have a value.
        float: Calculated lwma for the current datapoint.
    """
    if len(data) < min_periods:
        return None
    else:
        win_pts = win_pts_lst[i]
        win_values = data[(i + 1) - win_pts : i + 1]
        weights = np.linspace(1, min(len(data), win_pts), min(len(data), win_pts))
        lwma_val = np.dot(win_values, weights) / weights.sum()
    return lwma_val
