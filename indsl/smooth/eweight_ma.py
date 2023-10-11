# Copyright 2023 Cognite AS
from typing import Optional

import numpy as np
import pandas as pd

from indsl import versioning
from indsl.decorators import jit
from indsl.type_check import check_types
from indsl.validations import validate_series_is_not_empty

from . import eweight_ma_v1  # noqa


@versioning.register(version="2.0", changelog="Update parameter type")
@check_types
def ewma(
    data: pd.Series,
    time_window: pd.Timedelta = pd.Timedelta("60min"),
    min_periods: int = 1,
    adjust: bool = True,
    max_pt: int = 200,
    resample: bool = False,
    resample_window: pd.Timedelta = pd.Timedelta("60min"),
) -> pd.Series:
    """Exp. weighted moving average.

    The exponential moving average gives more weight to the more recent observations. The weights fall exponentially
    as the data point gets older. It reacts more than the simple moving average with regards to recent movements.
    The moving average value is calculated following the definition yt=(1−α)yt−1+αxt if adjust = False or
    yt=(xt+(1−α)*xt−1+(1−α)^2*xt−2+...+(1−α)^t*x0) / (1+(1−α)+(1−α)^2+...+(1−α)^t) if adjust = True.

    Args:
        data: Time series.
            Data with a pd.DatetimeIndex.
        time_window: Time window.
            Defines how important the current observation is in the calculation of the EWMA. The longer the period, the slowly it adjusts to reflect current trends. Defaults to '60min'.
            Accepted string format: '3w', '10d', '5h', '30min', '10s'.
            The time window is converted to the number of points for each of the windows. Each time window may have different number of points if the timeseries is not regular.
            The number of points specify the decay of the exponential weights in terms of span α=2/(span+1), for span≥1.
        min_periods: Minimum number of data points.
            Minimum number of data points inside a time window required to have a value (otherwise result is NA). Defaults to 1.
            If min_periods > 1 and adjust is False, the SMA is computed for the first observation.
        adjust: Adjust.
            If true, the exponential function is calculated using weights w_i=(1−α)^i.
            If false, the exponential function is calculated recursively using yt=(1−α)yt−1+αxt. Defaults to True.
        max_pt: Maximum number of data points.
            Sets the maximum number of points to consider in a window if adjust = True. A high number of points will require more time to execute. Defaults to 200.
        resample: Resample.
            If true, resamples the calculated exponential moving average series. Defaults to False.
        resample_window: Resampling window
            Time window used to resample the calculated exponential moving average series. Defaults to '60min'.

    Returns:
        pandas.Series: Smoothed time series.
    """
    validate_series_is_not_empty(data)

    win_pts_lst = np.array(data.rolling(time_window).count().values)
    ewma_vals = np.zeros_like(data)
    a = 2.0 / (win_pts_lst + 1)
    if not adjust:
        for i, value in enumerate(data):
            ewma_vals[i] = ewma_pt_not_adjust(data.values, value, min_periods, ewma_vals, i, a)
    else:
        ewma_vals[0] = data.iloc[0]
        for i, value in enumerate(data):
            if i == 0:
                continue
            ewma_vals[i] = ewma_pt_adjust(data.values, value, i, a, max_pt)
    bool_mask = win_pts_lst < min_periods
    ewma_vals[bool_mask] = None
    ewma_series = pd.Series(ewma_vals, index=data.index)
    if resample:
        ewma_series = ewma_series.resample(resample_window).mean()
    return ewma_series


@jit(nopython=True)
def ewma_pt_adjust(data: np.ndarray, value: float, i: int, a: np.ndarray, max_pt: int = 200) -> float:
    """Calculates ewma value using weights wi=(1−a)^i.

    Args:
        data (numpy.ndarray): Data values of the time series.
        value (float): Current datapoint.
        i (int): Index of the current value.
        a (np.ndarray): Alpha value to compute ewma.
        max_pt (int): Maximum number of points to consider when calculating ewma. Defaults to 200.

    Returns:
        float: Calculated ewma for the current datapoint.
    """
    values = data[:i] if len(data[:i]) <= max_pt else data[i - max_pt : i]
    exp = np.arange(1, len(values) + 1)[::-1]
    w_den = (1 - a[i]) ** exp
    w_nom = w_den * values
    ewma_val = (value + w_nom.sum()) / (1 + w_den.sum())
    return ewma_val


def ewma_pt_not_adjust(
    data: np.ndarray, value: float, min_periods: int, ewma_vals: np.ndarray, i: int, a: np.ndarray
) -> Optional[float]:
    """Calculates the ewma recursively using yt=(1−α)yt−1+αxt.

    Args:
        data (numpy.ndarray): Data values of the time series.
        value (float): Current datapoint.
        min_periods (int): Minimum number of observations in window required to have a value (otherwise result is NA).
            If min_periods > 1 the SMA is computed for the first observation.
        ewma_vals (numpy.ndarray): Array of the ewma values.
        i (int): Index of the current value.
        a (np.ndarray): Alpha value to compute ewma.

    Returns:
        None: Index of the current value is smaller than minimum number of observations in window required to have a value.
        float: Calculated ewma for the current datapoint.
    """
    if i < min_periods - 1:
        return None
    elif i == min_periods - 1:  # first value is the simple moving average
        start_sma = data[:min_periods].sum() / len(data[:min_periods])
        return start_sma
    else:
        ewma_val = (a[i] * value) + ((1 - a[i]) * ewma_vals[i - 1])
        return ewma_val
