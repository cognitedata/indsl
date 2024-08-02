import numpy as np

# Simple operations
import pandas as pd

from indsl.exceptions import UserValueError
from indsl.type_check import check_types


@check_types
def time_weighted_mean(data: pd.Series) -> pd.Series:
    """Time weighted mean.

    The time weighted mean for a time series. The returned timeseries has a constant value

    Args:
        data: Time series.

    Returns:
        pandas.Series: Time weighted mean.
    """
    from scipy.integrate import trapezoid

    n = len(data)
    if n == 0:
        raise UserValueError("Expected at least one item in data parameter, got zero instead.")
    # We need to convert the datetime to timestamp
    timestamps = np.array([val.timestamp() for val in data.index])

    # integrate over the time series to get the sum
    timeseries_sum = trapezoid(data.values, x=timestamps)
    # scale by the time range for the integration, giving the time weighted average
    timeseries_average = timeseries_sum / (timestamps[-1] - timestamps[0])
    # We need to convert it into a pandas Series
    time_weighted_mean = pd.Series(timeseries_average * np.ones(data.shape), index=data.index)

    return time_weighted_mean


@check_types
def time_weighted_std(data: pd.Series) -> pd.Series:
    """Time weighted std.

    The time weighted standard deviation for a time series. The returned timeseries has a constant value


    Args:
        data: Time series.

    Returns:
        pandas.Series: Time weighted std.
    """
    from scipy.integrate import trapezoid

    n = len(data)
    if n == 0:
        raise UserValueError("Expected at least one item in data parameter, got zero instead.")

    # We need to convert the datetime to timestamp
    timestamps = np.array([val.timestamp() for val in data.index])

    # first get the mean value
    # integrate over the time series to get the sum
    timeseries_sum = trapezoid(data.values, x=timestamps)
    # scale by the time range for the integration, giving the time weighted average
    timeseries_average = timeseries_sum / (timestamps[-1] - timestamps[0])

    # Then calculate the stadard deviation
    timeseries_sum = trapezoid((data.values - timeseries_average), x=timestamps)

    # Then calculate the stadard deviation
    timeseries_std = np.sqrt(trapezoid((data.values - timeseries_average), x=timestamps))

    # We need to convert it into a pandas Series
    time_weighted_std = pd.Series(timeseries_std * np.ones(data.shape), index=data.index)

    return time_weighted_std


@check_types
def timeseries_min(data: pd.Series) -> pd.Series:
    """Min value for the time series.

    The returned timeseries has a constant value

    Args:
        data: Time series.

    Returns:
        pandas.Series: Min value timeseries.
    """
    n = len(data)
    if n == 0:
        raise UserValueError("Expected at least one item in data parameter, got zero instead.")

    # Extract the smallest value
    timeseries_min = data.min()

    # We need to convert it into a pandas Series
    timeseries_min_series = pd.Series(timeseries_min * np.ones(data.shape), index=data.index)

    return timeseries_min_series


@check_types
def timeseries_max(data: pd.Series) -> pd.Series:
    """Max value for the time series.

    The returned timeseries has a constant value

    Args:
        data: Time series.

    Returns:
        pandas.Series: Maxi value timeseries.
    """
    n = len(data)
    if n == 0:
        raise UserValueError("Expected at least one item in data parameter, got zero instead.")

    # Extract the largest value
    timeseries_max = data.max()

    # We need to convert it into a pandas Series
    timeseries_max_series = pd.Series(timeseries_max * np.ones(data.shape), index=data.index)

    return timeseries_max_series
