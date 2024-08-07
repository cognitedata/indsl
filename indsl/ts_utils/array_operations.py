import numpy as np

# Simple operations
import pandas as pd

from indsl.type_check import check_types
from indsl.validations import validate_series_is_not_empty


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

    # Check if the time series is empty
    validate_series_is_not_empty(data)

    # We need to convert the datetime to timestamp
    timestamps = np.array([val.timestamp() for val in data.index])
    # We normalise as well to reduce rounding errors, since the timestamp is usually a very large number
    timestamps = timestamps - timestamps[0]
    timestamps = timestamps / timestamps[-1]

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

    # Check if the time series is empty
    validate_series_is_not_empty(data)

    # We need to convert the datetime to timestamp.
    timestamps = np.array([val.timestamp() for val in data.index])
    # We normalise as well to reduce rounding errors, since the timestamp is usually a very large number
    timestamps = timestamps - timestamps[0]
    timestamps = timestamps / timestamps[-1]

    # first get the mean value
    # integrate over the time series
    timeseries_sum = trapezoid(data.values, x=timestamps)
    # scale by the time range for the integration, giving the time weighted average
    timeseries_average = timeseries_sum / (timestamps[-1] - timestamps[0])

    # Then calculate the time-weighted standard deviation
    timeseries_std = np.sqrt(
        trapezoid((data.values - timeseries_average) ** 2, x=timestamps) / (timestamps[-1] - timestamps[0])
    )

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
    # Check if the time series is empty
    validate_series_is_not_empty(data)

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
    # Check if the time series is empty
    validate_series_is_not_empty(data)

    # Extract the largest value
    timeseries_max = data.max()

    # We need to convert it into a pandas Series
    timeseries_max_series = pd.Series(timeseries_max * np.ones(data.shape), index=data.index)

    return timeseries_max_series
