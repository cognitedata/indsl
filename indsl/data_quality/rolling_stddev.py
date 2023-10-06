# Copyright 2023 Cognite AS
import pandas as pd

from indsl.type_check import check_types
from indsl.validations import validate_series_has_time_index, validate_series_is_not_empty, validate_timedelta_unit


@check_types
def rolling_stddev_timedelta(
    data: pd.Series, time_window: pd.Timedelta = pd.Timedelta(minutes=15), min_periods: int = 1
) -> pd.Series:
    """Rolling stdev of time delta.

    Rolling standard deviation computed for the time deltas of the observations. This
    metric aims to measure the amount of variation or dispersion in the frequency of time series data points.

    Args:
        data: Time series.
        time_window: Time window.
            Length of the time period to compute the standard deviation for. Defaults to 'minutes=15'.
            Time unit should be in days, hours, minutes, or seconds. Accepted formats can be found here
            https://pandas.pydata.org/docs/reference/api/pandas.Timedelta.html.
        min_periods: Minimum samples.
            Minimum number of observations required in the given time window (otherwise, the result is set to 0).
            Defaults to 1.

    Returns:
        pandas.Series: Time series

    Raises:
        UserTypeError: data is not a time series
        UserValueError: data is empty
        UserTypeError: time_window is not of type pandas.Timedelta
    """
    validate_series_has_time_index(data)
    validate_series_is_not_empty(data)
    validate_timedelta_unit(time_window)

    timedelta_series = data.index.to_series().diff().astype("timedelta64[ms]")
    timedelta_series_ms = timedelta_series.dt.total_seconds().fillna(0) * 1000

    return timedelta_series_ms.rolling(window=time_window, min_periods=min_periods).std().fillna(0)
