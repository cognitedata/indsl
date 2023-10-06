# Copyright 2023 Cognite AS
import pandas as pd

from indsl.resample.auto_align import auto_align
from indsl.type_check import check_types
from indsl.validations import validate_series_has_time_index, validate_series_is_not_empty, validate_timedelta_unit


@check_types
def pearson_correlation(
    data1: pd.Series,
    data2: pd.Series,
    time_window: pd.Timedelta = pd.Timedelta(minutes=15),
    min_periods: int = 1,
    align_timesteps: bool = False,
) -> pd.Series:
    """Pearson correlation.

    This function measures the linear correlation between two time series along a rolling window.
    Pearson’s definition of correlation: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient

    Args:
        data1: Time series.
        data2: Time series.
        time_window: Time window.
            Length of the time period to compute the Pearson correlation. Defaults to 'minutes=15'.
            Time unit should be in days, hours, minutes or seconds. Accepted formats can be found here:
            https://pandas.pydata.org/docs/reference/api/pandas.Timedelta.html.
        min_periods: Minimum samples.
            Minimum number of observations required in the given time window (otherwise, the result is set to 0).
            Defaults to 1.
        align_timesteps (bool): Auto-align.
           Automatically align time stamp  of input time series. Default is False.

    Returns:
        pandas.Series: Time series

    Raises:
        UserTypeError: data is not a time series
        UserValueError: data is empty
        UserTypeError: time_window is not of type pandas.Timedelta
    """
    # validations
    for data in [data1, data2]:
        validate_series_has_time_index(data)
        validate_series_is_not_empty(data)
    validate_timedelta_unit(time_window)

    # align timesteps
    data1, data2 = auto_align([data1, data2], align_timesteps)

    # compute
    return data1.rolling(time_window, min_periods=min_periods).corr(data2)
