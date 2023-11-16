# Copyright 2023 Cognite AS
import numpy as np
import pandas as pd

from indsl.exceptions import UserValueError
from indsl.type_check import check_types
from indsl.validations import validate_series_has_time_index


@check_types
def resample_timeseries(data: pd.Series, is_step: bool = False) -> pd.Series:
    """Resample timeseries.

    Resamples a timeseries (provided as a pandas Series with datetime index)
    into a equally spaced timeseries with the frequency defined by the smallest
    delta time between each timestamp.

    Args:
        data: Time series
        is_step: Defines if the input time series is a step timeseries
                 Defaults to False

    Returns:
        pandas.Series: Resampled Time series
    """
    validate_series_has_time_index(data)
    if len(data) < 2:
        raise UserValueError(f"Expected data to be of length >= 2, got length {len(data)}")
    # compute delta time in seconds
    delta_time: np.ndarray = np.diff(data.index.values, n=1, axis=-1).astype(np.int64) * 1e-9
    # prevent timesteps smaller than 60s
    min_delta_time: float = max(delta_time.min().round(), 60)
    # define frequency
    frequency: pd.DateOffset = pd.DateOffset(seconds=min_delta_time)

    # compute min and max time
    min_time: pd.Timestamp = data.index.min()
    max_time: pd.Timestamp = data.index.max()

    # create a new series with equally spaced timestamps and join it with input data
    index: pd.DatetimeIndex = pd.date_range(start=min_time, end=max_time, freq=frequency)
    data_equally_spaced: pd.DataFrame = pd.DataFrame(index=index)
    data_missing_values: pd.Series = data_equally_spaced.join(data.rename("new"), how="outer").iloc[:, 0]

    # interpolate missing values
    if is_step:
        data_interpolated = data_missing_values.ffill()
    else:
        data_interpolated = data_missing_values.interpolate(method="time")

    # only keep data for the equally spaced timestamps
    data_clean: pd.Series = data_equally_spaced.join(data_interpolated.rename("new"), how="left").iloc[:, 0]

    return data_clean


@check_types
def constrain(value: float, min: float = 1.0e-4, max: float = 1.0e6) -> float:
    """Constrains a value to not exceed a maximum and minimum value.

    Args:
        value: The value to constrain
        min: The minimum limit. Defaults to 1.0e-4.
        max: The maximum limit. Defaults to 1.0e6.

    Returns:
        float: Value within the specified limits
    """
    if value > max:
        return max
    elif value < min:
        return min
    else:
        return value
