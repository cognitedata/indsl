# Copyright 2023 Cognite AS

import numpy as np
import pandas as pd

from indsl import versioning
from indsl.decorators import njit
from indsl.exceptions import UserValueError
from indsl.type_check import check_types
from indsl.validations import validate_series_has_time_index, validate_series_is_not_empty


@check_types
def _validate_time_series(x: pd.Series) -> None:
    validate_series_has_time_index(x)
    validate_series_is_not_empty(x)

    if not x.index.is_monotonic_increasing:
        raise UserValueError("Time series index is not increasing.")


@check_types
def _validate_threshold(threshold: float) -> None:
    if threshold < 0:
        raise UserValueError("Threshold should be a non-negative float.")


@njit
def _prolong_indicator(x_value: np.array, x_indicator: np.array) -> np.array:
    """Indicator adjustment.

    Function that prolongs the indicator value for bad data quality - indicator will be set to 1 for as long as data
    doesn't go back to "normal" (i.e., back to the value it had before the value decreased).

    Args:
         x_value: Numpy array with time series values
         x_indicator: Numpy array (with values 0 and 1) indicating bad data quality for given time series x

    Returns:
         np.array: Indicator time series
    """
    i, n = 0, len(x_value)

    while i < n - 1:
        if x_indicator[i] == 1:
            j = i + 1
            while j < n and x_value[i - 1] > x_value[j]:
                x_indicator[j] = 1
                j += 1
            i = j
        else:
            i += 1

    return x_indicator


@versioning.register(version="1.0", deprecated=True, name="negative_running_hours_check")
@check_types
def negative_running_hours_check(x: pd.Series, threshold: float = 0.0) -> pd.Series:
    """Negative running hours.

    The negative running hours model is created in order to automate data quality check for time series with values that
    shouldn't be decreasing over time. One example would be Running Hours (or Hour Count) time series - a specific type
    of time series that is counting the number of running hours in a pump. Given that we expect the number of running
    hours to either stay the same (if the pump is not running) or increase with time (if the pump is running), the
    decrease in running hours value indicates bad data quality. Although the algorithm is originally created for
    Running Hours time series, it can be applied to all time series where the decrease in value is a sign of bad data
    quality.

    Args:
        x: Time series
        threshold: Threshold for value drop.
            This threshold indicates for how many hours the time series value needs to drop (in hours) before we
            consider it bad data quality. The threshold must be a non-negative float. By default, the threshold is set to 0.

    Returns:
        pandas.Series: Time series
            The returned time series is an indicator function that is 1 where there is a decrease in time series
            value, and 0 otherwise. The indicator will be set to 1 until the data gets "back to normal" (that is,
            until time series reaches the value it had before the value drop).


    Raises:
        UserTypeError: x is not a time series
        UserValueError: x is empty
        UserTypeError: index of x is not a datetime
        UserValueError: index of x is not increasing
        UserTypeError: threshold is not a number
        UserValueError: threshold is a negative number
    """
    _validate_time_series(x)

    _validate_threshold(threshold)

    if len(x) < 2:
        return pd.Series([0] * len(x), index=x.index)

    x_diff = x.diff(1)
    x_indicator = (x_diff < -threshold).astype(np.uint8)

    return pd.Series(
        _prolong_indicator(
            x.to_numpy(),
            x_indicator.to_numpy(),
        ).astype(np.int64),
        index=x.index,
    )
