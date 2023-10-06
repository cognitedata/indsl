# Copyright 2023 Cognite AS
from typing import Union

import pandas as pd

from .exceptions import UserTypeError, UserValueError


def validate_series_has_time_index(data: pd.Series) -> None:
    """Helper method to validate if provided pandas.Series is of type pandas.DatetimeIndex."""
    if not isinstance(data.index, pd.DatetimeIndex):
        raise UserTypeError(f"Expected a time series, got index type {data.index.dtype}")


def validate_series_is_not_empty(series: Union[pd.Series, pd.DataFrame]) -> None:
    """Helper method to validate if provided pandas.Series has more than 0 values."""
    if len(series) == 0:
        raise UserValueError("Time series is empty.")


def validate_series_has_minimum_length(series: pd.Series, min_len: int) -> None:
    """Helper method to validate if provided pandas.Series has the minimum length specified."""
    if len(series) < min_len:
        raise UserValueError(f"Expected series with length >= {min_len}, got length {len(series)}")


def validate_timedelta_unit(timedelta: pd.Timedelta) -> None:
    """Helper method to validate if the provided pd.Timedelta is larger or equal to 1 second."""
    if timedelta < pd.Timedelta(seconds=1):
        raise UserValueError("Unit of timedelta should be in days, hours, minutes or seconds")


def validate_timedelta(timedelta: pd.Timedelta) -> None:
    """Helper method to validate if the provided pd.Timedelta valid: not NaT and strictly larger than zero."""
    if timedelta is pd.NaT:
        raise UserValueError("Timedelta is invalid (NaT).")

    if timedelta.value < 0:
        raise UserValueError("Timedelta must be strictly positive. The smallest possible value is '1ns'")
