# Copyright 2023 Cognite AS
import numpy as np
import pandas as pd

from indsl.exceptions import UserValueError
from indsl.type_check import check_types
from indsl.validations import validate_series_has_minimum_length, validate_series_has_time_index


MIN_DATA_PT = 10


@check_types
def find_period(x: pd.Series, method_period: str = "median") -> float:
    """Estimate Data Period.

    Estimates the period of a time series

    Args:
         x: Time series
         method_period: Method to estimate period
            can be 'median' or 'min' of the difference of timestamps
    Returns:
         float: period in nanoseconds

    Raises:
         TypeError: x is not a time series
         TypeError: index of x is not datetime
         ValueError: method_period is not median or min
    """
    validate_series_has_time_index(x)
    if method_period == "median":
        period = np.median(np.diff(x.index.to_numpy(np.int64))).astype(float)
    elif method_period == "min":
        period = np.min(np.diff(x.index.to_numpy(np.int64))).astype(float)
    else:
        raise UserValueError(f"Period calculation method can only be strings: median or min, not {method_period}")
    return period


@check_types
def calculate_completeness(x: pd.Series, time_start: pd.Timestamp, time_end: pd.Timestamp, period: float) -> float:
    """Completeness score.

    Calculates the completeness score of an input time series for a certain date range

    Args:
         x: Time series
         time_start: start time of date range
         time_end: end time of date range
         period: timeseries period in ns

    Returns:
         float: completeness score

    Raises:
         ValueError: completeness score more than 1
    """
    x = x.loc[(x.index >= time_start) & (x.index <= time_end)]

    total_time_span = time_end - time_start

    expected_num_points = int(total_time_span.total_seconds() * 10**9 / period)
    if x.index[0] == time_start:
        expected_num_points = expected_num_points + 1

    completeness = x.count() / expected_num_points
    if completeness > 1:
        raise ValueError(f"Completeness score {completeness}. Change period calculation method.")
    return completeness


@check_types
def completeness_score(
    x: pd.Series, cutoff_good: float = 0.80, cutoff_med: float = 0.60, method_period: str = "median"
) -> str:
    """Completeness score.

    This function determines the completeness of a time series from a completeness score.
    The score is a function of the inferred data sampling period
    (median or minimum of timestamp differences) and the expected total number of data points for the period
    from the sampling frequency.
    The completeness score is defined as the ratio between the actual number of data points
    to the expected number of data points based on the sampling frequency. The completeness
    is categorized as good if the score is above the
    specified cutoff ratio for good completeness. It is medium if the score falls between
    the cutoff ratio for good and medium completeness. It is characterized as poor if the
    score is below the medium completeness ratio.

    Args:
         x: Time series
         cutoff_good: Good cutoff
             Value between 0 and 1. A completeness score above this cutoff value
             indicates good data completeness. Defaults to 0.80.
         cutoff_med: Medium cutoff
             Value between 0 and 1 and lower than the good data completeness
             cutoff. A completeness score above this cutoff and below the good
             completeness cutoff indicates medium data completeness. Data
             with a completeness score below it are categorised as poor data
             completeness. Defaults to 0.60.
         method_period: Method
             Name of the method used to estimate the period of the time series, can be 'median' or 'min'.
             Default is 'median'

    Returns:
         string: Data quality
             The data quality is defined as
             Good, when completeness score >= cutoff_good,
             Medium, when cutoff_med <= completeness score < cutoff_good,
             Poor, when completeness score < cutoff_med

    Raises:
         TypeError: cutoff_good and cutoff_med are not a number
         ValueError: x has less than ten data points
         TypeError: x is not a time series
         TypeError: index of x is not datetime
         ValueError: method_period is not median or min
         ValueError: completeness score more than 1
    """
    validate_series_has_minimum_length(x, MIN_DATA_PT)

    if cutoff_med is not None and cutoff_good is not None and cutoff_med >= cutoff_good:
        raise UserValueError("cutoff_good should be higher than cutoff_med.")

    period = find_period(x, method_period)

    completeness = calculate_completeness(x, x.index[0], x.index[-1], period)

    if completeness >= cutoff_good:
        return f"Good: completeness score is {completeness:.2f}"
    elif completeness >= cutoff_med:
        return f"Medium: completeness score is {completeness:.2f}"
    else:
        return f"Poor: completeness score is {completeness:.2f}"
