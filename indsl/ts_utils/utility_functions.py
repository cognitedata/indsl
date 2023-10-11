# Copyright 2023 Cognite AS

from datetime import datetime, timedelta
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

from scipy.stats import shapiro

from indsl.exceptions import UserTypeError, UserValueError
from indsl.resample.auto_align import auto_align
from indsl.type_check import check_types


TimeUnits = Literal["ns", "us", "ms", "s", "m", "h", "D", "W"]


# Rounding and utility functions
@check_types
def round(x, decimals: int):
    """Round.

    Rounds a time series to a given number of decimals.

    Args:
        x: time series
        decimals: number of decimals

    Returns:
        pd.Series: time series

    """
    return np.round_(x, decimals=decimals)


def floor(x):
    """Round down.

    Rounds a time series down to the nearest integer smaller than
    or equal to the current value.

    Args:
        x: time series

    Returns:
        pd.Series: time series
    """
    return np.floor(x)


def ceil(x):
    """Round up.

    Rounds up a time series to the nearest integer greater than or
    equal to the current value.

    Args:
        x: time series

    Returns:
        pd.Series: time series
    """
    return np.ceil(x)


def sign(x):
    """Sign.

    Element-wise indication of the sign of a time series.

    Args:
        x: time series

    Returns:
        pd.Series: time series
    """
    return np.sign(x)


# Clipping functions
@check_types
def clip(x, low: float = -np.inf, high: float = np.inf):
    """Clip (low, high).

    Given an interval, values of the time series outside the
    interval are clipped to the interval edges.

    Args:
        x (pd.Series): time series
        low: Lower limit
            Lower clipping limit. Default: -infinity
        high: Upper limit
            Upper clipping limit. Default: +infinity

    Returns:
        pd.Series: time series
    """
    return np.clip(x, low, high)


@check_types
def maximum(x1, x2, align_timesteps: bool = False):
    """Element-wise maximum.

    Computes the maximum value of two time series or
    numbers.

    Args:
        x1: First time series or number
        x2: Second time series or number
        align_timesteps: Auto-align
          Automatically align time stamp  of input time series. Default is False.

    Returns:
        pd.Series: time series
    """
    x1, x2 = auto_align([x1, x2], align_timesteps)
    return np.maximum(x1, x2)


@check_types
def minimum(x1, x2, align_timesteps: bool = False):
    """Element-wise minimum.

    Computes the minimum value of two time series.

    Args:
        x1: First time series or number
        x2: Second time series or number
        align_timesteps: Auto-align
          Automatically align time stamp  of input time series. Default is False.

    Returns:
        pd.Series: time series
    """
    x1, x2 = auto_align([x1, x2], align_timesteps)
    return np.minimum(x1, x2)


@check_types
def union(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """Union.

    Takes the union of two time series. If a time stamp
    occurs in both series, the value of the first time series is used.

    Args:
        series1: time series
        series2: time series

    Returns:
        pd.Series: time series

    Raises:
        UserTypeError: series1 or series2 is not a time series
    """
    out = pd.concat([series1, series2]).sort_index()
    return out[~out.index.duplicated(keep="first")]


@check_types
def bin_map(x1, x2, align_timesteps: bool = False):
    """Element-wise greater-than.

    Maps to a binary array by checking if one time series is greater than another.

    Args:
        x1: First time series or number
        x2: Second time series or number
        align_timesteps: Auto-align
          Automatically align time stamp  of input time series. Default is False.

    Returns:
        np.ndarray: time series
    """
    x1, x2 = auto_align([x1, x2], align_timesteps)
    return out.astype(np.int64) if hasattr(out := x1 > x2, "astype") else out


@check_types
def set_timestamps(timestamp_series: pd.Series, value_series: pd.Series, unit: TimeUnits = "ms") -> pd.Series:
    """Set index of time series.

    Sets the time series values to the Unix timestamps.
    The timestamps follow the Unix convention (Number of seconds
    starting from January 1st, 1970). Both input time series
    must have the same length.


    Args:
        timestamp_series: Timestamp time series
        value_series: Value time series
        unit: Timestamp unit
          Valid values "ns|us|ms|s|m|h|D|W". Default "ms"

    Returns:
        pd.Series: time series

    Raises:
        UserTypeError: timestamp_series or value_series are not time series
        UserTypeError: unit is not a string
        UserValueError: timestamp_series and value_series do not have the same length
    """
    if not len(timestamp_series) == len(value_series):
        raise UserValueError("Length of input time series must be equal.")

    index = pd.to_datetime(timestamp_series.to_numpy(), unit=unit)
    return pd.Series(value_series.to_numpy(), index=index)


@check_types
def get_timestamps(series: pd.Series, unit: TimeUnits = "ms") -> pd.Series:
    """Get index of time series.

    Get timestamps of the time series as values.
    The timestamps follow the Unix convention (Number of seconds
    starting from January 1st, 1970). Precision loss in the order of
    nanoseconds may happen if unit is not nanoseconds.

    Args:
        series: Time-series
        unit: Timestamp unit
          Valid values "ns|us|ms|s|m|h|D|W". Default "ms"

    Returns:
        pd.Series: time series

    Raises:
        UserTypeError: series is not a time series
        UserTypeError: unit is not a string
    """
    if unit == "ns":
        values = series.index.to_numpy("datetime64[ns]").view(np.int64)
    else:
        values = series.index.view(np.int64) / pd.Timedelta(1, unit=unit).value

    return pd.Series(values, index=series.index)


@check_types
def time_shift(series: pd.Series, n_units: float = 0, unit: TimeUnits = "ms") -> pd.Series:
    """Shift time series.

    Shift time series by a time period

    Args:
        series: Time-series
        n_units: Time periods to shift
            Number of time periods to shift
        unit: Time period unit
          Valid values "ns|us|ms|s|m|h|D|W". Default "ms"

    Returns:
        pd.Series: time series

    Raises:
        UserTypeError: series is not a time series
        UserTypeError: n_units is not a number
        UserTypeError: unit is not a string
    """
    out = series.copy()
    out.index += pd.Timedelta(n_units, unit=unit)

    return out


@check_types
def replace(series: pd.Series, to_replace: Optional[List[float]] = None, value: Optional[float] = 0.0):
    """Replace.

    Replace values in a time series. The values to replace should be
    a semicolon-separated list. Undefined and infinity values can be replaced
    by using nan, inf and -inf (e.g. 1.0, 5, inf, -inf, 20, nan).

    Args:
        series: Time series
        to_replace: Replace
            List of values to replace. The values must be seperated by semicolons. Infinity and undefined values can be
            replaced by using the keywords inf, -inf and nan. The default is to replace no values.
        value: Value used as replacement.
            Default is 0.0

    Returns:
        pd.Series: time series

    Raises:
        UserTypeError: series is not a time series
        UserTypeError: to_replace is not a list
        UserTypeError: value is not a number

    """
    if to_replace is None:
        return series
    elif not isinstance(to_replace, list):
        raise UserTypeError("to_replace must be a list")
    if value is not None and not isinstance(value, (float, int)):
        raise UserTypeError("value must be a number")
    return series.replace(to_replace, value)


@check_types
def remove(
    series: pd.Series,
    to_remove: Optional[List[float]] = None,
    range_from: Optional[float] = None,
    range_to: Optional[float] = None,
):
    """Remove.

    Remove specific values or a range of values from a time series.

    Args:
        series: Time series
        to_remove: Values
            List of values to remove. The values must be seperated by semicolons. Infinity and undefined values can be
            replaced by using the keywords inf, -inf and nan. If empty, which is the default, all values are kept.
        range_from: Range start
            Only values above this parameter will be kept. If empty, which is the default, this range filter is deactivated.
        range_to: Range end
            Only values above this parameter will be kept. If empty, which is the default, the range filter is deactivated.

    Returns:
        pd.Series: time series

    Raises:
        UserTypeError: series is not a time series
        UserTypeError: to_remove is not a list
    """
    if to_remove is not None and not isinstance(to_remove, list):
        raise UserTypeError("to_replace must be a list")
    if to_remove is not None:
        series = series[~series.isin(to_remove)]
    if range_from is not None:
        series = series[series >= range_from]
    if range_to is not None:
        series = series[series <= range_to]
    return series


@check_types
def threshold(series: pd.Series, low: float = -np.inf, high: float = np.inf):
    """Threshold.

    Indicates if the input series exceeds the lower and higher limits. The output series
    is 1.0 if the input is between the (inclusive) limits, and 0.0 otherwise.

    Args:
        series: Time series
        low: Lower limit
           threshold. Default: -infinity
        high: Upper limit
           threshold. Default: +infinity

    Returns:
        pd.Series: Time series

    Raises:
        UserTypeError: series is not a time series
        UserTypeError: low or high are not floats
    """
    return series.between(low, high).astype(np.int64)


def generate_step_series(flag: pd.Series) -> pd.Series:
    """Step-wise time series.

    Construct a step-wise time series (with 0-1 values) from a flag time series.
    Consecutive 1 values are merged in one step.

    Args:
        flag: Binary time series.
            The length of the flag time series has to be the same as the original data points.

    Returns:
        pd.Series: Time series

    Example:
        Given 4 datapoints and a flag with values [0, 0, 1, 0]

            data points: x-------x-------x-------x
            flag:        0       0       1       0

        The resulting step series is represented as:

                                         x-------x
            step series: x---------------x
                         0               01      1
    """
    # add first point as the start of the step time series
    step_time_series = pd.Series([flag.iloc[0]], index=[flag.index[0]])

    # compute actual_value - previous_value
    flag_shift = flag.diff(1)

    # find end points of 1-0 (decreasng) or 0-1 (increasing) patterns
    end_point_increasing = flag_shift[flag_shift == 1]
    end_point_decreasing = flag_shift[flag_shift == -1]

    # add "1" value for 0-1 (increasing) pattern, add "0" value for 1-0 (decreasing) pattern
    end_point_decreasing_values = pd.Series(
        [0] * len(end_point_decreasing), index=end_point_decreasing.index, dtype=np.int64
    )
    end_point_increasing_values = pd.Series(
        [1] * len(end_point_increasing), index=end_point_increasing.index, dtype=np.int64
    )

    # add extra "corner" points
    extra_point_decreasing_values = pd.Series(
        [1] * len(end_point_decreasing), index=end_point_decreasing.index - pd.Timedelta(1, unit="ms"), dtype=np.int64
    )
    extra_point_increasing_values = pd.Series(
        [0] * len(end_point_increasing), index=end_point_increasing.index - pd.Timedelta(1, unit="ms"), dtype=np.int64
    )

    step_time_series = pd.concat(
        [
            step_time_series,
            end_point_increasing_values,
            end_point_decreasing_values,
            extra_point_increasing_values,
            extra_point_decreasing_values,
        ]
    )
    step_time_series = step_time_series.sort_index()

    # add last point if it doesn't exist
    if flag.index[-1] != step_time_series.index[-1]:
        step_time_series = pd.concat([step_time_series, pd.Series(flag.iloc[-1], index=[flag.index[-1]])])

    # remove the last point if the number of 1 values is odd
    if len(step_time_series[step_time_series == 1]) % 2 != 0:
        step_time_series = step_time_series[:-1]

    return step_time_series


def create_series_from_timesteps(timesteps: List[timedelta]) -> pd.Series:
    """Time series from timestamps.

    Create a time series which starts on the 2021-1-1 and contains data points
    with the provided timesteps in seconds. All values of the output series are zero.

    Args:
        timesteps: Time steps between points.

    Returns:
        pd.Series: Time series
    """
    current = datetime(2021, 1, 1)
    timestamps = [current]
    for step in timesteps:
        timestamps.append(timestamps[-1] + step)
    timestamps = pd.Series(timestamps)
    return pd.Series(np.zeros(len(timestamps)), index=timestamps)


def normality_assumption_test(
    series: Union[pd.Series, np.ndarray], max_data_points: int = 5000, min_p_value: float = 0.05, min_W: float = 0.5
) -> Optional[Tuple[float, float]]:
    """Test for normality assumption.

    This function performs a Shapiro-Wilk test to check if the data is normally distributed.

    Args:
        series: Time series
        max_data_points: Maximum number of data points.
            The test is only performed if the time series contains less data points than this value.
            Default to 5000.
        min_p_value: Minimum p value.
            Probability of the time series not being normally distributed.
            Default to 0.05.
        min_W: Minimum W value.
            W is between 0 and 1, small values lead to a rejection of the
            normality assumption. Ref: https://www.nrc.gov/docs/ML1714/ML17143A100.pdf
            Default to 0.5.

    Raises:
        UserValueError: time series has more than the maximum number of datapoints allowed for this test.
        UserValueError: time series is uniform and not normally distributed
        UserValueError: time series is not normally distributed.
    """
    if len(series) > max_data_points:
        raise UserValueError(f"This time series exceeds the limit of {max_data_points} datapoints to perform the test.")

    W, p_value = shapiro(series)

    if p_value == 1.0 and W == 1.0:
        raise UserValueError("This time series is uniform and not normally distributed")
    if p_value < min_p_value or W < min_W:
        raise UserValueError("This time series is not normally distributed")

    return None


def z_scores_test(x: np.ndarray, cutoff: float = 3.0, direction: Literal["greater", "less"] = "greater") -> np.ndarray:
    """Z-scores test.

    This functions performs a z-scores test given a cut-off value and returns a binary time series.

    Args:
        x: Data values.
        cutoff: Cut-off
            Number of standard deviations from the mean.
        direction: Direction of the test.
            Options:
            - "greater": the function will check if the z-scores are greater than the cut-off
            - "less": the function will check if the z-scores are less than the cut-off

    Returns:
        np.ndarray: Binary np.array with the results of the test.
            1 values indicate that the test has passed.
    """
    if (std := x.std()) == 0.0:
        z_scores = np.zeros(len(x))
    else:
        z_scores = (x - x.mean()) / std

    return threshold_test(z_scores, cutoff, direction)


def modified_z_scores_test(
    x: np.ndarray, cutoff: float = 3.0, direction: Literal["greater", "less"] = "greater"
) -> np.ndarray:
    """Z-scores test.

    This functions performs a modified z-scores test given a cut-off value and returns a binary time series.

    Args:
        x: Data values.
        cutoff: Cut-off value.
        direction: Direction of the test.
            Options:
            - "greater": the function will check if the modified z-scores are greater than the cut-off
            - "less": the function will check if the modified z-scores are less than the cut-off

    Returns:
        np.ndarray: Binary np.array with the results of the test.
                    1 values indicate that the test has passed.
    """
    median = np.median(x)
    if (mad := np.median(np.abs(x - median))) == 0.0:
        modified_z_scores = np.zeros(len(x))
    else:
        modified_z_scores = 0.6745 * (x - median) / mad

    return threshold_test(modified_z_scores, cutoff, direction)


def iqr_test(
    x: np.ndarray, limit: Literal["upper", "lower"] = "upper", direction: Literal["greater", "less"] = "greater"
) -> np.ndarray:
    """Z-scores test.

    This functions performs a iqr test and returns a binary time series.

    Args:
        x: Data values.
        limit: Cut-off limit.
        direction: Direction of the test.
            Options:
            - "greater": the function will check if the values are greater than the cut-off limit.
            - "less": the function will check if the values are less than the cut-off limit.

    Returns:
        np.ndarray: Binary np.array with the results of the test.
                    1 values indicate that the test has passed.
    """
    percentile25 = np.quantile(x, 0.25)
    percentile75 = np.quantile(x, 0.75)
    iqr = percentile75 - percentile25

    if limit == "upper":
        cutoff = percentile75 + 1.5 * iqr
    elif limit == "lower":
        cutoff = percentile25 - 1.5 * iqr

    return threshold_test(x, cutoff, direction)


def threshold_test(x: np.ndarray, cutoff: float, direction: Literal["greater", "less"]) -> np.ndarray:
    """Threshold test.

    This functions performs a threshold test based on a direction (greater than or less than).

    Args:
        x: Data values.
        cutoff: Cut-off value.
        direction: Direction of the test.
            Options:
            - "greater": the function will check if the values are greater than the cut-off limit.
            - "less": the function will check if the values are less than the cut-off limit.

    Returns:
        np.ndarray: Binary np.array with the results of the test.
                    1 values indicate that the test has passed.
    """
    if direction == "greater":
        return np.where(x > cutoff, 1, 0)
    elif direction == "less":
        return np.where(x < cutoff, 1, 0)
