# Copyright 2023 Cognite AS
import re
import warnings

from datetime import datetime
from typing import Callable, List, Literal, Optional, Union

import numpy as np
import pandas as pd
import scipy.integrate

from indsl.exceptions import UserRuntimeError, UserValueError
from indsl.type_check import check_types
from indsl.warnings import IndslUserWarning


_unit_in_ms_without_week = {"s": 1000, "m": 60000, "h": 3600000, "d": 86400000}
_unit_in_ms = {**_unit_in_ms_without_week, "w": 604800000}


def datetime_to_ms(dt):
    """Converts datetime to timestamp in milliseconds."""
    epoch = datetime.utcfromtimestamp(0)
    return int((dt - epoch).total_seconds() * 1000.0)


def time_string_to_ms(pattern, string, unit_in_ms):
    """Converts time from string to milliseconds representation."""
    pattern = pattern.format("|".join(unit_in_ms))
    res = re.fullmatch(pattern, string)
    if res:
        magnitude = int(res.group(1))
        unit = res.group(2)
        return magnitude * unit_in_ms[unit]
    return None


def granularity_to_ms(granularity: str) -> int:
    """Converts granularity from string to milliseconds representation."""
    ms = time_string_to_ms(r"(\d+)({})", granularity, _unit_in_ms_without_week)
    if ms is None:
        raise UserValueError(
            "Invalid granularity format: `{}`. Must be on format <integer>(s|m|h|d). E.g. '5m', '3h' or '1d'.".format(
                granularity
            )
        )
    return ms


def time_ago_to_ms(time_ago_string: str) -> int:
    """Returns millisecond representation of time-ago string."""
    if time_ago_string == "now":
        return 0
    ms = time_string_to_ms(r"(\d+)({})-ago", time_ago_string, _unit_in_ms)
    if ms is None:
        raise UserValueError(
            "Invalid time-ago format: `{}`. Must be on format <integer>(s|m|h|d|w)-ago or 'now'. E.g. '3d-ago' or '1w-ago'.".format(
                time_ago_string
            )
        )
    return ms


def get_fixed_freq(
    start_time: Union[datetime, str, int, float],
    end_time: Union[datetime, str, int, float],
    granularity: Union[str, pd.Timedelta],
):
    """Creates a time series index with a fixed frequency.

    Args:
        start_time (Union(pandas.Datetime,str, int, float)): Start time value to be converted to Timestamp.
        end_time (Union(pandas.Datetime,str, int, float)): End time value to be converted to Timestamp.
        granularity (str): Offset which Timestamp will have. Must be in format <integer>(s|m|h|d). E.g. '5m', '3h' or '1d'.".

    Returns:
        pd.DatetimeIndex: DatetimeIndex with fixed frequency.
    """
    if isinstance(granularity, str):
        return pd.date_range(start_time, end_time, freq=pd.Timedelta(milliseconds=granularity_to_ms(granularity)))
    else:
        return pd.date_range(start_time, end_time, freq=granularity)


def above_below(series: pd.Series, upper_limit: float, lower_limit: float):
    """Above/below.

    Counts the number of points above/below upper_limit/lower_limit.

    Args:
        series: Data with a pandas.DatetimeIndex.
        upper_limit: Value for upper limit.
        lower_limit: Value for lower limit.

    Returns:
        dict: The numbers above/below upper_limit/lower_limit.
    """
    result = {}
    if upper_limit is not None:
        result["above_upper"] = (series > upper_limit).sum()

    if lower_limit is not None:
        result["below_lower"] = (series < lower_limit).sum()

    return result


def num_vals_in_boxes(series: pd.Series, num_boxes: int):
    """Counts the number of points in each of the boxes between 0 and 100 %.

    Args:
        series: Data with a pandas.DatetimeIndex.
        num_boxes: Number of boxes.

    Returns:
        dict: A list with the number of points in each box in keyword "data".
    """
    max_val = series.max()
    min_val = series.min()
    delta = (max_val - min_val) / float(num_boxes)
    eps = 1.0e-6

    boxes = []
    low_limit = min_val
    for idx in range(num_boxes):
        if idx == 0:
            num_value_in_box = ((series >= low_limit * (1 - eps)) & (series <= low_limit + delta * (1 + eps))).sum()
        else:
            num_value_in_box = ((series > low_limit) & (series <= low_limit + delta * (1 + eps))).sum()
        boxes.append(num_value_in_box)
        low_limit += delta

    return {"data": boxes}


def functional_mean(function: Callable, x_vals: List) -> np.ndarray:
    """Convenience method to calculate the mean of a function between each x value.

    The last point is the function called at the last x value. This is required for the
    input and output lists to have the same lengths.

    Args:
        function: Function used to calculate the mean.
        x_vals: Data values used to calculate the mean of a function.

    Raises:
        RuntimeError: Checks that x_vals is not empty.

    Returns:
        numpy.ndarray: Array with the calculated values.
    """
    # Catch empty input
    if len(x_vals) == 0:
        raise UserValueError("No data in the input timeseries.")

    y_vals = np.zeros(len(x_vals))
    for i in range(len(x_vals) - 1):
        y_vals[i] = scipy.integrate.quad(function, x_vals[i], x_vals[i + 1])[0] / (x_vals[i + 1] - x_vals[i])
    y_vals[len(x_vals) - 1] = function(x_vals[-1])
    return y_vals


def is_na_all(data: Union[pd.DataFrame, pd.Series]) -> bool:
    """Convenience method to test if all values in dataframe/series are Nan.

    Args:
        data: Input data.

    Raises:
        UserValueError: Checks that the data type for the input data is pandas.DataFrame or pandas.Series.

    Returns:
        bool: True if all values are Nan, False if all values are not Nan.
    """
    if isinstance(data, pd.Series):
        return data.isnull().all()
    elif isinstance(data, pd.DataFrame):
        return pd.isna(data).all(axis=None)
    else:
        raise UserValueError("Convenience method only supports Series or DataFrame.")


def gaps_detector(timestamps: np.ndarray, threshold: int = 86400) -> np.ndarray:
    """Detect gaps bigger than a threshold in time series.

    Taken from:
    https://github.com/cognitedata/data-quality-
    functions/blob/master/functions/completeness/function/completeness.py.

    Args:
        timestamps: Time array_like.
        threshold: Threshold for gaps. Defaults to 86400 seconds, i.e one full day in epoch time.

    Returns:
        numpy.ndarray: 2D array of begin and end time of gaps in seconds epoch time.
    """
    if len(timestamps) == 0:
        return None

    mask = np.diff(timestamps) >= threshold
    gaps = np.column_stack([timestamps[:-1][mask], timestamps[1:][mask]])

    return gaps


def time_difference(data: pd.Series) -> pd.Series:
    """Calculate the time difference between two consecutive points.

    Args:
        data: Input time series.
        unit: Time unit of uniformity check. Follows Numpy DateTime Units convention. Defaults to "ns".

    Returns:
        pd.Series: Time difference between two consecutive points.

    """
    timedelta_series = data.index.astype("datetime64[ns]").to_series().diff().dropna()
    return timedelta_series.dt.total_seconds() * 1000  # ms


@check_types
def mad(
    data: pd.Series,
) -> Union[float, int, pd.Timedelta]:
    """Median absolute deviation (MAD).

    Median absolute deviation computed for a time series.

    Args:
        data: Time series.

    Returns:
        Median absolute deviation
    """
    return (data - data.median()).abs().median()


def check_uniform(data: pd.DataFrame, unit: str = "ns") -> bool:
    """Convenience method to verify input time series is uniform.

    Args:
        data: Time series to check. Object must have a datetime-like index.
        unit: Time unit of uniformity check. Follows Numpy DateTime Units convention. Defaults to "ns".

    Returns:
        bool: True if input timeseries is uniform, False if not uniform.
    """
    # Get time difference between successive observations
    timediff = np.diff(data.index)
    # Round to desired time unit (e.g. uniform on second scale)
    floored = timediff.astype(dtype=f"timedelta64[{unit}]")
    # Check whether uniform
    return np.all(floored[0] == floored[1:])


def make_uniform(data: pd.DataFrame, resolution: str = "1s", interpolation: Optional[str] = None) -> pd.DataFrame:
    """Make time series uniform.

    Convenience method to make an input time series dataframe uniform per specified temporal resolution.
    Object must have a datetime-like index.

    Args:
        data: Input time series dataframe.
        resolution: Temporal resolution of uniform time series. Follows Pandas DateTime convention. Defaults to "1s".
        interpolation: Method for optional interpolation. Follows pandas Resampler.interpolate. Defaults to None.

    Returns:
        pandas.DataFrame: Uniform Timeseries DataFrame.
    """
    # If no interpolation specified, do not perform interpolation
    if not interpolation:
        data = data.resample(resolution).mean()
    # Otherwise perform desired interpolation
    else:
        data = data.resample(rule=resolution).mean().interpolate(method=interpolation)
    return data


def time_parse(time_window: str = "60min", function_name: str = "") -> pd.Timedelta:
    """Check that time_window is given in a correct format.

    Args:
        time_window: Time window input. Defaults to "60min".
            Accepted string format: '3w', '10d', '5h', '30min', '10s'.
            If the user gives a number without unit (such as '60'), it will be considered as the number of minutes.
        function_name: Prints the name of the function that called time_parse in the error message if an error is raised. Defaults to "".

    Raises:
        UserValueError: Raise a value error if time_window is given in a wrong format.

    Returns:
        str: parsed time_window in string format.
    """
    if str.isdigit(time_window):  # if user gives '60' it will be considered as '60min'
        warnings.warn(
            f"Missing time unit in argument 'time_window' in {function_name} function, assuming {time_window} min.",
            category=IndslUserWarning,
        )
        time_window = str(time_window) + "min"
    try:
        return pd.Timedelta(time_window)
    except ValueError:
        raise UserValueError(
            f"Time window should be a string in weeks, days, hours or minutes format:'3w', '10d', '5h', '30min', '10s' not {time_window}."
        ) from None


def time_to_points(data: pd.Series, time_window="60min") -> int:
    """Find the average number of points given a time window.

    Args:
        data: Data with a pandas.DatetimeIndex.
        time_window: Length of the time period to compute the average of points. Defaults to "60min".
            Accepted string format: '3w', '10d', '5h', '30min', '10s'.

    Returns:
        int: average number of points in a time window.
    """
    time_window = time_parse(time_window)
    num_points = data.rolling(time_window).count().mean()

    return round(num_points)


def fill_gaps(
    data: pd.Series,
    granularity: Optional[pd.Timedelta] = None,
    ffill_resolution: Optional[pd.Timedelta] = None,
    interpolate_resolution: Optional[pd.Timedelta] = None,
    method: Literal["ffill", "backfill", "bfill", "pad"] = "ffill",
) -> pd.Series:
    """Mask nan/fill gaps based on the time size of missing data.

    Args:
        data: Input time series. Must be uniform.
        granularity: Temporal resolution of uniform time series, before resampling. Defaults to None. If not given as input, it will be inferred from the data.
        ffill_resolution: Gaps smaller than this will be filled with method. Defaults to None.
        interpolate_resolution: Gaps smaller than this will be interpolated, larger than this will be filled by noise. Defaults to None.
        method: Method used to fill nan. Defaults to 'ffill', but ‘backfill’, ‘bfill’ and ‘pad’ can also be used.

    Returns:
        pandas.DataFrame or pd.Series: Uniform, time series without nan values.

    Raises:
        UserRuntimeError: The input time series is not uniform
        UserValueError: granularity can not be 0
        UserValueError: ffill_resolution can not be 0
        UserValueError: interpolate_resolution can not be 0
    """
    if not check_uniform(data):
        raise UserRuntimeError("The input time series is not uniform")

    if not data.isnull().values.any():
        # no need to proceed
        return data

    if granularity is None:
        # If granularity is not given as input. Granularity will be inferred from the data
        granularity = pd.Timedelta(data.index.inferred_freq)
    if granularity == pd.Timedelta("0T"):
        raise UserValueError("granularity can not be 0")

    if ffill_resolution == pd.Timedelta("0T"):
        raise UserValueError("ffill_resolution can not be 0")

    if interpolate_resolution == pd.Timedelta("0T"):
        raise UserValueError("interpolate_resolution can not be 0")

    small_gap = 0
    inter_gap = None
    mask_gap = 0

    if ffill_resolution is not None:
        small_gap = granularity / ffill_resolution
    if interpolate_resolution is not None:
        mask_gap = granularity / interpolate_resolution
        # inter_gap has to be above 0, but can be None
        inter_gap = max(1, int(mask_gap))

    # ffill all, after interpolate and adding noise
    # create groups based on size of gap
    nan_cumsum = (
        data.isnull().astype("int64").groupby(data.notnull().astype("int64").cumsum()).cumsum().mask(lambda x: x == 0)
    )
    helper = data.notnull()
    nan_group_size = nan_cumsum.groupby(helper.cumsum()).transform("max").mask(helper)

    # start with random filling for the big gaps
    mask = nan_group_size > inter_gap
    n = len(data[mask])
    data[mask] = np.random.normal(loc=np.mean(data), scale=np.std(data), size=n)

    # interpolate the inteprolate range
    mask = np.logical_and(nan_group_size > small_gap, nan_group_size <= mask_gap)
    data[mask] = data.interpolate(method="linear", limit_direction="forward", limit=inter_gap)

    if method in ["ffill", "pad"]:
        return data.ffill()
    elif method in ["backfill", "bfill"]:
        return data.bfill()
    else:
        raise UserValueError(f"Method {method} is not supported. Use 'ffill', 'backfill', 'bfill' or 'pad'.")


def number_of_events(out: pd.Series):
    """Calculate the number of events detected from a step time series."""
    return len(np.where(out.values == 1)[0]) / 2


@check_types
def scalar_to_pandas_series(data: Union[float, pd.Series]) -> pd.Series:
    """Convert data to series if it is not already.

    Args:
        data: Data that needs to be series.

    Returns:
        pd.Series: Data as series.
    """
    if isinstance(data, float):
        data = pd.Series(data, index=pd.date_range(start="1970", end=pd.Timestamp.now(), periods=2))
    return data
