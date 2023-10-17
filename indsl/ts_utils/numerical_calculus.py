# Copyright 2023 Cognite AS
import numpy as np
import numpy.typing as npt
import pandas as pd

from scipy.integrate import cumulative_trapezoid

from indsl import versioning
from indsl.decorators import njit
from indsl.exceptions import NUMBA_REQUIRED, UserValueError
from indsl.type_check import check_types
from indsl.validations import validate_series_has_minimum_length, validate_series_is_not_empty, validate_timedelta

from . import numerical_calculus_v1  # noqa


try:
    from numba import prange
except ImportError:
    prange = None  # Only core dependencies installed. Will raise an error later


@versioning.register(version="2.0", changelog="Removed granlularity argument; updated parameter types")
@check_types
def trapezoidal_integration(series: pd.Series, time_unit: pd.Timedelta = pd.Timedelta("1 h")) -> pd.Series:
    """Integration.

    Cumulative integration using trapezoidal rule with an optional user-defined time unit.

    Args:
        series: Time series.
        time_unit: Frequency.
            User defined granularity to potentially override unit of time. Defaults to 1 h.
            Accepted formats can be found here: https://pandas.pydata.org/docs/reference/api/pandas.Timedelta.html.
            Some examples are: '1s', '5m', '3h' or '1d', but combinations also work: "1d 6h 43s"

    Returns:
        pandas.Series: Cumulative integral.
    """
    validate_series_is_not_empty(series)
    validate_timedelta(time_unit)
    arr = cumulative_trapezoid(series, series.index.view(np.int64) / time_unit.value, initial=0.0)
    return pd.Series(arr, index=series.index)


@versioning.register(version="2.0", changelog="Removed granlularity argument; updated parameter types")
@check_types
def differentiate(series: pd.Series, time_unit: pd.Timedelta = pd.Timedelta("1h")) -> pd.Series:
    """Differentiation.

    Differentiation (finite difference) using a second-order accurate numerical method (central difference).
    Boundary points are computed using a first-order accurate method.

    Args:
        series: Time series.
        time_unit: Frequency.
            User defined granularity to potentially override unit of time. Defaults to 1 h.
            Accepts integer followed by time unit string (ms|s|m|h|d). For example: '1s', '5m', '3h' or '1d'.

    Returns:
        pandas.Series: First order derivative.
    """
    validate_series_has_minimum_length(series, 2)
    arr = np.gradient(series, series.index.view(np.int64) / time_unit.value)
    return pd.Series(arr, index=series.index)


@njit(parallel=True)
def window_index(np_datetime_ns: npt.NDArray[np.float64], windowlength_in_ns: int) -> np.ndarray:
    """Sliding window indexing.

    Returns a np.ndarray where the index corresponds to the starting point of a window,
    the value at the index corresponds to numerical indexes for the end of the window.
    This gives the window span each window is supposed to integrate to.

    Args:
        np_datetime_ns: Datetime in integer ns.
        windowlength_in_ns: The length of the window in ns.

    Retruns:
        np.ndarray: indexing of timewindows
    """
    if prange is None:
        raise ImportError(NUMBA_REQUIRED)

    from_to_index = np.zeros(len(np_datetime_ns), dtype=np.int64)
    end = np_datetime_ns[len(np_datetime_ns) - 1]
    for i in prange(len(from_to_index)):
        end_of_window_ns = np_datetime_ns[i] + windowlength_in_ns
        from_to_index[i] = (np.abs(np_datetime_ns - end_of_window_ns)).argmin()  # finds the nearest index
        if end <= np_datetime_ns[i] + windowlength_in_ns:
            break
    return from_to_index


@njit(parallel=True)
def integrate_windows(
    values: np.ndarray, dt: np.ndarray, from_to_index: np.ndarray, number_of_windows: int
) -> np.ndarray:
    """Integrate the windows.

    Performs the integration, through the trapezoidal- or midpoint-method.
    Since all the from and to window indexes are available and all inputs are np.ndarrays, this is parallelized and sped up with numba.

    Args:
        values: Values to integrate corrected to the integrand rate.
        dt: The time in milliseconds between datapoints.
        from_to_index: index is the start of a window and value is the end index of the window.
        number_of_windows: the number of windows to integrate

    Returns:
        numpy.ndarray: The result of each integrated window.
    """
    if prange is None:
        raise ImportError(NUMBA_REQUIRED)

    res = np.empty(number_of_windows, dtype=np.float64)
    for i in prange(number_of_windows):
        cumulative = 0  # reset accumulator for new window (but numba does parallel magic in the background)
        for j in prange(i, from_to_index[i]):
            cumulative += ((values[j - 1] + values[j]) / 2) * dt[j - 1]
        res[i] = cumulative
    return res


@check_types
def sliding_window_integration(
    series: pd.Series,
    window_length: pd.Timedelta = pd.Timedelta("5m"),
    integrand_rate: pd.Timedelta = pd.Timedelta("1h"),
) -> pd.Series:
    """Sliding window integration.

    Siding window integration using trapezoidal rule.

    Args:
        series: Time series.
        window_length: window_length
            the length of time the window. Defaults to '5 minute'. Valid time units are:

                * ‘W’, ‘D’, ‘T’, ‘S’, ‘L’, ‘U’, or ‘N’
                * ‘days’ or ‘day’
                * ‘hours’, ‘hour’, ‘hr’, or ‘h’
                * ‘minutes’, ‘minute’, ‘min’, or ‘m’
                * ‘seconds’, ‘second’, or ‘sec’
                * ‘milliseconds’, ‘millisecond’, ‘millis’, or ‘milli’
                * ‘microseconds’, ‘microsecond’, ‘micros’, or ‘micro’
                * ‘nanoseconds’, ‘nanosecond’, ‘nanos’, ‘nano’, or ‘ns’.
        integrand_rate: integrand_rate.
            if the integrands rate is per sec, per hour, per day.
            Defaults to '1 hour'. Valid time units are:

                * ‘W’, ‘D’, ‘T’, ‘S’, ‘L’, ‘U’, or ‘N’
                * ‘days’ or ‘day’
                * ‘hours’, ‘hour’, ‘hr’, or ‘h’
                * ‘minutes’, ‘minute’, ‘min’, or ‘m’
                * ‘seconds’, ‘second’, or ‘sec’
                * ‘milliseconds’, ‘millisecond’, ‘millis’, or ‘milli’
                * ‘microseconds’, ‘microsecond’, ‘micros’, or ‘micro’
                * ‘nanoseconds’, ‘nanosecond’, ‘nanos’, ‘nano’, or ‘ns’.

    Returns:
        pandas.Series: Time series
    """
    validate_series_is_not_empty(series)

    if max(series.index) - min(series.index) <= window_length:
        raise UserValueError("Integration time window length too long compared to the time window of the dataset.")

    if window_length <= pd.Timedelta("0ms"):
        raise UserValueError("Insert non negative timedelta for window tolength")

    if integrand_rate <= pd.Timedelta("0ms"):
        raise UserValueError("Insert non negative timedelta for integrand rate")

    dt = np.diff(series.index.view(np.int64))  # dt in nanosec
    windowlength_in_ns = int(window_length.total_seconds() * 1e9)
    np_datetime_ns = series.index.view(np.int64)  # points to underlying np.ndarray with timestamps in nanoseconds
    from_to_index = window_index(np_datetime_ns, windowlength_in_ns)

    if from_to_index[0] == 0:
        raise UserValueError("Too small sliding window size, increase window or resample with higher frequency")

    # This code finds the number of windows to integrate
    if np.all(from_to_index != 0):
        n_windows = len(from_to_index) - 1
    else:
        n_windows = np.min(np.where(from_to_index == 0))

    integrand = series.to_numpy() / (integrand_rate.total_seconds() * 1e9)  # corrects the integrand rate to ns
    res = integrate_windows(integrand, dt, from_to_index, n_windows)
    new_index = series.index[
        len(series) - n_windows + 1 :
    ]  # difference in the index due to the first window must be calculated before values come in.
    return pd.Series(res[1:], index=new_index)
