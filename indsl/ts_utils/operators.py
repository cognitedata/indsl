# Copyright 2023 Cognite AS
import operator as op

from typing import List, Union

import numpy as np

# Simple operations
import pandas as pd

from indsl.exceptions import UserValueError
from indsl.resample.auto_align import auto_align
from indsl.type_check import check_types


@check_types
def add(a, b, align_timesteps: bool = False):
    """Add.

    Add any two time series or numbers.

    Args:
        a: Time-series or number.
        b: Time-series or number.
        align_timesteps (bool) : Auto-align
            Automatically align time stamp  of input time series. Default is False.

    Returns:
        pandas.Series: Time series.
    """
    a, b = auto_align([a, b], align_timesteps)
    return op.add(a, b)


@check_types
def sub(a, b, align_timesteps: bool = False):
    """Subtraction.

    The difference between two time series or numbers.

    Args:
        a: Time-series or number.
        b: Time-series or number.
        align_timesteps (bool) : Auto-align
           Automatically align time stamp  of input time series. Default is False.

    Returns:
        pandas.Series: Time series.
    """
    a, b = auto_align([a, b], align_timesteps)
    return op.sub(a, b)


@check_types
def mul(a, b, align_timesteps: bool = False):
    """Multiplication.

    Multiply two time series or numbers.

    Args:
        a: Time-series or number.
        b: Time-series or number.
        align_timesteps (bool): Auto-align
           Automatically align time stamp  of input time series. Default is False.

    Returns:
        pandas.Series: Time series.
    """
    a, b = auto_align([a, b], align_timesteps)
    return op.mul(a, b)


@check_types
def div(a, b, align_timesteps: bool = False):
    """Division.

    Divide two time series or numbers. If the time series in the
    denominator contains zeros, all instances are dropped from the final
    result.

    Args:
        a: Numerator
        b: Denominator
        align_timesteps (bool): Auto-align.
           Automatically align time stamp  of input time series. Default is False.

    Returns:
        pandas.Series: Time series.
    """
    a, b = auto_align([a, b], align_timesteps)

    if type(b) is pd.Series:
        res = op.truediv(a, b).replace([np.inf, -np.inf], np.nan).dropna()
    elif type(b) is np.array:
        b = b.astype("float")  # Make sure it is a float to replace zeros (int) by np.nan (float)
        b[b == 0] = np.nan
        res = op.truediv(a, b)
        res = res[~np.isnan(res)]
    else:
        res = op.truediv(a, b)

    return res


@check_types
def power(a, b, align_timesteps: bool = False):
    """Power.

    Power of time series or numbers.

    Args:
        a: base time series or number
        b: exponent time series or number
        align_timesteps (bool): Auto-align
           Automatically align time stamp  of input time series. Default is False.

    Returns:
        pandas.Series: Time series.
    """
    a, b = auto_align([a, b], align_timesteps)
    return op.pow(a, b)


def inv(x):
    """Inverse.

    Element-wise inverse of time series or numbers.

    Args:
        x: time series or numbers

    Returns:
        pandas.Series: Time series.
    """
    return 1 / x


def sqrt(x):
    """Square root.

    Square root of time series or numbers.

    Args:
        x: time series or numbers

    Returns:
        pandas.Series: Time series.
    """
    return np.sqrt(x)


def neg(x):
    """Negation.

    Negation of time series or numbers.

    Args:
        x: time series or numbers

    Returns:
        pandas.Series: Time series.
    """
    return op.neg(x)


def absolute(x):
    """Absolute value.

    The absolute value of time series or numbers.

    Args:
        x: time series or numbers

    Returns:
        pandas.Series: Time series.
    """
    return op.abs(x)


@check_types
def mod(a, b, align_timesteps: bool = False):
    """Modulo.

    Modulo of time series or numbers.

    Args:
        a: dividend time series or number
        b: divisor time series or number
        align_timesteps (bool): Auto-align
           Automatically align time stamp  of input time series. Default is False.

    Returns:
        pandas.Series: Time series.
    """
    a, b = auto_align([a, b], align_timesteps)
    return op.mod(a, b)


@check_types
def arithmetic_mean(a, b, align_timesteps: bool = False):
    """Arithmetic mean.

    The mean of two time series or numbers.

    Args:
        a: Time series or number
        b: Time series or number
        align_timesteps (bool): Auto-align
           Automatically align time stamp  of input time series. Default is False.

    Returns:
        pandas.Series: Time series.
    """
    return op.truediv(add(a, b, align_timesteps), 2)


@check_types
def arithmetic_mean_many(data: List[Union[pd.Series, float]], align_timesteps: bool = False) -> Union[pd.Series, float]:
    """Arithmetic mean many.

    The mean of multiple time series.

    Args:
        data: List of time series
        align_timesteps (bool): Auto-align
           Automatically align time stamp  of input time series. Default is False.

    Returns:
        pandas.Series: Time series.
    """
    n = len(data)
    if n == 0:
        raise UserValueError("Expected at least one item in data parameter, got zero instead.")
    timeseries_sum = 0
    for new_ts in data:
        timeseries_sum = add(timeseries_sum, new_ts, align_timesteps)
    return op.truediv(timeseries_sum, n)
