# Copyright 2023 Cognite AS
import numpy as np

from indsl.resample.auto_align import auto_align
from indsl.type_check import check_types


def exp(x):
    """Exp.

    Calculates the exponential of a time series.

    Args:
        x: time-series

    Returns:
        pandas.Series: time series
    """
    return np.exp(x)


def log(x):
    """Ln.

    Calculates the natural logarithm of a time series.

    Args:
        x: time-series

    Returns:
        pandas.Series: time series
    """
    return np.log(x)


def log2(x):
    """Log base 2.

    Calculates the logarithm with base 2 of a time series.

    Args:
        x: time-series

    Returns:
        pandas.Series: time series
    """
    return np.log2(x)


def log10(x):
    """Log base 10.

    Calculates the logarithm with base 10 of a time series.

    Args:
        x: time-series

    Returns:
        pandas.Series: time series
    """
    return np.log10(x)


@check_types
def logn(x, base, align_timesteps: bool = False):
    """Log, any base.

    Calculates the logarithm with base “n” of a time series.

    Args:
        x: Input time-series or number
        base: Base time-series or number
        align_timesteps (bool) : Auto-align
          Automatically align time stamp  of input time series. Default is False.

    Returns:
        pandas.Series: time series
    """
    x, base = auto_align([x, base], align_timesteps)
    return np.log(x) / np.log(base)
