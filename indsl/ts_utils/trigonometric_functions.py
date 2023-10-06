# Copyright 2023 Cognite AS
import numpy as np

from indsl.resample.auto_align import auto_align

# Trigonometric functions (using lambda because we can't inspect numpy's C funcs)
from indsl.type_check import check_types


def sin(x):
    """Sin.

    Calculates the trigonometric sine of a time series.

    Args:
        x: time series

    Returns:
        pandas.Series: time series
    """
    return np.sin(x)


def cos(x):
    """Cos.

    Calculates the trigonometric cosine of a time series.

    Args:
        x: time series

    Returns:
        pandas.Series: time series
    """
    return np.cos(x)


def tan(x):
    """Tan.

    Calculates the trigonometric tangent of a time series.

    Args:
        x: time series

    Returns:
        pandas.Series: time series
    """
    return np.tan(x)


def arcsin(x):
    """Arcsin.

    Calculates the trigonometric arcsine of a time series.

    Args:
        x: time series

    Returns:
        pandas.Series: time series
    """
    return np.arcsin(x)


def arccos(x):
    """Arccos.

    Calculates the trigonometric arccosine of a time series.

    Args:
        x: time series

    Returns:
        pandas.Series: time series
    """
    return np.arccos(x)


def arctan(x):
    """Arctan.

    Calculate inverse hyperbolic tangent of a time series.

    Args:
        x: time series

    Returns:
        pandas.Series: time series
    """
    return np.arctan(x)


@check_types
def arctan2(x1, x2, align_timesteps: bool = False):
    """Arctan(x1, x2).

    Element-wise arc tangent of x1/x2 choosing the quadrant
    correctly.

    Args:
        x1: First time series or number
        x2: Second time series or number
        align_timesteps (bool) : Auto-align
          Automatically align time stamp  of input time series. Default is False.

    Returns:
        pandas.Series: time series
    """
    x1, x2 = auto_align([x1, x2], align_timesteps)
    return np.arctan2(x1, x2)


def deg2rad(x):
    """Degrees to radians.

    Converts angles from degrees to radians.

    Args:
        x: time series

    Returns:
        pandas.Series: time series
    """
    return np.deg2rad(x)


def rad2deg(x):
    """Radians to degrees.

    Converts angles from radiants to degrees.

    Args:
        x: time series

    Returns:
        pandas.Series: time series
    """
    return np.rad2deg(x)


# Hyperbolic functions


def sinh(x):
    """Sinh.

    Calculates the hyperbolic sine of a time series.

    Args:
        x: time series

    Returns:
        pandas.Series: time series
    """
    return np.sinh(x)


def cosh(x):
    """Cosh.

    Calculates the hyperbolic cosine of a time series.

    Args:
        x: time series

    Returns:
        pandas.Series: time series
    """
    return np.cosh(x)


def tanh(x):
    """Tanh.

    Calculates the hyperbolic tangent of time series.

    Args:
        x: time series

    Returns:
        pandas.Series: time series
    """
    return np.tanh(x)


def arcsinh(x):
    """Arcsinh.

    Calculates the hyperbolic arcsine of a time series.

    Args:
        x: time series

    Returns:
        pandas.Series: time series
    """
    return np.arcsinh(x)


def arccosh(x):
    """Arccosh.

    Calculates the hyperbolic arccosine of a time series.

    Args:
        x: time series

    Returns:
        pandas.Series: time series
    """
    return np.arccosh(x)


def arctanh(x):
    """Arctanh.

    Calculates the hyperbolic arctangent of a time series.

    Args:
        x: time series

    Returns:
        pandas.Series: time series
    """
    return np.arctanh(x)
