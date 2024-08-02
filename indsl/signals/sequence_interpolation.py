# Copyright 2024 Cognite AS
from typing import List

import pandas as pd

from indsl.exceptions import UserValueError
from indsl.resample.auto_align import auto_align
from indsl.type_check import check_types


@check_types
def sequence_interpolation_1d(
    signal: pd.Series, x_values: List[float] = [0.0, 1.0], y_values: List[float] = [0.0, 1.0]
) -> pd.Series:
    """1D interpolation of a sequence.

    The input time serie is interpolated to the input sequence to create the return timeseries.
    The x_values represent the input timeseries and the y_values represent the output timeseries.
    The interpolation routine is a simple linear interpolation

    Args:
        signal (pandas.Series): Time series
        x_values (List[float]): The x-values
            List of coefficients separated by commas. The numbers must be entered separated by commas (e.g., 0, 1).
            The default is :math:`0.0, 1.0`, which returns the original time series.
            The number of parameters must match the y_values.
        y_values (List[float]): The y-values
            List of coefficients separated by commas. The numbers must be entered separated by commas (e.g., 0, 1).
            The default is :math:`0.0, 1.0`, which returns the original time series.
            The number of parameters must match the x_values.

    Returns:
        pd.Series: Output.
    """
    n_x, n_y = len(x_values), len(y_values)
    if not n_x == n_y:
        raise UserValueError("There is a different number of x and y parameters. len(x)=%d,len(y)=%d" % (n_x, n_y))
    if n_x < 2:
        raise UserValueError("We need at least two values to do the interpolation")

    from scipy.interpolate import interp1d

    interpolator = interp1d(x_values, y_values)
    output = interpolator(signal.values)
    output_series = pd.Series(output, index=signal.index)

    return output_series


@check_types
def sequence_interpolation_2d(
    signal_x: pd.Series,
    signal_y: pd.Series,
    interp_x: List[float] = [0.0, 1.0],
    interp_y: List[float] = [0.0, 1.0],
    interp_z: List[float] = [0.0, 1.0],
    align_timesteps: bool = False,
) -> pd.Series:
    """2D interpolation of a sequence.

    The input time series is interpolated to the input sequence to create the return timeseries.
    The x_values represent the input timeseries and the y_values represent the output timeseries.
    The interpolation routine is a simple linear interpolation
    Args:
        signal_x (pandas.Series): Time series x-value
        signal_y (pandas.Series): Time series y-value
        interp_x (List[float]): The x-values
            List of coefficients separated by commas. The numbers must be entered separated by commas (e.g., 0, 1).
            The default is :math:`0.0, 1.0`, which returns the original time series.
            The number of parameters must match the y- and z-values.
        interp_y (List[float]): The y-values
            List of coefficients separated by commas. The numbers must be entered separated by commas (e.g., 0, 1).
            The default is :math:`0.0, 1.0`, which returns the original time series.
            The number of parameters must match the x- and z-values.
        interp_z (List[float]): The z-values
            List of coefficients separated by commas. The numbers must be entered separated by commas (e.g., 0, 1).
            The default is :math:`0.0, 1.0`, which returns the original time series.
            The number of parameters must match the x- and y-values.
        align_timesteps: Auto-align.
            Automatically align time stamp  of input time series. Default is False.

    Returns:
        pd.Series: Output.
    """
    n_x, n_y, n_z = len(interp_x), len(interp_y), len(interp_z)
    if not n_x == n_y and not n_x == n_z:
        raise UserValueError(
            "There is a different number of x, y and z parameters. len(x)=%d,len(y)=%d,len(z)=%d" % (n_x, n_y, n_z)
        )
    if n_x < 2:
        raise UserValueError("We need at least two values to do the interpolation")

    # auto-align
    signal_x, signal_y = auto_align([signal_x, signal_y], align_timesteps)

    # x_values.dtype
    from scipy.interpolate import LinearNDInterpolator

    interpolator = LinearNDInterpolator(list(zip(interp_x, interp_y)), interp_z)
    output = interpolator(signal_x.values, signal_y.values)
    output_series = pd.Series(output, index=signal_x.index)
    return output_series
