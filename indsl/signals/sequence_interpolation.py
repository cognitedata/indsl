# Copyright 2024 Cognite AS
from typing import List

import pandas as pd

from indsl.exceptions import UserValueError
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
            The number of parameters must match the y_values.

    Returns:
        pd.Series: Output.
    """
    n_x, n_y = len(x_values), len(y_values)
    if not n_x == n_y:
        raise UserValueError("There is a different number of x and y parameters. len(x)=%d,len(x)=%d" % (n_x, n_y))
    if n_x or n_y < 2:
        raise UserValueError("We need at least two values to do the interpolation")
    # check is all values are floats
    # x_values.dtype

    from scipy.interpolate import interp1d

    interpolator = interp1d(x_values, y_values)
    output = interpolator(signal.values)
    output_series = pd.Series(output, index=signal.index)

    return output_series
