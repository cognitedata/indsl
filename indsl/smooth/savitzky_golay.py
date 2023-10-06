# Copyright 2023 Cognite AS
from typing import Optional

import pandas as pd

from scipy.signal import savgol_filter

from indsl.exceptions import UserValueError

# noinspection SpellCheckingInspection
from indsl.type_check import check_types


@check_types
def sg(data: pd.Series, window_length: Optional[int] = None, polyorder: int = 1) -> pd.Series:
    """Saviztky-Golay.

    Use this filter for smoothing data without distorting the data tendency. The method is independent of
    the sampling frequency. Hence, it is simple and robust to apply to data with non-uniform sampling. If you work with
    high-frequency data (e.g., sampling frequency ~> 1 Hz), we recommend that you provide the filter window length and
    polynomial order parameters to suit the requirements. Otherwise, if no parameters are provided, the function will
    estimate and set the parameters based on the characteristics of the input time series (e.g., sampling frequency).

    Args:
        data: Time series.
        window_length: Window.
            Point-wise length of the filter window (i.e., number of data points). A large window results in a stronger
            smoothing effect and vice-versa. If the filter window_length is not defined by the user, a
            length of about 1/5 of the length of time series is set.
        polyorder: Polynomial order.
            Order of the polynomial used to fit the samples. Must be less than filter window length. Defaults to 1.
            Hint: A small polyorder (e.g., polyorder = 1) results in a stronger data smoothing effect, representing the
            dominating trend and attenuating data fluctuations.

    Returns:
        pandas.Series: Smoothed time series.

    Raises:
        UserValueError: The window length must be a positive odd integer
        UserValueError: The window length must be less than or equal to the number of data points in your time series
        UserValueError: The polynomial order must be less than the window length
    """
    if window_length is None:
        window_length = len(data) // 5
    if window_length % 2 == 0:
        window_length += 1  # The filter requires the window to be odd
    if window_length <= 0:
        raise UserValueError("The window length must be a positive odd integer.")
    if window_length > len(data):
        raise UserValueError(
            "The window length must be less than or equal to the number of data points in your time series."
        )
    if polyorder >= window_length:
        raise UserValueError("The polynomial order must be less than the window length.")
    return pd.Series(savgol_filter(data, window_length=window_length, polyorder=polyorder), index=data.index)
