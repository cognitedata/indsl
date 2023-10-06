# Copyright 2023 Cognite AS
from typing import List, Union

import numpy as np
import pandas as pd

from indsl.exceptions import UserValueError
from indsl.type_check import check_types
from indsl.validations import validate_series_has_time_index, validate_series_is_not_empty


@check_types
def alma(data: pd.Series, window: int = 10, sigma: float = 6, offset_factor: float = 0.75) -> pd.Series:
    """Arnaud Legoux moving average.

    Moving average typically used in the financial industry, which aims to strike a good balance between smoothness
    and responsiveness (i.e., capture a general smoothed trend without allowing for significant lag). It can be
    interpreted as a Gaussian weighted moving average with an offset, where the offset, spread, and window size are
    user-defined.

    Args:
        data: Time series.
        window: Window size.
            Defaults to 10 data points or time steps for uniformly sample time series.
        sigma: Sigma.
            Parameter that controls the width of the Gaussian filter. Defaults to 6.
        offset_factor: Offset factor.
            Parameter that controls the magnitude of the weights for each past observation within the window.
            Defaults to 0.75.

    Raises:
        UserValueError: Time series is empty.
        UserValueError: Data has to be a time series.
        UserValueError: Not enough data to perform the calculation.

    Returns:
        pandas.Series: Smoothed data.
    """
    validate_series_is_not_empty(data)
    validate_series_has_time_index(data)

    # Check data
    if len(data) <= window:
        raise UserValueError(f"Not enough data to perform calculation. Expected {window} but got {len(data)}")

    # Check inputs
    if window == sigma == 0:
        raise UserValueError(
            "window or sigma can't be zero. Please change these user defined values to positive values."
        )

    # Calculate weights
    offset = int(offset_factor * window)
    k = np.array(range(0, window))
    weights = np.exp(-((k - offset) ** 2) / (sigma**2))

    # Apply smoothing function
    res = data.rolling(window=window).apply(lambda x: calculate_alma(x, weights))

    return res.dropna()


@check_types
def calculate_alma(values: Union[List, pd.Series], weights: np.ndarray) -> float:
    """Calculate alma value for a window time.

    Args:
        values: Datapoints in the window time.
        weights: Weights to calculate Alma value.

    Returns:
        float: Calculated Alma value.
    """
    weighted_sum = weights * values
    alma = weighted_sum.sum() / weights.sum()
    return alma
