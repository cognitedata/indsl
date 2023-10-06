# Copyright 2023 Cognite AS
from typing import List

import pandas as pd

from indsl.type_check import check_types


@check_types
def univariate_polynomial(signal: pd.Series, coefficients: List[float] = [0.0, 1.0]) -> pd.Series:
    """Univariate polynomial.

    Creates a univariate polynomial :math:`y`, of degree :math:`n`, from the time series :math:`x`, and a list of
    coefficients :math:`a_{n}`:

    .. math::

        y = a_0 + a_1x + a_2x^2 + a_3x^3 + ... + a_nx^n

    Args:
        signal (pandas.Series): Time series
        coefficients (List[float]): Coefficients
            List of coefficients separated by commas. The numbers must be entered separated by commas (e.g., 0, 1).
            The default is :math:`0.0, 1.0`, which returns the original time series.

    Returns:
        pd.Series: Output
    """
    poly = pd.Series(0, index=signal.index)
    n = 0
    for a in coefficients:
        poly = poly + a * signal**n
        n = n + 1

    return poly
