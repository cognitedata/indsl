# Copyright 2023 Cognite AS
import numpy as np
import pandas as pd

from scipy.integrate import cumulative_trapezoid

from indsl import versioning
from indsl.exceptions import UserValueError
from indsl.ts_utils.ts_utils import granularity_to_ms
from indsl.type_check import check_types


def granularity_to_ns(granularity, time_unit):  # noqa: D103
    if time_unit is None or time_unit.lower() != "auto":
        granularity = time_unit
    return granularity_to_ms(granularity) * 1e6


@versioning.register(version="1.0", deprecated=True)
@check_types
def trapezoidal_integration(series: pd.Series, granularity: str = "1h", time_unit: str = "auto"):
    """Integration.

    Cumulative integration using trapezoidal rule with an optional user-defined time unit.

    Args:
        series: Time series.
        granularity: Granularity.
            Current granularity for the chart on-screen (auto-given).
        time_unit: Frequency.
            User defined granularity to potentially override unit of time.
            Accepts integer followed by time unit string (s|m|h|d). For example: '1s', '5m', '3h' or '1d'.

    Returns:
        pandas.Series: Cumulative integral.
    """
    if len(series) < 1:
        raise UserValueError(f"Expected series to be of length > 0, got length {len(series)}")
    gran_ns = granularity_to_ns(granularity, time_unit)
    arr = cumulative_trapezoid(series, series.index.view(np.int64) / gran_ns, initial=0.0)
    return pd.Series(arr, index=series.index)


@versioning.register(version="1.0", deprecated=True)
@check_types
def differentiate(series: pd.Series, granularity: str = "1h", time_unit: str = "auto"):
    """Differentiation.

    Differentiation (finite difference) using a second-order accurate numerical method (central difference).
    Boundary points are computed using a first-order accurate method.

    Args:
        series: Time series.
        granularity: Granularity.
            Current granularity for the chart on-screen (auto-given).
        time_unit: Frequency.
            User defined granularity to potentially override unit of time.
            Accepts integer followed by time unit string (s|m|h|d). For example: '1s', '5m', '3h' or '1d'.

    Returns:
        pandas.Series: First order derivative.
    """
    if len(series) < 2:
        raise UserValueError(f"Expected series to be of length > 1, got length {len(series)}")
    gran_ns = granularity_to_ns(granularity, time_unit)
    arr = np.gradient(series, series.index.view(np.int64) / gran_ns)
    return pd.Series(arr, index=series.index)
