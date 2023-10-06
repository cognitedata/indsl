# Copyright 2021 Cognite AS
import math

import numpy as np
import pandas as pd
import pytest

from tests.detect.test_utils import RNGContext


@pytest.fixture
def create_data_arma():
    # Create data
    fx = "7H"
    x_dt = pd.date_range(start="1970", freq=fx, end="02-01-1970")
    x = np.linspace(0, 10, len(x_dt))

    y_hat = 1e-2 * x**2 - 1e-1 * x + 2
    with RNGContext():
        y_tilde = np.random.normal(size=len(x), scale=0.05)

    y = y_hat + y_tilde

    test_data = pd.Series(y, index=x_dt)
    perfect_data = pd.Series(y_hat, index=x_dt)

    return (perfect_data, test_data)


def _combine_single_periods_into_one(x, cycles, trend, seasonal_factor, trend_factor):
    one_period = np.sin(x)
    multiple_periods = one_period
    for i in range(1, cycles):
        one_period = one_period + trend
        multiple_periods = np.append(multiple_periods, seasonal_factor[i] * one_period * trend_factor[i])
    return multiple_periods


@pytest.fixture
def create_data_holt_winters(trend, seasonal_growth, trend_growth, fx: str, end_ts: str, periods: int, cycles: int):
    x_dt = pd.date_range(start="2021", freq=fx, end=end_ts)
    x = np.linspace(0, 2 * math.pi, len(x_dt) // cycles)

    cycles += 1  # add one cycle to account for missing days when fixed 30-day month periods are used
    seasonal_factor = np.linspace(1, seasonal_growth, cycles)
    trend_factor = np.linspace(1, trend_growth, cycles)

    multiple_periods = _combine_single_periods_into_one(x, cycles, trend, seasonal_factor, trend_factor)

    # lift time series so that all values are positive
    multiple_periods += abs(min(multiple_periods)) + 1

    with RNGContext():
        y_tilde = np.random.normal(size=len(multiple_periods), scale=0.05)
    y = multiple_periods + y_tilde

    return pd.Series(y[: len(x_dt)], index=x_dt)
