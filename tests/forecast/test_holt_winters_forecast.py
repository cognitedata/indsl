# Copyright 2021 Cognite AS
import warnings

import numpy as np
import pandas as pd
import pytest

from indsl.exceptions import UserRuntimeError, UserValueError
from indsl.forecast.holt_winters_predictor import holt_winters_predictor


@pytest.mark.extras
@pytest.mark.parametrize(
    "trend,seasonal_growth,trend_growth,trend_method,seasonal_method,fx,end_ts,periods,cycles",
    [
        (0.2, 1, 1, "add", "add", "1D", "12-31-2023", 365, 3),
        (0, 1, 1, "add", "add", "1D", "12-31-2023", 365, 3),
        (1, 1, 1, "add", "add", "1H", "03-25-2021 23:00", 7 * 24, 12),
        (0.5, 1, 1.5, "mul", "add", "1D", "12-31-2023", 30, 12 * 3),
        (0.1, 1.5, 1, "add", "mul", "1D", "12-31-2023", 30, 12 * 3),
        (0.2, 2, 1.5, "mul", "mul", "1D", "12-31-2025", 30, 12 * 5),
    ],
)
def test_holt_winters_predictor(
    create_data_holt_winters,
    trend,
    seasonal_growth,
    trend_growth,
    trend_method,
    seasonal_method,
    fx,
    end_ts,
    periods,
    cycles,
):
    warnings.filterwarnings("ignore")
    test_data = create_data_holt_winters

    training_fraction = 0.8
    steps_to_end_of_data = len(test_data) - int(len(test_data) * training_fraction)

    res = holt_winters_predictor(
        test_data,
        periods,
        seasonality=seasonal_method,
        trend=trend_method,
        steps=steps_to_end_of_data,
        train_fraction=training_fraction,
    )

    np.testing.assert_allclose(res, test_data.tail(steps_to_end_of_data), rtol=0.2)


@pytest.mark.core
def test_holt_winters_predictor_validation_errors():
    data = pd.Series(dtype=np.float64)
    with pytest.raises(UserRuntimeError) as excinfo:
        holt_winters_predictor(data, seasonal_periods=7)
    assert "No data passed to algorithm." in str(excinfo.value)

    data = pd.Series(data=[0.2, 1, 2], index=pd.to_datetime(["12-31-2023", "12-31-2024", "12-31-2025"]))
    with pytest.raises(UserValueError) as excinfo:
        holt_winters_predictor(data, seasonal_periods=0)
    assert "seasonal_periods must be an integer greater than 1. Got 0" in str(excinfo.value)

    with pytest.raises(UserValueError) as excinfo:
        holt_winters_predictor(data, seasonal_periods=2, train_fraction=1)
    assert "train_fraction needs to be a float between 0 and 1. Got 1" in str(excinfo.value)

    data = pd.Series(data=[-1, 1, 3], index=pd.to_datetime(["12-31-2023", "12-31-2024", "12-31-2025"]))
    with pytest.raises(UserValueError) as excinfo:
        holt_winters_predictor(data, seasonal_periods=2, trend="mul")
    assert """When using "mul" for trend or seasonality components, data must be strictly > 0""" in str(excinfo.value)

    with pytest.raises(UserValueError) as excinfo:
        holt_winters_predictor(data, seasonal_periods=2, seasonality="mul")
    assert """When using "mul" for trend or seasonality components, data must be strictly > 0""" in str(excinfo.value)
