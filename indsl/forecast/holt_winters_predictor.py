# Copyright 2023 Cognite AS

from typing import Literal

import numpy as np
import pandas as pd

from indsl import versioning
from indsl.exceptions import STATSMODELS_REQUIRED, UserRuntimeError, UserValueError
from indsl.type_check import check_types
from indsl.validations import validate_series_has_time_index


@versioning.register(version="1.0", deprecated=True)
@check_types
def holt_winters_predictor(
    data: pd.Series,
    seasonal_periods: int,
    seasonality: Literal["add", "mul"] = "add",
    trend: Literal["add", "mul"] = "add",
    dampen_trend: bool = False,
    steps: int = 1,
    train_fraction: float = 0.8,
) -> pd.Series:
    """Triple exponential smoothing.

    This technique (also known as Holt-Winters) can forecast time series data with a trend and
    seasonal variability.
    It works by utilizing exponential smoothing thrice - for the average value, the trend, and the seasonality.
    Values are predicted by combining the effects of these influences.

    Args:
        data: Time series.
        seasonal_periods: Number of periods per cycle.
            The value for the seasonal periods is chosen to denote the number of timesteps within a period,
            e.g., 7*24 for hourly data with a weekly seasonality or 365 for daily data with a yearly pattern. Note!
            A time series that shows a spike every day, but not on Sunday, does not have a daily seasonality, but a
            weekly seasonality!
        seasonality: Seasonality.
            Additive seasonality: Amount of adjustment is constant.
            Multiplicative seasonality: Amount of adjustment varies with the level of the series.
        trend: Trend.
            Additive seasonality: Amount of adjustment is constant.
            Multiplicative seasonality: Amount of adjustment varies with the level of the series.
        dampen_trend: Dampen the trend component.
            If the trend component shall be dampened. This method is useful to predict very far in the future,
            and it is reasonable to assume that the trend will not stay constant but flatten out.
        steps: Steps.
            Number of steps to forecast ahead of the training dataset.
        train_fraction: Fraction.
            Fraction of the input data used for training the model.

    Returns:
        pandas.Series: Prediction.
        Predicted data for the test fraction of the input data (e.g., 1 - train_fraction).

    Raises:
        UserRuntimeError: If an empty time series is passed into the function.
        UserTypeError: If a time series with the wrong index is provided.
        UserValueError: Incorrect values passed into keyword arguments.
    """
    if len(data) < 1:
        raise UserRuntimeError("No data passed to algorithm.")

    validate_series_has_time_index(data)

    if seasonal_periods <= 1:
        raise UserValueError(f"seasonal_periods must be an integer greater than 1. Got {seasonal_periods}")

    if not 0 < train_fraction < 1:
        raise UserValueError(f"train_fraction needs to be a float between 0 and 1. Got {train_fraction}")

    if trend == "mul" or seasonality == "mul":
        if any(data <= 0):
            raise UserValueError("""When using "mul" for trend or seasonality components, data must be strictly > 0""")

    n_obs = len(data)
    n_train = int(n_obs * train_fraction)
    dt_step_sec = int(data.reset_index()["index"].diff().mean().total_seconds())

    train_data = data.iloc[:n_train]
    pred_data = _train_and_return_forecast(
        train_data, seasonality, trend, seasonal_periods, dampen_trend, steps, dt_step_sec
    )

    return pred_data.dropna()


@check_types
def _train_and_return_forecast(
    train_data: pd.Series,
    seasonality: str,
    trend: str,
    seasonal_periods: int,
    dampen_trend: bool,
    steps: int,
    dt_step_sec: int,
) -> pd.Series:
    """Train and return forecast.

    Function to train an Holt-Winters model and return the forecasted value
    as a Pandas series with a datetime index.

    Args:
        train_data: data to train model on
        seasonality: Either "mul" or "add" - additive or multiplicative seasonality
        trend: Either "mul" or "add" - additive or multiplicative trend
        seasonal_periods: Number of periods in a complete seasonal cycle
        dampen_trend: If the trend should be dampened
        steps: number of steps to forecast ahead of training data
        dt_step_sec: expected granularity for each step (seconds)

    Returns:
        pandas.Series: Forecasted time series of length `steps`
    """
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing  # Lazy import
    except ImportError:
        raise ImportError(STATSMODELS_REQUIRED)

    fitted_model = ExponentialSmoothing(
        train_data,
        seasonal=seasonality,
        trend=trend,
        seasonal_periods=seasonal_periods,
        damped_trend=dampen_trend,
        initialization_method="heuristic",
    ).fit()
    y_pred = fitted_model.forecast(steps).values

    max_dt = train_data.index[-1]
    dt_index = max_dt + pd.TimedeltaIndex(np.linspace(1, steps, num=steps) * dt_step_sec, unit="s")

    return pd.Series(y_pred, index=dt_index)
