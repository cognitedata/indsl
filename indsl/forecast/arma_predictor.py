# Copyright 2023 Cognite AS
from enum import Enum

import numpy as np
import pandas as pd

from indsl import versioning
from indsl.exceptions import STATSMODELS_REQUIRED, UserRuntimeError, UserValueError
from indsl.type_check import check_types


class MethodType(Enum):
    """Method type."""

    ONESTEP = "onestep"
    MULTISTEP = "multistep"


@versioning.register(version="1.0", deprecated=True)
@check_types
def arma_predictor(
    data: pd.Series,
    ar_order: int = 2,
    ma_order: int = 2,
    train_fraction: float = 0.8,
    method: MethodType = MethodType.ONESTEP,
    steps: int = 1,
) -> pd.Series:
    """ARMA predictor.

    The ARMA predictor works by fitting constants to an auto regression (AR)  and a moving average (MA) equation and
    extrapolating the results.

    Args:
        data: Time series.
        ar_order: Number of AR terms.
        ma_order: Number of MA terms.
        train_fraction: Fraction.
            Fraction of the input data used for training the model.
        method: Method.
            Type of prediction to perform. MULTISTEP involves forecasting
            several steps ahead of the training dataset, while ONESTEP involves incrementally going through the test
            dataset, and appending it to the training dataset by performing a one-step forecast.
        steps: Steps.
            Number of steps to forecast ahead of the training dataset.

    Returns:
        pandas.Series: Prediction
        Predicted data for the test fraction of the input data (e.g., 1 - train_fraction)

    Raises:
        UserRuntimeError: If an empty time series is passed into the function.
        UserValueError: Incorrect values passed into keyword arguments. ar_order and ma_order should be integers, and train_fraction must be float between 0 and 1.
    """
    # Check length of data
    if len(data) < 1:
        raise UserRuntimeError("No data passed to algorithm.")

    # Check train-test split
    if not 0 < train_fraction < 1:
        raise UserValueError(f"train_fraction needs to be a float between 0 and 1. Got {train_fraction}")

    # Get key properties from data
    n_obs = len(data)
    n_train = int(n_obs * train_fraction)
    dt_step_sec = int(data.reset_index()["index"].diff().mean().total_seconds())

    # If method is multistep, then simply forecast some steps ahead of training dataset
    if method == MethodType.MULTISTEP:
        train_data = data.iloc[:n_train]
        pred_data = _train_and_return_forecast(ar_order, ma_order, train_data, steps, dt_step_sec)

    # If method is onestep, then iterate through test dataset and perform a one step prediction
    elif method == MethodType.ONESTEP:
        # Storage array
        store_forecast = []

        # Limiting number of cycles to 100
        cycles = min(n_obs - n_train, 100)

        for it in range(cycles + 1):
            # Append new observation
            train_data = data.iloc[it : it + n_train]
            res = _train_and_return_forecast(ar_order, ma_order, train_data, steps, dt_step_sec)
            store_forecast.append(res.tail(1))

        # Concatenate prediction dataframe
        pred_data = pd.concat(store_forecast)

    # If neither method, then raise value error
    else:
        raise UserValueError(f"Method needs to be either 'MethodType.ONESTEP' or 'MethodType.MULTISTEP'. Got {method}")

    # Return the prediction dataframe
    return pred_data.dropna()


@check_types
def _train_and_return_forecast(
    ar_order: int, ma_order: int, train_data: pd.Series, steps: int, dt_step_sec: int
) -> pd.Series:
    """Train and return forecast.

    Function to train an ARMA model and return the forecasted value as a Pandas series with a datetime index

    Args:
        ar_order: Number of AR terms in the equation
        ma_order: Number of MA terms in the equation
        train_data: data to train model on
        steps: number of steps to forecast ahead of training data
        dt_step_sec: expected granularity for each step (seconds)

    Returns:
        pandas.Series: Forecasted time series of length `steps`
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA  # Lazy import
    except ImportError:
        raise ImportError(STATSMODELS_REQUIRED)

    # Fit ARMA model
    model = ARIMA(train_data, order=(ar_order, 0, ma_order))
    fit_model = model.fit()

    # Perform prediction and return series with datetime index
    y_pred = fit_model.forecast(steps=steps).values
    max_dt = train_data.index[-1]
    # dt_index = pd.date_range(start=max_dt, freq=str(dt_step_sec) + "s", periods=steps + 1)
    dt_index = max_dt + pd.TimedeltaIndex(np.linspace(1, steps, num=steps) * dt_step_sec, unit="s")

    return pd.Series(y_pred, index=dt_index)
