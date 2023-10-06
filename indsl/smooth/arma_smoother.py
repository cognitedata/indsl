# Copyright 2023 Cognite AS
import pandas as pd

from indsl.exceptions import STATSMODELS_REQUIRED
from indsl.type_check import check_types
from indsl.validations import validate_series_is_not_empty


@check_types
def arma(data: pd.Series, ar_order: int = 2, ma_order: int = 2) -> pd.Series:
    """Autoregressive moving average.

    The autoregressive moving average (ARMA) is a popular model used in forecasting. It uses an autoregression (AR)
    analysis to characterize the effect of past values on current values and a moving average to quantify the effect of the
    previous day's error (variation).


    Args:
        data: Time series.
        ar_order: AR order.
            Number of past data points to include in the AR model. Defaults to 2.
        ma_order: MA order.
            Number of terms in the MA model.  Defaults to 2.

    Returns:
        pandas.Series: Smoothed data.
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError:
        raise ImportError(STATSMODELS_REQUIRED)

    validate_series_is_not_empty(data)

    # Train model on entire dataset and return fit on dataset
    model = ARIMA(data, order=(ar_order, 0, ma_order))
    fit_model = model.fit()
    y_pred = fit_model.predict()

    return y_pred
