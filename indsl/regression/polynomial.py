# Copyright 2023 Cognite AS

from typing import Literal

import numpy as np
import pandas as pd

from indsl.exceptions import SCIKIT_LEARN_REQUIRED, UserValueError
from indsl.type_check import check_types


@check_types
def poly_regression(
    data: pd.Series,
    order: int = 2,
    method: Literal["Lasso", "Ridge", "No regularisation"] = "No regularisation",
    alpha: float = 0.1,
) -> pd.Series:
    """Polynomial.

    Fit a polynomial curve of a specified degree to the data. Default method corresponds to a ordinary least squares
    fitting procedure but method can be changed to allow for L1 or L2 regularisation.

    Args:
        data: Time series.
            Data to fit the polynomial regression
        order: Polynomial order.
        method: Method.
            Type of regularisation to apply (Lasso or Ridge). Default is simple linear least squares with no regularisation.
        alpha: Alpha.
            Only applies to either Ridge or Lasso methods which sets the penalty for either L2 or L1 regularisation.
            Value of 0 means that there is no penalty and this essentially equivalent to ordinary least squares.

    Returns:
        pd.Series: Fitted data.
    """
    try:
        from sklearn.linear_model import Lasso, LinearRegression, Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import PolynomialFeatures
    except ImportError:
        raise ImportError(SCIKIT_LEARN_REQUIRED)

    # Check alpha
    if not 0 < alpha < 1:
        raise UserValueError("Alpha needs to be a float between 0 and 1")

    # Select method and build pipeline
    if method == "No regularisation":
        fitter = LinearRegression()
    elif method == "Ridge":
        fitter = Ridge(alpha=alpha)
    elif method == "Lasso":
        fitter = Lasso(alpha=alpha)

    model = make_pipeline(PolynomialFeatures(order), fitter)

    # Transform variables
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    if len(data) < (order + 1):
        raise UserValueError(
            f"Not enough data (got {len(data)} values) to perform operation (min {order + 1} values required!)"
        )

    x = (np.array(data.index, dtype=np.int64) - data.index[0].value) / 1e9
    y = data.to_numpy()

    # Fit, transform and return result
    model.fit(x.reshape(-1, 1), y)
    y_pred = model.predict(x.reshape(-1, 1))
    return pd.Series(y_pred, index=data.index)
