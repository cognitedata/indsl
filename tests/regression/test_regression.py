# Copyright 2021 Cognite AS
import warnings

import numpy as np
import pandas as pd
import pytest

from indsl.exceptions import UserValueError
from indsl.regression.polynomial import poly_regression
from tests.detect.test_utils import RNGContext


@pytest.mark.parametrize("method", ["No regularisation", "Lasso", "Ridge"])
def test_poly_regression(method):
    warnings.filterwarnings("ignore")

    # Setting up test polynomial data
    x = np.linspace(0, 10, 100)
    x_dt = pd.date_range(start="1970", periods=len(x), freq="1T")

    # Creating a polynomial of degree 2
    y_hat = 1e-2 * x**2 - 1e-1 * x + 2

    with RNGContext():
        y_tilda = np.random.normal(size=len(x), scale=0.05)

    y = y_hat + y_tilda

    test_data = pd.Series(y, index=x_dt)
    exp_res = pd.Series(y_hat, index=x_dt)

    # Call the regression
    res = poly_regression(test_data, method=method)

    # Allow for max 5% difference from the mean
    max_diff = np.sqrt(max((res - exp_res) ** 2))
    assert max_diff * 100 / np.mean(res) < 5


def create_test_data():
    # Setting up test polynomial data
    x = np.linspace(0, 10, 100)
    x_dt = pd.date_range(start="1970", periods=len(x), freq="1T")

    # Creating a polynomial of degree 2
    y_hat = 1e-2 * x**2 - 1e-1 * x + 2

    with RNGContext():
        y_tilda = np.random.normal(size=len(x), scale=0.05)

    y = y_hat + y_tilda
    test_data = pd.Series(y, index=x_dt)
    return test_data


@pytest.mark.extras
def test_poly_regression_validation_errors():
    data = create_test_data()
    with pytest.raises(UserValueError) as excinfo:
        poly_regression(data, alpha=0)
        assert "Alpha needs to be a float between 0 and 1" in str(excinfo.value)

    data = pd.Series(dtype=np.float64)
    with pytest.raises(UserValueError) as excinfo:
        poly_regression(data)
        assert f"Not enough data (got {len(data)} values) to perform operation (min 3 values required!)" in str(
            excinfo.value
        )
