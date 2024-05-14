# Copyright 2021 Cognite AS
import warnings

import numpy as np
import pandas as pd
import pytest

from indsl.smooth.arma_smoother import arma
from tests.detect.test_utils import RNGContext


@pytest.mark.extras
def test_arma_smoother():
    warnings.filterwarnings("ignore")

    # Create data
    x = np.linspace(0, 10, 1000)
    x_dt = pd.date_range(start="1970", periods=len(x), freq="1h")

    y_hat = 1e-2 * x**2 - 1e-1 * x + 2
    with RNGContext():
        y_tilde = np.random.normal(size=len(x), scale=0.05)

    y = y_hat + y_tilde

    test_data = pd.Series(y, index=x_dt)
    perfect_data = pd.Series(y_hat, index=x_dt)

    # Run ARMA smoother
    res = arma(test_data)

    # Check discrepancy
    res_sse = sum((res - perfect_data) ** 2)
    data_sse = sum((test_data - perfect_data) ** 2)

    assert res_sse < 0.1 * data_sse
    assert res.shape == perfect_data.shape == test_data.shape
