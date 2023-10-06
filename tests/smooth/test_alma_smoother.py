# Copyright 2021 Cognite AS
import numpy as np
import pandas as pd
import pytest

from indsl.smooth import alma
from tests.detect.test_utils import RNGContext


@pytest.mark.core
def test_alma_smooth_poly():
    # Creating a polynomial of degree 2
    x = np.linspace(0, 10, 1000)
    x_dt = pd.date_range(start="1970", periods=len(x), freq="1H")

    y_hat = 1e-2 * x**2 - 1e-1 * x + 2
    with RNGContext():
        y_tilda = np.random.normal(size=len(x), scale=0.05)

    y = y_hat + y_tilda

    test_data = pd.Series(y, index=x_dt)
    perfect_data = pd.Series(y_hat, index=x_dt)

    # Compute the SSE of both
    res = alma(test_data, sigma=6, offset_factor=0.75, window=10)

    data_SSE = ((perfect_data - test_data) ** 2).sum()
    res_SSE = ((perfect_data - res) ** 2).sum()

    # Check for an 80% reduction in the SSE
    assert res_SSE < 0.2 * data_SSE


@pytest.mark.core
def test_alma_smooth_linear():
    # Creating a polynomial of degree 2
    x = np.linspace(0, 10, 1000)
    x_dt = pd.date_range(start="1970", periods=len(x), freq="1H")

    with RNGContext():
        y_hat = 120 * np.ones(len(x))
        y_tilda = np.random.normal(size=len(x), scale=0.05)

    y = y_hat + y_tilda

    test_data = pd.Series(y, index=x_dt)
    perfect_data = pd.Series(y_hat, index=x_dt)

    # Compute the SSE of both
    res = alma(test_data, sigma=6, offset_factor=0.75, window=10)

    data_SSE = ((perfect_data - test_data) ** 2).sum()
    res_SSE = ((perfect_data - res) ** 2).sum()

    # Check for an 80% reduction in the SSE
    assert res_SSE < 0.2 * data_SSE
