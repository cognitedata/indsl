# Copyright 2021 Cognite AS
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def create_data():
    # Create data
    nx = 3
    x = np.linspace(0, 10, nx)
    x_dt = pd.date_range(start="01-01-1970 00:00:00", periods=nx, end="01-02-1970 00:0:00")

    constant_y = 0 * x + 1
    linear_y = x
    quadratic_y = x**2

    constant_data = pd.Series(constant_y, index=x_dt)
    linear_data = pd.Series(linear_y, index=x_dt)
    quadratic_data = pd.Series(quadratic_y, index=x_dt)

    return {"constant": constant_data, "linear": linear_data, "quadratic": quadratic_data}
