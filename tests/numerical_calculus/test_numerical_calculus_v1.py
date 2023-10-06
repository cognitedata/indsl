# Copyright 2021 Cognite AS

import numpy as np
import pandas as pd
import pytest

from indsl.exceptions import UserValueError
from indsl.ts_utils.numerical_calculus_v1 import differentiate, trapezoidal_integration


@pytest.mark.core
@pytest.mark.parametrize(
    "data_type,granularity,expected_result",
    [
        ("constant", "1h", [0.0, 0.0, 0.0]),
        ("constant", "12h", [0.0, 0.0, 0.0]),
        ("linear", "1h", [10.0 / 12 / 2, 10.0 / 12 / 2, 10.0 / 12 / 2]),
        ("linear", "12h", [5.0, 5.0, 5.0]),
    ],
)
def test_differentiate(create_data, data_type, granularity, expected_result):
    data = create_data[data_type]
    res = differentiate(data, granularity)
    assert (res == expected_result).all()


@pytest.mark.core
@pytest.mark.parametrize("periods", [0, 1])
def test_differentiate_fails(periods):
    index = pd.date_range(start="01-01-1970 00:00:00", periods=periods, end="01-02-1970 00:0:00")
    data = pd.Series([0] * periods, index=index, dtype=np.float64)

    with pytest.raises(UserValueError):
        differentiate(data, "1h")


@pytest.mark.core
@pytest.mark.parametrize(
    "data_type,granularity,expected_result",
    [
        ("constant", "1h", [0.0, 12.0, 24.0]),
        ("constant", "12h", [0.0, 1.0, 2.0]),
        ("linear", "1h", [0.0, 30, 120.0]),
        ("linear", "12h", [0.0, 2.5, 10.0]),
    ],
)
def test_trapezoidal_integration(create_data, data_type, granularity, expected_result):
    data = create_data[data_type]
    res = trapezoidal_integration(data, granularity)
    assert (res == expected_result).all()


@pytest.mark.core
def test_trapezoidal_integration_fails():
    data = pd.Series([], dtype=np.float64)

    with pytest.raises(UserValueError):
        trapezoidal_integration(data, "1h")
