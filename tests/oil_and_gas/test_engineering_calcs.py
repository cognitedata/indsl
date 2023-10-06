# Copyright 2021 Cognite AS
import numpy as np
import pandas as pd
import pytest

from pandas.testing import assert_series_equal

from indsl.oil_and_gas.engineering_calcs import productivity_index


@pytest.mark.core
def test_PI_calc():
    # Define input ts
    p_res = pd.Series([10, 25, 3, 7], index=[0, 1, 2, 5])
    p_bh = pd.Series([8, 2.5, 3.9, 7], index=[0, 1, 4, 5])
    Q_gas = pd.Series([78, 21, 2, 5], index=[0, 1, 2, 3])

    # Expected res
    exp_res = pd.Series([38.9998869, 0.93333296], index=[0, 1])

    # Check result
    res = productivity_index(p_res, p_bh, Q_gas)
    assert_series_equal(res, exp_res)


@pytest.mark.core
def test_PI_calc_raise_error():
    # Define input ts
    p_res = pd.Series([10, 25, 3, 7], index=[0, 1, 2, 5])
    p_bh = pd.Series([8, 2.5, 3.9, 7], index=[0, 1, 4, 5])
    Q_gas = pd.Series(dtype=np.int64)

    # Check if exception is raised
    with pytest.raises(RuntimeError, match="One of the inputs has no data. Please check all time series inputs."):
        _ = productivity_index(p_res, p_bh, Q_gas)
