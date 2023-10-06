import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def test_data():
    data = [5.6, 7.8, np.nan, 56]
    index = pd.date_range(start="1970", periods=4, freq="1h")

    return pd.Series(data, index=index)


@pytest.fixture
def test_data_integrated():
    # Expected imputation is linear for test data (i.e. nan value will be be 31.9)
    # Data below is calculated by hand using trapezoidal integration technique
    data = [0, 6.7, 26.55, 70.5]
    index = pd.date_range(start="1970", periods=4, freq="1h")

    return pd.Series(data, index=index)
