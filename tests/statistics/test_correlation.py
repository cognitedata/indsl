# Copyright 2021 Cognite AS
import numpy as np
import pandas as pd
import pytest

from indsl.statistics.correlation import pearson_correlation
from tests.detect.test_utils import RNGContext


@pytest.fixture()
def generate_data(aligned):
    """Generate data.

    Generates two correlated time series. RNGContext is used
    to set the seeds and then return them to normal in the Python environment.

    Args:
        aligned: Boolean flag to determine if the index of the time series returned are aligned.

    Returns:
        Tuple of correlated Time Series
    """
    # generate the data
    with RNGContext(seed=10):
        y1 = np.random.randn(10)
        # create data2 from data1 with some noise
        y2 = y1.copy()
        y2 += 5
        y2 += np.random.randn(10) * 0.5  # add noise

    index1 = pd.date_range(start="1970", periods=10, freq="1min")

    if not aligned:
        index2 = pd.date_range(start="1970-01-01 00:00:30", periods=10, freq="1min")
    else:
        index2 = index1.copy()
    return pd.Series(y1, index=index1), pd.Series(y2, index=index2)


@pytest.mark.core
@pytest.mark.parametrize("aligned, last_expected_value", [(True, 0.909), (False, 0.53)])
def test_pearson_correlation(generate_data, last_expected_value):
    """Test pearson result."""
    data1, data2 = generate_data
    corr = pearson_correlation(data1, data2, time_window=pd.Timedelta(minutes=15), align_timesteps=True)
    assert round(corr.iloc[-1], 3) == last_expected_value


@pytest.mark.core
def test_pearson_errors():
    """Test errors.

    Unit test for the pearson correlation function.

    The data expected are two non empty pandas.Series with a non empty
    DateTime index.
    """
    with pytest.raises(TypeError):
        pearson_correlation(data1=[], data2=[])

    with pytest.raises(TypeError):
        x1 = pd.Series([1, 2], dtype=np.float64)
        x2 = pd.Series([1, 2], dtype=np.float64)
        pearson_correlation(data1=x1, data2=x2)

    with pytest.raises(ValueError):
        x1 = pd.Series([], index=pd.to_datetime([]), dtype=np.float64)
        x2 = pd.Series([], index=pd.to_datetime([]), dtype=np.float64)
        pearson_correlation(data1=x1, data2=x2)
