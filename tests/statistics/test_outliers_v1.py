# Copyright 2021 Cognite AS

import numpy as np
import pandas as pd
import pytest

from pandas.testing import assert_series_equal

from indsl.exceptions import UserValueError
from indsl.statistics.outliers_v1 import detect_outliers, remove_outliers


def generate_data_with_outliers(num_points: int = 100):
    """Generate a pandas series with a timestamp and value from a standard
    normal distribution containing outliers."""
    rng1 = np.random.default_rng(0)
    mu = 0
    sigma = 1
    values = rng1.normal(0, 1, 100)
    outliers_positive = rng1.uniform(low=3 * sigma, high=5 * sigma, size=10)
    outliers_negative = rng1.uniform(low=-5 * sigma, high=-3 * sigma, size=10)
    values = np.concatenate((outliers_positive, outliers_negative, rng1.normal(mu, sigma, 1000)), axis=0)
    rng1.shuffle(values)

    return pd.Series(values, index=pd.date_range("2021-02-09 00:00:00", "2021-03-09 09:00:00", periods=1020))


@pytest.mark.extras
def test_outlier_removal():
    """
    Unit test for the outlier removal function. For chosen default parameters and a series of 1020 values,
    only 31 outliers are expected to be removed. This unit test checks that the series that is retuned has removed 31
    datapoints.
    """
    test_data = generate_data_with_outliers()
    result = remove_outliers(test_data)
    assert len(result) == 989


@pytest.mark.parametrize(
    "steady_state",
    ([pd.Series([1] * i, index=pd.date_range(0, periods=i, freq="1s"), dtype=np.float64) for i in range(10)]),
)
@pytest.mark.extras
def test_outlier_removal_steady_state_input_returns_idendity(steady_state):
    result = remove_outliers(steady_state)
    assert_series_equal(steady_state, result)


@pytest.mark.parametrize(
    "input",
    (
        [1, 1, 1, 1, 6],
        [1, 1, 1, 1, 1, 6],
        [1, 1, 1, 1, 1, 1, 1, 6],
    ),
)
@pytest.mark.extras
def test_outlier_removal_single_outlier_is_removed(input):
    test_data = pd.Series(input, index=pd.date_range(0, periods=len(input), freq="1s"))
    result = remove_outliers(test_data)
    assert_series_equal(result, test_data[:-1])


@pytest.mark.extras
def test_outlier_detection():
    """
    Unit test for the outlier detection function. For chosen default parameters and a series of 1020 values,
    only 31 outliers are expected to be removed. This unit test checks that the series that is retuned has identified
    31 datapoints as outliers.
    """

    test_data = generate_data_with_outliers()
    result = detect_outliers(test_data)

    assert (len(np.where(result == 1)[0])) == 31


@pytest.mark.extras
def test_outliers_errors():
    """Unit test for dbscan errors."""
    test_data = generate_data_with_outliers()
    with pytest.raises(UserValueError) as excinfo:
        remove_outliers(data=test_data, eps=0.0)
    assert "eps should be > 0.0." == str(excinfo.value)

    with pytest.raises(UserValueError) as excinfo:
        remove_outliers(data=test_data, time_window="a")
    assert (
        "Time window should be a string in weeks, days, hours or minutes format:'3w', '10d', '5h', '30min', '10s' not a."
        == str(excinfo.value)
    )
