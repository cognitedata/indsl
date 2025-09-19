# Copyright 2025 Cognite AS

import numpy as np
import pandas as pd
import pytest

from indsl.statistics.rolling_standard_deviation import rolling_stddev
from indsl.exceptions import UserValueError


@pytest.mark.core
def test_rolling_stddev_predefined_values_min_periods_1():
    idx = pd.date_range("2022-01-01 00:00:00", periods=5, freq="1min")
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)

    window = pd.Timedelta(minutes=3)
    result = rolling_stddev(s, time_window=window, min_periods=1)

    # Expected sample standard deviation (ddof=1) over trailing 3-minute window
    # Windows:
    # t0: [1] -> 0 (filled from NaN)
    # t1: [1,2] -> sqrt(0.5)
    # t2: [1,2,3] -> 1
    # t3: [2,3,4] -> 1
    # t4: [3,4,5] -> 1
    expected = np.array([0.0, np.sqrt(0.5), 1.0, 1.0, 1.0])

    np.testing.assert_allclose(
        result.values,
        expected,
        rtol=0,
        atol=1e-12,
        err_msg="rolling_stddev should match predefined sample standard deviations",
    )


@pytest.mark.core
def test_rolling_stddev_predefined_values_min_periods_3():
    idx = pd.date_range("2022-01-01 00:00:00", periods=5, freq="1min")
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)

    window = pd.Timedelta(minutes=3)
    result = rolling_stddev(s, time_window=window, min_periods=3)

    # With min_periods=3, first two entries are filled to 0; then std is 1
    expected = np.array([0.0, 0.0, 1.0, 1.0, 1.0])

    np.testing.assert_allclose(
        result.values,
        expected,
        rtol=0,
        atol=1e-12,
        err_msg="rolling_stddev with min_periods=3 should match predefined values",
    )


@pytest.mark.core
def test_rolling_stddev_without_time_unit_raises():
    idx = pd.date_range("2022-01-01 00:00:00", periods=5, freq="1min")
    s = pd.Series(np.arange(5, dtype=float), index=idx)

    with pytest.raises(UserValueError):
        rolling_stddev(s, time_window=pd.Timedelta(15), min_periods=1)


