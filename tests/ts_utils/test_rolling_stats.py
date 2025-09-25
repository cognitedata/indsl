# Copyright 2025 Cognite AS

import numpy as np
import pandas as pd
import pytest

from indsl.ts_utils.rolling_stats import rolling_stddev
from indsl.exceptions import UserValueError


@pytest.mark.core
@pytest.mark.parametrize(
    "min_periods, expected",
    [
        pytest.param(1, np.array([0.0, np.sqrt(0.5), 1.0, 1.0, 1.0]), id="minimum periods one"),
        pytest.param(3, np.array([0.0, 0.0, 1.0, 1.0, 1.0]), id="minimum periods three"),
    ],
)
def test_rolling_stddev_different_minimum_periods(min_periods, expected):
    idx = pd.date_range("2022-01-01 00:00:00", periods=5, freq="1min")
    data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)

    window = pd.Timedelta(minutes=3)
    result = rolling_stddev(data, time_window=window, min_periods=min_periods)

    np.testing.assert_allclose(
        result.values,
        expected,
        rtol=0,
        atol=1e-12,
        err_msg="rolling_stddev should match predefined sample standard deviations",
    )


@pytest.mark.core
def test_rolling_stddev_without_time_unit_raises():
    idx = pd.date_range("2022-01-01 00:00:00", periods=5, freq="1min")
    data = pd.Series(np.arange(5, dtype=float), index=idx)

    with pytest.raises(UserValueError):
        rolling_stddev(data, time_window=pd.Timedelta(15), min_periods=1)


@pytest.mark.core
@pytest.mark.parametrize(
    "window_minutes, expected",
    [
        pytest.param(1, np.array([0.0, 0.0, 0.0, 0.0, 0.0]), id="window one minute"),
        pytest.param(
            2, np.array([0.0, np.sqrt(0.5), np.sqrt(0.5), np.sqrt(0.5), np.sqrt(0.5)]), id="window two minutes"
        ),
        pytest.param(3, np.array([0.0, np.sqrt(0.5), 1.0, 1.0, 1.0]), id="window three minutes"),
        pytest.param(
            4, np.array([0.0, np.sqrt(0.5), 1.0, 1.2909944487358056, 1.2909944487358056]), id="window four minutes"
        ),
        pytest.param(5, np.array([0.0, np.sqrt(0.5), 1.0, 1.2909944487358056, np.sqrt(2.5)]), id="window five minutes"),
    ],
)
def test_rolling_stddev_different_time_windows(window_minutes, expected):
    idx = pd.date_range("2022-01-01 00:00:00", periods=5, freq="1min")
    data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)

    window = pd.Timedelta(minutes=window_minutes)
    result = rolling_stddev(data, time_window=window, min_periods=1)

    np.testing.assert_allclose(
        result.values,
        expected,
        rtol=0,
        atol=1e-12,
        err_msg="rolling_stddev should produce the expected series for the given time window",
    )


@pytest.mark.core
@pytest.mark.parametrize(
    "freq, expected",
    [
        pytest.param(
            "1min",
            np.array([0.0, np.sqrt(0.5), np.sqrt(0.5), np.sqrt(0.5), np.sqrt(0.5)]),
            id="frequency one minute",
        ),
        pytest.param(
            "30s",
            np.array([0.0, np.sqrt(0.5), 1.0, 1.2909944487358056, 1.2909944487358056]),
            id="frequency thirty seconds",
        ),
        pytest.param(
            "2min",
            np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            id="frequency two minutes",
        ),
    ],
)
def test_rolling_stddev_different_input_frequencies(freq, expected):
    # Keep the time window fixed at 2 minutes and vary the sampling frequency
    idx = pd.date_range("2022-01-01 00:00:00", periods=5, freq=freq)
    data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)

    window = pd.Timedelta(minutes=2)
    result = rolling_stddev(data, time_window=window, min_periods=1)

    np.testing.assert_allclose(
        result.values,
        expected,
        rtol=0,
        atol=1e-12,
        err_msg="rolling_stddev should produce the expected series for different sampling frequencies",
    )
