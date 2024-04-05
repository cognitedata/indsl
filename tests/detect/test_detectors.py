# Copyright 2021 Cognite AS
import numpy as np
import pandas as pd
import pytest

from indsl.detect.change_point_detector import cpd_ed_pelt
from indsl.detect.steady_state import ssd_cpd, ssid
from indsl.detect.utils import resample_timeseries
from indsl.exceptions import UserTypeError, UserValueError
from tests.detect.test_utils import RNGContext


@pytest.mark.core
def test_invalid_inputs_throws_user_exception():
    df = pd.Series([], dtype=np.float64)
    with pytest.raises(UserTypeError) as e:
        resample_timeseries(df)
    assert "Expected a time series, got index type int64" in str(e.value)

    df = pd.Series([1], index=pd.to_datetime([0]))
    with pytest.raises(UserValueError) as e:
        resample_timeseries(df)
    assert "Expected data to be of length >= 2, got length 1" in str(e.value)


@pytest.mark.core
def test_steady_state_detector():
    """Unit test for the steady-state detector.

    For chosen default values, only 2 periods of transience are
    expected. This unit tests checks that they occur when expected and
    that only 2 are found. RNGContext is used to set the seeds and then
    return them to normal in the Python environment.
    """

    # Create artificial data with transient behaviour
    t = pd.date_range(start=pd.to_datetime("2019-01-01 00:00:00"), end=pd.to_datetime("2019-01-02 00:00:00"), freq="1min")
    with RNGContext():
        sig = np.random.normal(size=1000, scale=0.5) + 10
        disturbance = np.random.normal(size=200, scale=0.5) + 20
        sig2 = np.random.normal(size=len(t) - 1200, scale=0.5) + 10

    res = np.concatenate((sig, disturbance, sig2))

    df = pd.Series(res, index=t)

    # Call steady-state detector
    ss_res = ssid(df)

    # Check that algorithm has found the expected transition periods
    change_points = ss_res[ss_res.diff() == 1].index.tolist()

    exp_change_point_1 = t[1001]
    exp_change_point_2 = t[1201]

    assert exp_change_point_1 in change_points
    assert exp_change_point_2 in change_points

    # Expect only 2 change points
    assert len(change_points) == 2


@pytest.mark.core
def test_steady_state_detector_cpd():
    """Unit test for the steady-state detector.

    For chosen default values, only 2 periods of transience are
    expected. This unit tests checks that they occur when expected and
    that only 2 are found. RNGContext is used to set the seeds and then
    return them to normal in the Python environment.
    """
    # Create artificial data with transient behaviour
    t = pd.date_range(
        start=pd.to_datetime("2019-01-01 00:00:00"),
        end=pd.to_datetime("2019-01-02 00:00:00"),
        freq=pd.DateOffset(seconds=60),
    )
    with RNGContext():
        sig = np.random.normal(size=1000, scale=0.5) + 10
        disturbance = np.random.normal(size=200, scale=0.5) + 60
        sig2 = np.random.normal(size=len(t) - 1200, scale=0.5) + 10

    res = np.concatenate((sig, disturbance, sig2))

    df = pd.Series(res, index=t)

    # Call steady-state detector using the wrong argument types to test the input parsing
    ss_res = ssd_cpd(data=df, min_distance=1, var_threshold=15, slope_threshold=-3)

    # Check that algorithm has found the expected transition periods
    change_points = ss_res[ss_res.diff().abs() == 1].index.tolist()

    exp_change_point_1 = t[1000]
    exp_change_point_2 = t[1200]

    assert exp_change_point_1 in change_points
    assert exp_change_point_2 in change_points

    # Expect only 2 change points
    assert len(change_points) == 2


@pytest.mark.core
def test_change_point_detector():
    """Unit test for the change point detector.

    For chosen default values, only 2 periods of transitions are
    expected. This unit tests checks that they occur when expected and
    that only 2 are found. RNGContext is used to set the seeds and then
    return them to normal in the Python environment.
    """
    # Create artificial data with transient behaviour
    t = pd.date_range(
        start=pd.to_datetime("2019-01-01 00:00:00"),
        end=pd.to_datetime("2019-01-02 00:00:00"),
        freq=pd.DateOffset(seconds=60),
    )
    with RNGContext():
        sig = np.random.normal(size=1000, scale=0.5) + 10
        disturbance = np.random.normal(size=200, scale=0.5) + 60
        sig2 = np.random.normal(size=len(t) - 1200, scale=0.5) + 10

    res = np.concatenate((sig, disturbance, sig2))

    df = pd.Series(res, index=t)

    # Call the change point detector
    cpd_res = cpd_ed_pelt(data=df, min_distance=1)

    # Expected change points
    change_points = [1000, 1200]
    expected_ts = pd.Series(index=t, data=[0] * len(t))
    for cp in change_points:
        # sets the value of 1 to the timestamp of the current change point
        expected_ts.iloc[cp] = 1
        # add the value of 0 to 1ns before and after the timestamp of the current change point
        expected_ts = pd.concat(
            [
                expected_ts,
                pd.Series(
                    index=[
                        t[cp] - pd.Timedelta(value=1, unit="nanoseconds"),
                        t[cp] + pd.Timedelta(value=1, unit="nanoseconds"),
                    ],
                    data=[0, 0],
                ),
            ]
        )

    # Check that algorithm has found the expected change points
    assert cpd_res.equals(expected_ts.sort_index())
