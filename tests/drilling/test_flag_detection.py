# Copyright 2022 Cognite AS
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from pandas.testing import assert_series_equal

from indsl.drilling.flag_detection import (
    circulation_detection,
    inhole_detection,
    onbottom_detection,
    rotation_detection,
)
from indsl.exceptions import UserValueError
from indsl.ts_utils.utility_functions import generate_step_series


@pytest.mark.core
def test_rotation_normal():
    # basic rotation test
    # Define input ts
    date_range = pd.date_range(start=datetime(2000, 10, 1), end=datetime(2000, 10, 10), periods=6)
    rotation_ts = pd.Series([120, 120, 70, 20, 0, 0], index=date_range)

    # Expected res
    exp_res = generate_step_series(pd.Series([1, 1, 1, 1, 0, 0], index=date_range))

    # Check result
    res = rotation_detection(rotation_ts)
    assert_series_equal(res, exp_res, check_names=False)


@pytest.mark.core
def test_rotation_nan():
    # basic rotation test with nan behavior
    # Define input ts
    date_range = pd.date_range(start=datetime(2000, 10, 1), end=datetime(2000, 10, 10), periods=6)
    rotation_ts = pd.Series([120, 120, np.nan, 20, 0, 0], index=date_range)

    # Expected res
    exp_res = generate_step_series(pd.Series([1, 1, 0, 1, 0, 0], index=date_range))

    # Check result
    res = rotation_detection(rotation_ts)
    assert_series_equal(res, exp_res, check_names=False)


@pytest.mark.core
def test_rotation_nan_start():
    # basic rotation test with np.nan to start
    # Define input ts
    date_range = pd.date_range(start=datetime(2000, 10, 1), end=datetime(2000, 10, 10), periods=6)
    rotation_ts = pd.Series([np.nan, 120, 120, 0, 0, 0], index=date_range)

    # Expected res
    exp_res = generate_step_series(pd.Series([0, 1, 1, 0, 0, 0], index=date_range))

    # Check result
    res = rotation_detection(rotation_ts)
    assert_series_equal(res, exp_res, check_names=False)


@pytest.mark.core
def test_inhole_normal():
    # basic inhole test
    # Define input ts
    date_range = pd.date_range(start=datetime(2000, 10, 1), end=datetime(2000, 10, 10), periods=6)
    inhole_ts = pd.Series([0, 49, 51, 75, 100, 40], index=date_range)

    # Expected res
    exp_res = generate_step_series(pd.Series([0, 0, 1, 1, 1, 0], index=date_range))

    # Check result
    res = inhole_detection(inhole_ts)
    assert_series_equal(res, exp_res, check_names=False)


@pytest.mark.core
def test_inhole_nan():
    # basic inhole test
    # Define input ts
    date_range = pd.date_range(start=datetime(2000, 10, 1), end=datetime(2000, 10, 10), periods=6)
    inhole_ts = pd.Series([0, np.nan, 51, 75, 100, np.nan], index=date_range)

    # Expected res
    exp_res = generate_step_series(pd.Series([0, 0, 1, 1, 1, 0], index=date_range))

    # Check result
    res = inhole_detection(inhole_ts)
    assert_series_equal(res, exp_res, check_names=False)


@pytest.mark.core
def test_onbottom_normal():
    # basic onbottom test
    # Define input ts
    date_range = pd.date_range(start=datetime(2000, 10, 1), end=datetime(2000, 10, 10), periods=6)
    measured_depth_ts = pd.Series([0, 49, 51, 75, 100, 105], index=date_range)
    hole_depth_ts = pd.Series([100, 100, 100, 100, 100, 105], index=date_range)

    # Expected res
    exp_res = generate_step_series(pd.Series([0, 0, 0, 0, 1, 1], index=date_range))

    # Check result
    res = onbottom_detection(measured_depth_ts, hole_depth_ts)
    assert_series_equal(res, exp_res, check_names=False)


@pytest.mark.core
def test_onbottom_nan():
    # basic onbottom test
    # Define input ts
    date_range = pd.date_range(start=datetime(2000, 10, 1), end=datetime(2000, 10, 10), periods=6)
    measured_depth_ts = pd.Series([0, np.nan, 51, 75, np.nan, 105], index=date_range)
    hole_depth_ts = pd.Series([100, 100, 100, 100, 100, 105], index=date_range)

    # Expected res
    exp_res = generate_step_series(pd.Series([0, 0, 0, 0, 0, 1], index=date_range))
    # print(exp_res)

    # Check result
    res = onbottom_detection(measured_depth_ts, hole_depth_ts)
    # print(res)
    assert_series_equal(res, exp_res, check_names=False)


@pytest.mark.core
def test_onbottom_delta_depth_error():
    # basic onbottom test
    # Define input ts
    date_range = pd.date_range(start=datetime(2000, 10, 1), end=datetime(2000, 10, 10), periods=6)
    measured_depth_ts = pd.Series([0, 200, 51, 75, 200, 105], index=date_range)
    hole_depth_ts = pd.Series([100, 100, 100, 100, 100, 105], index=date_range)

    # Check if exception is raised when bit_depth excees the hole depth
    with pytest.raises(UserValueError, match="Bit depth cannot be greater than the hole depth"):
        _ = onbottom_detection(measured_depth_ts, hole_depth_ts)


@pytest.mark.core
def test_onbottom_nan_error():
    # basic onbottom test
    # Define input ts
    date_range = pd.date_range(start=datetime(2000, 10, 1), end=datetime(2000, 10, 10), periods=6)
    measured_depth_ts = pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], index=date_range)
    hole_depth_ts = pd.Series([100, 100, 100, 100, 100, 105], index=date_range)

    # Check if exception is raised when bit_depth excees the hole depth
    with pytest.raises(UserValueError, match="Bit depth contains all nan values"):
        _ = onbottom_detection(measured_depth_ts, hole_depth_ts)


@pytest.mark.core
def test_circulation_normal():
    # basic circulation test
    # Define input ts
    date_range = pd.date_range(start=datetime(2000, 10, 1), end=datetime(2000, 10, 10), periods=6)
    flow_rate_ts = pd.Series([0, 0, 5, 0, 10, 15], index=date_range)

    # Expected res
    exp_res = generate_step_series(pd.Series([0, 0, 1, 0, 1, 1], index=date_range))

    # Check result
    res = circulation_detection(flow_rate_ts)
    assert_series_equal(res, exp_res, check_names=False)


@pytest.mark.core
def test_circulation_nan():
    # basic circulation test
    # Define input ts
    date_range = pd.date_range(start=datetime(2000, 10, 1), end=datetime(2000, 10, 10), periods=6)
    flow_rate_ts = pd.Series([0, np.nan, 5, 0, 10, np.nan], index=date_range)

    # Expected res
    exp_res = generate_step_series(pd.Series([0, 0, 1, 0, 1, 0], index=date_range))

    # Check result
    res = circulation_detection(flow_rate_ts)
    assert_series_equal(res, exp_res, check_names=False)
