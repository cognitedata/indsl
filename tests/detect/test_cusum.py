# Copyright 2022 Cognite AS
import re
import unittest.mock as mock

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from pandas.testing import assert_series_equal

from indsl.detect.cusum import Cusum, cusum
from indsl.exceptions import MATPLOTLIB_REQUIRED, UserValueError
from tests.detect.test_utils import RNGContext


@pytest.fixture
def cusum_data(scope="module"):
    """Generate data.

    RNGContext is used to set the seeds and then return them
    to normal in the Python environment.
    """
    # generate the data
    with RNGContext():
        y = np.random.randn(300)
    y[100:200] += 6
    index = pd.date_range(start="1970", periods=300, freq="1min")
    return pd.Series(y, index=index)


@pytest.fixture
def cusum_params(scope="module"):
    return {
        "threshold": 4,
        "drift": 1.5,
        "detect": "both",
        "predict_ending": True,
        "alpha": 0.05,
    }


@pytest.mark.core
@pytest.mark.parametrize(
    "expected_res",
    [
        (
            [
                pd.Timestamp("1970-01-01 01:39:00"),
                pd.Timestamp("1970-01-01 01:56:00"),
                pd.Timestamp("1970-01-01 03:37:00"),
                pd.Timestamp("1970-01-01 03:47:00"),
            ]
        ),
    ],
)
def test_cusum(cusum_params, expected_res, cusum_data):
    """Unit test for the CUSUM change point detector.

    For chosen default values, only 2 change periods are expected. This
    unit tests checks that the initial and final time of the occur when
    expected.
    """
    # call method
    res = cusum(
        data=cusum_data,
        threshold=cusum_params["threshold"],
        drift=cusum_params["drift"],
        detect=cusum_params["detect"],
        predict_ending=cusum_params["predict_ending"],
        alpha=cusum_params["alpha"],
        return_series_type="cusum_binary_result",
    )
    change_points = res[res.diff().abs() == 1].index.tolist()

    # assertions
    assert change_points == expected_res


@pytest.mark.core
@pytest.mark.parametrize(
    "return_series_type, expected_0_values", [("positive_cumulative_sum", 263), ("negative_cumulative_sum", 259)]
)
def test_cusum_cumulative_values(cusum_params, return_series_type, expected_0_values, cusum_data):
    """Unit test for the calculated cumulative values.

    This unit test checks the number of times the cumulative series
    restarts from 0.
    """
    # call method
    res = cusum(
        data=cusum_data,
        threshold=cusum_params["threshold"],
        drift=cusum_params["drift"],
        detect=cusum_params["detect"],
        predict_ending=cusum_params["predict_ending"],
        alpha=cusum_params["alpha"],
        return_series_type=return_series_type,
    )
    restart_times = (res == 0).sum()

    # assertions
    assert restart_times == expected_0_values


@pytest.mark.core
def test_cusum_errors():
    """Unit test for the CUSUM change point detector.

    The data expected is a non empty pandas.Series with a non empty
    DateTime index.
    """
    with pytest.raises(TypeError):
        cusum([])

    with pytest.raises(TypeError):
        x = pd.Series([1, 2], dtype=np.float64)
        cusum(x)

    with pytest.raises(ValueError):
        x = pd.Series([], index=pd.to_datetime([]), dtype=np.float64)
        cusum(x)

    with pytest.raises(ValueError):
        x = pd.Series([], dtype=np.float64)
        cusum(x)


@pytest.mark.core
def test_plot_cusum(cusum_data):
    cusum_obj = Cusum(data=cusum_data, threshold=4, drift=1.5, detect="both", predict_ending=True, alpha=0.05)

    # test ImportError
    with mock.patch.dict("sys.modules", {"matplotlib": None}):
        with pytest.raises(ImportError, match=re.escape(MATPLOTLIB_REQUIRED)):
            cusum_obj._plot_cusum()

    # test plot functionality
    pytest.importorskip("matplotlib.pyplot")  # skip the test in case matplotlib is not installed
    with mock.patch("matplotlib.pyplot.show", return_value=None):
        cusum_obj.cusum(plot_cusum=True)


@pytest.mark.core
def test_cusum_time_index_not_increasing():
    not_increasing_time_index_series = pd.Series(
        np.zeros(3),
        index=pd.DatetimeIndex(
            [
                datetime(2010, 1, 1),
                datetime(2009, 1, 1),
                datetime(2011, 1, 1),
            ]
        ),
    )

    with pytest.raises(UserValueError) as excinfo:
        Cusum(data=not_increasing_time_index_series)
    expected = "Time series index is not increasing."
    assert expected in str(excinfo.value)


def create_cusum_series(number1: int = 0, number2: int = 0):
    y = np.zeros(20)
    y[4:9] += number1
    y[10:20] += number2
    datetime_index = pd.date_range("1980", periods=20, freq="1s")
    increasing_series = pd.Series(y, index=datetime_index)
    return increasing_series


@pytest.mark.parametrize("detect", ["increase", "both"])
def test_cusum_detect_changes_above_threshold(detect):
    increasing_series = create_cusum_series(number1=-0, number2=3)
    expected_alarm_index = 12

    Cusum_increasing_object = Cusum(data=increasing_series, threshold=5.0, detect=detect, drift=0.7)
    Cusum_increasing_object._detect_changes()
    assert expected_alarm_index == Cusum_increasing_object.time_alarm[0]


@pytest.mark.core
def test_cusum_detect_changes_decrease_alarm():
    decreasing_series = create_cusum_series(number1=0, number2=-3)
    expected_alarm_index = 12

    Cusum_decreasing_object = Cusum(data=decreasing_series, threshold=5.0, detect="decrease", drift=0.7)
    Cusum_decreasing_object._detect_changes()
    assert expected_alarm_index == Cusum_decreasing_object.time_alarm[0]


@pytest.mark.core
def test_cusum_detect_ending():
    class OverwriteAttributesCusum(Cusum):
        def time_alarm_initial_is_largest(self):
            self.time_alarm_initial = np.array([11, 15, 19])
            self.time_alarm_final = np.array([13, 17])
            self.time_alarm = np.array([12, 16])

        def time_alarm_final_is_largest(self):
            self.time_alarm_initial = np.array([11, 15])
            self.time_alarm_final = np.array([13, 18, 22])
            self.time_alarm = np.array([12, 16])

        def time_alarm_intercalated_change(self):
            self.time_alarm_initial = np.array([11, 15])
            self.time_alarm_final = np.array([16, 18])
            self.time_alarm = np.array([12, 16])

        def test_detect_ending(self):
            assert self.time_alarm_initial.size == self.time_alarm_final.size

        def test_alarm_intercalated_change(self):
            assert self.time_alarm_initial.size == 1 and self.time_alarm_final.size == 1

    no_time_alarm_series = create_cusum_series()
    Cusum_object = OverwriteAttributesCusum(data=no_time_alarm_series)

    # time_alarm_final is larger than time_alarm_initial
    Cusum_object.time_alarm_final_is_largest()
    Cusum_object._fix_time_alarm_size_errors()
    Cusum_object.test_detect_ending()

    # time_alarm_initial is larger than time_alarm_final
    Cusum_object.time_alarm_initial_is_largest()
    Cusum_object._fix_time_alarm_size_errors()
    Cusum_object.test_detect_ending()

    # ending of change is after beginning of next change
    Cusum_object.time_alarm_intercalated_change()
    Cusum_object._fix_time_alarm_size_errors()
    Cusum_object.test_alarm_intercalated_change()


@pytest.mark.core
def test_create_binary_output_no_alarm():
    Cusum_object1 = Cusum(data=create_cusum_series())
    res = Cusum_object1._create_binary_output()
    assert res.mean() == 0


@pytest.mark.core
def test_cusum_mean_data():
    increasing_series = pd.Series([0.0, 2.0, 4.0], index=pd.date_range("1980", periods=3, freq="1s"))

    res = cusum(data=increasing_series, return_series_type="mean_data")
    exp_res = pd.Series([0.0, 1.0, 2.0], index=pd.date_range("1980", periods=3, freq="1s"))
    assert_series_equal(res, exp_res, rtol=0.1)


@pytest.mark.parametrize("alpha", [(0), (1.1)])
def test_cusum_invalid_alpha_values(alpha):
    """Test that the Cusum class raises an error when alpha is not between 0 and 1."""
    increasing_series = create_cusum_series(number1=0, number2=3)
    with pytest.raises(UserValueError) as excinfo:
        Cusum_increasing_object = Cusum(
            data=increasing_series,
            threshold=5.0,
            detect="increase",
            drift=0.7,
            alpha=alpha,
        )
    expected = f"Invalid alpha value: {alpha}. Alpha should be in the range (0, 1]."
    assert expected in str(excinfo.value)
