from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from pandas.testing import assert_series_equal

from indsl.exceptions import UserRuntimeError, UserValueError
from indsl.ts_utils.ts_utils import (
    check_uniform,
    datetime_to_ms,
    fill_gaps,
    functional_mean,
    gaps_detector,
    granularity_to_ms,
    is_na_all,
    mad,
    make_uniform,
    scalar_to_pandas_series,
    time_ago_to_ms,
    time_difference,
    time_parse,
    time_string_to_ms,
    time_to_points,
)
from indsl.ts_utils.utility_functions import create_series_from_timesteps


@pytest.mark.core
def test_datetime_to_ms():
    # Arrange
    datetime_example = datetime(2020, 7, 13)
    # Act
    result = datetime_to_ms(datetime_example)
    # Assert
    assert result == 1594598400000


@pytest.mark.parametrize("string, expected_result", [("2days", None), ("2m", 120000)])
def test_time_to_string(string, expected_result):
    # Arrange
    pattern = r"(\d+)({})"
    _unit_in_ms_without_week = {"s": 1000, "m": 60000, "h": 3600000, "d": 86400000}
    # Act
    result = time_string_to_ms(pattern, string, _unit_in_ms_without_week)
    # Assert
    if expected_result is None:
        assert result is expected_result
    else:
        assert result == expected_result


@pytest.mark.core
def test_granularity_to_ms():
    # Arrange
    invalid_granularity = "4days"
    # Act
    with pytest.raises(UserValueError) as excinfo:
        granularity_to_ms(invalid_granularity)
    # Assert
    assert "Invalid granularity format: `4days`. Must be on format <integer>(s|m|h|d). E.g. '5m', '3h' or '1d'." in str(
        excinfo.value
    )


@pytest.mark.core
def test_time_ago_to_ms():
    assert time_ago_to_ms("now") == 0
    # Arrange
    invalid_time_ago_string = "3days-ago"
    # Act
    with pytest.raises(UserValueError) as excinfo:
        time_ago_to_ms(invalid_time_ago_string)
    # Assert
    assert "Invalid time-ago format: `{}`. Must be on format <integer>(s|m|h|d|w)-ago or 'now'. E.g. '3d-ago' or '1w-ago'.".format(
        invalid_time_ago_string
    ) in str(
        excinfo.value
    )

    # Arrange
    valid_time_ago_string = "5s-ago"
    # Act
    valid_result = time_ago_to_ms(valid_time_ago_string)
    # Assert
    assert valid_result == 5 * 1000


@pytest.mark.core
def test_functional_mean_fails_with_invalid_input_function():
    # Arrange
    empty_x_vals = []

    def mock(*args, **kwargs) -> None:
        return

    # Act
    with pytest.raises(UserValueError) as excinfo:
        functional_mean(mock, empty_x_vals)
    # Assert
    assert "No data in the input timeseries." in str(excinfo.value)


@pytest.mark.core
def test_is_na_all():
    # Arrange
    not_series_or_df = [1, 2, 3]
    # Act
    with pytest.raises(UserValueError) as excinfo:
        is_na_all(not_series_or_df)
    # Assert
    assert "Convenience method only supports Series or DataFrame." in str(excinfo.value)

    # Arrange
    na_all_data = {"numbers": [np.NaN, np.NaN]}
    dataframe = pd.DataFrame(na_all_data, columns=["numbers"])
    # Act
    result_dataframe = is_na_all(dataframe)
    # Assert
    assert result_dataframe

    # Arrange
    series = pd.Series([np.NaN, np.NaN])
    # Act
    result_series = is_na_all(series)
    # Assert
    assert result_series


@pytest.mark.core
def test_gaps_detector():
    assert gaps_detector(np.array([])) is None
    # Arrange
    timestamps = np.array([1329968892, 1329968892 + 1, 1329968892 + 86401])
    expected_result = np.column_stack(([1329968892 + 1], [1329968892 + 86401]))
    # Act
    result = gaps_detector(timestamps)
    # Assert
    assert np.array_equal(result, expected_result)


@pytest.mark.core
def test_check_uniform():
    # Arrange
    uniform_df = pd.DataFrame(
        {
            "datetime": [
                datetime(2020, 7, 13, 1, 0, 0),
                datetime(2020, 7, 14, 1, 0, 0),
                datetime(2020, 7, 15, 1, 0, 0),
            ],
            "values": [1, 2, 3],
        }
    )
    uniform_df.set_index("datetime", drop=True, inplace=True)
    # Act
    true_result = check_uniform(uniform_df)
    # Assert
    assert true_result

    # Arrange
    not_uniform_df = pd.DataFrame(
        {
            "datetime": [
                datetime(2020, 7, 13, 1, 0, 0),
                datetime(2020, 7, 14, 1, 0, 0),
                datetime(2020, 7, 16, 1, 0, 0),
            ],
            "values": [1, 2, 3],
        }
    )
    not_uniform_df.set_index("datetime", drop=True, inplace=True)
    # Act
    false_result = check_uniform(not_uniform_df)
    # Assert
    assert not false_result


@pytest.mark.core
def test_make_uniform():
    # Arrange
    not_uniform_df = pd.DataFrame(
        {
            "datetime": [
                datetime(2020, 7, 13, 1, 0, 1),
                datetime(2020, 7, 13, 1, 0, 3),
                datetime(2020, 7, 13, 1, 0, 4),
            ],
            "values": [1.0, 2.0, 3.0],
        }
    )
    not_uniform_df.set_index("datetime", drop=True, inplace=True)
    # Act
    result_interpolation = make_uniform(not_uniform_df, interpolation="linear")
    # Assert
    assert result_interpolation["values"][1] == 1.5
    assert not result_interpolation.isnull().values.any()

    # Act
    result_no_interpolation = make_uniform(not_uniform_df)
    # Assert
    assert result_no_interpolation.isnull().values.any()


@pytest.mark.core
def test_time_parse():
    # Arrange
    time_window = "40"
    # Act
    result = time_parse(time_window)
    # Assert
    assert result == pd.Timedelta("40min")

    # Arrange
    time_window = "days"
    # Act
    with pytest.raises(UserValueError) as excinfo:
        time_parse(time_window)
    # Assert
    assert (
        f"Time window should be a string in weeks, days, hours or minutes format:'3w', '10d', '5h', '30min', '10s' not {time_window}."
        == str(excinfo.value)
    )


@pytest.mark.core
def test_time_to_points():
    # Arrange
    time_window = "4s"
    data = pd.Series(
        [1, 1, 1, 1, 1],
        index=pd.DatetimeIndex(
            [
                datetime(2020, 7, 13, 1, 0, 1),
                datetime(2020, 7, 13, 1, 0, 3),
                datetime(2020, 7, 13, 1, 0, 4),
                datetime(2020, 7, 13, 1, 0, 7),
                datetime(2020, 7, 13, 1, 0, 8),
            ]
        ),
    )
    # Act
    result = time_to_points(data, time_window)
    # Assert
    assert 2 == result


@pytest.mark.core
def test_fill_gates():
    # Arrange
    not_uniform_data = pd.Series(
        [1, 2, np.NaN],
        index=pd.DatetimeIndex(
            [
                datetime(2020, 7, 13, 1, 0, 1),
                datetime(2020, 7, 13, 1, 0, 3),
                datetime(2020, 7, 13, 1, 0, 6),
            ]
        ),
    )
    # Act
    with pytest.raises(UserRuntimeError) as excinfo:
        fill_gaps(not_uniform_data)
    # Assert
    assert "The input time series is not uniform" in str(excinfo.value)

    # Arrange
    data = pd.Series(
        [1, 2, 3, 4, 5],
        index=pd.DatetimeIndex(
            [
                datetime(2020, 7, 13, 1, 0, 1),
                datetime(2020, 7, 13, 1, 0, 3),
                datetime(2020, 7, 13, 1, 0, 5),
                datetime(2020, 7, 13, 1, 0, 7),
                datetime(2020, 7, 13, 1, 0, 9),
            ]
        ),
    )
    # Act
    result = fill_gaps(data)
    # Assert
    assert_series_equal(data, result)

    # Arrange
    data = data.replace(4, np.NaN)

    # Act
    with pytest.raises(UserValueError) as excinfo:
        fill_gaps(data, interpolate_resolution=pd.Timedelta("0T"))
    # Assert
    assert "interpolate_resolution can not be 0" in str(excinfo.value)

    # Act
    with pytest.raises(UserValueError) as excinfo:
        fill_gaps(data, ffill_resolution=pd.Timedelta("0T"))
    # Assert
    assert "ffill_resolution can not be 0" in str(excinfo.value)

    # Act
    with pytest.raises(UserValueError) as excinfo:
        fill_gaps(data, granularity=pd.Timedelta("0T"))
    # Assert
    assert "granularity can not be 0" in str(excinfo.value)

    # Act
    result = fill_gaps(data, ffill_resolution=pd.Timedelta("3s"), interpolate_resolution=pd.Timedelta("4s"))
    expected_result = data.replace(np.NaN, 3)
    # Assert
    assert_series_equal(result, expected_result)


@pytest.mark.core
def test_scalar_to_pandas_series():
    # Arrange
    float_data = 3.4
    # Act
    float_result = scalar_to_pandas_series(float_data)
    # Assert
    assert float_result[0] == float_data

    # Arrange
    series_data = pd.Series(3.4, index=pd.date_range(start="1970", end=pd.Timestamp.now(), periods=2))
    # Act
    series_result = scalar_to_pandas_series(series_data)
    # Assert
    assert_series_equal(series_result, series_data)


@pytest.mark.core
def test_mad():
    rng = np.random.default_rng(1)
    data = pd.Series(rng.standard_normal(1000))

    assert round(mad(data), 2) == 0.67


@pytest.mark.core
def test_time_difference():
    second = timedelta(seconds=1)
    timesteps = 6 * [second] + [10 * second] + 6 * [second]
    x = create_series_from_timesteps(timesteps)
    time_difference_values = time_difference(x).values
    np.testing.assert_array_equal(time_difference_values, np.array([1000] * 6 + [10000] + [1000] * 6))
