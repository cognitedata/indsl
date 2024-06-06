# Tests for the calculate_operational_availability function in the equipment module
import pytest
import pandas as pd
from indsl.equipment.operational_availability import operational_availability


@pytest.mark.core
@pytest.mark.parametrize(
    "up_time_data, down_time_data, expected",
    [
        (
            pd.Series([1] * 365, index=pd.date_range(start="2023-01-01", periods=365, freq="D")),
            pd.Series([1] * 365, index=pd.date_range(start="2023-01-01", periods=365, freq="D")),
            pd.Series([0.5] * 365, index=pd.date_range(start="2023-01-01", periods=365, freq="D")).astype(float),
        ),
        (
            pd.Series([1] * 365, index=pd.date_range(start="2023-01-01", periods=365, freq="D")),
            pd.Series([0] * 365, index=pd.date_range(start="2023-01-01", periods=365, freq="D")),
            pd.Series([1] * 365, index=pd.date_range(start="2023-01-01", periods=365, freq="D")).astype(float),
        ),
    ],
)
def test_calculate_operational_availability(up_time_data, down_time_data, expected):
    result = operational_availability(up_time_data=up_time_data, down_time_data=down_time_data)
    pd.testing.assert_series_equal(result, expected)
