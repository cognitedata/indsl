import pandas as pd
import pytest

from indsl.equipment.mean_time_between_failures_ import mean_time_between_failures


@pytest.mark.parametrize(
    "mean_time_to_failure, mean_time_to_resolution, expected",
    [
        (
            pd.Series([1] * 365, index=pd.date_range(start="2023-01-01", periods=365, freq="D")),
            pd.Series([1] * 365, index=pd.date_range(start="2023-01-01", periods=365, freq="D")),
            pd.Series([2] * 365, index=pd.date_range(start="2023-01-01", periods=365, freq="D")),
        ),
        (
            pd.Series([1] * 365, index=pd.date_range(start="2023-01-01", periods=365, freq="D")),
            pd.Series([0] * 365, index=pd.date_range(start="2023-01-01", periods=365, freq="D")),
            pd.Series([1] * 365, index=pd.date_range(start="2023-01-01", periods=365, freq="D")),
        ),
    ],
)
def test_mean_time_between_failures(mean_time_to_failure, mean_time_to_resolution, expected):
    """Test mean time between failures function."""
    mtbf = mean_time_between_failures(
        mean_time_to_failure=mean_time_to_failure, mean_time_to_resolution=mean_time_to_resolution
    )
    assert mtbf.equals(expected)
