# Tests for the calculate_operational_availability function in the equipment module
import inspect

import pytest
import pandas as pd
from indsl.equipment.operational_availability_ import operational_availability


@pytest.mark.parametrize(
    "availability, output, expected",
    [
        # Test case 1: Alternating availability pattern
        (
            pd.Series([1, 0, 1, 0, 1, 0, 1, 0], index=pd.date_range(start="2023-01-01", periods=8, freq="D")),
            "Uptime",
            pd.Series([1, 0, 1, 0, 1, 0, 1, 0], index=pd.date_range(start="2023-01-01", periods=8, freq="D")).astype(
                float
            ),
        ),
        # Test case 2: All ones (always available)
        (
            pd.Series([1, 1, 1, 1, 1, 1, 1, 1], index=pd.date_range(start="2023-01-01", periods=8, freq="D")),
            "Uptime",
            pd.Series([1, 1, 1, 1, 1, 1, 1, 1], index=pd.date_range(start="2023-01-01", periods=8, freq="D")).astype(
                float
            ),
        ),
        # Test case 3: All zeros (never available)
        (
            pd.Series([0, 0, 0, 0, 0, 0, 0, 0], index=pd.date_range(start="2023-01-01", periods=8, freq="D")),
            "Uptime",
            pd.Series([0, 0, 0, 0, 0, 0, 0, 0], index=pd.date_range(start="2023-01-01", periods=8, freq="D")).astype(
                float
            ),
        ),
        # Test case 4: Mixed availability
        (
            pd.Series([1, 0, 0, 1, 1, 0, 0, 1], index=pd.date_range(start="2023-01-01", periods=8, freq="D")),
            "Uptime",
            pd.Series([1, 0, 0, 1, 1, 0, 0, 1], index=pd.date_range(start="2023-01-01", periods=8, freq="D")).astype(
                float
            ),
        ),
        # Test case 5: Mixed availability, output DT
        (
            pd.Series([1, 0, 0, 1, 1, 0, 0, 1], index=pd.date_range(start="2023-01-01", periods=8, freq="D")),
            "Downtime",
            pd.Series([0, 1, 1, 0, 0, 1, 1, 0], index=pd.date_range(start="2023-01-01", periods=8, freq="D")).astype(
                float
            ),
        ),
    ],
)
@pytest.mark.core
def test_operational_availability(availability, output, expected):
    result = operational_availability(availability=availability, output=output)
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.core
def test_operational_availability_output_default_is_provided():
    """Test that the default value for 'output' parameter is provided."""
    sig = inspect.signature(operational_availability)
    param = sig.parameters["output"]
    assert param.default is not inspect.Parameter.empty, "Default value for 'output' must be provided"
