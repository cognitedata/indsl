import pandas as pd
import pytest

from pandas.testing import assert_series_equal

from indsl.equipment.pump_parameters import total_head


@pytest.mark.core
def test_total_head():
    suction = pd.Series([3.17, 3.21, 3.0]) * 100000
    discharge = pd.Series([3.2, 3.24, 6.65]) * 100000
    den = pd.Series([1100] * 3)
    res = total_head(discharge, suction, den)
    expected = pd.Series([0.28, 0.28, 33.82])

    assert_series_equal(res, expected, atol=5e-3)
