import pandas as pd
import pytest

from pandas.testing import assert_series_equal

from indsl.equipment.pump_parameters import pump_hydraulic_power


@pytest.mark.core
def test_pump_hydraulic_power():
    flowrate = pd.Series([30.32, 24.64, 148.5]) / 3600
    total_head = pd.Series([0.29, 0.35, 37.24])
    den = pd.Series([1100] * 3)
    res = pump_hydraulic_power(flowrate, total_head, den) / 1000
    expected = pd.Series([0.02636, 0.02585, 16.57659])

    assert_series_equal(res, expected, atol=5e-6)
