import pandas as pd
import pytest

from pandas.testing import assert_series_equal

from indsl.equipment.pump_parameters import pump_shaft_power


@pytest.mark.core
def test_pump_shaft_power():
    pump_hydraulic_power = pd.Series([0.02635, 0.02585, 16.57535]) * 1000
    pump_liquid_flowrate = pd.Series([30.32, 24.64, 148.5])
    eff_parameter_1 = pd.Series([-8.00781603e-06] * 3)
    eff_parameter_2 = pd.Series([5.19564490e-02] * 3)
    eff_intercept = pd.Series([3.89930657e00] * 3)

    res = pump_shaft_power(pump_hydraulic_power, pump_liquid_flowrate, eff_parameter_1, eff_parameter_2, eff_intercept)
    expected = pd.Series([0.482, 0.5, 144.912]) * 1000

    assert_series_equal(res, expected, atol=0.5)
