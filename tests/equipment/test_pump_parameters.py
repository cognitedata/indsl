import pandas as pd
import pytest

from indsl.equipment.pump_parameters import pump_hydraulic_power, recycle_valve_power_loss


@pytest.mark.core
def test_recycle_valve_power_loss():
    valve_params = {
        "Q_valve": pd.Series([1, 2]),
        "total_head": pd.Series([1, 2]),
        "den": pd.Series([1, 2]),
    }
    W_expected = pump_hydraulic_power(
        pump_liquid_flowrate=valve_params["Q_valve"], total_head=valve_params["total_head"], den=valve_params["den"]
    )
    W_calc = recycle_valve_power_loss(**valve_params)
    pd.testing.assert_series_equal(W_expected, W_calc)
