import pandas as pd
import pytest

from indsl.equipment.valve_parameters import flow_through_valve
from indsl.validations import UserValueError


@pytest.fixture
def valve_params():
    return {
        "inlet_P": pd.Series([4, 1]),
        "outlet_P": pd.Series([0, 0]),
        "valve_opening": pd.Series([0.5, 0.6]),
        "type": "Linear",
        "SG": 1,
        "min_opening": 0.1,
        "max_opening": 0.9,
        "min_Cv": 10,
        "max_Cv": 90,
    }


@pytest.fixture
def compressible_params():
    return {
        "compressible": True,
        "gas_expansion_factor": 1,
        "inlet_T": 300,
        "Z": 1,
    }


@pytest.mark.core
def test_flow_through_valve_incompressible_fluid_linear_cv(valve_params):
    Q_calc = flow_through_valve(**valve_params)
    Q_expected = pd.Series([86.5, 51.9])
    pd.testing.assert_series_equal(Q_calc, Q_expected)


@pytest.mark.core
def test_flow_through_valve_incompressible_fluid_eq_cv(valve_params):
    valve_params["type"] = "EQ"
    Q_calc = flow_through_valve(**valve_params)
    Q_expected = pd.Series([72.84162784043666, 45.27995734273004])
    pd.testing.assert_series_equal(Q_calc, Q_expected)


def test_flow_through_valve_negative_sg(valve_params):
    valve_params["SG"] = -1
    with pytest.raises(UserValueError):
        flow_through_valve(**valve_params)

    valve_params["SG"] = pd.Series([-1, -1])
    with pytest.raises(UserValueError):
        flow_through_valve(**valve_params)


@pytest.mark.core
def test_flow_through_valve_compressible_fluid_linear_cv(valve_params, compressible_params):
    Q_calc = flow_through_valve(**valve_params, **compressible_params)
    Q_expected = pd.Series([4815.101245, 1444.530374])
    pd.testing.assert_series_equal(Q_calc, Q_expected)
