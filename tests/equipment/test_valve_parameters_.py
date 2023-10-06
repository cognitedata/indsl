import numpy as np
import pandas as pd
import pytest

from pandas.testing import assert_series_equal

from indsl.equipment.valve_parameters_ import flow_through_valve, flow_through_valve_initial_incompressible
from indsl.validations import UserValueError, UserTypeError


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


@pytest.mark.parametrize("flow_func", [flow_through_valve, flow_through_valve_initial_incompressible])
@pytest.mark.core
class TestFlowThroughValve:
    def test_flow_through_valve_linear_cv(self, valve_params, flow_func):
        Q_calc = flow_func(**valve_params)
        Q_expected = pd.Series([86.5, 51.9])
        pd.testing.assert_series_equal(Q_calc, Q_expected)

    def test_flow_through_valve_eq_cv(self, valve_params, flow_func):
        valve_params["type"] = "EQ"
        Q_calc = flow_func(**valve_params)
        Q_expected = pd.Series([72.84162784043666, 45.27995734273004])
        pd.testing.assert_series_equal(Q_calc, Q_expected)

    def test_flow_through_valve_negative_sg(self, valve_params, flow_func):
        valve_params["SG"] = -1
        with pytest.raises(UserValueError):
            flow_func(**valve_params)


def create_test_series():
    inlet_P = pd.Series([2, 2, 2, 2], pd.date_range("1970-01-01 00:00:01", periods=4, freq="1s"))
    outlet_P = pd.Series([1, 1, 1, 1], pd.date_range("1970-01-01 00:00:00", periods=4, freq="1s"))
    valve_opening = pd.Series([3, 3, 3, 3], pd.date_range("1970-01-01 00:00:02", periods=4, freq="1s"))
    return inlet_P, outlet_P, valve_opening


@pytest.mark.core
def test_flow_through_valve_validation_tests():
    inlet_P, outlet_P, valve_opening = create_test_series()
    with pytest.raises(UserValueError) as excinfo:
        flow_through_valve(
            inlet_P=inlet_P,
            outlet_P=outlet_P,
            valve_opening=valve_opening,
            SG=-1.0,
            type="Linear",
            min_opening=0.0,
            max_opening=3.0,
            min_Cv=0.0,
            max_Cv=4.0,
        )
    exp_res = "Specific gravity cannot be negative."
    assert exp_res in str(excinfo.value)

    with pytest.raises(UserTypeError):
        flow_through_valve(
            inlet_P=inlet_P,
            outlet_P=outlet_P,
            valve_opening=valve_opening,
            SG=1.0,
            type="Invalid method",
            min_opening=0.0,
            max_opening=3.0,
            min_Cv=0.0,
            max_Cv=4.0,
        )


@pytest.mark.parametrize("method", ["Linear", "EQ"])
def test_flow_through_valve_linear_and_eq(method):
    # without align_timestamps
    inlet_P, outlet_P, valve_opening = create_test_series()
    res = flow_through_valve(
        inlet_P=inlet_P,
        outlet_P=outlet_P,
        valve_opening=valve_opening,
        SG=1.0,
        type=method,
        min_opening=1.0,
        max_opening=3.0,
        min_Cv=1.0,
        max_Cv=4.0,
        align_timestamps=False,
    )
    exp_res = pd.Series(
        [np.NaN, np.NaN, 3.46, 3.46, np.NaN, np.NaN], index=pd.date_range("1970-01-01 00:00:00", periods=6, freq="1s")
    )
    assert_series_equal(res, exp_res)

    # with align_timestamps
    res = flow_through_valve(
        inlet_P=inlet_P,
        outlet_P=outlet_P,
        valve_opening=valve_opening,
        SG=1.0,
        type=method,
        min_opening=1.0,
        max_opening=3.0,
        min_Cv=1.0,
        max_Cv=4.0,
        align_timestamps=True,
    )
    exp_res = pd.Series([3.46, 3.46], index=pd.date_range("1970-01-01 00:00:02", periods=2, freq="1s"))
    assert_series_equal(res, exp_res)
