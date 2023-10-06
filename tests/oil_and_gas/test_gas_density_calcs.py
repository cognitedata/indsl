import numpy as np
import pandas as pd
import pytest

from pandas.testing import assert_series_equal

from indsl.exceptions import UserValueError
from indsl.oil_and_gas.gas_density_calcs import (
    calculate_compressibility,
    calculate_critical_prop,
    calculate_gas_density,
    calculate_reduced_prop,
)


@pytest.mark.core
def test_density_calc_raises_error_with_empty_inputs():
    # Define input
    pressure = temp = sg = pd.Series([1.0])
    pressure_empty = temp_empty = sg_empty = pd.Series([])
    with pytest.raises(UserValueError, match="Time series is empty"):
        calculate_gas_density(pressure, temp, sg_empty)
    with pytest.raises(UserValueError, match="Pressure and Temperature data are empty or out of range"):
        calculate_gas_density(pressure_empty, temp, sg)
    with pytest.raises(UserValueError, match="Pressure and Temperature data are empty or out of range"):
        calculate_gas_density(pressure, temp_empty, sg)


@pytest.mark.parametrize("pressure", [pd.Series(np.ones(N)) for N in range(1, 5)])
@pytest.mark.parametrize("temp", [pd.Series(np.ones(N)) for N in range(1, 5)])
@pytest.mark.parametrize("sg", [pd.Series(np.ones(N)) for N in range(1, 5)])
@pytest.mark.core
def test_density_calc_few_data(pressure, temp, sg):
    try:
        calculate_gas_density(pressure, temp, sg)
    except Exception as e:
        pytest.fail(f"Unexpected error: {e}")


@pytest.mark.core
def test_density_calc():
    # Define input
    pressure = pd.Series([14.5, 100, 150, 100], index=[0, 3, 5, 6])
    temp = pd.Series([40, 50, 60, 70], index=[0, 1, 2, 3])
    sg = pd.Series([0.5534])
    res = calculate_gas_density(pressure, temp, sg)

    # Expected res
    exp_res = pd.Series([0.043153, 0.283269], index=[0, 3])
    assert_series_equal(res, exp_res)


@pytest.mark.core
def test_density_calc_raise_error_range():
    # Define input
    pressure = pd.Series([150, 1000000], index=[0, 1])
    temp = pd.Series([100000, 70], index=[0, 1])
    temp_0 = pd.Series([-459.4, -459.4], index=[0, 1])
    sg = pd.Series([0.5534])
    # Check if exception is raised
    with pytest.raises(UserValueError, match="Pressure and Temperature data are empty or out of range"):
        calculate_gas_density(pressure, temp, sg)
    with pytest.raises(UserValueError, match="Pressure and Temperature data are empty or out of range"):
        calculate_gas_density(pressure, temp_0, sg)
    with pytest.raises(UserValueError, match="Specific gravity cannot be zero"):
        calculate_gas_density(pressure, temp_0, pd.Series([0]))


@pytest.mark.core
def test_calculate_critical_prop():
    # Define input
    sg = 0.5534
    res = pd.Series(calculate_critical_prop(sg))
    # exp res
    exp_res = pd.Series([4710.206393, 188.861491], index=[0, 1])
    assert_series_equal(res, exp_res)


@pytest.mark.core
def test_calculate_reduced_prop():
    # Define input
    pressure = pd.Series([14.5, 50, 150, 1000], index=[0, 1, 2, 3])
    temp = pd.Series([40, 50, 60, 70], index=[0, 1, 2, 3])
    Pc = 4710.206393
    Tc = 188.861491
    res = calculate_reduced_prop(pressure, temp, Pc, Tc)[0]
    # exp res
    exp_res = pd.Series(
        [0.0030784213663224923, 0.010615246090767215, 0.031845738272301645, 0.2123049218153443], index=[0, 1, 2, 3]
    )
    assert_series_equal(res, exp_res)


@pytest.mark.core
def test_calculate_compressibility():
    # Define input
    ppr = pd.Series([6.7, 7, 8, 8.5], index=[0, 1, 2, 3])
    tpr = pd.Series([1, 1.2, 1.4, 2], index=[0, 1, 2, 3])

    res = calculate_compressibility(ppr, tpr)
    # return(res)
    # exp res
    exp_res = pd.Series(
        [0.8292891996292637, 0.9032231734572864, 0.9862768764448215, 1.0626146943711818], index=[0, 1, 2, 3]
    )
    assert_series_equal(res, exp_res)
