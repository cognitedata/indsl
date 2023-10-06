# Copyright 2021 Cognite AS
import pickle as pkl

import pandas as pd
import pytest

from pandas.testing import assert_series_equal

from indsl.oil_and_gas.live_fluid_properties import retrieve_fluid_properties


@pytest.mark.core
def test_fluid_property_retrieve():
    # testing main interpolation functionality
    pvt_data = pkl.load(open("datasets/data/pvt_data.pkl", "rb"))
    pressure = pd.Series([20000, 18560000.0, 4320000, 2500000])
    temperature = pd.Series([-100, 198.9, 25, 3])

    res = retrieve_fluid_properties(pressure, temperature, pvt_data, "CPG")
    exp_res = pd.Series(
        [1908.019209512195, 2961.575217804878, 2438.253338788485, 2285.7074686325473],
        name="Gas thermal capacity [J/kg·°C]",
    )

    assert_series_equal(res, exp_res)


@pytest.mark.core
def test_error_handling():
    # testing the error handling function to make sure pressure and temperature sensors are within
    # the range of the fluid property table.
    pvt_data = pkl.load(open("datasets/data/pvt_data.pkl", "rb"))
    pressure = pd.Series([1000000, 5000000, 4320000, 2500000])
    temperature = pd.Series([-110, 100, 25, 3])

    with pytest.raises(
        Exception,
        match="min temperature of the sensor is outside the window of the pvt table, min temperature PVT table = -100.0, min temperature sensor = -110",
    ):
        retrieve_fluid_properties(pressure, temperature, pvt_data, "ROG")

    pressure = pd.Series([1000000, 50000000000, 0, 2500000])
    temperature = pd.Series([-1000, 10000, 25, 3])

    with pytest.raises(
        Exception,
        match="max pressure of the sensor is outside the window of the pvt table, max pressure PVT table = 20100000.0, max pressure sensor = 50000000000\n           min pressure of the sensor is outside the window of the pvt table, min pressure PVT table = 10000.0, min pressure sensor = 0",
    ):
        retrieve_fluid_properties(pressure, temperature, pvt_data, "ROG")


@pytest.mark.core
def test_false_parameter():
    # check is the input `param` is a fluid property that is in the fluid files.
    pvt_data = pkl.load(open("datasets/data/pvt_data.pkl", "rb"))
    pressure = pd.Series([1000000, 5000000, 4320000, 2500000])
    temperature = pd.Series([-50, 100, 25, 3])

    with pytest.raises(AssertionError, match="Parameter is not a property of the fluid file."):
        retrieve_fluid_properties(pressure, temperature, pvt_data, "TEST")
