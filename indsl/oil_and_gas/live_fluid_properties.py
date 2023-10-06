# Copyright 2023 Cognite AS
import json
import os

from typing import Literal

import numpy as np
import pandas as pd

from scipy import interpolate as intp

from indsl.type_check import check_types


@check_types
def _error_handling(pvt_table: np.ndarray, pressure: pd.Series, temperature: pd.Series) -> None:
    max_pressure, min_pressure = max(pvt_table[:, [0]])[0], min(pvt_table[:, [0]])[0]
    max_temperature, min_temperature = max(pvt_table[:, [1]])[0], min(pvt_table[:, [1]])[0]

    errors = []
    if max(pressure) > max_pressure:
        errors.append(
            f"max pressure of the sensor is outside the window of the pvt table, max pressure PVT table = {max_pressure}, max pressure sensor = {max(pressure)}"
        )
    if min(pressure) < min_pressure:
        errors.append(
            f"min pressure of the sensor is outside the window of the pvt table, min pressure PVT table = {min_pressure}, min pressure sensor = {min(pressure)}"
        )
    if max(temperature) > max_temperature:
        errors.append(
            f"max temperature of the sensor is outside the window of the pvt table, max temperature PVT table = {max_temperature}, max temperature sensor = {max(temperature)}"
        )
    if min(temperature) < min_temperature:
        errors.append(
            f"min temperature of the sensor is outside the window of the pvt table, min temperature PVT table = {min_temperature}, min temperature sensor = {min(temperature)}"
        )

    if errors:
        msg = "\n           ".join(errors)
        raise Exception(msg)


@check_types
def retrieve_fluid_properties(
    pressure: pd.Series,
    temperature: pd.Series,
    pvt_data: pd.DataFrame,
    param: str,
    interp_method: Literal["linear", "nearest", "cubic"] = "linear",
) -> pd.Series:
    """Retrieve fluid properties.

    This function obtains the value for the selected fluid property (`param`) corresponding to the pressure/temp pairs input as series. The input fluid file type is a .tab file that is an output
    of an equation of state simulator (i.e PVTSim, MultiFlash) and is an input file specifically for OLGA. The 31 fluid property parameters included in the .tab file are the options as an output for this function.
    Note that pressure and temperature sensors have to come from the same location.

    Args:
        pressure: Pressure time series.
        temperature: Temperature time series.
        pvt_data: PVT data.
            This is a PVT table that has been parsed out from a .tab file which is a standardized generated output of
            MultiFlash and PVTSim.
        param: Fluid parameter of interest.
            Fluid property to be estimated.
        interp_method: Method.
            The method used to interpolate the pvt table to obtain fluid properties of a given pressure and temperature.

    Returns:
        pandas.Series: Time series
            Fluid property of selected fluid property ("param") for corresponding pressure and temperature sensors.
            There are 31 fluid properties currently available to output from the fluid file.

    Raises:
        AssertionError: If data in sequence is not from a .tab file.
        AssertionError: If `param` input is not one of the 31 fluid properties available.
        Exception: If the range of the pressure and temperature sensor is outside the pvt table window, the function
        cannot interpolate for fluid property parameters.
    """
    assert (pvt_data.attrs["file_type"]) == "tab", "File is not a '.tab' file"

    column_names, pvt_table = pvt_data.columns, pvt_data.values
    assert param in column_names, (
        "Parameter is not a property of the fluid file. Valid inputs are %s" % column_names.values
    )

    pvt_table_dict_file = os.path.join(os.path.dirname(__file__), "tab_fp_identifier.json")
    pvt_table_dict = json.load(open(pvt_table_dict_file, "rb"))
    description = pvt_table_dict[param]["description"]
    unit = pvt_table_dict[param]["unit"]

    param_idx = next(idx for idx, val in enumerate(column_names) if param == val)

    _error_handling(pvt_table, pressure, temperature)

    return pd.Series(
        intp.griddata(pvt_table[:, [0, 1]], pvt_table[:, param_idx], (pressure, temperature), method=interp_method),
        index=pressure.index,
        name=f"{description} [{unit}]",
    )
