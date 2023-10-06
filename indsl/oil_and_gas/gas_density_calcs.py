# Copyright 2023 Cognite AS
from typing import Tuple

import numpy as np
import pandas as pd

from indsl.exceptions import UserValueError
from indsl.resample.auto_align import auto_align
from indsl.type_check import check_types
from indsl.validations import validate_series_is_not_empty


CONVERT_R_TO_K = 5 / 9  # conversion from R to K
CONVERT_PSI_TO_KPA = 6.8947  # conversion from psi to kpa
CONVERT_KGM3_TO_PCF = 0.062  # convion from kg/m3 to pcf
R_U = 8.314  # universal gas constant KJ/Kmol.K
M_AIR = 28.97  # Molar mass of air (Kg/kmol)


@check_types
def calculate_gas_density(
    pressure: pd.Series, temperature: pd.Series, sg: pd.Series, align_timesteps: bool = False
) -> pd.Series:
    """Gas density calculator.

    The gas density is calculated from real gas laws.The psuedo critical tempreature and pressure is
    calculated from specific gravity following `Sutton (1985) <https://doi.org/10.2118/14265-MS>`_. The
    `Beggs and Brill (1973) <https://onepetro.org/JPT/article-abstract/25/05/607/165212/A-Study-of-Two-Phase-Flow-in-Inclined-Pipes>`_
    method (explicit) is used to calculate the compressibility factor. All equations used here can be found in one place at
    `Kareem et. al. <https://link.springer.com/article/10.1007/s13202-015-0209-3>`_. The gas equation *Pv = zRT*
    is used to calculate the gas density.

    Args:
       pressure: Pressure [psi].
           Pressure time series in psi units.
       temperature: Temperature [degF].
           Temperature time series in degrees Fahrenheit units.
       sg: Specific gravity [-].
           Specific gravity of the gas.
       align_timesteps: Auto-align
          Automatically align time stamp  of input time series. Default is False.

    Returns:
       pandas.Series: Gas density [lbm/ft3]
           Estimated gas density in pound-mass per cubic foot (pcf).

    Raises:
       UserValueError: When all values of the pressure and temperature are out of range for calculating compressibility.
    """
    # check specific gravity value
    validate_series_is_not_empty(sg)
    sg_ = sg[0]
    if sg_ == 0:
        raise UserValueError("Specific gravity cannot be zero")

    # auto-align
    pressure, temperature = auto_align([pressure, temperature], align_timesteps)

    M = M_AIR * sg_  # Molar mass of gas
    R = R_U / M  # gas constant

    # calculate the pseudo - crtitical pressure and temperature
    (Pc, Tc) = calculate_critical_prop(sg_)

    # convert pressure to kPa and temperature to K
    pressure_Kpa = CONVERT_PSI_TO_KPA * pressure
    temperature_K = 273 + ((temperature - 32) * (5 / 9))

    # calculate the pseudo - reduced pressure and temperature
    (Ppr, Tpr) = calculate_reduced_prop(pressure_Kpa, temperature_K, Pc, Tc)

    # calculate compressibility
    compressibility = calculate_compressibility(Ppr, Tpr)

    # check division by zero
    compressibility = compressibility[compressibility.values > 0]
    temperature_K = temperature_K[temperature_K.values > 0]
    ixs = compressibility.index.intersection(temperature_K.index)
    compressibility, temperature_K, pressure_Kpa = (
        compressibility.loc[ixs],
        temperature_K.loc[ixs],
        pressure_Kpa.loc[ixs],
    )
    if compressibility.empty or temperature_K.empty:
        raise UserValueError("Compressibility or temperature series are empty after filtering zeros")
    if R == 0:
        raise UserValueError("Gas constant is zero. Division by zero is not allowed")

    # calculate gas density
    rho = pressure_Kpa / R / temperature_K / compressibility
    rho_out = CONVERT_KGM3_TO_PCF * rho
    return rho_out.dropna()


@check_types
def calculate_critical_prop(sg: float) -> Tuple[float, float]:
    """Critical properties.

    Returns the crtitcal pressure and temperature as a function of specific gravity (Sutton - 1973).

    Args:
       sg: Specific gravity
           Specific gravity of the gas.

    Returns:
       float: Critical pressure
           Critical pressure in KPa.
       float: Critical temperature
           Critical temperature in Kelvin.
    """
    Tc_R = 169.2 + 349.5 * sg - 74 * sg**2
    Tc = CONVERT_R_TO_K * Tc_R  # convert from Rankine to K
    Pc_psi = 756.8 - 131.07 * sg - 3.6 * sg**2
    Pc = Pc_psi * CONVERT_PSI_TO_KPA  # convert from psi to Kpa
    return (Pc, Tc)


@check_types
def calculate_reduced_prop(
    pressure_Kpa: pd.Series, temperature_K: pd.Series, Pc: float, Tc: float
) -> Tuple[pd.Series, pd.Series]:
    """Reduced properties.

    Returns the reduced pressure and temperature from input pressure, temperature, and critical prperties.

    Args:
       pressure_Kpa: Pressure
           Time series containing pressure data in KPa.
       temperature_K: Temperature
           Time series containing temperature data in Kelvin.
       Pc: Critical pressure
           Critical pressure in KPa.
       Tc: Critical temperature
           Critical temperature in Kelvin.

    Returns:
       pandas.Series: Reduced pressure
           Non-dimensional series.
       pandas.Series: Reduced temperature
           Non-dimensional series.
    """
    # calculate the pseudo-reduced pressure and temperature
    Ppr = pressure_Kpa / Pc
    Tpr = temperature_K / Tc
    return (Ppr, Tpr)


@check_types
def calculate_compressibility(Ppr: pd.Series, Tpr: pd.Series) -> pd.Series:
    """Gas Compressibility.

    Returns the compressibility factor from reduced pressure and temperature.
    Also deletes points in input temperature and pressure that are out of range.

    Args:
       Ppr: Reduced pressure
            Non-dimensional series.
       Tpr: Reduced temperature
            Non-dimensional series.

    Returns:
       pandas.Series: Compressibility factor
            Non-dimensional series.

    Raises:
       UserValueError: When all values of the pressure and temperature are out of range for calculating compressibility.
    """
    # delete points that are out of range, raise error if empty
    Tpr = Tpr[(Tpr.values >= 1) & (Tpr.values <= 3)]
    Ppr = Ppr[(Ppr.values >= 0) & (Ppr.values <= 15)]
    idx = Tpr.index.intersection(Ppr.index)
    Tpr, Ppr = Tpr.loc[idx], Ppr.loc[idx]

    if Tpr.empty or Ppr.empty:
        raise UserValueError("Pressure and Temperature data are empty or out of range")

    A = 1.39 * (Tpr - 0.92) ** 0.5 - 0.36 * Tpr - 0.10
    E = 9 * (Tpr - 1)
    B = (0.62 - 0.23 * Tpr) * Ppr + ((0.066 / (Tpr - 0.86)) - 0.037) * Ppr**2 + ((0.32 * Ppr**2) / (10**E))
    C = 0.132 - 0.32 * np.log10(Tpr)
    F = 0.3106 - 0.49 * Tpr + 0.1824 * Tpr**2
    D = 10**F
    compressibility = A + ((1 - A) / np.exp(B)) + C * Ppr**D

    return compressibility
