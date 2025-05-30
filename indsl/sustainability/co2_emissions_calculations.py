# Copyright 2023 Cognite AS
# Determine rate of emissions from combustor
from typing import Optional

import pandas as pd

from indsl.resample.resample import resample_to_granularity
from indsl.ts_utils.numerical_calculus import trapezoidal_integration
from indsl.type_check import check_types


@check_types
def emissions_factor_combustor(
    emissions_factor: Optional[float] = None,
    heating_value: Optional[float] = None,
    carbon_content: Optional[float] = None,
) -> float:
    r"""Combustor emissions factor.

    This function calculates the emissions factor of a combustor (e.g., gas turbine, flare, etc.). Three different methods can be used
    to perform this calculation:

    * Method 1: Specify the emissions factor in CO2 emitted per mass or volume unit.
    * Method 2: Specify the emissions factor in CO2 per energy unit and convert it to mass or volume rate using the heating value.

    .. math::
        \mathrm{EF [per\ mass\ or\ volume\ of\ fuel] = EF [per\ unit\ of\ energy] \times Heating\ Value}

    * Method 3: Specify the carbon content of the fuel and use that as the emissions factor.

    Args:
        emissions_factor: CO2 emitted
            CO2 emitted either per mass (e.g., kg CO2/kg fuel), volume (e.g., kg CO2/m^3 fuel), or energy unit (e.g., kg CO2/MJ).
        heating_value: Heating value of the fuel
            Heating value of the fuel (e.g., MJ/kg fuel or MJ/m^3 fuel).
        carbon_content: Carbon content of fuel
            Carbon content of the fuel (e.g., kg C/kg fuel).Carbon content of fuel (e.g., kg C/kg fuel or kg C/m^3 fuel).

    Returns:
        float: Emissions factor
            Emissions factor per mass or volume of fuel (e.g., kg CO2/kg fuel or kg CO2/m^3 fuel).
    """
    # Method 1
    if emissions_factor and (heating_value is None) and (carbon_content is None):
        return emissions_factor

    # Method 2
    elif emissions_factor and heating_value and (carbon_content is None):
        return heating_value * emissions_factor

    # Method 3
    elif (emissions_factor is None) and (heating_value is None) and carbon_content:
        return carbon_content

    raise RuntimeError(
        "Incorrect arguments defined. Please see documentation in detail for how to specify the correct inputs."
    )


# Rate of emissions
@check_types
def rate_of_emissions(data: pd.Series, emissions_factor: float) -> pd.Series:
    r"""Rate of emissions.

    This function calculates the rate of emissions generated by a power consumer or a combustor.

    .. math::
        \mathrm{Rate\ of\ emissions = Emissions\ Factor \times Power\ (or\ Fuel\ Flow\ Rate)}

    The units for the time series and the emissions factor should be consistent to deliver an expected unit for the output in kg (or tonnes) CO2/time.

    Args:
        data: Time series
            Time series representing either power (e.g., MJ/time) or rate of fuel consumption (e.g., kg fuel/time or m^3 fuel/time).
        emissions_factor: Emissions factor
            CO2 emitted per unit of energy (e.g., kg CO2/MJ), mass (e.g., kg CO2/kg fuel), or volume (e.g., kg CO2/m^3 fuel).

    Returns:
        pandas.Series: Rate of emissions
            Rate of emissions (e.g., kg CO2/time or tonnes CO2/time).
    """
    return data * emissions_factor


@check_types
def cumulative_co2_production(rate_of_emissions: pd.Series, start_date: Optional[pd.Timestamp] = None) -> pd.Series:
    r"""Cumulative CO2 production.

    This function calculates the total CO2 production according to the rate of emissions. The total is calculated by performing trapezoidal integration
    over time (granularity of 1 hour). The rate of emissions is resampled to 1-hour granularity accordingly. If no start time is specified,
    it will default to the start of the current year.

    Args:
        rate_of_emissions: Rate of CO2 released
            Rate of CO2 released over time (e.g., kg CO2/time or tonnes CO2/time).
        start_date: Start date
            Start date to begin cumulative calculation.

    Returns:
        pandas.Series: Cumulative CO2 emissions
            Cumulative CO2 emissions (e.g., kg CO2 or tonnes CO2).
    """
    # Assign start_date to be start of year if not defined
    if start_date is None:
        start_date = pd.Timestamp(year=pd.Timestamp("now").year, month=1, day=1)

    data = resample_to_granularity(rate_of_emissions.loc[start_date:], granularity=pd.Timedelta("1h"))

    return trapezoidal_integration(data, time_unit=pd.Timedelta("1h"))


@check_types
def cumulative_co2_cost(
    data: pd.Series, co2_cost_factor: float, emissions_factor: float, start_date: Optional[pd.Timestamp] = None
) -> pd.Series:
    r"""Cumulative CO2 cost.

    This function calculates the cumulative cost of CO2 for either a combustor or power consumer. It calculates the rate of CO2 emitted
    and then uses that to calculate total CO2 emitted. This is then multiplied by the cost factor to get the total cumulative cost. Note that the co2_cost_factor, emissions_factor, and data must have consistent units to generate a currency output.

    Args:
        data: Power or fuel consumption
            Power consumption (e.g., MJ/time) or fuel consumption (e.g., kg fuel/time or m^3 fuel/time).
        co2_cost_factor: Cost per mass of CO2 emitted
            Cost per mass of CO2 emitted (e.g., USD/kg CO2 or USD/tonne CO2).
        emissions_factor: Mass of CO2 emitted
            Mass of CO2 emitted per mass (e.g., kg CO2/kg fuel), volume (e.g., kg CO2/m^3 fuel), or energy (e.g., kg CO2/MJ) of power source consumed.
        start_date: Start date
            Start date to begin cumulative calculation.

    Returns:
        pandas.Series: Cumulative cost
            Cumulative cost of CO2 emissions (e.g., USD).
    """
    rate_co2_produced = rate_of_emissions(data, emissions_factor)
    total_co2_produced = cumulative_co2_production(rate_co2_produced, start_date=start_date)

    return total_co2_produced * co2_cost_factor
