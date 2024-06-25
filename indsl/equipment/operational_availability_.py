# Copyright 2024 Cognite AS
from typing import Literal

import pandas as pd

from indsl.type_check import check_types


@check_types
def operational_availability(availability: pd.Series, output: Literal["Uptime", "Downtime"]) -> pd.Series:
    r"""Operational availability.

    Calculate the operational availability of a system based on
    the availability time series. This time series will either be 1 or 0.
    1 represents the system is operational and 0 represents the system is not operational.
    This function will calculate the number of hours the system was operational or down
    in a given time period.

    Args:
        availability: Availability.
            Time series data of the availability of the system.
        output: Output type.
            A string representing the output of the function. Either 'Uptime' for uptime or 'Downtime' for downtime.

    Returns:
        pd.Series: Total hours per time period.
            Time series data of the operational availability or downtime of the system.
    """
    if output not in ["Uptime", "Downtime"]:
        raise ValueError("Output must be either 'UT' for uptime or 'DT' for downtime.")

    if output == "Downtime":
        # Invert the availability to get downtime (1 when down, 0 when up)
        status_series = 1 - availability
    else:
        # Use the availability directly for uptime
        status_series = availability

    # Resample the data to the specified frequency and sum the values to get total hours per period
    total_hours = status_series.resample("D").sum()

    # Take the absolute value of each point
    total_hours = total_hours.abs().astype(float)

    return total_hours
