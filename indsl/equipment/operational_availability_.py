import pandas as pd

from indsl.type_check import check_types


@check_types
def operational_availability(up_time_data: pd.Series, down_time_data: pd.Series) -> pd.Series:
    r"""Operational availability.

    Calculate the operational availability of a system based on
    the mean up time (MUT) and mean down time (MDT). The formula for the operational
    availability is given by

    .. math::
        A_o = \frac{MUT}{MUT + MDT}

    Args:
      up_time_data: Up time data.
          Time series data of equipment up time.
      down_time_data: Down time data.
          Time series data of equipment down time.

    Returns:
      pd.Series
          Time series data of the operational availability of the system.
    """
    MUT = up_time_data.resample("D").mean()
    MDT = down_time_data.resample("D").mean()

    operational_availability = MUT / (MUT + MDT)

    # Fill NaN values with 0
    operational_availability = operational_availability.fillna(0)

    return operational_availability
