# Copyright 2023 Cognite AS
import numpy as np
import pandas as pd

from indsl.exceptions import UserRuntimeError
from indsl.resample.auto_align import auto_align
from indsl.type_check import check_types


@check_types
def status_flag_filter(
    data: pd.Series, filter_by: pd.Series, int_to_keep: int = 0, align_timesteps: bool = False
) -> pd.Series:
    """Status flag filter.

    This function filters any given data series by a series with integers denoting different states. A typical example of
    such a series is a series of 0 and 1 where 1 would indicate the presence of an anomaly.
    The status flag filter retrieves all relevant indices and matches these to the data series.

    Args:
        data: Time series.
        filter_by: Status flag time series.
            Time series values are expected to be integers. If not, values are cast to integer automatically.
        int_to_keep: Value.
            Value to filter by in the boolean filter
        align_timesteps: Auto-align.
          Automatically align time stamp of input time series. Default is False.

    Returns:
        pandas.Series: Filtered time series

    Raises:
        UserRuntimeError: Time series returns no data. This could be due to insufficient data in either `data`
            or `filter_by`, or `filter_by` series contains no values of `int_to_keep`.
    """
    data, filter_by = auto_align([data, filter_by], align_timesteps)

    # Select index to keep
    filter_index = filter_by[filter_by == int_to_keep].index

    # Match data index to filter index
    data_index = data.index[data.index.isin(filter_index)]

    # Check for no entries
    if len(data_index) == 0:
        raise UserRuntimeError("Current filter returns no entries for data.")

    # Check the filter_by contains integers
    if filter_by.dtypes is not np.dtype("int64"):
        filter_by = filter_by.astype(np.int64)
        # warnings.warn(
        #    "Input data `filter_by` contains types other than integers. Casting values to int and continuing!",
        #    RuntimeWarning,
        # )

    return data[data_index]
