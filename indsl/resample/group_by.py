# Copyright 2023 Cognite AS

from typing import Literal

import numpy as np
import pandas as pd

from indsl.exceptions import UserValueError
from indsl.resample.auto_align import auto_align
from indsl.type_check import check_types


@check_types
def group_by_region(
    data: pd.Series,
    filter_by: pd.Series,
    int_to_keep: int = 1,
    aggregate: Literal["Mean", "Median", "Standard deviation", "Count", "Min", "Max"] = "Mean",
    timestamp: Literal["Region center", "Region start", "Region end", "Entire region"] = "Region center",
) -> pd.Series:
    """Group by region.

    This function groups any given data series by a series with integers denoting different states. A typical example of
    such a series is a series of 0 and 1 where 1 would indicate the presence of steady process conditions.

    Args:
        data (pd.Series): Time series.
        filter_by (pd.Series): Region flag time series.
            Time series values are expected to be integers. If not, values are cast to integer automatically.
        int_to_keep (int, optional): Value.
            Value that identifies the region of interest.
        aggregate (str, optional): Aggregate.
            Indicates the aggregation to be performed for each identified region.
        timestamp (str, optional): Timestamp.
            Indicates the location of the timestamps for the aggregated data.

    Returns:
        pd.Series: Grouped time series

    Raises:
        UserRuntimeError: Time series returns no data. This could be due to insufficient data in either `data`
            or `filter_by`, or `filter_by` series contains no values of `int_to_keep`.
        ValueError: The provided `aggregate` or `timestamp` inputs are not valid options.
    """
    options = {
        "Mean": "mean",
        "Median": "median",
        "Standard deviation": "std",
        "Count": "count",
        "Min": "min",
        "Max": "max",
    }
    if aggregate not in options.keys():
        raise UserValueError(f"`{aggregate=}` not recognized. Must be one of {list(options.keys())}")

    # sort time series
    data = data.sort_index()
    filter_by = filter_by.sort_index()

    # remove duplicate indexes
    data = data[~data.index.duplicated(keep="first")]
    filter_by = filter_by[~filter_by.index.duplicated(keep="first")]

    data, filter_by = auto_align([data, filter_by])

    # Check the filter_by contains integers
    if filter_by.dtypes is not np.dtype("int64"):
        filter_by = filter_by.astype(np.int64)

    # Find the region groups
    groups = filter_by.ne(filter_by.shift()).cumsum()[filter_by == int_to_keep]
    region_indexes = [group.index for _, group in filter_by.groupby(groups)]

    # Aggregate the data for each group
    result = pd.Series(dtype=np.float64)
    for region_index in region_indexes:
        # select the region index
        if timestamp == "Region center":
            index = [region_index[0] + (region_index[-1] - region_index[0]) / 2]
        elif timestamp == "Region start":
            index = [region_index[0]]
        elif timestamp == "Region end":
            index = [region_index[-1]]
        elif timestamp == "Entire region":
            index = region_index
        else:
            raise UserValueError(
                f"`{timestamp=}` not recognized. Must be one of [`Region center`, `Region start`, `Region end`, `Entire region`]"
            )
        result = pd.concat([result, pd.Series(data[region_index].agg(options[aggregate]), index=index)])

    return result
