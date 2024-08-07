# Copyright 2024 Cognite AS
from typing import List

import pandas as pd

from indsl.exceptions import UserValueError
from indsl.type_check import check_types


# from indsl.validations import validate_series_is_not_empty


@check_types
def user_specified_timeseries(
    input_time: List[float] = [0.0, 1.0], input_values: List[float] = [0.0, 1.0]
) -> pd.Series:
    """Convert input to a timeseries.

    The input is converted to a time series, assuning the input time is provied as Unix time, in seconds.

    Args:
        input_time (List[float]): Timestamps.
            The timestamps follow the Unix convention (Number of seconds starting from January 1st, 1970). Precision loss in the order of nanoseconds may happen if unit is not nanoseconds.
        input_values (List[float]): y values.
            The values matching the timestamps
    Returns:
        pd.Series: Output.
    """
    n_t, n_y = len(input_time), len(input_values)
    if not n_t == n_y:
        raise UserValueError("There is a different number of timestamps and values. len(x)=%d,len(x)=%d" % (n_t, n_y))
    if n_t < 1:
        raise UserValueError("Empty input. We need at least one point to show")
    # Convert input timestamps to datetime
    timestamps = []
    for time_ in input_time:
        timestamps.append(pd.Timestamp(time_, unit="s"))
    output_series = pd.Series(input_values, index=timestamps)
    return output_series
