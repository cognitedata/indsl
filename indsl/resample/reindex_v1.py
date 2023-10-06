# Copyright 2023 Cognite AS
from enum import Enum
from typing import List

import numpy as np
import pandas as pd
import pandas.core.indexes.datetimes
import scipy

from indsl import versioning
from indsl.exceptions import UserTypeError, UserValueError
from indsl.ts_utils.ts_utils import functional_mean


class Kind(Enum):  # noqa: D101
    POINTWISE = "pointwise"
    AVERAGE = "average"


class Method(Enum):  # noqa: D101
    ZERO_PREVIOUS = "zero order spline interpolation (forward filling)"
    ZERO_NEXT = "zero order spline interpolation (backward filling)"
    LINEAR = "first order spline interpolation"
    QUADRATIC = "second order plines interpolation"
    CUBIC = "third order plines interpolation"

    def to_interpd1_arg(self):
        """Get method used for scipy.interpolation.

        Returns:
            str: Interpolation method.
        """
        if self is Method.ZERO_PREVIOUS:
            return "zero"
        if self is Method.ZERO_NEXT:
            return "next"
        if self is Method.LINEAR:
            return "slinear"
        if self is Method.QUADRATIC:
            return "quadratic"
        if self is Method.CUBIC:
            return "cubic"


@versioning.register(version="1.0", deprecated=True)
# @check_types # typeguard expects str, not class Methor or Kind as a data tyepe
# it is commented out to avoid changing the signature of this deprecated function, so that it is compatible with Charts
def reindex(
    data1: pd.Series,
    data2: pd.Series,
    method: Method = Method.LINEAR,
    kind: Kind = Kind.POINTWISE,
    bounded: bool = False,
) -> List[pd.Series]:
    """Reindex.

    This method offers data reindexing onto a common index and fills missing data points.
    If bounded is false, the common index is the union of the the input time-series indices.
    If bounded is true, the common index is restricted to the period where the time-series overlap.
    All not-a-number (NaN) values are removed in the output time series.

    Args:
        data1: First time series.

        data2: Second time series.

        method: Method.
            Specifies the interpolation method. Defaults to "linear". Possible inputs are :

            * 'zero_previous': zero order spline interpolation with forward filling mode, i.e. the previous known value of any point is used.
            * 'zero_next': zero order spline interpolation with backward filling mode, i.e. the next known value of any point is used.
            * 'linear': linear order spline interpolation.
            * 'quadratic': quadratic order spline interpolation.
            * 'cubic': cubic order spline interpolation.

        kind: Kind.
            Specifies the kind of returned data points. Defaults to "pointwise".  Possible inputs are:

            * 'pointwise': returns the pointwise value of the interpolated function for each timestamp.
            * 'average': returns the average of the interpolated function within each time period.

        bounded: Bounded.
            Specifies if the output should be bounded to avoid extrapolation. Defaults to False. Possible inputs are:

            * True: Return the intersection of the timeperiods of the input time series.
            * False: Return the union of the timeperiods of the input time series. Extraplolate points outside of the data range.


    Returns:
        pandas.Series: First reindexed time series
        pandas.Series: Second reindexed time series
        Reindexed time series with common indices.

    Raises:
        UserTypeError: data1 or data2 are not a time series
        UserTypeError: Method has to be of type Method
        UserTypeError: Kind has to be of type Kind
        UserTypeError: Bounded has to be of type bool
        UserValueError: All time series must have at least two values
    """
    return reindex_many([data1, data2], method, kind, bounded)


@versioning.register(version="1.0", deprecated=True)
# @check_types # typeguard expects str, not class Methor or Kind as a data tyepe
# it is commented out to avoid changing the signature of this deprecated function, so that it is compatible with Charts
def reindex_many(
    data: List[pd.Series],
    method: Method = Method.LINEAR,
    kind: Kind = Kind.POINTWISE,
    bounded: bool = False,
) -> List[pd.Series]:
    """Re-indexes list.

    Args:
        data: List[pd.Series]
        method: Method, defaults to Method.LINEAR
        kind: Kind, defaults to Kind.POINTWISE
        bounded: bool, defaults to False

    Raises:
        UserTypeError: Only DateTimeIndex index supported.
        UserValueError: One of the time series has less than two values.
        UserTypeError:  The re-indexed time series is empty.

    Returns:
        List[pd.Series]: re-indexed list
    """
    # Nothing to do if data contains less than 2 series
    if len(data) < 2:
        return data

    # Return if all indices are already aligned and
    # none of the input series contain NaNs
    indices = [d.index for d in data]
    if all([indices[0].equals(idx) for idx in indices[1:]]) and not any([d.isnull().values.any() for d in data]):
        return data

    # check input types
    if not all(isinstance(d.index, pd.DatetimeIndex) for d in data):
        raise UserTypeError("Only DateTimeIndex index supported.")

    # Check validity of input time series
    if any([len(d.dropna()) < 2 for d in data]):
        raise UserValueError("One of the time series has less than two values.")

    # If bounded=True, we need to remove Nan values here.
    # Otherwise, we might end up extrapolating if there are NaN values at
    # the beginning/end of a time-series (which is in conflict with the bounded=True flag)
    if bounded:
        data = [d.dropna() for d in data]
        indices = [d.index for d in data]

    # Get the union of the indices
    idx_common = indices[0]
    for index in indices[1:]:
        idx_common = idx_common.union(index)

    # If bounded is true, restrict the dataset to a range where both time-series have data
    if bounded:
        start = max([idx.min() for idx in indices])
        stop = min([idx.max() for idx in indices])

        start_loc = idx_common.get_loc(start)
        stop_loc = idx_common.get_loc(stop)

        idx_common = idx_common[start_loc : stop_loc + 1]

    if len(idx_common) == 0:
        raise UserTypeError("The re-indexed time series is empty.")

    # Reindex data to common index
    out = [_reindex(d, idx_common, method, kind, bounded) for d in data]

    return out


# @check_types # typeguard expects str, not class Methor or Kind as a data type
# it is commented out to avoid changing the signature of this deprecated function, so that it is compatible with Charts
def _reindex(
    data: pd.Series,
    new_index: pandas.core.indexes.datetimes.DatetimeIndex,
    method: Method,
    kind: Kind,
    bounds_error: bool,
) -> pd.Series:
    # Convert Enum types to scipy compatible types
    method = method.to_interpd1_arg()

    # Create x values for output time series
    x_uniform = new_index.view(np.int64)

    # extract time series as pd.Series and drop NaNs
    observations = data.dropna()

    # x and y datapoints used to construct linear piecewise function
    x_observed = observations.index.view(np.int64)
    y_observed = observations.values

    if bounds_error:
        fill_value = None
    else:
        fill_value = "extrapolate"

    # interpolator function
    interper = scipy.interpolate.interp1d(
        x_observed, y_observed, kind=method, bounds_error=bounds_error, fill_value=fill_value, assume_sorted=True
    )

    # If pointwise, sample directly from interpolated (or original) points
    if kind is Kind.POINTWISE:
        y_uniform = interper(x_uniform)
    else:
        y_uniform = functional_mean(interper, x_uniform)

    series = pd.Series(data=y_uniform, index=new_index, name=data.name)

    return series
