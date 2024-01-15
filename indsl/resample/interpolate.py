# Copyright 2023 Cognite AS
import warnings

from contextlib import suppress
from typing import Literal, Union

import numpy as np
import pandas as pd
import scipy.interpolate

from indsl import versioning
from indsl.exceptions import UserValueError
from indsl.ts_utils.ts_utils import functional_mean, is_na_all
from indsl.type_check import check_types
from indsl.warnings import IndslUserWarning

from . import interpolate_v1  # noqa


@versioning.register(version="1.1", changelog="Removed outside_fill parameter")
@check_types
def interpolate(
    data: pd.Series,
    method: Literal["linear", "ffill", "stepwise", "zero", "slinear", "quadratic", "cubic"] = "linear",
    kind: Literal["pointwise", "average"] = "pointwise",
    granularity: pd.Timedelta = pd.Timedelta("1s"),
    bounded: bool = True,
) -> Union[pd.DataFrame, pd.Series]:
    """Interpolation.

    This function interpolates and resamples data with a uniform sampling frequency.

    Args:
        data: Time series.

        method: Method.
            Specifies the interpolation method. Defaults to "linear". Possible inputs are :

                * 'linear': linear interpolation.
                * 'ffill': forward filling.
                * 'stepwise': yields same result as ffill.
                * 'zero', 'slinear', 'quadratic', 'cubic': spline interpolation of zeroth, first, second or third order.

        kind: Kind.
            Specifies the kind of returned data points. Defaults to "pointwise".  Possible inputs are:

                * 'pointwise': returns the pointwise value of the interpolated function for each timestamp.
                * 'average': returns the average of the interpolated function within each time period.

        granularity: Frequency.
            Sampling frequency or granularity of the output (e.g. '1s' or '2h'). Defaults to "1s".

        bounded: Bounded.
            Specifies behaviour for requested points outside of the data range. Defaults to True.

                * True: Extrapolate for requested points outside of the data range.
                * False: Ignore points outside of the data range.

    Returns:
        pandas.Series: Interpolated time series.

    Raises:
        UserTypeError: data is not a time series
        Warning: Empty data time series
        Warning: All data in timeseries is nan
    """
    # Check if all values are NaN
    if is_na_all(data):
        warnings.warn("All values in the time series are NaN.", category=IndslUserWarning)
        return data

    # Allow for other ways of defining forward filling for stepwise functions
    method_ = "previous" if method in ("ffill", "stepwise") else method

    # Get outside fill value
    if bounded:
        fill_value = np.nan
    else:
        fill_value = "extrapolate"

    # Get start and end dates and store as datetime
    start_dt = _validate(data.index[0])
    end_dt = _validate(data.index[-1])

    # Output timestamps for uniform time series
    timestamps = pd.date_range(start_dt, end_dt, freq=granularity)

    # Create uniform x values for output time series
    x_uniform = np.array([timestamp.timestamp() for timestamp in timestamps])

    # extract time series as pd.Series and drop NaNs
    observations = data.dropna()

    # Check for empty time series
    if len(observations) < 2:
        warnings.warn("The time series contains less than two values.", category=IndslUserWarning)
        return data

    # x and y datapoints used to construct linear piecewise function
    x_observed = np.array([index.timestamp() for index in observations.index])
    y_observed = observations.values.squeeze()

    # interpolator function
    interper = scipy.interpolate.interp1d(
        x_observed, y_observed, kind=method_, bounds_error=False, fill_value=fill_value
    )

    # If pointwise, sample directly from interpolated (or original) points
    if kind == "pointwise":
        y_uniform = interper(x_uniform)
    elif kind == "average":
        y_uniform = functional_mean(interper, x_uniform)

    series = pd.Series(data=y_uniform, index=timestamps, name=data.name)

    return series


@check_types
def _validate(date_text: Union[str, pd.Timestamp]) -> pd.Timestamp:
    """Validates data format.

    Args:
        date_text (Union[str, pd.Timestamp]): Date in string format.

    Raises:
        UserValueError: If date string format does not match YYYY-MM-DD hh:mm:ss.

    Returns:
        pandas.Timestamp: String converted to pandas.Timestamp.
    """
    with suppress(ValueError):
        return pd.Timestamp(date_text)
    raise UserValueError("Incorrect data format, should be YYYY-MM-DD hh:mm:ss")
