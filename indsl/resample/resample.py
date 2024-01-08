# Copyright 2023 Cognite AS
import operator as op
import warnings

from typing import Literal, Optional

import pandas as pd

from scipy import signal

from indsl import versioning
from indsl.exceptions import UserTypeError
from indsl.ts_utils.ts_utils import fill_gaps, get_fixed_freq, is_na_all
from indsl.type_check import check_types
from indsl.validations import validate_series_has_time_index, validate_series_is_not_empty
from indsl.warnings import IndslUserWarning

from . import resample_v1  # noqa


@versioning.register(version="2.0", changelog="update data types")
@check_types
def resample(
    data: pd.Series,
    method: Literal["fourier", "polyphase", "interpolate", "min", "max", "sum", "count", "mean"] = "fourier",
    granularity_current: Optional[pd.Timedelta] = None,
    granularity_next: Optional[pd.Timedelta] = pd.Timedelta("1s"),
    num: Optional[int] = None,
    downsampling_factor: Optional[int] = None,
    interpolate_resolution: Optional[pd.Timedelta] = None,
    ffill_resolution: Optional[pd.Timedelta] = None,
) -> pd.Series:
    """Resample.

    This method offers a robust filling of missing data points and data resampling a given sampling frequency. Multiple
    data resampling options are available:

        * Fourier
        * Polynomial phase filtering
        * Linear interpolation (for up-sampling)
        * Min, max, sum, count, mean (for down-sampling)

    Args:
        data: Time series.

        method: Method.
            Resampling method. Valid options are:

                * "fourier" for Fourier method (default)
                * "polyphase" for polyphase filtering
                * "interpolate" for linear interpolation when upsampling
                * "min", "max", "sum", "count", "mean" when downsampling

        granularity_current: Current temporal resolution.
            Temporal resolution of uniform time series, before resampling. Defaults to None.
            If not specified, the frequency will be implied, which only works if no data is missing.
            Follows Pandas DateTime convention.

        granularity_next: Final temporal resolution.
            Temporal resolution of uniform time series, after resampling. Defaults to "1s".
            Either "Number of Samples" or "Final temporal resolution" must be provided.

        num: Number of Samples.
            The number of samples in the resampled signal. If this is set, the time deltas will be inferred. Defaults
            to None. Either "Number of Samples" or "Final temporal resolution" must be provided.

        downsampling_factor: Down-sampling factor.
            The down-sampling factor is required for the polyphase filtering. Defaults to None.

        interpolate_resolution: Interpolation threshold.
            Gaps smaller than threshold will be interpolated, larger than this will be filled by noise.
            Defaults to None.

        ffill_resolution: Forward fill threshold.
            Gaps smaller than this threshold will be forward filled. Defaults to None.

    Returns:
        pandas.Series: Interpolated time series
        Uniform, resampled time series with specified number of data points.

    Raises:
        UserTypeError: data is not a time series
        UserTypeError: Either num or granularity_next has to be set
        UserTypeError: If specified, outside_fill must be either 'None' or 'extrapolate'.
        UserTypeError: Method has to be in 'fourier', 'polyphase', 'interpolate', 'min', 'max', 'sum', 'count'
        UserTypeError: Empty data time series
        UserTypeError: All values in the time series are NaN
        Warning: Can't infer time series resolution with missing data. Please provide resolution
    """
    if granularity_next is None and (num is None or num <= 0):
        raise UserTypeError("Either num or granularity_next has to be set.")

    validate_series_has_time_index(data)
    validate_series_is_not_empty(data)
    if is_na_all(data):
        raise UserTypeError("All values in the time series are NaN.")

    # To resample data to uniform distribution
    if not granularity_current:
        # it returns none if it isn't able to infer the resolution
        granularity_current = pd.infer_freq(data.index)

        if not granularity_current:
            # TODO: pick max resolution and apply to the rest of the timeseries?
            warnings.warn(
                "Can't infer time series resolution with missing data. Please provide resolution",
                category=IndslUserWarning,
            )
            return data

    # make sure that it is uniform
    data = data.asfreq(freq=granularity_current)

    # add 1 if number is missing, in order to help Timedelta read it
    granularity_current = (
        granularity_current if any(char.isdigit() for char in granularity_current) else "1" + granularity_current
    )

    # remove nan
    data = fill_gaps(
        data=data,
        granularity=granularity_current,
        ffill_resolution=ffill_resolution,
        interpolate_resolution=interpolate_resolution,
    )

    if not num:
        num = 1 + int((data.index.max() - data.index.min()) / granularity_next)

    if method == "fourier":
        series = pd.Series(signal.resample(data, num), index=pd.date_range(data.index.min(), data.index.max(), num))
    elif method == "polyphase":
        if not downsampling_factor:
            downsampling_factor = len(data)
        sig = signal.resample_poly(data, num, downsampling_factor)
        series = pd.Series(sig, index=pd.date_range(data.index.min(), data.index.max(), len(sig)))
    elif method == "interpolate":
        series = data.resample(granularity_next).interpolate()
    else:
        series = data.resample(granularity_next, origin="epoch")
        series = op.methodcaller(method)(series)

    return series


@versioning.register(version="2.0", changelog="Describe here how the function changed compared to the previous version")
@check_types
def resample_to_granularity(
    series: pd.Series,
    granularity: pd.Timedelta = pd.Timedelta("1h"),
    aggregate: Literal["mean", "interpolation", "stepInterpolation", "max", "min", "count", "sum"] = "mean",
) -> pd.Series:
    """Resample to granularity.

    Resample time series to a given fixed granularity (time delta) and aggregation type
    (`read more about aggregation <https://docs.cognite.com/dev/concepts/aggregation/>`_)

    Args:
        series: Time series.

        granularity: Granularity.
            Granularity defines the time range that each aggregate is calculated from. It consists of a time unit and a
            size. Valid time units are day or d, hour h, minute or m and second or s. For example, 2h means that each
            time range should be 2 hours wide, 3m means 3 minutes, and so on.

        aggregate: Aggregate.
            Type of aggregation to use when resampling. Valid options are:

                * "interpolation" for linear interpolation when upsampling
                * "stepInterpolation" for stepwise interpolation when upsampling
                * "min", "max", "sum", "count", "mean" when downsampling

    Returns:
        pandas.Series: Resampled time series.
    """
    validate_series_has_time_index(series)
    validate_series_is_not_empty(series)

    # Translates from cdf aggregates to pandas method names:
    # TODO: 'mean' (point-weighted) does not equal CDF average (time-weighted).
    # Read more in the docs: https://docs.cognite.com/dev/concepts/aggregation/
    start_time = series.index[0]
    end_time = series.index[-1]
    if aggregate in ("min", "max", "sum", "count", "mean"):
        resampler = series.resample(granularity, origin="epoch")
        series_out = op.methodcaller(aggregate)(resampler)
        if aggregate == "mean":
            # We only fill inner holes with lin.interp for average:
            series_out = series_out.interpolate("slinear")
        return series_out.dropna()

    elif aggregate in ("interpolation", "stepInterpolation"):
        # We adhere to the CDF definition of interp. and step interp:
        fixed_freq = get_fixed_freq(start_time, end_time, granularity)
        series = series.reindex(series.index.union(fixed_freq))  # Note: reindex, not resample
        if aggregate == "interpolation":
            return series.interpolate("slinear")[fixed_freq].dropna()
        else:
            return series.ffill().bfill()[fixed_freq]
