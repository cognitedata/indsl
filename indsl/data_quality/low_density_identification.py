# Copyright 2023 Cognite AS
import numpy as np
import pandas as pd

from indsl.ts_utils.utility_functions import (
    generate_step_series,
    modified_z_scores_test,
    normality_assumption_test,
    z_scores_test,
)
from indsl.type_check import check_types
from indsl.validations import validate_series_has_time_index, validate_series_is_not_empty


@check_types
def get_densities(data: pd.Series, time_window: pd.Timedelta = pd.Timedelta("5m")) -> pd.Series:
    """Point density.

    Counts the number of points on a rolling time window.

    Args:
        data: Time series.
        time_window: Window
            Length of the time period to compute the density of points. Defaults to 5min.

    Returns:
        pd.Series: Time series with the point density for each rolling window.
    """
    # make sure index is in ns
    index = data.index.astype("datetime64[ns]")
    data_ns = pd.Series(data.values, index=index)
    return data_ns.rolling(time_window, min_periods=1).count()


@check_types
def low_density_identification_z_scores(
    data: pd.Series,
    time_window: pd.Timedelta = pd.Timedelta("5m"),
    cutoff: float = -3.0,
    test_normality_assumption: bool = False,
) -> pd.Series:
    """Low density, Z-scores.

    Detect periods with low density of data points using `Z-scores <https://en.wikipedia.org/wiki/Standard_score>`_. Z-score stands for
    the number of standard deviations by which the value of a raw score (i.e., an observed value or data point) is
    above or below the mean value of what is being observed or measured. This method assumes that the densities over a rolling window
    are normally distributed. Low density periods are defined as time periods where the Z-score is lower than cutoff.

    Args:
        data: Time series
        time_window: Rolling window.
            Length of the time period to compute the density of points. Defaults to 5 min.
        cutoff: Cut-off.
            Number of standard deviations from the mean.
            Low density periods are detected if the Z-score is below this cut-off value. Default -3.0.
        test_normality_assumption: Test for normality.
            Raises an error if the data densities over the rolling windows are not normally distributed.
            The Shapiro-Wilk test is used. The test is only performed if the time series contains less than 5000 data points.
            Default to False.

    Returns:
        pd.Series: Time series
            The returned time series is an indicator function that is 1 where there is a low density period, and 0 otherwise.

    Raises:
        UserTypeError: data is not a time series
        UserTypeError: cutoff is not a number
        UserValueError: data is empty
        UserValueError: time series is not normally distributed
    """
    validate_series_has_time_index(data)
    validate_series_is_not_empty(data)

    if len(data) < 2:
        return pd.Series([0] * len(data), index=data.index)

    x_freq = get_densities(data, time_window=time_window)

    if test_normality_assumption:
        normality_assumption_test(series=x_freq, max_data_points=5000, min_p_value=0.05, min_W=0.5)

    is_low_density = z_scores_test(x_freq.values, cutoff=cutoff, direction="less")

    return generate_step_series(pd.Series(is_low_density, index=data.index))


@check_types
def low_density_identification_modified_z_scores(
    data: pd.Series, time_window: pd.Timedelta = pd.Timedelta("5m"), cutoff: float = -3.5
) -> pd.Series:
    """Low density, mod.Z-scores.

    Detect periods with a low density of data points using modified Z-scores.
    Low density periods are defined as time periods where the Z-score is lower than the cutoff.

    Args:
        data: Time series
        time_window: Rolling window.
            Length of the time period to compute the density of points. Defaults to 5 min.
        cutoff: Cut-off.
            Low density periods are detected if the modified Z-score is below this cut-off value. Default -3.5.

    Returns:
        pd.Series: Time series
            The returned time series is an indicator function that is 1 where there is a low density period, and 0 otherwise.

    Raises:
        UserTypeError: data is not a time series
        UserTypeError: cutoff has to be of type float
        UserValueError: data is empty
    """
    validate_series_has_time_index(data)
    validate_series_is_not_empty(data)

    if len(data) < 2:
        return pd.Series([0] * len(data), index=data.index)

    x_freq = get_densities(data, time_window=time_window)

    is_low_density = modified_z_scores_test(x_freq.values, cutoff=cutoff, direction="less")

    return generate_step_series(pd.Series(is_low_density, index=data.index))


@check_types
def low_density_identification_iqr(data: pd.Series, time_window: pd.Timedelta = pd.Timedelta("5m")) -> pd.Series:
    """Low density, IQR.

    Detect periods with a low density of data points using the `interquartile range (IQR)
    <https://en.wikipedia.org/wiki/Interquartile_range>`_ method. The IQR is a measure of statistical
    dispersion, which is the spread of the data. Densities that are more than 1.5 IQR below Q1 are considered
    as low density periods in the data.

    Args:
        data: time series
        time_window: Rolling window.
            Length of the time period to compute the density of points. Defaults to 5 min.

    Returns:
        pd.Series: time series
            The returned time series is an indicator function that is 1 where there is a low density period, and 0 otherwise.

    Raises:
        UserTypeError: data is not a time series
        UserValueError: data is empty
    """
    validate_series_has_time_index(data)
    validate_series_is_not_empty(data)

    if len(data) < 2:
        return pd.Series([0] * len(data), index=data.index)

    x_freq = get_densities(data, time_window=time_window)

    percentile25 = np.quantile(x_freq, 0.25)
    percentile75 = np.quantile(x_freq, 0.75)

    iqr = percentile75 - percentile25
    lower_limit = percentile75 - 1.5 * iqr
    is_low_density = np.where(x_freq < lower_limit, 1, 0)

    return generate_step_series(pd.Series(is_low_density, index=data.index))


@check_types
def low_density_identification_threshold(
    data: pd.Series, time_window: pd.Timedelta = pd.Timedelta("5m"), cutoff: int = 10
) -> pd.Series:
    """Low density, threshold.

    Detect periods with a low density of points using a time delta threshold as a cut-off value.

    Args:
        data: time series
        time_window: Rolling window.
            Length of the time period to compute the density of points. Defaults to 5 min.
        cutoff: Density cut-off.
            Low density periods are detected if the number of points is less than this cut-off value. Default is 10.

    Returns:
        pd.Series: time series
            The returned time series is an indicator function that is 1 where there is a low density period, and 0 otherwise.

    Raises:
        UserTypeError: data is not a time series
        UserValueError: data is empty
    """
    validate_series_has_time_index(data)
    validate_series_is_not_empty(data)

    x_freq = get_densities(data, time_window=time_window)
    check_freq_series = x_freq < cutoff
    check_freq_series = check_freq_series.astype(int)

    return generate_step_series(check_freq_series)
