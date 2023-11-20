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
def gaps_identification_z_scores(
    data: pd.Series, cutoff: float = 3.0, test_normality_assumption: bool = False
) -> pd.Series:
    """Gaps detection, Z-scores.

    This function detects gaps in the time stamps using `Z-scores <https://en.wikipedia.org/wiki/Standard_score>`_. Z-score stands for
    the number of standard deviations by which the value of a raw score (i.e., an observed value or data point) is
    above or below the mean value of what is being observed or measured. This method assumes that the time step sizes
    are normally distributed. Gaps are defined as time periods where the Z-score is larger than cutoff.

    Args:
        data: Time series
        cutoff: Cut-off
            Time periods are considered gaps if the Z-score is over this cut-off value. Default 3.0.
        test_normality_assumption: Test for normality.
            Raise a warning if the data is not normally distributed.
            The Shapiro-Wilk test is used. The test is only performed if the time series contains less than 5000 data points.
            Default to False.

    Returns:
        pd.Series: Time series
            The returned time series is an indicator function that is 1 where there is a gap, and 0 otherwise.

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

    timestamps = data.index.to_numpy(np.int64)
    diff = np.diff(timestamps)

    if test_normality_assumption:
        normality_assumption_test(series=diff, max_data_points=5000, min_p_value=0.05, min_W=0.5)

    is_gap = z_scores_test(diff, cutoff=cutoff, direction="greater")

    # duplicate the last flag value since the flag series should have the same length as the original data points
    flag_series = pd.concat([pd.Series(is_gap, index=data.index[:-1]), pd.Series(is_gap[-1], index=[data.index[-1]])])

    return generate_step_series(flag_series)


@check_types
def gaps_identification_modified_z_scores(data: pd.Series, cutoff: float = 3.5) -> pd.Series:
    """Gaps detection, mod. Z-scores.

    Detect gaps in the time stamps using modified Z-scores. Gaps are defined as time periods
    where the Z-score is larger than cutoff.

    Args:
        data: Time series
        cutoff: Cut-off
            Time-periods are considered gaps if the modified Z-score is over this cut-off value. Default 3.5.

    Returns:
        pd.Series: Time series
            The returned time series is an indicator function that is 1 where there is a gap, and 0 otherwise.

    Raises:
        UserTypeError: data is not a time series
        UserTypeError: cutoff has to be of type float
        UserValueError: data is empty
    """
    validate_series_has_time_index(data)
    validate_series_is_not_empty(data)

    if len(data) < 2:
        return pd.Series([0] * len(data), index=data.index)

    timestamps = data.index.to_numpy(np.int64)
    diff = np.diff(timestamps)

    is_gap = modified_z_scores_test(diff, cutoff=cutoff, direction="greater")

    # duplicate the last flag value since the flag series should have the same length as the original data points
    flag_series = pd.concat([pd.Series(is_gap, index=data.index[:-1]), pd.Series(is_gap[-1], index=[data.index[-1]])])

    return generate_step_series(flag_series)


@check_types
def gaps_identification_iqr(data: pd.Series) -> pd.Series:
    """Gaps detection, IQR.

    Detect gaps in the time stamps using the `interquartile range (IQR)
    <https://en.wikipedia.org/wiki/Interquartile_range>`_ method. The IQR is a measure of statistical
    dispersion, which is the spread of the data. Any time steps more than 1.5 IQR above Q3 are considered
    gaps in the data.

    Args:
        data: time series

    Returns:
        pd.Series: time series
            The returned time series is an indicator function that is 1 where there is a gap, and 0 otherwise.

    Raises:
        UserTypeError: data is not a time series
        UserValueError: data is empty
    """
    validate_series_has_time_index(data)
    validate_series_is_not_empty(data)

    if len(data) < 2:
        return pd.Series([0] * len(data), index=data.index)

    timestamps = data.index.to_numpy(np.int64)
    diff = np.diff(timestamps)

    percentile25 = np.quantile(diff, 0.25)
    percentile75 = np.quantile(diff, 0.75)

    iqr = percentile75 - percentile25
    upper_limit = percentile75 + 1.5 * iqr
    is_gap = np.where(diff > upper_limit, 1, 0)

    # duplicate the last flag value since the flag series should have the same length as the original data points
    flag_series = pd.concat([pd.Series(is_gap, index=data.index[:-1]), pd.Series(is_gap[-1], index=[data.index[-1]])])

    return generate_step_series(flag_series)


@check_types
def gaps_identification_threshold(data: pd.Series, time_delta: pd.Timedelta = pd.Timedelta("5m")) -> pd.Series:
    """Gaps detection, threshold.

    Detect gaps in the time stamps using a timedelta threshold.

    Args:
        data: time series
        time_delta: Time threshold
            Maximum time delta between points. Defaults to 5min.

    Returns:
        pd.Series: time series
            The returned time series is an indicator function that is 1 where there is a gap, and 0 otherwise.

    Raises:
        UserTypeError: data is not a time series
        UserValueError: data is empty
        UserTypeError: time_delta is not a pd.Timedelta
    """
    validate_series_has_time_index(data)
    validate_series_is_not_empty(data)

    if len(data) < 2:
        return pd.Series([0] * len(data), index=data.index)

    last_index = data.index[-1]

    diff = data.index.astype("datetime64[ns]").to_series().diff().shift(-1).dropna()
    is_gap_series = diff > time_delta
    is_gap_series = is_gap_series.astype(int)

    # duplicate the last flag value since the flag series should have the same length as the original data points
    flag_series = pd.concat([is_gap_series, pd.Series(is_gap_series.iloc[-1], index=[last_index])])

    return generate_step_series(flag_series)
