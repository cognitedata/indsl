# Copyright 2023 Cognite AS
import pandas as pd

from indsl.ts_utils.utility_functions import generate_step_series
from indsl.type_check import check_types
from indsl.validations import (
    validate_series_has_minimum_length,
    validate_series_has_time_index,
    validate_series_is_not_empty,
    validate_timedelta_unit,
)


@check_types
def datapoint_diff_over_time_period(
    data: pd.Series,
    time_period: pd.Timedelta = pd.Timedelta("1D"),
    difference_threshold: int = 24,
    tolerance: pd.Timedelta = pd.Timedelta("1H"),
) -> pd.Series:
    """Diff. between two datapoints.

    The function is created in order to automate data quality check for time series with values that shouldn't be
    increasing more than a certain threshold over a certain amount of hours. For each data point in a given
    time series, the function will calculate the difference in value between that data point and the data point at the
    defined length of period ago (i.e. it calculates the value change over a period).

    An example is Running Hours (or Hour Count) time series - a specific type of time series that is counting the number
    of running hours in a pump. Given that we expect the number of running hours to stay within 24 over a period of 24
    hours, a breach in the number of running hours over the last 24 hours would indicate poor data quality. In short,
    the value difference over 24 hours should not be higher than 24 for an Hour Count time series.

    Although the algorithm is originally created for Running Hours time series, it can be applied to all time series
    where the breach in the threshold defined for difference between two datapoints at a certain length of period apart
    is a sign of bad data quality.

    Args:
        data: Time series.
        time_period: Time period.
            The amount of period over which to calculate the difference between two data points. The value must be a
            non-negative float. Defaults to 1 day.
        difference_threshold: Threshold for data point diff.
            The threshold for difference calculation between two data points. Defaults to 24.
        tolerance: Tolerance range
            The tolerance period to allow between timestamps while looking for closest timestamp.

    Returns:
        pandas.Series: Time series
            The returned time series is an indicator function that is 1 where the difference between two datapoints over
            given number of hours exceeds the defined threshold, and 0 otherwise.

    Raises:
        UserTypeError: data is not a time series
        UserValueError: data is empty
        UserValueError: invalid unit of time delta
        UserTypeError: time_window is not of type pandas.Timedelta
    """
    validate_series_has_time_index(data)
    validate_series_is_not_empty(data)
    validate_series_has_minimum_length(data, 2)
    validate_timedelta_unit(time_period)
    validate_timedelta_unit(tolerance)

    # Convert series to dataframe
    df = data.to_frame()

    df.columns = ["value"]
    df["time_value"] = data.index

    df["timestamp_w_timedelta"] = df["time_value"] - time_period
    merged_df = pd.merge_asof(
        df,
        df,
        left_on="timestamp_w_timedelta",
        right_on="time_value",
        direction="nearest",
        tolerance=tolerance,
        suffixes=("", "_y"),
    )

    merged_df["delta"] = merged_df["value"] - merged_df["value_y"]
    merged_df = merged_df[["time_value", "value", "delta"]]

    # set the alert if the difference in datapoint values over the given time period exceeds the defined limit
    merged_df["alert"] = 0
    merged_df.loc[merged_df["delta"] > difference_threshold, "alert"] = 1

    merged_df = merged_df.set_index(merged_df.time_value)
    merged_df = merged_df[["alert"]]
    data = merged_df.squeeze()

    return generate_step_series(data)
