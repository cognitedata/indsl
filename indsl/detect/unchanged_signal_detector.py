# Copyright 2023 Cognite AS
import pandas as pd

from indsl.ts_utils.utility_functions import generate_step_series
from indsl.type_check import check_types
from indsl.validations import validate_series_has_time_index, validate_series_is_not_empty


@check_types
def unchanged_signal_detector(
    data: pd.Series, duration: pd.Timedelta = pd.Timedelta(minutes=60), min_nr_data_points: int = 3
) -> pd.Series:
    """Unchanged signal detection.

    Detect periods of time when the data stays at a constant value for longer than a given time window.

    Args:
        data: Time series.
        duration: Time window.
            Length of the time period to check for unchanged time series values. Defaults to 'minutes=60'.
        min_nr_data_points: Data points.
            The least number of data points to avoid alerting on missing data. Defaults to 3.

    Returns:
        pd.Series: Time series.
            The returned time series is an indicator function that is 1 where the time series value has remained
            unchanged, 0 if it has changed.

    Raises:
        UserTypeError: data is not a time series
        UserValueError: data is empty
    """
    validate_series_has_time_index(data)
    validate_series_is_not_empty(data)

    # Convert series to dataframe
    df = data.to_frame()

    duration_in_hours = duration.total_seconds() / 3600

    # Rename dataframe columns
    df.columns = ["value"]
    df["datetime"] = df.index

    # Calculate cumulative sum of consecutive time series value differences to see how much the values change over time
    df["value_grp"] = (df.value.diff(1) != 0).astype("int").cumsum()

    # Count the number of consecutive unchanged values and their start and end time stamps
    df1 = pd.DataFrame(
        {
            "BeginDate": df.groupby("value_grp").datetime.first(),
            "EndDate": df.groupby("value_grp").datetime.last(),
            "Consecutive": df.groupby("value_grp").size(),
        }
    ).reset_index(drop=True)

    # Calculate the time between unchanged signal values - in hours
    df1["time_passed"] = (df1["EndDate"] - df1["BeginDate"]).dt.total_seconds() / 3600

    # Merge the original dataframe with the new based on 'datetime' and 'BeginDate'
    df = pd.merge(df, df1, how="left", left_on="datetime", right_on="BeginDate")

    df["is_unchanged"] = 0

    # If the time passed between the occurrence of unchanged data points in the signal is greater than the provided
    # duration, set 'is_unchanged' to 1, else 0.
    # Also check that there are at least 'min_nr_data_points' data points to avoid alerting about missing data
    # The default number of data points is set to 3 so that we have at least one data point between start and stop

    for i in range(len(df)):
        if df["time_passed"].iloc[i] >= duration_in_hours and df["Consecutive"].iloc[i] >= min_nr_data_points:
            df.at[i, "is_unchanged"] = 1

    # Set value of 'is_unchanged' to 1 for the whole duration of unchanged signal
    for i in range(len(df)):
        if df["is_unchanged"][i] == 1:
            j = 1
            while j < df["Consecutive"].iloc[i] and i + j < len(df):
                df.at[i + j, "is_unchanged"] = 1
                j = j + 1

    df = df[["datetime", "is_unchanged"]]

    # Convert dataframe to series
    df = df.set_index("datetime")
    data = df.squeeze()

    return generate_step_series(data)
