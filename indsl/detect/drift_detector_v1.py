# Copyright 2023 Cognite AS
import numpy as np
import pandas as pd

from indsl import versioning
from indsl.type_check import check_types


@versioning.register(version="1.0", deprecated=True)
@check_types
def drift(
    data: pd.Series,
    long_interval: str = "3d",
    short_interval: str = "4h",
    std_threshold: float = 3,
    detect: str = "both",
) -> pd.Series:
    """Drift.

    This function detects data drift (deviation) by comparing two rolling averages, short and long interval, of the signal. The
    deviation between the short and long term average is considered significant if it is above a given threshold
    multiplied by the rolling standard deviation of the long term average.

    Args:
        data: Time series.
        long_interval: Long length.
            Length of long term time interval.
        short_interval: Short length.
            Length of short term time interval.
        std_threshold: Threshold.
            Parameter that determines if the signal has changed significantly enough to be considered drift. The threshold
            is multiplied by the long term rolling standard deviation to take into account the recent condition of the
            signal.
        detect: Type.
            Parameter to determine if the model should detect significant decreases, increases or both. Options are:
            "decrease", "increase", or "both". Defaults to "both".

    Returns:
      pandas.Series: Boolean time series.
        Drift = 1, No drift = 0.
    """
    # Compute long term average and std
    long_average = data.rolling(long_interval).mean()
    long_stds = data.rolling(long_interval).std()

    # Compute short term average
    short_average = data.rolling(short_interval).mean()

    # Calculate drift masks
    drift_mask_increase = short_average > long_average + std_threshold * long_stds
    drift_mask_decrease = short_average < long_average - std_threshold * long_stds

    # set mask according to setting
    if detect == "increase":
        drift_mask = drift_mask_increase
    elif detect == "decrease":
        drift_mask = drift_mask_decrease
    else:
        drift_mask = np.logical_or(drift_mask_increase, drift_mask_decrease)

    return drift_mask.dropna().astype(int)
