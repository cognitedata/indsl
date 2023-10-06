# Copyright 2023 Cognite AS
from datetime import timedelta

import numpy as np
import pandas as pd

from indsl.exceptions import UserValueError
from indsl.type_check import check_types


@check_types
def calculate_shutin_variable(
    variable_signal: pd.Series, shutin_signal: pd.Series, hrs_after_shutin: float
) -> pd.Series:
    """Shut-in variable calculator.

    The shut-in variable calculator is a function to compute the variable of interest at specific times after the shut-in
    onset. Typically, variables of interest are pressure and temperature. The function is the dependency of the shut-in
    detector. Based on the detected shut-in interval, the function uses specified number of hours indicating the time after
    the onset of each shut-in. It calculates the variable of interest at that time instance using interpolation (method - time).

    Args:
        variable_signal: Signal of interest.
            Typically pressure or temperature signal.
        shutin_signal: Shut-in signal.
            The signal comes from a shut-in detector function or a signal indicating shut-in condition
            (0 - well in shut-in state, 1 - well in flowing state). We suggest using the
            :meth:`indsl.oil_and_gas.calculate_shutin_interval`.
        hrs_after_shutin: Hours after.
            Hours after shut-in onset at which to calculate the signal of interest.

    Returns:
        pandas.Series: Output.
        Signal of interest at specific time after shut-in onset.
    """
    if shutin_signal.empty or variable_signal.empty:
        return pd.Series(dtype=np.float64, index=pd.DatetimeIndex([]))

    if shutin_signal.dtype not in [int, np.int64, np.int32]:
        raise UserValueError("The results from shut-in detector contain non-integer numbers")

    # condition1: guarantee the timeseries always start from 1 (flowing state)
    if shutin_signal.iloc[0] == 0:
        time_idx = shutin_signal[shutin_signal.values == 1].index
        if time_idx.empty:
            raise UserValueError("The signal does not contain flowing data")
        shutin_signal = shutin_signal[shutin_signal.index > time_idx[0]]

    # get changes of the shut-in signal (-1 indicates the start and 1 indicates the end of shut-in)
    changepts = shutin_signal.diff()
    ss_start_shifted = shutin_signal[changepts == -1].index + timedelta(hours=hrs_after_shutin)
    ss_end = shutin_signal[changepts == 1].index

    # condition1 guarantees that len(ss_start_shifted) is greater or equal to len(ss_end)
    if len(ss_start_shifted) > len(ss_end):
        if ss_start_shifted[-1] <= shutin_signal.index[-1]:
            ss_end = ss_end.union(shutin_signal.index[-1:])
        else:
            ss_start_shifted = ss_start_shifted[:-1]

    # condition2: filter shut-ins where shifted start is earlier than finish
    shut_ins = ss_start_shifted[ss_start_shifted <= ss_end]
    if shut_ins.empty:
        return pd.Series(dtype=np.float64, index=pd.DatetimeIndex([]))

    return (
        variable_signal.reindex(variable_signal.index.union(shut_ins))
        .interpolate(method="time")[shut_ins]
        .astype(np.float64)
        .rename(None)
    )
