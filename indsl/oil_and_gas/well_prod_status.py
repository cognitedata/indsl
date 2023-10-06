# Copyright 2023 Cognite AS
from typing import Union

import numpy as np
import pandas as pd

from indsl.exceptions import UserValueError
from indsl.resample.auto_align import auto_align
from indsl.type_check import check_types


@check_types
def _pct_to_fraction(pct_value: Union[float, int]):
    return pct_value / 100.0


@check_types
def calculate_well_prod_status(
    master_valve: pd.Series,
    wing_valve: pd.Series,
    choke_valve: pd.Series,
    threshold_master: float = 1,
    threshold_wing: float = 1,
    threshold_choke: float = 5,
    align_timesteps: bool = False,
) -> pd.Series:
    """Check if the well is producing.

    Determine if the well is producing. In order for this to be the case, the following has to happen:

        * All Master, Wing and Choke data have to come from the same well.
        * Check if the master, wing and choke valve openings are above their respective threshold values at a given time.
        * If any of the valves are below the threshold opening, then the well is closed.
        * If all of the valves are above the threshold opening, then the well is open.
        * Threshold values should be between 0-100.

    Args:
        master_valve: Master Valve
            Time series of the master valve.
        wing_valve:  Wing Valve
            Time series of the wing valve.
        choke_valve:  Choke Valve
            Time series of the choke valve.
        threshold_master: Master threshold
            Threshold percentage value from 0%-100%.
        threshold_wing: Wing threshold
            Threshold percentage value from 0%-100%.
        threshold_choke: Choke threshold
            Threshold percentage value from 0%-100%.
        align_timesteps: Auto-align
            Automatically align time stamp  of input time series. Default is False.

    Returns:
        pandas.Series: Well Status
            Well production status (1 means open, 0 means closed).
    """
    master_valve, wing_valve, choke_valve = auto_align([master_valve, wing_valve, choke_valve], align_timesteps)

    # Check if any of the Series objects are empty
    if any(valve.empty for valve in [master_valve, wing_valve, choke_valve]):
        raise UserValueError("Empty Series are not allowed for valve inputs")

    valves = [master_valve, wing_valve, choke_valve]
    thresholds = [threshold_master, threshold_wing, threshold_choke]

    if any([i < 0 for i in thresholds]):
        raise UserValueError(
            "Threshold value has to be greater than or equal to 0"
        )  # it is not physical to have an opening less than 0

    if any([i > 100 for i in thresholds]):
        raise UserValueError(
            "Threshold value has to be less than or equal to 100"
        )  # it is not physical to have an opening more than 100

    thresholds = [  # make modifications for threshold if range is between 0-1.
        _pct_to_fraction(threshold)
        if max(valve) <= 1.0
        else float(threshold)  # if the valve series is between 0-1 then it should have a
        for threshold, valve in zip(thresholds, valves)  # max value of 1 so we change the threshold from percentage
    ]  # to fraction.
    threshold_master, threshold_wing, threshold_choke = thresholds

    is_master_gt_threshold = master_valve >= threshold_master
    is_wing_gt_threshold = wing_valve >= threshold_wing
    is_choke_gt_threshold = choke_valve >= threshold_choke

    return pd.Series(
        np.logical_and.reduce((is_master_gt_threshold, is_wing_gt_threshold, is_choke_gt_threshold)).astype(int),
        index=master_valve.index,
    )
