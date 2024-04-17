# Copyright 2023 Cognite AS
from typing import Optional, Union

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
        (
            _pct_to_fraction(threshold) if max(valve) <= 1.0 else float(threshold)
        )  # if the valve series is between 0-1 then it should have a
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


@check_types
def merge_valves(valves: list[pd.Series]) -> pd.Series:
    """Merge n number of valve time series on the same flow into one.

    The lowest value of the valves is taken at each time step.

    Args:
        valves: Valves
            List of time series of the valves.

    Returns:
        pd.Series: Merged Valve
            Merged time series of the valves.
    """
    if valves:
        return pd.concat(valves).groupby(level=0).min()
    else:
        return pd.Series([])


# @check_types
def calculate_xmt_prod_status(
    choke_valve: pd.Series,  # Can be partially opened. The others are binary (0 or 1)
    master_valves: Optional[list[pd.Series]] = None,
    annulus_valves: Optional[list[pd.Series]] = None,
    xover_valves: Optional[list[pd.Series]] = None,
    threshold_master: float = 1.0,  # Per cent. Range 0-100. Default 1%
    threshold_annulus: float = 1.0,  # Per cent. Range 0-100. Default 1%
    threshold_xover: float = 1.0,  # Per cent. Range 0-100. Default 1%
    threshold_choke: float = 5.0,  # Per cent. Range 0-100. Default 5%
    align_timesteps: bool = False,
) -> pd.Series:
    """Determine if the well is producing.

    This function is an improvement of the 'calculate_well_prod_status' function, and this new function is recommended for calculating production status for xmas trees.

    In order for this to be the case, the following has to happen:

            * All Master, Annulus, Xover and Choke data have to come from the same well.
            * Check if the master, annulus, xover and choke valve openings are above their respective threshold values at a given time.
            * If any of the valves are below the threshold opening, then the well is closed.
            * If all of the valves are above the threshold opening, then the well is open.
            * Threshold values should be between 0-100.

    Args:
        choke_valve:  Choke Valve
            Time series of the choke valve.
        master_valves: Master Valves
            Time series of the master valves.
        annulus_valves:  Annulus Valves
            Time series of the annulus valves.
        xover_valves:  Xover Valves
            Time series of the xover valves.

        threshold_master: Master threshold
            Threshold percentage value from 0%-100%.
        threshold_annulus: Annulus threshold
            Threshold percentage value from 0%-100%.
        threshold_xover: Xover threshold
            Threshold percentage value from 0%-100%.
        threshold_choke: Choke threshold
            Threshold percentage value from 0%-100%.
        align_timesteps: Auto-align
            Automatically align time stamp  of input time series. Default is False.

    Returns:
        pandas.Series: Well Status
            Well production status (1 means open, 0 means closed).
    """
    wellhead_super_valves = []
    wellhead_thresholds = []
    if master_valves:
        wellhead_super_valves.append(merge_valves(master_valves))
        wellhead_thresholds.append(threshold_master)
    if annulus_valves:
        wellhead_super_valves.append(merge_valves(annulus_valves))
        wellhead_thresholds.append(threshold_annulus)
    if xover_valves:
        wellhead_super_valves.append(merge_valves(xover_valves))
        wellhead_thresholds.append(threshold_xover)

    # Input validation
    if not wellhead_super_valves:
        raise UserValueError(
            "At least one of the wellhead valve time series (master, annulus or xover) must be provided"
        )

    if any(valve.empty for valve in [*wellhead_super_valves, choke_valve]):
        raise UserValueError("Empty Series are not allowed for valve inputs")

    if any([i < 0 for i in [*wellhead_thresholds, threshold_choke]]):
        raise UserValueError("Threshold value has to be greater than or equal to 0")

    if any([i > 100 for i in [*wellhead_thresholds, threshold_choke]]):
        raise UserValueError("Threshold value has to be less than or equal to 100")

    # Aligning time series on time index
    *wellhead_super_valves, choke_valve = auto_align(
        [*wellhead_super_valves, choke_valve],
        align_timesteps,
    )

    is_wellhead_super_valves_open = [
        valve >= threshold for valve, threshold in zip(wellhead_super_valves, wellhead_thresholds)
    ]
    is_choke_valve_open = choke_valve >= threshold_choke

    is_wellhead_open = pd.Series(
        np.logical_or.reduce(array=is_wellhead_super_valves_open).astype(int),
        index=choke_valve.index,
    )

    return pd.Series(
        np.logical_and.reduce((is_wellhead_open, is_choke_valve_open)).astype(int),
        index=choke_valve.index,
    )
