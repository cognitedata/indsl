# Copyright 2023 Cognite AS
from typing import Optional

import numpy as np
import pandas as pd

from indsl.exceptions import UserValueError
from indsl.type_check import check_types


@check_types
def calculate_shutin_interval(
    shut_valve: pd.Series,
    min_shutin_len: float = 6,
    min_time_btw_shutins: float = 1,
    shutin_state_below_threshold: bool = True,
    shutin_threshold: Optional[float] = None,
) -> pd.Series:
    """Shut-in interval.

    The shut-in interval is defined as the period when the valve is in closed state. The close state is determined based
    on the calculated manually-given threshold. The threshold is calculated based on the analysis of the valve signal
    histogram.

    Args:
        shut_valve: Shut-in valve.
            Time series with the shut-in valve opening.
        min_shutin_len: Minimum time.
            Minimum shut-in length in hours to be considered for detection.
        min_time_btw_shutins: Time between.
            Minimum time between consecutive shut-ins in hours to validate change of state.
        shutin_state_below_threshold: Below threshold.
            Indicator to tell the algorithm if the shut-in state is below the threshold.
        shutin_threshold: Threshold.
            Threshold between the valve open and close states. Defaults to None, meaning that the threshold is calculated.

    Returns:
        pandas.Series: Shut-in periods.
        Binary time series indicating open state or closed state: Open state= 1, Close state = 0.

    Raises:
        RuntimeError: If threshold cannot be determined in the evaluation window, the user is asked to increase the evaluation period. Recommended period is 30 days.
    """
    # calculate threshold
    if not shutin_threshold:
        shutin_threshold = calculate_threshold(shut_valve)

    # create binary timeseries
    wvalve = shut_valve.to_frame(name="valve_state_org")
    wvalve["timestamp"] = wvalve.index
    if shutin_state_below_threshold:
        wvalve["valve_state"] = np.where(wvalve["valve_state_org"] > shutin_threshold, 1, 0)
    else:
        wvalve["valve_state"] = np.where(wvalve["valve_state_org"] > shutin_threshold, 0, 1)
    wvalve["valve_diff"] = wvalve["valve_state"].diff()
    if wvalve["valve_diff"].std() == 0:
        # no change in state, return the original state
        return wvalve["valve_state"].rename(None)
    wvalve["final_indicator"] = 1
    wvalve = wvalve.dropna()

    # get shut-in starts and ends
    sh_start = list(wvalve[wvalve["valve_diff"] == -1]["timestamp"])
    sh_end = list(wvalve[wvalve["valve_diff"] == 1]["timestamp"])
    # create time pairs
    if sh_start and sh_end:
        # case 1: several shut-ins exist - the shut-in comes from the past - make a pseudo-start of the shut-in
        if sh_end[0] < sh_start[0]:
            sh_start = [wvalve.iloc[0, :]["timestamp"], *sh_start]
        # case 2: several shut-ins exist - the shut-in goes to the future - make a pseudo-end of shut-in
        if sh_end[-1] < sh_start[-1]:
            sh_end = [*sh_end, wvalve.iloc[-1, :]["timestamp"]]
    elif not sh_start:
        # case 3: only one shut-in exist - the shut-in comes from the past - make a pseudo-start of the shut-in
        sh_start = [wvalve.iloc[0, :]["timestamp"], *sh_start]
    else:
        # case 4: only one shut-in exist - the shut-in comes from the past - make a pseudo-start of the shut-in
        sh_end = [*sh_end, wvalve.iloc[-1, :]["timestamp"]]
    time_pairs = list(zip(sh_start, sh_end))

    # merge the shut-ins based on the minimum distance between consecutive shut-ins
    time_pairs_upd = []
    time_pairs.append(time_pairs[-1])

    i = 0
    new_pair = time_pairs[i]
    time_pairs_upd.append(new_pair)
    while i < len(time_pairs) - 1:
        delta_hr = (time_pairs[i + 1][0] - new_pair[1]).total_seconds() / 3600
        if delta_hr < min_time_btw_shutins:
            new_pair = (new_pair[0], time_pairs[i + 1][1])
            if time_pairs_upd[-1][0] == new_pair[0]:
                time_pairs_upd[-1] = new_pair
        else:
            new_pair = time_pairs[i + 1]
            time_pairs_upd.append(new_pair)
        i = i + 1

    # filter the shut-ins based on the minimum duration condition
    time_pairs_upd = [pair for pair in time_pairs_upd if (pair[1] - pair[0]).total_seconds() / 3600 > min_shutin_len]
    for start, end in time_pairs_upd:
        mask = np.logical_and(
            wvalve["timestamp"] >= start,
            wvalve["timestamp"] <= end,
        )
        wvalve.loc[mask, "final_indicator"] = 0
    return wvalve["final_indicator"].dropna().rename(None)


@check_types
def calculate_threshold(shut_valve: pd.Series) -> float:
    """Calculate valve threshold.

    The valve threshold refers to the threshold between valve open and close states. The threshold is calculated as
    a mean of two histogram peaks (max values). The histogram should have two peaks indicating two states.

    Args:
        shut_valve: pandas.Series
            Shut-in valve signal

    Returns:
        float: Threshold

    Raises:
        RuntimeError: If threshold cannot be determined in the evaluation window, the user is asked to increase the evaluation period. Recommended period is 30 days.
    """
    freqs, edges = np.histogram(shut_valve.values, bins=5)
    # get indices of 2 most frequent bins
    freqs_sorted = [el[0] for el in sorted(enumerate(freqs), key=lambda x: x[1])]
    max1_idx, max2_idx = freqs_sorted[-2], freqs_sorted[-1]
    # get mean values between bin edges
    mean_edge = np.array([(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)])
    # calculate threshold
    if 1.5 * min(mean_edge[max1_idx], mean_edge[max2_idx]) > max(mean_edge[max1_idx], mean_edge[max2_idx]):
        raise UserValueError("Not enough data to detect the threshold")
    else:
        threshold = (mean_edge[max1_idx] + mean_edge[max2_idx]) / 2
    return threshold
