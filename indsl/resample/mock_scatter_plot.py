# Copyright 2024 Cognite AS
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd

from indsl.resample.auto_align import auto_align
from indsl.type_check import check_types


@check_types
def __reindex_scatter_core(
    signal_x: pd.Series, x_min_value: float = 0, x_max_value: float = 1
) -> Tuple[np.array, np.array]:
    """Reindex scatterplot core.

    It returns a reindexed array of timestamp. The timestamps are creates such that the values from signal_x
    are scaled to the range of signal_x.index, and then applied as index.
    This is a way of creating a scatterplot inside a chart

    Args:
        signal_x: x-value.
            The time series where the values are used as the x-value
        x_min_value: Minimum x value.
            Minimum x value, used to scale the x values.
        x_max_value: Maximum x value.
            Maximum x value, used to scale the x value.

    Returns:
        np.array: Scatter plot index, Scatter plot sort index
    """
    # convert timestamps to epoc
    epoc = np.array([val.timestamp() for val in signal_x.index])
    d_epoc = epoc[-1] - epoc[0]
    # We will now map the values in singla_x to the epoc and then convert it back to datetime
    sequence_epoc = signal_x.values / (x_max_value - x_min_value) * d_epoc
    index_x_epoc = sequence_epoc + epoc[0]  # translate
    index_x = np.array([datetime.fromtimestamp(epoc_) for epoc_ in index_x_epoc])
    # create a sort index, such that the order is increasing
    index_sort = np.argsort(index_x_epoc)

    return index_x[index_sort], index_sort


@check_types
def reindex_scatter(
    signal_x: pd.Series,
    signal_y: pd.Series,
    x_min_value: float = 0,
    x_max_value: float = 1,
    align_timesteps: bool = False,
) -> pd.Series:
    """Reindex scatterplot.

    It returns the values from signal_y with the timestamps as the values from signal_x,
    where the timestamps has been scaled to the range of timestamps from signal_x.
    The timestamps are sorted in ascending order, and the values are sorted with the same sort-index

    This is a way of creating a scatterplot inside a chart

    Args:
        signal_x: x-value.
            The time series where the values are used as the x-value
        signal_y: y-value.
            The time series where the values are used as the y-value
        x_min_value: Minimum x value.
            Minimum x value, used to scale the x values.
        x_max_value: Maximum x value.
            Maximum x value, used to scale the x value.
        align_timesteps (bool) : Auto-align
            Automatically align time stamp  of input time series. Default is False.

    Returns:
        pandas.Series: Scatter plot
    """
    if align_timesteps:
        signal_x, signal_y = auto_align([signal_x, signal_y], align_timesteps)

    index_x_sorted, index_sort = __reindex_scatter_core(signal_x, x_min_value=x_min_value, x_max_value=x_max_value)

    signal_scatter = pd.Series(signal_y.array[index_sort], index=index_x_sorted)

    return signal_scatter


@check_types
def reindex_scatter_x(signal_x: pd.Series, x_min_value: float = 0, x_max_value: float = 1) -> pd.Series:
    """Reindex scatterplot x-values.

    It returns the values from signal_y with the timestamps as the values from signal_x,
    where the timestamps has been scaled to the range of timestamps from signal_x.
    The timestamps are sorted in ascending order, and the values are sorted with the same sort-index
    In effect this is a straight line going from x_min_value to x_max_value over the time range of signal_x

    Args:
        signal_x: x-value.
            The time series where the values are used as the x-value
        x_min_value: Minimum x value.
            Minimum x value, used to scale the x values.
        x_max_value: Maximum x value.
            Maximum x value, used to scale the x value.

    Returns:
        pandas.Series: Scatter plot
    """
    index_x_sorted, index_sort = __reindex_scatter_core(signal_x, x_min_value=x_min_value, x_max_value=x_max_value)

    signal_scatter_x = pd.Series(signal_x.array[index_sort], index=index_x_sorted)

    return signal_scatter_x