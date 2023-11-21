# Copyright 2023 Cognite AS
from typing import List

import numpy as np
import pandas as pd

from indsl.decorators import jit
from indsl.detect.utils import resample_timeseries
from indsl.exceptions import UserRuntimeError, UserTypeError, UserValueError
from indsl.type_check import check_types
from indsl.validations import validate_series_has_time_index


@check_types
def cpd_ed_pelt(data: pd.Series, min_distance: int = 1) -> pd.Series:
    """Change Point Detection.

    This function detects change points in a time series. The time series is split into "statistically homogeneous" segments using the
    ED Pelt change point detection algorithm while observing the minimum distance argument.

    Args:
        data: Time series

        min_distance: Minimum distance.
            Specifies the minimum point wise distance for each segment that will be considered in the Change
            Point Detection algorithm.

    Returns:
        pandas.Series: Time series.
        Binary time series.
    """
    validate_series_has_time_index(data)

    data_resampled: pd.Series = resample_timeseries(data=data)
    # the maximum allowable distance is half the number of datapoints
    max_distance: int = int(np.floor(len(data_resampled) / 2))
    if min_distance > max_distance:
        raise UserRuntimeError(
            f"Minimum segment distance argument ({min_distance}) is larger than the maximum allowable segment distance ({max_distance})"
        )

    change_points: np.ndarray = ed_pelt(data=data_resampled.values, min_distance=min_distance)

    # create the binary time series full of zeros
    change_points_ts: pd.Series = pd.Series(index=data_resampled.index, data=[0] * len(data_resampled))

    # add the change points to the time series
    for cp in change_points:
        change_points_ts.iloc[cp] = 1
        # add the value of 0 to 1ns before and after the timestamp of the current change point
        change_points_ts = pd.concat(
            [
                change_points_ts,
                pd.Series(
                    index=[
                        data_resampled.index[cp] - pd.Timedelta(value=1, unit="nanoseconds"),
                        data_resampled.index[cp] + pd.Timedelta(value=1, unit="nanoseconds"),
                    ],
                    data=[0, 0],
                ),
            ]
        )

    # returns a time series with values at the change points and nan's everywhere else
    # this will be rendered as vertical lines in the change point locations
    return change_points_ts.sort_index()


@check_types
def get_partial_sums(data: np.ndarray, k: int) -> np.ndarray:
    """Partial sums.

    Partial sums for empirical CDF (formula (2.1) from Section 2.1 "Model" in [Haynes2017])

    Args:
        data: Input data
        k: Number of quantiles

    Returns:
        np.ndarray: Partial Sums
    """
    n: int = len(data)
    partial_sums: np.ndarray = np.zeros(shape=(k, n + 1), dtype=int)
    sorted_data: np.ndarray = np.sort(data)

    for i in range(k):
        z: float = -1 + (2 * i + 1.0) / k  # values from (-1 + 1 / k) to (1 - 1 / k) with step = 2 / k
        p: float = 1.0 / (1 + np.power(2 * n - 1, -z))  # values from 0.0 to 1.0
        t: float = sorted_data[int(np.trunc((n - 1) * p))]  # quantile value, formula (2.1) in[Haynes2017]

        for tau in range(1, n + 1):
            partial_sums[i, tau] = partial_sums[i, tau - 1]
            if data[tau - 1] < t:
                partial_sums[i, tau] += 2  # we use doubled value (2) instead of original 1.0
            if data[tau - 1] == t:
                partial_sums[i, tau] += 1  # we use doubled value (1) instead of original 0.5

    return partial_sums


@jit(nopython=True)
def get_segment_cost(partial_sums: np.ndarray, tau1: int, tau2: int, k: int, n: int) -> float:
    """Calculates the cost of the (tau1; tau2] segment.

    Args:
        partial_sums: Partial sums for empirical CDF
        tau1: Start of the segment
        tau2: End of the segment
        k: Number of quantiles
        n: Number of datapoints

    Returns:
        float: Segment cost
    """
    sum: float = 0
    for i in range(k):
        # actual_sum is (count(data[j] < t) * 2 + count(data[j] == t) * 1) for j=tau1..tau2-1
        actual_sum: int = partial_sums[i, tau2] - partial_sums[i, tau1]

        # we skip these two cases (correspond to fit = 0 or fit = 1) because of invalid np.log values
        if (actual_sum != 0) & (actual_sum != (tau2 - tau1) * 2):
            # empirical CDF F_i(t) (Section 2.1 "Model" in [Haynes2017])
            fit: float = actual_sum * 0.5 / (tau2 - tau1)
            # segment cost L_np (Section 2.2 "Nonparametric maximum likelihood" in [Haynes2017])
            lnp: float = (tau2 - tau1) * (fit * np.log(fit) + (1 - fit) * np.log(1 - fit))
            sum += lnp

    c: float = -np.log(2 * n - 1)  # Constant from Lemma 3.1 in [Haynes2017]
    return 2.0 * c / k * sum  # See Section 3.1 "Discrete approximation" in [Haynes2017]


@check_types
def ed_pelt(data: np.ndarray, min_distance: int = 1) -> np.ndarray:
    """The ED-PELT algorithm for change point detection.

    For a given array of `float` values, this algorithm detects locations of change points that splits original series of values into
    "statistically homogeneous" segments. Such points correspond to moments when statistical properties of the
    distribution are changing. This method supports nonparametric distributions and has O(N*log(N)) algorithmic
    complexity.

    Args:
        data: An array with sensor values.
        min_distance: Minimum distance between changepoints. Defaults to 1.

    Raises:
        UserTypeError: data needs to be a numpy array containing float elements.
        UserTypeError: min_distance needs to be a value between 1 and len(data).

    Returns:
        np.ndarray: Returns an array with 1-based indexes of changepoint. Changepoints correspond to the end of the
        detected segments.

    References:
        1. [Haynes2017] Haynes, Kaylea, Paul Fearnhead, and Idris A. Eckley. "A computationally efficient nonparametric
        approach for changepoint detection." Statistics and Computing 27, no. 5 (2017): 1293-1305.
        https://doi.org/10.1007/s11222-016-9687-5

        2. [Killick2012] Killick, Rebecca, Paul Fearnhead, and Idris A. Eckley. "Optimal detection of changepoints with
        a linear computational cost." Journal of the American Statistical Association 107, no. 500 (2012): 1590-1598.
        https://arxiv.org/pdf/1101.1438.pdf

    Based on the ED-Pelt C# implementation from (c) 2019 Andrey Akinshin
    Licensed under The MIT License https://opensource.org/licenses/MIT
    """
    # check if the provided data is in the correct format
    if data.dtype.kind not in np.typecodes["AllFloat"]:
        raise UserTypeError("data argument need to be a numpy array containing float elements")

    # we will use `n` as the number of elements in the `data` array
    n: int = len(data)

    # check corner cases
    if n <= 2:
        return np.array([])
    if not 1 <= min_distance <= n:
        raise UserValueError(f"min_distance ({min_distance}) should be in range from 1 to {n}")

    # the penalty which we add to the final cost for each additional changepoint
    # here we use the Modified Bayesian Information Criterion
    penalty: float = 3 * np.log(n)

    # `k` is the number of quantiles that we use to approximate an integral during the segment cost evaluation
    # we use `k = ceiling(4 * log(n))` as suggested in the Section 4.3 "Choice of K in ED-PELT" in [Haynes2017]
    # `k` can't be greater than `n`, so we should always use the `Min` function here (important for n <= 8)
    k: int = min(n, int(np.ceil(4 * np.log(n))))

    # we should precalculate sums for empirical CDF, it will allow fast evaluating of the segment cost
    partial_sums: np.ndarray = get_partial_sums(data, k)

    # since we use the same values of `partial_sums`, `k`, `n` all the time,
    # we introduce a shortcut `cost(tau1, tau2)` for segment cost evaluation.
    # hereinafter, we use `tau` to name variables that are changepoint candidates.
    def cost(tau1: int, tau2: int) -> float:
        return get_segment_cost(partial_sums, tau1, tau2, k, n)

    # we will use dynamic programming to find the best solution; `best_cost` is the cost array.
    # `best_cost[i]` is the cost for subarray `data[0..i - 1]`.
    # it's a 1-based array (`data[0]`..`data[n-1]` correspond to `best_cost[1]`..`best_cost[n]`)
    best_cost: np.ndarray = np.zeros(n + 1)
    best_cost[0] = -penalty
    for current_tau in range(min_distance, 2 * min_distance):
        best_cost[current_tau] = cost(0, current_tau)

    # `previous_change_point_index` is an array of references to previous changepoints. If the current segment ends at
    # the position `i`, the previous segment ends at the position `previous_change_point_index[i]`. It's a 1-based
    # array(`data[0]`..`data[n - 1]` correspond to the `previous_change_point_index[1]`..
    # `previous_change_point_index[n]`)
    previous_change_point_index: np.ndarray = np.zeros(shape=n + 1, dtype=int)

    # we use PELT (Pruned Exact Linear Time) approach which means that instead of enumerating all possible previous
    # tau values, we use a whitelist of "good" tau values that can be used in the optimal solution. If we are 100%
    # sure that some of the tau values will not help us to form the optimal solution, such values should be
    # removed. See [Killick2012] for details.
    previous_taus: List[int] = [0, min_distance]

    # following the dynamic programming approach, we enumerate all tau positions. For each `current_tau`, we pretend
    # that it's the end of the last segment and trying to find the end of the previous segment.
    for current_tau in range(2 * min_distance, n + 1):
        # for each previous tau, we should calculate the cost of taking this tau as the end of the previous
        # segment. This cost equals the cost for the `previous_tau` plus cost of the new segment (from `previous_tau`
        # to `current_tau`) plus penalty for the new changepoint.
        cost_for_previous_tau: List[float] = [
            best_cost[previous_tau] + cost(previous_tau, current_tau) + penalty for previous_tau in previous_taus
        ]

        # Now we should choose the tau that provides the minimum possible cost.
        best_previous_tau_index: np.int64 = np.argmin(cost_for_previous_tau).astype(np.int64)
        best_cost[current_tau] = cost_for_previous_tau[best_previous_tau_index]
        previous_change_point_index[current_tau] = previous_taus[best_previous_tau_index]

        # Prune phase: we remove "useless" tau values that will not help to achieve minimum cost in the future
        current_best_cost: float = best_cost[current_tau]
        new_previous_taus_size: int = 0
        for i in range(len(previous_taus)):
            if cost_for_previous_tau[i] < current_best_cost + penalty:
                previous_taus[new_previous_taus_size] = previous_taus[i]
                new_previous_taus_size += 1
        previous_taus = previous_taus[:new_previous_taus_size]

        # We add a new tau value that is located on the `min_distance` distance from the next `current_tau` value
        previous_taus.append(current_tau - (min_distance - 1))

    # here we collect the result list of changepoint indexes `change_point_indexes` using `previous_change_point_index`
    change_point_indexes: List[int] = []
    current_index: np.int64 = np.int64(
        previous_change_point_index[n]
    )  # The index of the end of the last segment is `n`
    while current_index != 0:
        change_point_indexes.append(current_index)
        current_index = np.int64(previous_change_point_index[current_index])

    # sort the change_point_indexes and transform it to a numpy ndarray
    result: np.ndarray = np.asarray(change_point_indexes, dtype=int)
    result_sorted: np.ndarray = np.sort(result)

    return result_sorted
