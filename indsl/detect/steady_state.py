# Copyright 2023 Cognite AS
import numpy as np
import pandas as pd

from scipy import stats

from indsl.decorators import njit
from indsl.detect.change_point_detector import ed_pelt
from indsl.detect.utils import constrain, resample_timeseries
from indsl.exceptions import UserValueError

# Detectors
from indsl.type_check import check_types
from indsl.validations import validate_series_has_time_index


@check_types
def ssd_cpd(
    data: pd.Series, min_distance: int = 15, var_threshold: float = 2.0, slope_threshold: float = -3.0
) -> pd.Series:
    """Steady State Detection (CPD).

    Detect steady state periods in a time series based on a change point detection algorithm.  The time series is split
    into "statistically homogeneous" segments using the ED Pelt change point detection algorithm. Then each segment is tested with regard
    to a normalized standard deviation and the slope of the line of best fit to determine if the segment can be
    considered a steady or transient region.

    Args:
        data: Time series.

        min_distance: Minimum distance.
            Specifies the minimum point-wise distance for each segment that will be considered in the Change
            Point Detection algorithm.

        var_threshold: Variance threshold.
            Specifies the variance threshold. If the normalized variance calculated for a given segment is greater than
            the threshold, the segment will be labeled as transient (value = 0).

        slope_threshold: Slope threshold.
            Specifies the slope threshold. If the slope of a line fitted to the data of a given segment is greater than
            10 to the power of the threshold value, the segment will be labeled as transient (value = 0).

    Returns:
        pandas.Series: Binary time series.
        Steady state = 1, Transient = 0.
    """
    validate_series_has_time_index(data)

    # resample the provided timeseries into an equally spaced one
    data_resampled: pd.Series = resample_timeseries(data=data)

    # store the arrays with datapoints and timestamp (in ms since epoch)
    y: np.ndarray = data_resampled.values
    x: np.ndarray = data_resampled.index.to_numpy("datetime64[ms]").view(np.int64)

    # the maximum allowable distance is half the number of datapoints
    max_distance: int = int(np.floor(len(x) / 2))
    if min_distance > max_distance:
        raise UserValueError(
            f"min_distance ({min_distance}) is larger than the maximum allowable distance ({max_distance})"
        )

    # compute the change points
    change_points: np.ndarray = ed_pelt(data=y, min_distance=min_distance)

    # include the endpoints to slice the segments
    change_points = np.insert(change_points, 0, 0)  # index of the first datapoint
    change_points = np.insert(change_points, len(change_points), len(y))  # index of the last datapoint

    # compute mean of the data
    avg: float = np.mean(y)
    # constrains the mean of the data into predefined limits
    # this will prevent generating infinite values on the var calculation below
    divisor: float = constrain(value=avg, min=1.0e-4, max=1.0e6)

    # loop over all changepoint segments
    # we consider a region as transient (value = 0) unless it passes the subsequent logical tests
    ss_map: np.ndarray = np.zeros_like(y)
    for i in range(1, len(change_points)):
        i0: np.int64 = np.int64(change_points[i - 1])
        i1: np.int64 = np.int64(change_points[i])
        xi: np.ndarray = x[i0:i1]
        yi: np.ndarray = y[i0:i1]

        # compute the segment standard deviation and normalize it by the segment length
        std: float = np.std(yi) / (i1 - i0)
        # calculate a normalized variance (this is an effort to make the var_threshold more generic between different
        # timeseries)
        var: float = 1.0e5 * std / divisor

        # we first check if the variance criteria is met
        if abs(var) < var_threshold:
            # fit a line to the segment data
            lr = stats.linregress(xi, yi)
            slope = lr.slope

            # then we check if the slope criteria is met
            if abs(slope) < np.power(10.0, slope_threshold):
                ss_map[i0:i1] = 1.0

    # we finally create the series that is returned by the function
    ss_map_series: pd.Series = pd.Series(data=ss_map, index=data_resampled.index)

    return ss_map_series


@check_types
def ssid(
    data: pd.Series, ratio_lim: float = 2.5, alpha1: float = 0.2, alpha2: float = 0.1, alpha3: float = 0.1
) -> pd.Series:
    """Steady state (variance).

    The steady state detector is based on the ration of two variances estimated from the same signal [#]_ . The algorithm first
    filters the data using the factor "Alpha 1" and calculates two variances (long and short term) based on the
    parameters "Alpa 2" and "Alpha 3". The first variance is an exponentially weighted moving variance based on the
    difference between the data and the average. The second is also an exponentially weighted moving “variance” but
    based on sequential data differences. Larger Alpha values imply that fewer data are involved in the analysis,
    which has the benefit of reducing the time for the identifier to detect a process change (average run length, ARL)
    but has an undesired impact of increasing the variability on the results, broadening the distribution and
    confounding interpretation. Lower λ values undesirably increase the average run length to detection but increase
    precision (minimizing Type-I and Type-II statistical errors) by reducing the variability of the distributions
    and increasing the signal-to-noise ratio of a TS to SS situation.

    Args:
        data: Time series.
        ratio_lim: Threshold.
            Specifies the variance ratio threshold if it is in steady state or not. A variance ratio greater than the
            threshold labels the state as transient.
        alpha1: Alpha 1.
            Filter factor for the mean. Value should be between 0 and 1. Recommended value is 0.2.
            Defaults to 0.2.
        alpha2: Alpha 2.
            Filter factor for variance 1. Value should be between 0 and 1. Recommended value is 0.1.
            Defaults to 0.1.
        alpha3: Alpha 3.
            Filter factor for variance 2. Value should be between 0 and 1. Recommended value is 0.1.
            Defaults to 0.1.

    Returns:
        pandas.Series: Binary time series.
        Steady state = 0, transient = 1.

    References:
        .. [#] Rhinehart, R. Russell. (2013). Automated steady and transient state identification in noisy processes.
               Proceedings of the American Control Conference. 4477-4493. 10.1109/ACC.2013.6580530
    """
    # Determine number of datapoints to be used to initialize values (must have at least 10 within range)
    Ls = int(min(len(data) * 0.05, 50))

    if Ls < 10:
        raise UserValueError("There are too few datapoints to detect steady states.")

    # Check the bounds of alpha parameters
    for par, par_val in zip([alpha1, alpha2, alpha3], ["alpha1", "alpha2", "alpha3"]):
        if not (0 <= par <= 1):
            raise UserValueError(f"Alpha parameters should be between 0 and 1. Received {par_val} for {par}")

    # Split data into first sample points and rest of the data
    data_ini = data[0:Ls]
    df = data[Ls + 1 :]

    # Initialize filter and variance
    filt0 = data_ini[0:Ls].mean()
    var0 = data_ini[0:Ls].var()
    initial = data_ini.iloc[Ls - 1]

    start_val = df.iloc[0]
    filt = np.array([alpha1 * start_val + (1 - alpha1) * filt0])
    var1 = np.array([alpha2 * (start_val - filt0) ** 2 + (1 - alpha2) * var0])
    var2 = np.array([alpha3 * (start_val - initial) ** 2 + (1 - alpha3) * var0])

    # Calculate filter
    values = df.tolist()
    for i in range(1, len(values)):
        filt = np.append(filt, alpha1 * values[i] + (1 - alpha1) * filt[i - 1])
        var1 = np.append(var1, alpha2 * (values[i] - filt[i - 1]) ** 2 + (1 - alpha2) * var1[i - 1])
        var2 = np.append(var2, alpha3 * (values[i] - values[i - 1]) ** 2 + (1 - alpha3) * var2[i - 1])

    # Calculate Variance Ratio and find unsteady regions
    ratio = (2 - alpha1) * (var1 / var2)
    unsteady_flag = ratio > ratio_lim

    return pd.Series(unsteady_flag, index=df.index).astype(int)


@check_types
def vma(series: pd.Series, window_length: int = 10) -> pd.Series:
    """Steady state (vma).

    This moving average is designed to become flat (constant value) when the data
    within the lookup window does not vary significantly. It can also be state detector. The calculation is based on
    the variability of the signal in a lookup window.

    Args:
        series: Time series.
        window_length: Lookup window.
            Window length in data points used to estimate the variability of the signal.

    Returns:
        pandas.Series: Moving average.
        If the result has the same value as the previous moving average result, the signal can be considered to
        be on steady state.
    """
    k = 1.0 / window_length
    N = len(series)

    # Estimate iS:
    diff = np.diff(series)
    iS = _estimate_iS(diff, k, N)

    # Estimate hhv, llv, vI:
    iS_df = pd.Series(iS, index=series.index).rolling(window_length, min_periods=1).aggregate(["min", "max"]).fillna(0)
    hhv, llv = iS_df["max"], iS_df["min"]
    vI = (iS - llv) / (hhv - llv)

    # Finally, estimate variable moving average:
    return pd.Series(_estimate_vma(vI.values, series.values, k, N), index=series.index)


@njit
def _estimate_iS(diff: np.ndarray, k: float, N: int) -> np.ndarray:
    # Estimate pdm and mdm:
    pdm, mdm = np.zeros(N), np.zeros(N)
    pdm[1:] = np.maximum(diff, 0)
    mdm[1:] = np.maximum(-diff, 0)
    # Estimate pdmS, mdmS:
    pdmS, mdmS = np.zeros(N), np.zeros(N)
    for i in range(1, N):
        pdmS[i] = (1 - k) * pdmS[i - 1] + k * pdm[i]
        mdmS[i] = (1 - k) * mdmS[i - 1] + k * mdm[i]
    pdi, mdi = np.zeros(N), np.zeros(N)
    s = mdmS + pdmS
    pdi[1:] = pdmS[1:] / s[1:]
    mdi[1:] = mdmS[1:] / s[1:]
    # Estimate pdiS, mdiS, d, s1:
    pdiS, mdiS = np.zeros(N), np.zeros(N)
    for i in range(1, N):
        pdiS[i] = (1 - k) * pdiS[i - 1] + k * pdi[i]
        mdiS[i] = (1 - k) * mdiS[i - 1] + k * mdi[i]
    d = np.abs(pdiS - mdiS)
    s1 = pdiS + mdiS
    # Estimate iS:
    iS = np.zeros(N)
    for i in range(1, N):
        iS[i] = (1 - k) * iS[i - 1] + k * d[i] / s1[i]
    return iS


@njit
def _estimate_vma(vI: np.ndarray, arr: np.ndarray, k: float, N: int):
    vma = np.zeros(N)
    for i in range(1, N):
        vma[i] = (1 - k * vI[i]) * vma[i - 1] + k * vI[i] * arr[i]
    return vma
