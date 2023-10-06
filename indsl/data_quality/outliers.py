# Copyright 2023 Cognite AS
from typing import List

import numpy as np
import pandas as pd

from numpy.polynomial import polynomial
from scipy.stats import t as student_dist

from indsl.exceptions import UserValueError
from indsl.resample.reindex import reindex
from indsl.smooth import sg
from indsl.type_check import check_types
from indsl.validations import validate_series_has_minimum_length


@check_types
def extreme(data: pd.Series, alpha: float = 0.05, bc_relaxation: float = 0.167, poly_order: int = 3) -> pd.Series:
    r"""Extreme outliers removal.

    Outlier detector and removal based on the `paper from Gustavo A. Zarruk
    <https://iopscience.iop.org/article/10.1088/0957-0233/16/10/012/meta>`_. The procedure is as follows:

         * Fit a polynomial curve to the model using all the data
         * Calculate the studentized deleted (or externally studentized) residuals
         * These residuals follow a t distribution with degrees of freedom n - p - 1
         * Bonferroni critical value can be computed using the significance level (alpha) and t distribution
         * Any values that fall outside of the critical value are treated as anomalies

    Use of the hat matrix diagonal allows for the rapid calculation of deleted residuals without having to refit
    the predictor function each time.

    Args:
        data: Time series.
        alpha: Significance level.
            This is a number higher than or equal to 0 and lower than 1. In statistics, the significance level is the
            probability of rejecting the null hypothesis when true. For example, a significance level of 0.05 means that
            there is a 5% risk detecting an outlier that is not a true outlier.
        bc_relaxation: Relaxation factor
            for the Bonferroni critical value. Smaller values will make anomaly detection more conservative. Defaults
            to 1/6.
        poly_order: Polynomial order.
            It represents the order of the polynomial function fitted to the original time series.
            Defaults to 3.

    Returns:
        pandas.Series: Time series.

    Raises:
        UserValueError: Alpha must be a number between 0 and 1
    """
    # Check inputs
    data = data.dropna()

    validate_series_has_minimum_length(data, 3)
    if not 0 <= alpha < 1:
        raise UserValueError("The significance level must be a number higher than or equal to 0 and lower than 1")

    x, y = _split_timeseries_into_time_and_value_arrays(data)

    # Create a polynomial fit and apply the fit to data
    coefs = polynomial.polyfit(x, y, poly_order)
    y_pred = polynomial.polyval(x, coefs)

    hat_diagonal = _calculate_hat_diagonal(x)

    # Calculate degrees of freedom (n-p-1)
    n = len(y)
    dof = n - 3  # Using p = 2 for a model based on a single time series

    # Calculate Bonferroni critical value
    bc = student_dist.ppf(1 - alpha / (2 * n), df=dof) * bc_relaxation

    t_res = _calculate_residuals_and_normalize_them(dof, hat_diagonal, y, y_pred)

    # Return filtered dataframe with the anomalies removed
    mask = np.logical_and(t_res < bc, t_res > -bc)
    return pd.Series(y[mask], index=data.index[mask])


@check_types
def out_of_range(
    data: pd.Series,
    window_length: List[int] = [20, 20],
    polyorder: List[int] = [3, 3],
    alpha: List[float] = [0.05, 0.05],
    bc_relaxation: List[float] = [0.25, 0.5],
    return_outliers: bool = True,
) -> pd.Series:
    r"""Out of range.

    The main objective of this function is to detect data points outside the typical range or  unusually far
    from the main trend. The method is data adaptive; i.e. it  should work independent of the data characteristics. The
    only exception, the method is designed for *non-linear, non-stationary sensor data*, one of the most common type
    of time series in physical processes.

    Outliers are detected using an iterative and data-adaptive method. Additional details on the analytical methods are provided below. But it is basically a
    three step process, carried out in two iterations. The three steps are:

    1. Estimate the main trend of the time series using the method:

        * Savitzky-Golay smoothing (`SG <https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter>`_).

    2. Estimate the `studentized residuals <https://en.wikipedia.org/wiki/Studentized_residual>`_.

    3. Identify outliers using the `bonferroni correction <https://mathworld.wolfram.com/BonferroniCorrection.html>`_
       or bonferroni outlier test.

    The results from the first iteration (new time series without the detected outliers) are used to carry out the
    second iteration.
    The `Student's t-distribution <https://en.wikipedia.org/wiki/Student%27s_t-distribution>`_ is used as it useful when
    estimating the mean of a normally distributed population in situations where the sample size is small and
    the population's standard deviation is unknown. Finally, the bonferroni correction or Boneferroni test is
    a simple and efficient method to tests for extreme outliers. Additional details on each of these methods is
    provided below.

    **Savitzky-Golay Smoother**:

    The SG smoother is a digital filter ideal for smoothing data without distorting the data tendency. But its true
    value comes from being independent of the sampling frequency (unlike most digital filters). Hence, simple
    and robust to apply on data with non-uniform sampling (i.e. industrial sensor data). Two parameters are required
    to apply the SG smoother, a point-wise `window length` and a `polynomial order`. The `window length` is the number
    of data points used to estimate the local trend and the `polynomial order` is used to fit those data points with a
    linear (order 1) or non-linear (order higher than 1) fit.

    **Studentized residuals and the Bonferroni Correction**:

    The `Student's t-distribution <https://en.wikipedia.org/wiki/Student%27s_t-distribution>`_ is typically used when
    the sample size is small and the standard deviation is
    unknown. In this case we assume that there will be a small amount of out of range values and the statistical
    properties of the data are not known. By studentizing the residuals is analogous to normalizing the data and it
    a useful technique for detecting outliers. Furthermore, we apply the bonferroni correction to test if a
    residual is an outlier or not. To studentize the residual an `significance lebel` must be defined. Furthermore, the
    Bonferroni Correction is very conservative before it classifies a data point as an outlier. Consequently, a
    `sensitivity` factor is used to relax the test and identifying points that are located close to the main trend
    and that can be removed in the second iteration.

    Args:
        data: Time series.
        window_length: Window length.
            Point-wise window length used to estimate the local trend using the SG smoother. Two integer values are
            required, one for each iteration. If the SG smoother is not used, these values are ignored. Default value
            is 20 data points for both iterations.
        polyorder: Polynomial order.
            It represents the order of the polynomial function fitted to the data when using the SG smoother.
            Default value is 3 data points for both iterations.
        alpha: Significance level.
            Number higher than or equal to 0 and lower than 1. Statistically speaking, the significance level
            is the probability of detecting an outlier that is not a true outlier. A value of 0.05 means that
            there is a 5% risk of detecting an outlier that is not a true outlier. Defaults to 0.05 for both
            iterations.
        bc_relaxation: Sensitivity.
            Number higher than 0 used to make outlier detection more or less sensitive. Smaller values will make
            the detection more conservative. Defaults 0.25 and 0.5 for the first and second iterations, respectively.
        return_outliers: Output outliers.
            If selected (`True`) the method outputs the detected outliers, otherwise the filtered (no-outliers)
            time series is returned. Defaults to `True`.

    Returns:
        pandas.Series: Time series.
    """
    # Check inputs
    data = data.dropna()

    validate_series_has_minimum_length(data, 3)
    _validations(alpha, bc_relaxation, window_length, polyorder)

    x, y = _split_timeseries_into_time_and_value_arrays(data)

    # ==========
    # FIRST PASS
    # ==========

    # Estimate the trend using Savitzky-Golay smoother
    trend_pass01 = sg(data, window_length=window_length[0], polyorder=polyorder[0])

    # Detect outliers based on first trend estimate
    x, y = _split_timeseries_into_time_and_value_arrays(data)
    y_pred_pass01 = trend_pass01.to_numpy()
    hat_diagonal = _calculate_hat_diagonal(x)

    # Calculate degrees of freedom (n-p-1)
    n = len(y)
    dof = n - 3  # Using p = 2 for a model based on a single time series

    # Calculate Bonferroni critical value and studentized residuals
    bc = student_dist.ppf(1 - alpha[0] / (2 * n), df=dof) * bc_relaxation[0]
    t_res = _calculate_residuals_and_normalize_them(dof, hat_diagonal, y, y_pred_pass01)

    # Boolean mask where outliers are detected
    mask = np.logical_and(t_res < bc, t_res > -bc)
    filtered_ts_pass01 = pd.Series(y[mask], index=data.index[mask])  # Remove detected outliers from time series

    # ===========
    # SECOND PASS
    # ===========

    # Filtering parameters
    trend_pass02 = sg(filtered_ts_pass01, window_length=window_length[1], polyorder=polyorder[1])

    y_pred_pass02 = reindex(trend_pass02, data)
    y_pred_pass02 = y_pred_pass02.to_numpy()

    bc_pass02 = student_dist.ppf(1 - alpha[1] / (2 * n), df=dof) * bc_relaxation[1]
    t_res_pass02 = _calculate_residuals_and_normalize_them(dof, hat_diagonal, y, y_pred_pass02)
    #
    # Boolean mask where outliers are detected
    mask_pass02 = np.logical_and(t_res_pass02 < bc_pass02, t_res_pass02 > -bc_pass02)
    filtered_ts_pass02 = pd.Series(
        y[mask_pass02], index=data.index[mask_pass02]
    )  # Remove detected outliers from time series
    outliers_pass02 = pd.Series(y[~mask_pass02], index=data.index[~mask_pass02])

    # Assign result selected in the arguments
    if return_outliers:
        res = outliers_pass02
    else:
        res = filtered_ts_pass02

    return res


def _validations(alpha, bc_relaxation, window_length, polyorder):
    if not all(0 <= i < 1 for i in alpha):
        raise UserValueError(
            "The Significance Level (alpha) must be a number higher than or equal to 0 and lower " "than 1"
        )
    if not all(i > 0 for i in bc_relaxation) > 0:
        raise UserValueError("The Relaxation Factor must be a number higher than 0")
    if len(window_length) != 2:
        raise UserValueError(f"The window length requires two values, got {window_length}")
    if len(polyorder) != 2:
        raise UserValueError(f"The polynomial order parameter requires two values, got {polyorder}")
    if len(alpha) != 2:
        raise UserValueError(f"The Significance Level (alpha) parameter requires two values, got {alpha}")
    if len(bc_relaxation) != 2:
        raise UserValueError(f"The sensitivity parameter requires two values, got {bc_relaxation}")


def _split_timeseries_into_time_and_value_arrays(data: pd.Series) -> tuple:
    """Split ts into time and value arrays.

    Method that takes a time series with a datetime index and splits it into two numpy arrays. One with the datetime
    index converted to integer values, starting from zero, and the other array contains the values of the time series.

    Args:
        data (pandas.Seres): Time series with a datetime index

    Returns:
        tuple: Tuple containing two 1-D numpy arrays:
            x -> Datetime index converted to integers starting at 0
            y -> Time series values
    """
    x = (np.array(data.index, dtype=np.int64) - data.index[0].value) / 1e9
    y = data.to_numpy()  # Just to please pandas devs
    return x, y


def _calculate_residuals_and_normalize_them(
    dof: int, hat_diagonal: np.ndarray, y: np.ndarray, y_pred: np.ndarray
) -> np.ndarray:
    """Calculate residuals and normalize.

    Calculate studentized residuals between a signal and fitted (e.g. regression) values.

    Args:
        dof (int): Degrees of freedom of the Student's t-distribution
        hat_diagonal (np.ndarray): Diagonal of the hat matrix
        y (np.ndarray) : Original data points
        y_pred (np.ndarray) : Fitted data points

    Returns:
        numpy.ndarray: Studentized residuals
    """
    res = y - y_pred
    sse = np.sum(res**2)
    t_res = res * np.sqrt(dof / (sse * (1 - hat_diagonal) - res**2))
    return t_res


def _calculate_hat_diagonal(x: np.ndarray) -> np.ndarray:
    """Hat diagonal.

    Generates the hat matrix from 1-D numpy array

    Args:
        x (numpy.ndarray): 1-D array

    Returns:
        numpy.ndarray: hat diagonal

    """
    X_mat = np.vstack((np.ones_like(x), x)).T
    X_hat = X_mat @ np.linalg.inv(X_mat.T @ X_mat) @ X_mat.T

    return X_hat.diagonal()
