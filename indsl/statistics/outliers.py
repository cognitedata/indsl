# Copyright 2023 Cognite AS

from typing import Optional

import numpy as np
import pandas as pd

from indsl import versioning
from indsl.exceptions import CSAPS_REQUIRED, KNEED_REQUIRED, SCIKIT_LEARN_REQUIRED, UserValueError
from indsl.type_check import check_types
from indsl.validations import validate_series_has_time_index, validate_series_is_not_empty

from . import outliers_v1  # noqa


@versioning.register(
    version="2.0",
    changelog="handles steady_state inputs without crashing, improves performance and implements new input types",
)
@check_types
def detect_outliers(
    data: pd.Series,
    reg_smooth: float = 0.9,
    min_samples: int = 4,
    eps: Optional[float] = None,
    time_window: pd.Timedelta = pd.Timedelta("60min"),
    del_zero_val: bool = False,
) -> pd.Series:
    """Outlier detection.

    Identifies outliers combining two methods, dbscan and csap.

    - **dbscan**: Density-based clustering algorithm used to identify clusters of varying shape and size within a data
      set. Does not require a pre-determined set number of clusters. Able to identify outliers as noise, instead of
      classifying them into a cluster. Flexible when it comes to the size and shape of clusters, which makes it more
      useful for noise, real life data.

    - **csaps regression**: Cubic smoothing spline algorithm. Residuals from the regression are computed. Data points with
      high residuals (3 Standard Deviations from the Mean) are considered as outliers.

    Args:
        data: Time series.
            The data has to be non-uniform.
        reg_smooth: Smoothing factor.
            The smoothing parameter that determines the weighted sum of terms in the regression, and it is limited by
            the range [0,1]. Defaults to 0.9. Ref: https://csaps.readthedocs.io/en/latest/formulation.html#definition
        min_samples: Minimum samples.
            Minimum number of data points required to form a distinct cluster. Defaults to 4.
            Defines the minimum number of data points required to form a distinct cluster. Rules of thumb for selecting
            the minimum samples value:

            * The larger the data set, the larger the value of MinPts should be.
            * If the data set is noisier, choose a larger value of MinPts Generally, MinPts should be greater than or
              equal to the dimensionality of the data set. For 2-dimensional data, use DBSCAN’s default value of 4
              (Ester et al., 1996).
            * If your data has more than 2 dimensions, choose MinPts = 2*dim, where dim= the dimensions of your data
              set (Sander et al., 1998).
        eps: Distance threshold.
            Defines the maximum distance between two samples for one to be considered as in the
            neighborhood of the other (i.e. belonging to the same cluster). This is the most important DBSCAN parameter
            to choose appropriately for your dataset and distance function. If no value is given, it is set automatically
            using Nearest Neighbors algorithm to calculate the average distance between each point and its k
            nearest neighbors, where k = min_samples (minimum samples). In ascending order on a k-distance graph, the
            optimal value for the threshold is at the point of maximum curvature (i.e. after plotting the average
            k-distances in where the graph has the greatest slope). This is not a maximum bound on the distances of
            points within a cluster.
            Defaults to None, eps value has to be > 0.0.
        time_window: Time window.
            Length of the time period to compute the rolling mean. The rolling mean and the data point value are the two features considered when calculating the distance to the furthest neighbour.
            This distance allows us to find the right epsilon when training dbscan. Defaults to '60min'.
            Accepted format: '3w', '10d', '5h', '30min', '10s'.
            If a number without unit (such as '60')is given, it will be considered as the number of minutes.
        del_zero_val: Remove zeros.
            Removes data points containing a value of 0. Defaults to False.

    Returns:
        pandas.Series: Time series.
            Binary time series indicating outliers: Outlier= 1, Not an outlier = 0
    """
    outlier_indices = _get_outlier_indices(data, min_samples, eps, time_window, del_zero_val, reg_smooth)

    df_with_outliers = pd.DataFrame(data=1, index=outlier_indices, columns=["isOutlier"])

    # Convert original time series to DataFrame
    df = data.to_frame()

    # Left join original time series and timeseries with outliers
    outlier_indicator_df = df.join(df_with_outliers)

    # Fill NaN with zeroes
    outlier_indicator_df["isOutlier"] = outlier_indicator_df["isOutlier"].fillna(0).astype(int)

    # Convert DataFrame to Series
    outlier_indicator_ts = outlier_indicator_df["isOutlier"].squeeze()

    return outlier_indicator_ts


@versioning.register(version="2.0", changelog="handle steady_state inputs, better performance; new input types")
@check_types
def remove_outliers(
    data: pd.Series,
    reg_smooth: float = 0.9,
    min_samples: int = 4,
    eps: Optional[float] = None,
    time_window: pd.Timedelta = pd.Timedelta("60min"),
    del_zero_val: bool = False,
) -> pd.Series:
    """Outlier removal.

    Identifies and removes outliers combining two methods, dbscan and csap.

    - **dbscan**: Density-based clustering algorithm used to identify clusters of varying shape and size within a data
      set. Does not require a pre-determined set number of clusters. Able to identify outliers as noise, instead of
      classifying them into a cluster. Flexible when it comes to the size and shape of clusters, which makes it more
      useful for noise, real life data.

    - **csaps regression**: Cubic smoothing spline algorithm. Residuals from the regression are computed. Data points with
      high residuals (3 Standard Deviations from the Mean) are considered as outliers.

    Args:
        data: Time series.
        reg_smooth: Smoothing factor.
            The smoothing parameter that determines the weighted sum of terms in the regression, and it is limited by
            the range [0,1]. Defaults to 0.9. Ref: https://csaps.readthedocs.io/en/latest/formulation.html#definition
        min_samples: Minimum samples.
            Minimum number of data points required to form a distinct cluster. Defaults to 4.
            Defines the minimum number of data points required to form a distinct cluster. Rules of thumb for selecting
            the minimum samples value:

            * The larger the data set, the larger the value of MinPts should be.
            * If the data set is noisier, choose a larger value of MinPts Generally, MinPts should be greater than or
              equal to the dimensionality of the data set. For 2-dimensional data, use DBSCAN’s default value of 4
              (Ester et al., 1996).
            * If your data has more than 2 dimensions, choose MinPts = 2*dim, where dim= the dimensions of your data
              set (Sander et al., 1998).
        eps: Distance threshold.
            Defines the maximum distance between two samples for one to be considered as in the
            neighborhood of the other (i.e. belonging to the same cluster). This is the most important DBSCAN parameter
            to choose appropriately for your dataset and distance function. If no value is given, it is set automatically
            using Nearest Neighbors algorithm to calculate the average distance between each point and its k
            nearest neighbors, where k = min_samples (minimum samples). In ascending order on a k-distance graph, the
            optimal value for the threshold is at the point of maximum curvature (i.e. after plotting the average
            k-distances in where the graph has the greatest slope). This is not a maximum bound on the distances of
            points within a cluster.
            Defaults to None, eps value has to be > 0.0.
        time_window: Time window.
            Length of the time period to compute the rolling mean. The rolling mean and the data point value are the two features considered when calculating the distance to the furthest neighbour.
            This distance allows us to find the right epsilon when training dbscan. Defaults to '60min'.
            Accepted format: '3w', '10d', '5h', '30min', '10s'.
            If a number without unit (such as '60')is given, it will be considered as the number of minutes.
        del_zero_val: Remove zeros.
            Removes data points containing a value of 0. Defaults to False.

    Returns:
        pandas.Series: Time series without outliers.
    """
    outlier_indices = _get_outlier_indices(data, min_samples, eps, time_window, del_zero_val, reg_smooth)

    # Drop rows with outlier indices
    ts_without_indices = data.drop(outlier_indices)

    return ts_without_indices


@check_types
def _get_outlier_indices(
    data: pd.Series,
    min_samples,
    eps: Optional[float],
    time_window: pd.Timedelta,
    del_zero_val: bool,
    reg_smooth: float,
) -> pd.DatetimeIndex:
    """Get outlier indices.

    Identifies the indices of outliers in the given time series combining two methods, dbscan and csap.

    Args:
        data: Time series.
        reg_smooth: Smoothing factor.
        min_samples: Minimum samples.
        eps: Distance threshold.
        time_window: Time window.
            Accepted format: '3w', '10d', '5h', '30min', '10s'.
            If a number without unit (such as '60')is given, it will be considered as the number of minutes.
        del_zero_val: Remove zeros.
            Defaults to False.

    Returns:
        pandas.DatetimeIndex: An array of indices of outliers in the given time series.
    """
    try:
        from csaps import csaps  # Lazy import to avoid csaps being a core dependency
    except ImportError:
        raise ImportError(CSAPS_REQUIRED)

    try:
        from kneed import KneeLocator  # Lazy import to avoid kneed being a core dependency
    except ImportError:
        raise ImportError(KNEED_REQUIRED)

    try:
        from sklearn.cluster import DBSCAN
        from sklearn.neighbors import NearestNeighbors
    except ImportError:
        raise ImportError(SCIKIT_LEARN_REQUIRED)

    # Handle special cases
    if len(data) <= 4:
        return pd.DatetimeIndex([])
    if (data.values == data.values[0]).all():  # if data is uniform, return 0 outliers
        return pd.DatetimeIndex([])

    # Validations
    validate_series_has_time_index(data)

    if eps is not None and eps <= 0.0:
        raise UserValueError("eps should be > 0.0.")

    df = data.to_frame()
    df = df.rename(columns={df.columns[0]: "val"})

    df["rolling_mean"] = df["val"].rolling(time_window, min_periods=1).mean()  # calculate features

    if del_zero_val:  # delete 0 values if user requests it
        df = df[df["val"] != 0.0]

    df_dbscan = df[["val"]].copy()
    validate_series_is_not_empty(df_dbscan)

    if eps is None:
        # calculate distance to the further neighbor in order to find best epsilon (radius) parameter for dbscan
        n_neighbors = min(len(df) - 3, 6)
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(df)
        distances, _ = nbrs.kneighbors(df)
        df["knn_dist"] = distances[:, -1]

        dist = np.sort(distances[:, -1])
        i = np.arange(len(dist))
        knee = KneeLocator(i, dist, S=1, curve="convex", direction="increasing", interp_method="polynomial")
        eps = dist[knee.knee] if dist[knee.knee] > 0 else 0.5  # if calculated eps is 0, use 0.5 as default value

    # train dbscan
    dbscan_model = DBSCAN(eps=eps, min_samples=min_samples).fit(df_dbscan)
    labels = dbscan_model.labels_

    outlier_positions = np.where(labels == -1)[0]

    # Get indices of outlier data points calculated using dbscan
    outlier_indices_dbscan = df.iloc[outlier_positions].index

    # Delete outliers detected by dbscan
    df_without_dbscan_outliers = df.drop(outlier_indices_dbscan)

    # Apply regression on the remaining data points
    df_reg = df.loc[df_without_dbscan_outliers.index, :]
    date_int = df_reg.index.to_series().astype(np.int64)
    date_int_stand = (date_int - date_int.mean()) / date_int.std()
    csaps_data = csaps(date_int_stand, df_reg["val"], date_int_stand, smooth=reg_smooth)

    # Delete points with high residuals
    res = pd.DataFrame(abs(df_reg["val"] - csaps_data))
    res["val_stand"] = (res["val"] - res["val"].mean()) / res["val"].std()
    res_stand_outliers = res[res["val_stand"] >= 3]

    outlier_indices_res_std = res_stand_outliers.index

    all_outliers = outlier_indices_dbscan.append(outlier_indices_res_std)

    return all_outliers


def outlier_percent(
    data: pd.Series,
    min_samples: int = 4,
    eps: Optional[float] = None,
    time_window: pd.Timedelta = pd.Timedelta("60min"),
    del_zero_val: bool = False,
    reg_smooth: float = 0.9,
) -> Optional[float]:
    """Calculates the percentage of outliers in the given time series.

    Args:
        data: Time series.
        min_samples: Minimum samples.
        eps: Distance threshold.
        time_window: Time window.
            Defaults to '60min'.
            Accepted format: '3w', '10d', '5h', '30min', '10s'.
            If a number without unit (such as '60')is given, it will be considered as the number of minutes.
        del_zero_val: Remove zeros.
            Defaults to False.
        reg_smooth: Smoothing factor.

    Returns:
        float: Percentage of outliers.
    """
    try:
        outlier_indices = _get_outlier_indices(data, min_samples, eps, time_window, del_zero_val, reg_smooth)
        return len(outlier_indices) / len(data) * 100
    except ZeroDivisionError:
        return None


OUTLIERS_REMOVE = remove_outliers
