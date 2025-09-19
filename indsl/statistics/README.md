### 0. [Confidence bands](confidence.md)


### 1. [Outlier detection and removal](outliers.md)
This function combines two methods to 1) detect, and 2) remove outliers.

1. Density-based clustering algorithm (dbscan) used to identify clusters of varying shape and size within a data set and delete those datapoints that are not assign to a cluster.

2. Cubic smooting spline algorithm to detect those datapoints with high residuals that were previously assigned to a cluster after using dbscan.


### 2. Rolling standard deviation

Compute the rolling standard deviation of a time series over a time-based window.

Implementation details:

- Uses pandas' time-based rolling windows via [`Series.rolling`](https://pandas.pydata.org/docs/reference/api/pandas.Series.rolling.html) with a `Timedelta` window and computes the sample standard deviation with [`Rolling.std`](https://pandas.pydata.org/docs/reference/api/pandas.core.window.rolling.Rolling.std.html) (ddof=1 by default).