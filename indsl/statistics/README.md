### 0. [Confidence bands](confidence.md)


### 1. [Outlier detection and removal](outliers.md)
This function combines two methods to 1) detect, and 2) remove outliers.

1. Density-based clustering algorithm (dbscan) used to identify clusters of varying shape and size within a data set and delete those datapoints that are not assign to a cluster.

2. Cubic smooting spline algorithm to detect those datapoints with high residuals that were previously assigned to a cluster after using dbscan.
