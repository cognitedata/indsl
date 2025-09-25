# Copyright 2025 Cognite AS
"""
=================
Rolling variance
=================

This example computes the rolling variance of a single time series
using a time-based window.

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.dates import DateFormatter

from indsl.ts_utils.rolling_stats import rolling_variance


# Generate a synthetic time series
rng = np.random.default_rng(12345)
num_datapoints = 200
index = pd.date_range(start="1970-01-01", periods=num_datapoints, freq="1min")

# Random walk with a small seasonal component
y = np.cumsum(rng.standard_normal(num_datapoints)) + 0.1 * np.sin(np.linspace(0, 10 * np.pi, num_datapoints))
data = pd.Series(y, index=index)

# Compute rolling variance over a 5-minute window
window = pd.Timedelta(minutes=5)
var_series = rolling_variance(data, time_window=window, min_periods=1)

# Plot both on the same axes (overlaid)
fig, ax = plt.subplots(figsize=[15, 6])
ax.plot(data, label="Time series")
ax.plot(var_series, color="tab:orange", label="Rolling variance (5 min window)")
ax.set_title("Time series with rolling variance")
ax.legend(loc="best")

# Formatting
date_fmt = DateFormatter("%b %d, %H:%M")
ax.xaxis.set_major_formatter(date_fmt)
_ = plt.setp(ax.get_xticklabels(), rotation=45)

plt.tight_layout()
plt.show()


