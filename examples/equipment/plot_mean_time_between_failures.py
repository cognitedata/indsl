"""
==========================
Mean time between failures
==========================
This example demonstrates how to calculate and plot the mean time between failures (MTBF)
of a system using the `MTBF` function.
"""

import pandas as pd
import matplotlib.pyplot as plt

from indsl.equipment.mean_time_between_failures_ import mean_time_between_failures

mean_time_to_failure = pd.Series([1] * 365, index=pd.date_range(start="2023-01-01", periods=365, freq="D"))
mean_time_to_resolution = pd.Series([1] * 365, index=pd.date_range(start="2023-01-01", periods=365, freq="D"))


mtbf = mean_time_between_failures(
    mean_time_to_failure=mean_time_to_failure, mean_time_to_resolution=mean_time_to_resolution
)

plt.plot(mtbf)
plt.xlabel("Date")
plt.ylabel("Mean Time Between Failures [Time]")
plt.title("Mean Time Between Failures")
plt.show()
