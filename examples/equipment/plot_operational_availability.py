"""
========================
Operational Availability
========================
This example demonstrates how to calculate and plot the operational availability
of a system using the `operational_availability` function.

"""

import pandas as pd
import matplotlib.pyplot as plt

from indsl.equipment.operational_availability_ import operational_availability

# Create some dummy data
up_time_data = pd.Series([1] * 365, index=pd.date_range(start="2023-01-01", periods=365, freq="D"))
down_time_data = pd.Series([1] * 365, index=pd.date_range(start="2023-01-01", periods=365, freq="D"))

# Calculate the operational availability

_operational_availability = operational_availability(up_time_data=up_time_data, down_time_data=down_time_data)

# Plot the operational availability

plt.plot(_operational_availability)
plt.xlabel("Date")
plt.ylabel("Operational Availability")
plt.title("Operational Availability")
plt.show()
