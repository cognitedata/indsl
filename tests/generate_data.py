# Copyright 2021 Cognite AS
import random

from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd


def create_uniform_data(
    values: npt.ArrayLike,
    start_date: datetime = datetime(2020, 7, 23, 15, 27, 0),
    end_date: Optional[datetime] = None,
    frequency: Optional[str] = "1s",
) -> pd.Series:
    values = np.hstack(values)
    return pd.DataFrame(
        {"value": values}, index=pd.date_range(start=start_date, end=end_date, freq=frequency, periods=len(values))
    ).iloc[:, 0]


def create_non_uniform_data(
    values: tuple, start_date: datetime = datetime(2020, 7, 23, 15, 27, 0), time_delta: timedelta = timedelta(seconds=1)
) -> pd.Series:
    values = np.hstack(values)
    indices = [start_date]
    for i in range(1, len(values)):
        indices.append(indices[i - 1] + time_delta * random.randint(1, 1000))
    return pd.DataFrame({"value": values}, index=indices).iloc[:, 0]


def set_na_random_data(data: pd.Series, percentage: float = 0.5) -> pd.Series:
    new_data = data.copy()
    num_points_to_change = min(int(len(new_data) * percentage), len(new_data) - 2)
    indices = np.random.choice(new_data.index[1:-1], size=num_points_to_change, replace=False)
    new_data.loc[indices] = np.nan
    return new_data
