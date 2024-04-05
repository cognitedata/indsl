# Copyright 2021 Cognite AS
import random

import numpy as np
import pandas as pd
import pytest

from indsl.exceptions import UserTypeError, UserValueError
from indsl.resample.resample_v1 import AggregateEnum, resample_to_granularity


# Test for empty data
@pytest.mark.core
def test_empty_data():
    with pytest.raises(UserTypeError) as e:
        resample_to_granularity(pd.Series([1], dtype="float64"))
    assert "Expected a time series, got index type int64" in str(e.value)

    with pytest.raises(UserValueError) as e:
        resample_to_granularity(pd.Series([], index=pd.to_datetime([]), dtype="float64"))
    assert "Expected series to be of length > 0, got length 0" in str(e.value)


def test_resample_to_granularity_count():
    data = pd.Series([random.random() * i for i in range(24)], index=pd.date_range("2020-02-03", periods=24, freq="1h"))
    resampled_data = resample_to_granularity(data, granularity="2h", aggregate=AggregateEnum.COUNT)
    assert len(resampled_data) == len(data) // 2
    assert all(resampled_data == 2)


@pytest.mark.parametrize(
    "aggregate, expected_resampled_data",
    [
        (AggregateEnum.INTERPOLATION, np.array(range(24 * 2 - 1)) / 2),
        (AggregateEnum.STEP_INTERPOLATION, np.array(range(24 * 2 - 1)) // 2),
    ],
)
def test_resample_to_granularity_interpolate(aggregate, expected_resampled_data):
    data = pd.Series(list(range(24)), index=pd.date_range("2020-02-03", periods=24, freq="1h"))
    resampled_data = resample_to_granularity(data, granularity="30m", aggregate=aggregate)
    assert len(resampled_data) == len(data) * 2 - 1
    assert all(resampled_data.values == expected_resampled_data)
