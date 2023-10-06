# Copyright 2021 Cognite AS
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from pandas.testing import assert_series_equal

from indsl.data_quality.low_density_identification import (
    low_density_identification_iqr,
    low_density_identification_modified_z_scores,
    low_density_identification_threshold,
    low_density_identification_z_scores,
)
from indsl.exceptions import UserValueError
from indsl.signals.generator import insert_data_gaps
from indsl.ts_utils.ts_utils import number_of_events
from indsl.ts_utils.utility_functions import create_series_from_timesteps


@pytest.mark.core
@pytest.mark.parametrize(
    "density_identification_method, kwargs",
    [
        (low_density_identification_z_scores, {"cutoff": "s"}),
        (low_density_identification_modified_z_scores, {"cutoff": "s"}),
        (low_density_identification_iqr, {}),
        (low_density_identification_threshold, {"cutoff": "s"}),
    ],
)
def test_density_check_errors(density_identification_method, kwargs):
    with pytest.raises(ValueError):  # validate is not empty
        x = pd.Series([1, 2], index=pd.to_datetime([]))
        density_identification_method(x)

    with pytest.raises(TypeError):  # validate has time index
        x = pd.Series([1, 2])
        density_identification_method(x)

    with pytest.raises(TypeError):  # validate typing
        x = pd.Series([1, 2], index=pd.date_range("2021-01-01", periods=2, freq="h"))
        density_identification_method(data=x, time_window="s", **kwargs)


@pytest.mark.core
@pytest.mark.parametrize(
    "density_identification_method, kwargs, num_events_expected",
    [
        (low_density_identification_z_scores, {"time_window": pd.Timedelta("30min"), "cutoff": -3.0}, 1),
        (low_density_identification_iqr, {"time_window": pd.Timedelta("10min")}, 1),
        (low_density_identification_threshold, {"time_window": pd.Timedelta("5min"), "cutoff": 2}, 2),
    ],
)
def test_low_density_identification(density_identification_method, kwargs, num_events_expected):
    step = timedelta(minutes=5)
    timesteps = 1000 * [step] + [step / 2] + [step / 2] + [step / 2] + 1000 * [step]
    x = create_series_from_timesteps(timesteps)

    out = density_identification_method(x, **kwargs)

    assert number_of_events(out) == num_events_expected


@pytest.mark.core
def test_low_density_identification_modified_z_scores():
    step = timedelta(minutes=5)
    timesteps = 1000 * [step] + [step / 2] + [step / 2] + [step / 2] + 1000 * [step]
    x = create_series_from_timesteps(timesteps)

    x = insert_data_gaps(data=x, fraction=0.35, method="Random")
    out = low_density_identification_modified_z_scores(x, pd.Timedelta("2h"))

    assert number_of_events(out) == 7


@pytest.mark.core
def test_low_density_identification_z_scores_short_list():
    short_timeseries = pd.Series([1], index=pd.date_range("2020-02-01", periods=1))
    result = low_density_identification_z_scores(data=short_timeseries)
    exp_result = short_timeseries.replace(1, 0)
    assert_series_equal(result, exp_result)


@pytest.mark.core
def test_test_low_density_identification_z_normality_assumption():
    np.random.seed(21)
    # uniform time series
    data = np.zeros(10)
    time_index = pd.date_range("2020-01-01", periods=10, freq="s")
    uniform_distribution_series = pd.Series(data=data, index=time_index)
    with pytest.raises(UserValueError) as excinfo:
        low_density_identification_z_scores(
            data=uniform_distribution_series, time_window=pd.Timedelta("1s"), test_normality_assumption=True
        )
    assert "This time series is uniform and not normally distributed" in str(excinfo.value)

    # Normally distributed time series
    length = 1000
    data = np.zeros(length)
    time_index = pd.date_range("2010-01-01", periods=length, freq="s")
    normal_series = pd.Series(data=data, index=time_index)

    # creating gaps
    while True:
        try:
            normal_series = insert_data_gaps(
                data=normal_series, fraction=max(0, np.random.normal(loc=0.5, scale=0.2, size=1)[0]), method="Single"
            )
        except UserValueError:
            break

    # run without error -> test_normality_assumption works
    low_density_identification_z_scores(
        data=normal_series, test_normality_assumption=True, time_window=pd.Timedelta("5h")
    )

    # Not uniform, not normally distributed
    not_uniform_or_normal_series = pd.Series(data=data, index=time_index)

    # creating gaps
    for x in range(1, 5):
        not_uniform_or_normal_series = insert_data_gaps(
            data=not_uniform_or_normal_series, fraction=0.05, method="Single"
        )

    with pytest.raises(UserValueError) as excinfo:
        low_density_identification_z_scores(
            data=not_uniform_or_normal_series, test_normality_assumption=True, time_window=pd.Timedelta("5h")
        )
    expected = "This time series is not normally distributed"
    assert expected in str(excinfo.value)


@pytest.mark.core
def test_low_density_identification_iqr_with_short_list_as_input():
    short_timeseries = pd.Series([1], index=pd.date_range("2020-02-01", periods=1))
    result = low_density_identification_iqr(data=short_timeseries)
    exp_result = short_timeseries.replace(1, 0)
    assert_series_equal(result, exp_result)
