# Copyright 2021 Cognite AS
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from indsl.data_quality import (
    gaps_identification_iqr,
    gaps_identification_modified_z_scores,
    gaps_identification_z_scores,
)
from indsl.data_quality.gaps_identification import gaps_identification_threshold
from indsl.ts_utils.ts_utils import number_of_events
from indsl.ts_utils.utility_functions import create_series_from_timesteps


second = timedelta(seconds=1)


def test_gaps_identification_z_scores_errors():
    with pytest.raises(TypeError):
        gaps_identification_z_scores([])

    with pytest.raises(TypeError):
        x = pd.Series([1, 2], dtype=np.float64)
        gaps_identification_z_scores(x, "s")

    with pytest.raises(ValueError):
        x = pd.Series([], index=pd.to_datetime([]), dtype=np.float64)
        gaps_identification_z_scores(x)

    with pytest.raises(TypeError):
        x = pd.Series([], dtype=np.float64)
        gaps_identification_z_scores(x)


def test_gaps_identification_modified_z_scores_errors():
    with pytest.raises(TypeError):
        gaps_identification_modified_z_scores([])

    with pytest.raises(TypeError):
        x = pd.Series([1, 2], dtype=np.float64)
        gaps_identification_modified_z_scores(x, "s")

    with pytest.raises(ValueError):
        x = pd.Series([], index=pd.to_datetime([]), dtype=np.float64)
        gaps_identification_modified_z_scores(x)

    with pytest.raises(TypeError):
        x = pd.Series([], dtype=np.float64)
        gaps_identification_modified_z_scores(x)


def test_gaps_identification_iqr_errors():
    with pytest.raises(TypeError):
        gaps_identification_iqr([])

    with pytest.raises(TypeError):
        x = pd.Series([], dtype=np.float64)
        gaps_identification_iqr(x)

    with pytest.raises(ValueError):
        x = pd.Series([], index=pd.to_datetime([]), dtype=np.float64)
        gaps_identification_iqr(x)


def test_gaps_identification_threshold_errors():
    with pytest.raises(TypeError):
        gaps_identification_threshold([])

    with pytest.raises(TypeError):
        x = pd.Series([1, 2], dtype=np.float64)
        gaps_identification_z_scores(x, "s")

    with pytest.raises(TypeError):
        x = pd.Series([], dtype=np.float64)
        gaps_identification_threshold(x)

    with pytest.raises(ValueError):
        x = pd.Series([], index=pd.to_datetime([]), dtype=np.float64)
        gaps_identification_threshold(x)


@pytest.mark.parametrize(
    "gaps_identification_method, kwargs",
    [
        (gaps_identification_z_scores, {}),
        (gaps_identification_modified_z_scores, {}),
        (gaps_identification_iqr, {}),
        (gaps_identification_threshold, {"time_delta": pd.Timedelta("2h")}),
    ],
)
def test_gaps_identification_no_gap(gaps_identification_method, kwargs):
    x = pd.Series(index=pd.date_range("2021-01-01 00:00:00", periods=3, freq="h"), dtype=np.float64)
    out = gaps_identification_method(x, **kwargs)

    assert len(out) == 2
    assert all(out == 0)


@pytest.mark.parametrize(
    "gaps_identification_method",
    [
        gaps_identification_z_scores,
        gaps_identification_modified_z_scores,
        gaps_identification_iqr,
        gaps_identification_threshold,
    ],
)
def test_gaps_identification_no_gap2(gaps_identification_method):
    timesteps = 6 * [second] + [1.0 / 60 / 60 * second] + 6 * [second]
    x = create_series_from_timesteps(timesteps)
    out = gaps_identification_method(x)

    assert len(out) == 2
    assert all(out == 0)


@pytest.mark.parametrize(
    "gaps_identification_method, kwargs, nb_gaps_detected",
    [
        (gaps_identification_z_scores, {}, 0),
        (gaps_identification_modified_z_scores, {}, 0),
        (gaps_identification_iqr, {}, 0),
        (gaps_identification_threshold, {}, 0),
    ],
)
@pytest.mark.parametrize("periods", [1, 2])
def test_gaps_identification_very_little_data(gaps_identification_method, kwargs, nb_gaps_detected, periods):
    x = pd.Series(index=pd.date_range("2021-01-01 00:00:00", periods=periods, freq="s"), dtype=np.float64)
    out = gaps_identification_method(x, **kwargs)

    assert number_of_events(out) == nb_gaps_detected


@pytest.mark.parametrize(
    "gaps_identification_method, kwargs, nb_gaps_detected",
    [
        (gaps_identification_z_scores, {}, 0),
        (gaps_identification_z_scores, {"cutoff": 2}, 2),
        (gaps_identification_modified_z_scores, {}, 0),
        (gaps_identification_iqr, {}, 2),
        (gaps_identification_threshold, {"time_delta": pd.Timedelta("1s")}, 2),
    ],
)
def test_gaps_identification_little_data(gaps_identification_method, kwargs, nb_gaps_detected):
    timesteps = 6 * [second] + [2 * second] + 6 * [second] + [2 * second] + 6 * [second]
    x = create_series_from_timesteps(timesteps)

    out = gaps_identification_method(x, **kwargs)
    assert number_of_events(out) == nb_gaps_detected


@pytest.mark.parametrize(
    "gaps_identification_method, kwargs, nb_gaps_detected",
    [
        (gaps_identification_z_scores, {}, 0),
        (gaps_identification_modified_z_scores, {}, 1),
        (gaps_identification_iqr, {}, 1),
        (gaps_identification_threshold, {"time_delta": pd.Timedelta("1s")}, 4),
    ],
)
def test_gaps_identification_little_data3(gaps_identification_method, kwargs, nb_gaps_detected):
    # Test case from Iglewicz, Boris and David C. Hoaglin (1993), How to Detect and Handle Outliers. American Society for Quality Control, Vol 16.
    timesteps = np.array([1.03, 0.96, 1.11, 0.76, 1.02, 0.98, 0.89, 2.34, 1.01, 1.00]) * second
    x = create_series_from_timesteps(timesteps)

    out = gaps_identification_method(x, **kwargs)
    assert number_of_events(out) == nb_gaps_detected


@pytest.mark.parametrize(
    "gaps_identification_method, kwargs, nb_gaps_detected",
    [
        (gaps_identification_z_scores, {}, 0),
        (gaps_identification_modified_z_scores, {}, 1),
        (gaps_identification_iqr, {}, 1),
        (gaps_identification_threshold, {"time_delta": pd.Timedelta("5s")}, 1),
    ],
)
def test_gaps_identification_little_data2(gaps_identification_method, kwargs, nb_gaps_detected):
    # Test case from Iglewicz, Boris and David C. Hoaglin (1993), How to Detect and Handle Outliers. American Society for Quality Control, Vol 16.
    timesteps = np.array([2.1, 2.6, 2.4, 2.5, 2.3, 2.1, 2.3, 2.6, 8.2, 8.3]) * second
    x = create_series_from_timesteps(timesteps)

    out = gaps_identification_method(x, **kwargs)
    assert number_of_events(out) == nb_gaps_detected


@pytest.mark.parametrize(
    "gaps_identification_method, kwargs, nb_gaps_detected",
    [
        (gaps_identification_z_scores, {}, 1),
        (gaps_identification_modified_z_scores, {}, 0),
        (gaps_identification_iqr, {}, 1),
        (gaps_identification_threshold, {"time_delta": pd.Timedelta("1s")}, 1),
    ],
)
def test_gaps_identification_no_gap_much_data(gaps_identification_method, kwargs, nb_gaps_detected):
    # The challenge of this test is that it uses a time-series with equidistant
    # timesteps, where one of the timesteps has an eps-variation.

    timesteps = 1000 * [second] + [1.0001 * second] + 1000 * [second]
    x = create_series_from_timesteps(timesteps)
    out = gaps_identification_method(x, **kwargs)
    assert number_of_events(out) == nb_gaps_detected


@pytest.mark.parametrize(
    "gaps_identification_method, kwargs, nb_gaps_detected",
    [
        (gaps_identification_z_scores, {}, 0),
        (gaps_identification_z_scores, {"cutoff": 17}, 0),
        (gaps_identification_modified_z_scores, {}, 0),
        (gaps_identification_iqr, {}, 0),
        (gaps_identification_threshold, {"time_delta": pd.Timedelta("1s")}, 1),
    ],
)
def test_gaps_identification_increasing_timesteps1(gaps_identification_method, kwargs, nb_gaps_detected):
    # This tests consists of two sub time series with increasing time steps
    timesteps = 100 * [second] + 100 * [2 * second]
    x = create_series_from_timesteps(timesteps)

    out = gaps_identification_method(x, **kwargs)
    assert number_of_events(out) == nb_gaps_detected


@pytest.mark.parametrize(
    "gaps_identification_method, kwargs, nb_gaps_detected",
    [
        (gaps_identification_z_scores, {}, 0),
        (gaps_identification_z_scores, {"cutoff": 17}, 0),
        (gaps_identification_modified_z_scores, {}, 0),
        (gaps_identification_iqr, {}, 0),
        (gaps_identification_threshold, {"time_delta": pd.Timedelta("1s")}, 1),
    ],
)
def test_gaps_identification_increasing_timesteps2(gaps_identification_method, kwargs, nb_gaps_detected):
    # Test with four sub time series with increasing time steps
    timesteps = 100 * [second] + 100 * [2 * second] + 100 * [3 * second] + 100 * [4 * second]
    x = create_series_from_timesteps(timesteps)

    out = gaps_identification_method(x, **kwargs)
    assert number_of_events(out) == nb_gaps_detected
