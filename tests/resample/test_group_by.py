# Copyright 2023 Cognite AS

import numpy as np
import pandas as pd
import pytest

from pandas.testing import assert_series_equal

from indsl.resample import group_by_region


# Some examples of input types
time_index = pd.date_range(start="2022-07-01 00:00:00", end="2022-07-01 07:00:00", periods=8)
A = pd.Series(index=time_index, data=[0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7])
B = pd.Series(index=time_index, data=[1, 1, 0, 0, 1, 1, 0, 1])


# aggregate and results for int_to_keep = 1
results_1 = [
    ("Mean", [0.55, 4.95, 7.7]),
    ("Median", [0.55, 4.95, 7.7]),
    ("Standard deviation", [0.77781746, 0.77781746, np.nan]),
    ("Count", [2, 2, 1]),
    ("Min", [0.0, 4.4, 7.7]),
    ("Max", [1.1, 5.5, 7.7]),
]
results_1_entire_region = [
    ("Mean", [0.55, 0.55, 4.95, 4.95, 7.7]),
    ("Median", [0.55, 0.55, 4.95, 4.95, 7.7]),
    ("Standard deviation", [0.77781746, 0.77781746, 0.77781746, 0.77781746, np.nan]),
    ("Count", [2, 2, 2, 2, 1]),
    ("Min", [0.0, 0.0, 4.4, 4.4, 7.7]),
    ("Max", [1.1, 1.1, 5.5, 5.5, 7.7]),
]


@pytest.mark.parametrize(
    "region, expected_index, aggregate_and_result",
    [
        ("Region center", ["2022-07-01 00:30:00", "2022-07-01 04:30:00", "2022-07-01 07:00:00"], results_1),
        ("Region start", ["2022-07-01 00:00:00", "2022-07-01 04:00:00", "2022-07-01 07:00:00"], results_1),
        ("Region end", ["2022-07-01 01:00:00", "2022-07-01 05:00:00", "2022-07-01 07:00:00"], results_1),
        (
            "Entire region",
            [
                "2022-07-01 00:00:00",
                "2022-07-01 01:00:00",
                "2022-07-01 04:00:00",
                "2022-07-01 05:00:00",
                "2022-07-01 07:00:00",
            ],
            results_1_entire_region,
        ),
    ],
)
def test_group_by_1(region, expected_index, aggregate_and_result):
    int_to_keep = 1

    for item in aggregate_and_result:
        aggregate = item[0]
        expected_result = item[1]

        res = group_by_region(data=A, filter_by=B, int_to_keep=int_to_keep, aggregate=aggregate, timestamp=region)
        expected = pd.Series(index=pd.to_datetime(expected_index), data=expected_result)
        assert_series_equal(res, expected)


# aggregate and results for int_to_keep = 0
results_0 = [
    ("Mean", [2.75, 6.60]),
    ("Median", [2.75, 6.60]),
    ("Standard deviation", [0.77781746, np.nan]),
    ("Count", [2, 1]),
    ("Min", [2.2, 6.6]),
    ("Max", [3.3, 6.6]),
]
results_0_entire_region = [
    ("Mean", [2.75, 2.75, 6.60]),
    ("Median", [2.75, 2.75, 6.60]),
    ("Standard deviation", [0.77781746, 0.77781746, np.nan]),
    ("Count", [2, 2, 1]),
    ("Min", [2.2, 2.2, 6.6]),
    ("Max", [3.3, 3.3, 6.6]),
]


@pytest.mark.parametrize(
    "region, expected_index, aggregate_and_result",
    [
        ("Region center", ["2022-07-01 02:30:00", "2022-07-01 06:00:00"], results_0),
        ("Region start", ["2022-07-01 02:00:00", "2022-07-01 06:00:00"], results_0),
        ("Region end", ["2022-07-01 03:00:00", "2022-07-01 06:00:00"], results_0),
        (
            "Entire region",
            ["2022-07-01 02:00:00", "2022-07-01 03:00:00", "2022-07-01 06:00:00"],
            results_0_entire_region,
        ),
    ],
)
def test_group_by_0(region, expected_index, aggregate_and_result):
    int_to_keep = 0

    for item in aggregate_and_result:
        aggregate = item[0]
        expected_result = item[1]

        res = group_by_region(data=A, filter_by=B, int_to_keep=int_to_keep, aggregate=aggregate, timestamp=region)
        expected = pd.Series(index=pd.to_datetime(expected_index), data=expected_result)
        assert_series_equal(res, expected)
