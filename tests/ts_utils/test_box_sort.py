# Copyright 2021 Cognite AS
import pandas as pd
import pytest

from numpy.random import MT19937, RandomState, SeedSequence

from indsl.ts_utils.ts_utils import num_vals_in_boxes


def alternative_num_in_box(series: pd.Series, num_boxes: int):
    max_val = series.max()
    min_val = series.min()

    delta = (max_val - min_val) / float(num_boxes)
    low_limit = min_val
    eps = 1.0e-6

    result = []
    for _ in range(num_boxes):
        num_in_box = 0
        for val in series:
            if val > low_limit * (1 - eps) and val <= low_limit + delta * (1 + eps):
                num_in_box += 1
        result.append(num_in_box)
        low_limit += delta

    return result


@pytest.mark.core
def test_box_sort_functionality_with_2_boxes():
    rs = RandomState(MT19937(SeedSequence(1975)))
    r_series = pd.Series(rs.random(200))
    num_box = 2

    ref_res = alternative_num_in_box(r_series, num_box)
    result = num_vals_in_boxes(r_series, num_box)

    assert result["data"][0] == ref_res[0]
    assert result["data"][1] == ref_res[1]


@pytest.mark.core
def test_box_sort_functionality_with_10_boxes():
    rs = RandomState(MT19937(SeedSequence(1975)))
    r_series = pd.Series(rs.random(200))
    num_box = 10

    ref_res = alternative_num_in_box(r_series, num_box)
    result = num_vals_in_boxes(r_series, num_box)

    for idx in range(num_box):
        assert result["data"][idx] == ref_res[idx]


@pytest.mark.core
def test_box_sort_functionality_with_100_boxes():
    rs = RandomState(MT19937(SeedSequence(1975)))
    r_series = pd.Series(rs.random(200))
    num_box = 100

    ref_res = alternative_num_in_box(r_series, num_box)
    result = num_vals_in_boxes(r_series, num_box)

    for idx in range(num_box):
        assert result["data"][idx] == ref_res[idx]
