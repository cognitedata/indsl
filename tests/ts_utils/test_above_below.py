# Copyright 2021 Cognite AS
import pandas as pd
import pytest

from numpy.random import MT19937, RandomState, SeedSequence

from indsl.ts_utils.ts_utils import above_below


@pytest.mark.core
def test_above_below_functionality_median():
    rs = RandomState(MT19937(SeedSequence(1975)))
    r_series = pd.Series(rs.random(20))
    median = r_series.median()

    result = above_below(r_series, median, median)

    num_elem = len(r_series)
    assert result["above_upper"] == num_elem / 2  # integer division intentional
    assert result["below_lower"] == num_elem / 2  # integer division intentional


@pytest.mark.core
def test_above_below_functionality_quart():
    rs = RandomState(MT19937(SeedSequence(1975)))
    r_series = pd.Series(rs.random(20))
    q25, q75 = r_series.quantile([0.25, 0.75]).tolist()

    result = above_below(r_series, q75, q25)

    num_elem = len(r_series)
    assert result["above_upper"] == num_elem / 4  # integer division intentional
    assert result["below_lower"] == num_elem / 4  # integer division intentional


@pytest.mark.core
def test_above_below_functionality_perc():
    rs = RandomState(MT19937(SeedSequence(1975)))
    r_series = pd.Series(rs.random(20))
    q10, q90 = r_series.quantile([0.10, 0.90]).tolist()

    result = above_below(r_series, q90, q10)

    num_elem = len(r_series)
    assert result["above_upper"] == num_elem / 10  # integer division intentional
    assert result["below_lower"] == num_elem / 10  # integer division intentional
