# Copyright 2021 Cognite AS

import pandas as pd
import pytest

from indsl.data_quality.completeness import completeness_score, find_period
from indsl.exceptions import UserValueError
from indsl.signals.generator import insert_data_gaps, line


@pytest.mark.core
def test_completeness_score():
    # Create data with gaps
    start = pd.Timestamp("1975/05/09")
    end = pd.Timestamp("1975/05/20")
    data = line(start_date=start, end_date=end, slope=0, intercept=0, sample_freq=pd.Timedelta("1 h"))
    data_gap = insert_data_gaps(data=data, fraction=0.30, method="Random", num_gaps=10)

    res = completeness_score(data_gap)
    exp_res = "Medium: completeness score is 0.70"
    assert res == exp_res

    res = completeness_score(x=data_gap, cutoff_good=0.2, cutoff_med=0.1)
    exp_res = "Good: completeness score is 0.70"
    assert res == exp_res


@pytest.mark.core
def test_completeness_score_raise_error():
    # Create data with gaps
    start = pd.Timestamp("1975/05/09")
    end = pd.Timestamp("1975/05/20")
    data = line(start_date=start, end_date=end, slope=0, intercept=0, sample_freq=pd.Timedelta("1 h"))
    data_sparse = data.iloc[:5]

    # Check if exception is raised
    with pytest.raises(ValueError, match="Expected series with length >= 10, got length 5"):
        completeness_score(data_sparse)

    # check for index type error
    data_2 = pd.Series(range(1000), index=range(1000))
    with pytest.raises(TypeError, match="Expected a time series, got index type int64"):
        completeness_score(data_2)

    with pytest.raises(UserValueError) as excinfo:
        completeness_score(x=data_2, cutoff_good=0.5, cutoff_med=0.5)
    assert "cutoff_good should be higher than cutoff_med." in str(excinfo.value)


@pytest.mark.core
def test_completeness_period_type_median():
    # create set with bimodal distribution of time periods
    start_end = dict(start="2020", end="2021")
    dense_ts = pd.date_range(start="2021-01-01 00:00:01", end="2021-01-02", periods=50000)
    s2 = pd.Series(range(100002), index=pd.date_range(**start_end, periods=50002).union(dense_ts))

    # check for error when calculated completeness score exceeds 1
    with pytest.raises(ValueError, match="Completeness score 1.9945350831704496. Change period calculation method."):
        _ = completeness_score(s2)


@pytest.mark.core
def test_completeness_period_type_min():
    # create set with bimodal distribution of time periods
    start_end = dict(start="2020", end="2021")
    dense_ts = pd.date_range(start="2021-01-01 00:00:01", end="2021-01-02", periods=50000)
    s2 = pd.Series(range(100002), index=pd.date_range(**start_end, periods=50002).union(dense_ts))

    # check completeness score when the period calculation method is from 'min'
    res_2 = completeness_score(s2, method_period="min")
    exp_res_2 = "Poor: completeness score is 0.00"
    assert res_2 == exp_res_2


@pytest.mark.core
def test_find_period():
    # Create data with gaps
    start = pd.Timestamp("1975/05/09")
    end = pd.Timestamp("1975/05/20")
    data = line(start_date=start, end_date=end, slope=0, intercept=0, sample_freq=pd.Timedelta("1 h"))
    data_gap = insert_data_gaps(data=data, fraction=0.30, method="Random", num_gaps=10)

    res = find_period(data_gap)

    exp_res = 3600000000000
    assert res == exp_res

    with pytest.raises(UserValueError) as excinfo:
        find_period(x=data, method_period="Unvalid method")
    assert "Period calculation method can only be strings: median or min, not Unvalid method" in str(excinfo.value)
