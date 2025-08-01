# Copyright 2021 Cognite AS
import random

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from pandas.testing import assert_index_equal

from indsl.resample.reindex_v1 import Kind, Method, reindex

from ..generate_data import create_uniform_data


# Test for empty data
@pytest.mark.core
def test_empty_data():
    data1 = pd.Series(dtype="float64")
    data2 = pd.Series(dtype="float64")

    out1, out2 = reindex(data1, data2)
    assert id(out1) == id(data1)
    assert id(out2) == id(data2)


# Test already aligned
@pytest.mark.core
def test_aligned_data():
    data1 = pd.Series([1], dtype="float64", index=[1])
    data2 = pd.Series([2], dtype="float64", index=[1])

    out1, out2 = reindex(data1, data2)
    assert id(out1) == id(data1)
    assert id(out2) == id(data2)


# Test for empty data with non-overlappying time-series
def test_empty_data_with_bounds():
    data1 = create_uniform_data(
        np.ones(5),
        start_date=datetime(2020, 7, 23, 16, 27, 0),
        end_date=datetime(2020, 7, 23, 17, 27, 0),
        frequency=None,
    )
    data2 = create_uniform_data(
        np.ones(5),
        start_date=datetime(2020, 7, 24, 16, 27, 0),
        end_date=datetime(2020, 7, 24, 17, 27, 0),
        frequency=None,
    )

    with pytest.raises(TypeError):
        reindex(data1, data2, bounded=True)


# Test with a single data point
def test_single():
    data1 = create_uniform_data(
        np.ones(5),
        start_date=datetime(2020, 7, 23, 16, 27, 0),
        end_date=datetime(2020, 7, 23, 17, 27, 0),
        frequency=None,
    )
    data2 = create_uniform_data(
        np.ones(1),
        start_date=datetime(2020, 7, 23, 16, 57, 0),
        end_date=datetime(2020, 7, 23, 17, 27, 0),
        frequency=None,
    )

    with pytest.raises(ValueError):
        reindex(data1, data2, bounded=True)


# Test for all NaN data
@pytest.mark.core
def test_all_nan_data():
    data1 = create_uniform_data(np.ones(10) * np.nan)
    data2 = create_uniform_data(np.ones(10))

    with pytest.raises(ValueError):
        reindex(data1, data2)


# test correct output size
@pytest.mark.parametrize("method", list(Method))
@pytest.mark.parametrize("kind", list(Kind))
@pytest.mark.parametrize("len1, len2, len_out", [(10, 10, 10), (3, 5, 5), (5, 8, 11)])
def test_reindex_length(len1, len2, method, kind, len_out):
    if len1 == 3 and method == Method.CUBIC:
        # Too few data points for cubic interpolation - skip
        return

    data1 = create_uniform_data(np.arange(len1), end_date=datetime(2020, 7, 23, 16, 27, 0), frequency=None)
    data2 = create_uniform_data(np.arange(len2), end_date=datetime(2020, 7, 23, 16, 27, 0), frequency=None)

    out1, out2 = reindex(data1, data2, method=method, kind=kind, bounded=True)

    assert len(out1) == len(out2)
    assert len(out1) == len_out


# test correct output size with bounds
@pytest.mark.parametrize("method", list(Method))
@pytest.mark.parametrize("kind", list(Kind))
def test_reindex_length_bounded(method, kind):
    """
    Here we test the bounded case. The time stamps of the 2 data series look like this:
    ts1:   x -- x --- x --- x --- x
    ts2:   x -- x --- x --- x --- x -- x --- x --- x --- x

    If bounded=True, we expect that the re-indexed time steps are the same as for ts1
    """

    data1 = create_uniform_data(np.ones(10), frequency="1min")
    data2 = create_uniform_data(np.ones(20), frequency="1min")

    out1, out2 = reindex(data1, data2, method=method, kind=kind, bounded=True)

    assert len(out1) == len(out2)
    assert len(data1) == len(out2)


# test distribution of random data when re-indexing should be smaller
@pytest.mark.parametrize("method", [Method.LINEAR])
@pytest.mark.parametrize("kind", list(Kind))
def test_reindex_distribution(method, kind):
    periods1 = 200
    periods2 = 400

    data1 = pd.Series(
        [random.random() * i for i in range(periods1)], index=pd.date_range("2020-02-03", periods=periods1, freq="3h")
    )
    data2 = pd.Series(
        [random.random() * i for i in range(periods2)], index=pd.date_range("2020-02-03", periods=periods2, freq="2h")
    )

    std_d1 = np.std(data1)
    std_d2 = np.std(data2)

    # Re-index
    reindexed1, reindexed2 = reindex(data1, data2, method=method, kind=kind, bounded=True)

    std_up1 = np.std(reindexed1)
    std_up2 = np.std(reindexed2)

    assert std_d1 >= std_up1
    assert std_d2 >= std_up2


@pytest.mark.core
def test_reindexed_data_contains_no_nans():
    data1 = pd.Series(
        [random.random() * i for i in range(24)], index=pd.date_range("2020-02-03", periods=24, freq="1h")
    )
    data2 = pd.Series(
        [random.random() * i for i in range(24)], index=pd.date_range("2020-02-03", periods=24, freq="1h")
    )

    data1[1:5] = np.nan

    out1, out2 = reindex(data1, data2)

    assert not out1.isna().any()
    assert not out2.isna().any()


@pytest.mark.core
def test_if_data_starts_with_nan_values_and_bounded_is_true_then_output_range_is_reduced():
    data1 = pd.Series(
        [random.random() * i for i in range(24)], index=pd.date_range("2020-02-03", periods=24, freq="1h")
    )
    data2 = pd.Series(
        [random.random() * i for i in range(24)], index=pd.date_range("2020-02-03", periods=24, freq="1h")
    )

    data1[0:5] = np.nan

    out1, out2 = reindex(data1, data2, bounded=True)

    assert_index_equal(out1.index, data1.index[5:])
    assert_index_equal(out2.index, data2.index[5:])
