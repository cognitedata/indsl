# Copyright 2021 Cognite AS
import numpy as np
import pandas as pd
import pytest
from indsl.exceptions import UserTypeError

from numpy.testing import assert_array_equal

from indsl.ts_utils import remove, replace


def test_replace_errors():
    series = pd.Series(np.array([1.0, 1.0, -np.inf, np.inf, np.nan]))

    with pytest.raises(UserTypeError):
        replace([1], [1.0], 0.0)

    with pytest.raises(TypeError):
        replace(series, 1.0, 0.0)

    with pytest.raises(UserTypeError):
        replace(series, [1.0], [0.0])


def test_remove_errors():
    series = pd.Series(np.array([1.0, 1.0, -np.inf, np.inf, np.nan]))

    with pytest.raises(UserTypeError):
        remove([1], [1.0])

    with pytest.raises(UserTypeError):
        remove(series, 1.0)


@pytest.mark.core
def test_replace():
    series = pd.Series(np.array([1.0, 1.0, -np.inf, np.inf, np.nan]))

    assert_array_equal(replace(series, [1.0], 0.0).values, np.array([0.0, 0.0, -np.inf, np.inf, np.nan]))
    assert_array_equal(replace(series, [-np.inf], 0.0).values, np.array([1.0, 1.0, 0.0, np.inf, np.nan]))
    assert_array_equal(replace(series, [np.inf], 0.0).values, np.array([1.0, 1.0, -np.inf, 0.0, np.nan]))
    assert_array_equal(replace(series, [np.nan], 0.0).values, np.array([1.0, 1.0, -np.inf, np.inf, 0.0]))
    assert_array_equal(replace(series, [1.0, np.inf], 0.0).values, np.array([0.0, 0.0, -np.inf, 0.0, np.nan]))
    assert_array_equal(replace(series, [1.0], np.inf).values, np.array([np.inf, np.inf, -np.inf, np.inf, np.nan]))
    assert_array_equal(replace(series, [1.0], np.nan).values, np.array([np.nan, np.nan, -np.inf, np.inf, np.nan]))
    # Test default parameter values
    assert_array_equal(replace(series).values, series)


@pytest.mark.core
def test_remove_values():
    series = pd.Series(np.array([1.0, 1.0, -np.inf, np.inf, np.nan]))

    assert_array_equal(remove(series, [1.0]).values, np.array([-np.inf, np.inf, np.nan]))
    assert_array_equal(remove(series, [-np.inf]).values, np.array([1.0, 1.0, np.inf, np.nan]))
    assert_array_equal(remove(series, [np.inf]).values, np.array([1.0, 1.0, -np.inf, np.nan]))
    assert_array_equal(remove(series, [np.nan]).values, np.array([1.0, 1.0, -np.inf, np.inf]))
    assert_array_equal(remove(series, [1.0, np.inf]).values, np.array([-np.inf, np.nan]))
    # Test default parameter values
    assert_array_equal(remove(series).values, series.values)


@pytest.mark.core
def test_remove_range():
    series = pd.Series(np.array([1.0, 2.0, 5.0, -1.0, -np.inf]))

    assert_array_equal(remove(series, range_from=1.0, range_to=4.5).values, np.array([1.0, 2.0]))
    assert_array_equal(remove(series, range_to=4.5).values, np.array([1.0, 2.0, -1.0, -np.inf]))
    assert_array_equal(remove(series, range_from=1.0, range_to=4.5).values, np.array([1.0, 2.0]))
    assert_array_equal(remove(series, to_remove=[2.0], range_from=1.0, range_to=4.5).values, np.array([1.0]))
