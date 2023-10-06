# Copyright 2021 Cognite AS
import numpy as np
import pandas as pd
import pytest

from numpy.testing import assert_almost_equal

from indsl.ts_utils import exp, log, log2, log10, logn, power


@pytest.mark.core
def test_log_and_exp():
    assert log(1) == 0
    # Natural logorithm of array
    num1 = np.array([-100, 100])
    assert np.all(log(exp(num1)) == num1)

    # Natural logorithm of pandas series
    arr1 = pd.Series(data=np.array([-100, 100]))
    assert np.all(log(exp(arr1)) == arr1)

    # Natural logorithm of data frame
    df1 = pd.DataFrame(data={"a": np.array([-100, 100]), "b": np.array([100, -100])})
    assert np.all(log(exp(df1.a)) == df1.a)
    assert np.all(log(exp(df1)) == df1)


@pytest.mark.core
def test_log2():
    assert log2(4) == 2
    # Logarithm with base 2 of array
    num1 = np.array([8, 16])
    assert np.all(log2(power(2, num1)) == num1)

    # Logarithm with base 2 of pandas series
    arr1 = pd.Series(data=np.array([8, 16]))
    assert np.all(log2(power(2, arr1)) == arr1)

    # Logarithm with base 2  of data frame
    df1 = pd.DataFrame(data={"a": np.array([2, 16]), "b": np.array([32, 0])}, dtype=np.int64)
    assert np.all(log2(power(2, df1.a)) == df1.a)
    assert np.all(log2(power(2, df1)) == df1)


@pytest.mark.core
def test_log10():
    assert log10(100000) == 5
    # Logarithm with base 10 of array
    num1 = np.array([1000, 100000])
    assert np.all(log10(num1) == np.array([3, 5]))

    # Logarithm with base 10 of pandas series
    arr1 = pd.Series(data=np.array([1000, 100000]))
    assert np.all(log10(arr1) == np.array([3, 5]))

    # Logarithm with base 10  of data frame
    df1 = pd.DataFrame(data={"a": np.array([1000, 100000]), "b": np.array([1, 10])})
    assert np.all(log10(df1.a) == np.array([3, 5]))
    assert np.all(log10(df1) == pd.DataFrame(data={"a": np.array([3, 5]), "b": np.array([0, 1])}))


@pytest.mark.core
def test_logn():
    # assert logn(27, 3) == 3
    assert_almost_equal(logn(27, 3), 3)

    # Logarithm with base n of array
    num1 = np.array([27, 81])
    assert_almost_equal(logn(num1, 3), np.array([3, 4]))

    # Logarithm with base n of pandas series
    arr1 = pd.Series(data=np.array([27, 81]))
    assert_almost_equal(logn(arr1, 3), np.array([3, 4]))

    # Logarithm with base n of data frame
    df1 = pd.DataFrame(data={"a": np.array([27, 81]), "b": np.array([729, 243])})
    assert_almost_equal(logn(df1.a, 3), np.array([3, 4]))
    assert np.all(logn(df1, 3).compare(pd.DataFrame(data={"a": np.array([3.0, 4.0]), "b": np.array([6.0, 5.0])})))
