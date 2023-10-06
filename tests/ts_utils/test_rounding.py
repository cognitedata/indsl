# Copyright 2021 Cognite AS
import numpy as np
import pandas as pd
import pytest

from indsl.ts_utils import ceil, floor, round, sign


@pytest.mark.core
def test_round():
    assert round(0.987654321, 5) == 0.98765
    # Rounds array
    num1 = np.array([0.987654, 0.123456])
    assert np.all(round(num1, 3) == np.array([0.988, 0.123]))

    # Rounds pandas series
    arr1 = pd.Series(data=np.array([0.111, 0.115, 0.1145, 0.1159]))
    assert np.all(round(arr1, 2) == pd.Series(data=np.array([0.11, 0.12, 0.11, 0.12])))

    # Rounds pandas data frame
    df1 = pd.DataFrame(
        data={"a": np.array([0.111, 0.115, 0.1145, 0.1159]), "b": np.array([0.9876, 0.8765, 0.7654, 0.6543])}
    )
    assert np.all(round(df1.a, 2) == np.array([0.11, 0.12, 0.11, 0.12]))
    assert np.all(
        round(df1, 2)
        == pd.DataFrame(data={"a": np.array([0.11, 0.12, 0.11, 0.12]), "b": np.array([0.99, 0.88, 0.77, 0.65])})
    )


@pytest.mark.core
def test_floor():
    assert floor(0.987654321) == 0
    # Rounds down array
    num1 = np.array([-0.987654, 5.123456])
    assert np.all(floor(num1) == np.array([-1, 5]))

    # Rounds down pandas series
    arr1 = pd.Series(data=np.array([-9.1, 9.1, -10.6, 10.6]))
    assert np.all(floor(arr1) == pd.Series(data=np.array([-10, 9, -11, 10])))

    # Rounds down pandas data frame
    df1 = pd.DataFrame(data={"a": np.array([9.1, 10.5, 10.6, 10.1]), "b": np.array([-9.1, -10.5, -10.6, -10.1])})
    assert np.all(floor(df1.a) == np.array([9, 10, 10, 10]))
    assert np.all(
        floor(df1) == pd.DataFrame(data={"a": np.array([9, 10, 10, 10]), "b": np.array([-10, -11, -11, -11])})
    )


@pytest.mark.core
def test_ceil():
    assert ceil(0.987654321) == 1
    # Rounds up array
    num1 = np.array([-0.987654, 5.123456])
    assert np.all(ceil(num1) == np.array([0, 6]))

    # Rounds up pandas series
    arr1 = pd.Series(data=np.array([-9.1, 9.1, -10.6, 10.6]))
    assert np.all(ceil(arr1) == pd.Series(data=np.array([-9, 10, -10, 11])))

    # Rounds up pandas data frame
    df1 = pd.DataFrame(data={"a": np.array([9.1, 10.5, 10.6, 10.1]), "b": np.array([-9.1, -10.5, -10.6, -10.1])})
    assert np.all(ceil(df1.a) == np.array([10, 11, 11, 11]))
    assert np.all(ceil(df1) == pd.DataFrame(data={"a": np.array([10, 11, 11, 11]), "b": np.array([-9, -10, -10, -10])}))


@pytest.mark.core
def test_sign():
    assert sign(100) == 1
    # Indication of the sign of numbers in array
    num1 = np.array([-100, 100])
    assert np.all(sign(num1) == np.array([-1, 1]))

    # Indication of the sign of numbers in pandas series
    arr1 = pd.Series(data=np.array([-100, 100]))
    assert np.all(sign(arr1) == pd.Series(data=np.array([-1, 1])))

    # Indication of the sign of numbers in pandas data frame
    df1 = pd.DataFrame(data={"a": np.array([-100, 100]), "b": np.array([100, -100])})
    assert np.all(sign(df1.a) == np.array([-1, 1]))
    assert np.all(sign(df1) == pd.DataFrame(data={"a": np.array([-1, 1]), "b": np.array([1, -1])}))
