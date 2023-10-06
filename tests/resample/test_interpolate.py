# Copyright 2021 Cognite AS
import numpy as np
import pandas as pd
import pytest

from indsl.resample.interpolate import interpolate

from ..generate_data import create_non_uniform_data, create_uniform_data, set_na_random_data


# Test for empty data
@pytest.mark.parametrize(
    "input",
    [
        pd.Series(dtype=np.float64),
        pd.Series([0], index=pd.date_range(0, periods=1, freq="10s")),
    ],
)
def test_too_few_data_warnings(input):
    with pytest.warns(UserWarning):
        interpolated_data = interpolate(input)
        assert len(interpolated_data) == len(input)


# Test for all NaN data
def test_all_nan_data():
    with pytest.warns(UserWarning):
        test_data = create_uniform_data(np.ones(10) * np.nan)
        interpolated_data = interpolate(test_data)
        assert len(interpolated_data) == len(test_data)
        assert interpolated_data.isna().all()


# Test for linear, uniform data
def test_linear_uniform_data():
    test_data = create_uniform_data(np.linspace(-10, 10, 200))
    interpolated_data = interpolate(test_data)
    pd.testing.assert_series_equal(test_data, interpolated_data)


# Test for uniform data with NaN
def test_non_linear_data():
    expected = create_uniform_data(np.linspace(0, 10, 10))
    test_data = set_na_random_data(expected.copy())
    interpolated_data = interpolate(test_data.copy())
    assert (expected.index.to_numpy() == interpolated_data.index.to_numpy()).all()
    np.testing.assert_almost_equal(interpolated_data.values, expected.values)


# Test for series data
def test_series_data():
    data = create_uniform_data(np.linspace(0, 10, 10))
    data.name = "nicolas"
    res = interpolate(data)
    assert res.equals(data)
    assert isinstance(res, pd.Series)


# Test for linear, uniform data with 1h granularity
def test_linear_uniform_data_1h():
    test_data = create_uniform_data(np.linspace(-10, 10, 200), frequency="1h")
    interpolated_data = interpolate(test_data, granularity=pd.Timedelta("1h"))
    assert test_data.equals(interpolated_data)


# Test for linear, uniform data with 1h granularity
def test_linear_non_uniform_data():
    test_data = create_non_uniform_data(np.linspace(-10, 10, 200))
    interpolated_data = interpolate(test_data, granularity=pd.Timedelta("1s"))
    assert test_data.values[0] == interpolated_data.values[0]
    assert test_data.values[-1] == interpolated_data.values[-1]


# Test for uniform data with NaN and 1h granularity
def test_non_linear_data_1h():
    expected = create_uniform_data(np.linspace(0, 10, 10), frequency="1h")
    test_data = set_na_random_data(expected.copy())
    interpolated_data = interpolate(test_data.copy(), granularity=pd.Timedelta("1h"))
    assert (expected.index.to_numpy() == interpolated_data.index.to_numpy()).all()
    np.testing.assert_almost_equal(interpolated_data.values, expected.values)


# Test for linear, uniform data with forwardfill
def test_linear_uniform_data_ffill():
    test_data = create_uniform_data(np.linspace(-10, 10, 200))
    interpolated_data = interpolate(test_data, method="ffill")
    assert test_data.equals(interpolated_data)


# Test for uniform data with NaN with forwardfill
def test_non_linear_data_ffill():
    values = [1.0, np.nan, np.nan, 3.9, -2.3]
    expected_values = [1.0, 1.0, 1.0, 3.9, -2.3]
    test_data = create_uniform_data(values)
    expected = create_uniform_data(expected_values)
    interpolated_data = interpolate(test_data, method="ffill")
    assert (expected.index.to_numpy() == interpolated_data.index.to_numpy()).all()
    np.testing.assert_almost_equal(interpolated_data.values, expected.values)


# Test for uniform data with NaN with forwardfill
@pytest.mark.parametrize("method", ["ffill", "linear"])
@pytest.mark.parametrize("kind", ["pointwise", "average"])
def test_non_linear_data_ffill_nan_endpoint(method, kind):
    values = [1.0, np.nan, np.nan, 3.9, -2.3, np.nan, 1.0]
    test_data = create_uniform_data(values)
    out = interpolate(test_data, method=method, kind=kind)
    assert len(out) == len(out.dropna())


# Test method average
@pytest.mark.parametrize(
    "test_values, expected_values",
    [
        ([1.0, 2.0, 3.0, 4.0, 5.0], [1.5, 2.5, 3.5, 4.5, 5.0]),
        ([1.0, np.nan, 3.0, np.nan, 5.0], [1.5, 2.5, 3.5, 4.5, 5.0]),
    ],
)
def test_linear_average_method(test_values, expected_values):
    test_data = create_uniform_data(test_values)
    expected = create_uniform_data(expected_values)
    interpolated_data = interpolate(test_data, method="linear", kind="average")
    pd.testing.assert_series_equal(expected, interpolated_data)


# Test for outside fill method
@pytest.mark.parametrize(
    "test_values, expected_values, bounded",
    [
        ([np.nan, 2.0, np.nan, 4.0, np.nan], [1.0, 2.0, 3.0, 4.0, 5.0], False),
        ([np.nan, 2.0, np.nan, 4.0, np.nan], [np.nan, 2.0, 3.0, 4.0, np.nan], True),
    ],
)
def test_outside_fill_method(test_values, expected_values, bounded):
    test_data = create_uniform_data(test_values)
    expected = create_uniform_data(expected_values)
    interpolated_data = interpolate(test_data, method="linear", bounded=bounded)
    pd.testing.assert_series_equal(expected, interpolated_data)
