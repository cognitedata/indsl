# Copyright 2021 Cognite AS
import numpy as np
import pandas as pd
import pytest

from pandas.testing import assert_series_equal

from indsl.exceptions import UserValueError
from indsl.resample.auto_align import auto_align

from indsl.signals.sequence_interpolation import sequence_interpolation_1d, sequence_interpolation_2d


@pytest.mark.core
def test_sequence_interpolation_1d():

    import pickle

    WHP_series = pd.pandas.read_pickle("./tests/signals/pd_series_WHP.pkl")

    # defining the interpolation curve
    x_values = np.linspace(WHP_series.min() * 0.99, WHP_series.max() * 1.01, 20)
    y_values = (np.sin(x_values / x_values.max() * np.pi - np.pi * 0.5) + 1) * 0.5

    from scipy.interpolate import interp1d

    interpolator = interp1d(x_values, y_values)
    output = interpolator(WHP_series.values)

    interpolation_test_result = pd.Series(output, index=WHP_series.index)

    interpolation_function_result = sequence_interpolation_1d(WHP_series, x_values.tolist(), y_values.tolist())

    pytest.approx(interpolation_test_result, interpolation_function_result)


@pytest.mark.core
def test_sequence_interpolation_2d():

    import pickle

    WHP_series = pd.pandas.read_pickle("./tests/signals/pd_series_WHP.pkl")
    WHT_series = pd.pandas.read_pickle("./tests/signals/pd_series_WHT.pkl")
    # auto-align
    WHP_series, WHT_series = auto_align([WHP_series, WHT_series], True)

    # defining the interpolation curve
    from numpy.random import uniform

    n = 200
    params_P = uniform(WHP_series.min() * 0.99, WHP_series.max() * 1.01, size=n)
    params_T = uniform(WHT_series.min() * 0.99, WHT_series.max() * 1.01, size=n)

    params_z = (
        (np.sin(params_P / params_P.max() * np.pi - np.pi * 0.5) + 1)
        * 0.5
        * (np.sin(params_T / params_T.max() * np.pi - np.pi * 0.5) + 1)
        * 0.5
    )

    from scipy.interpolate import LinearNDInterpolator

    interpolator = LinearNDInterpolator(list(zip(params_P, params_T)), params_z)
    output = interpolator(WHP_series.values, WHT_series.values)
    output_series = pd.Series(output, index=WHP_series.index)
    interpolation_test_result = pd.Series(output, index=WHP_series.index)

    interpolation_function_result = sequence_interpolation_2d(
        WHP_series, WHT_series, params_P.tolist(), params_T.tolist(), params_z.tolist()
    )

    pytest.approx(interpolation_test_result, interpolation_function_result)


@pytest.mark.core
def test_sequence_interpolation_2d_outside():
    """
    Contains input parameters that are outside the convec hull of the interpolation input points
    """
    import pickle

    WHP_series = pd.pandas.read_pickle("./tests/signals/pd_series_WHP.pkl")
    WHT_series = pd.pandas.read_pickle("./tests/signals/pd_series_WHT.pkl")
    # auto-align
    WHP_series, WHT_series = auto_align([WHP_series, WHT_series], True)

    # defining the interpolation curve
    from numpy.random import uniform

    n = 200
    params_P = uniform(WHP_series.min() * 0.99, WHP_series.max() * 0.99, size=n)
    params_T = uniform(WHT_series.min() * 0.99, WHT_series.max() * 0.99, size=n)

    params_z = (
        (np.sin(params_P / params_P.max() * np.pi - np.pi * 0.5) + 1)
        * 0.5
        * (np.sin(params_T / params_T.max() * np.pi - np.pi * 0.5) + 1)
        * 0.5
    )

    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

    interpolator = LinearNDInterpolator(list(zip(params_P, params_T)), params_z)
    output = interpolator(WHP_series.values, WHT_series.values)
    output_series = pd.Series(output, index=WHP_series.index)
    interpolation_test_result = pd.Series(output, index=WHP_series.index)

    # Do the input parameters outside the convex hull using nearest neighbor
    idx_nan = np.isnan(output)
    interpolator_nnd = NearestNDInterpolator(list(zip(params_P, params_T)), params_z, rescale=True)
    # Only run it on the values that has nan
    input_x_nan = WHP_series.values[idx_nan]
    input_y_nan = WHT_series.values[idx_nan]
    output[idx_nan] = interpolator_nnd(input_x_nan, input_y_nan)

    interpolation_function_result = sequence_interpolation_2d(
        WHP_series, WHT_series, params_P.tolist(), params_T.tolist(), params_z.tolist()
    )

    pytest.approx(interpolation_test_result, interpolation_function_result)


@pytest.mark.core
def test_sequence_interpolation_return_user_value_error():
    """
    Test that the function raises a UserValueError when the number of parameters is not equal or less than 2.
    """
    WHP_series = pd.Series(
        np.random.uniform(1, 80, size=100),
        index=pd.date_range("2022-01-01 10:00:00", periods=100, freq="min"),
        name="value",
    )
    x_values = [0, 1, 2]
    y_values = [0, 1]
    z_values = [0]

    with pytest.raises(UserValueError):
        sequence_interpolation_1d(WHP_series, x_values, y_values)

    with pytest.raises(UserValueError):
        sequence_interpolation_2d(WHP_series, WHP_series, x_values, y_values, z_values)

    with pytest.raises(UserValueError):
        sequence_interpolation_2d(WHP_series, WHP_series, x_values, x_values, z_values)

    with pytest.raises(UserValueError):
        sequence_interpolation_2d(WHP_series, WHP_series, x_values, y_values, y_values)

    with pytest.raises(UserValueError):
        sequence_interpolation_2d(WHP_series, WHP_series, z_values, z_values, z_values)
