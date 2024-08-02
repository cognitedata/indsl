# Copyright 2021 Cognite AS
import numpy as np
import pandas as pd
import pytest

from pandas.testing import assert_series_equal

from indsl.signals.sequence_interpolation import sequence_interpolation_1d


@pytest.mark.core
def test_sequence_interpolation_1d():

    import pickle

    "tests/moving_averages/variable_ma_test_data.csv",

    WHP_series = pd.pandas.read_pickle("./tests/signals/pd_series_WHP.pkl")

    # defining the interpolation curve
    x_values = np.linspace(WHP_series.min() * 0.99, WHP_series.max() * 1.01, 20)
    y_values = (np.sin(x_values / x_values.max() * np.pi - np.pi * 0.5) + 1) * 0.5

    from scipy.interpolate import interp1d

    interpolator = interp1d(x_values, y_values)
    output = interpolator(WHP_series.values)

    interpolation_test_result = pd.Series(output, index=WHP_series.index)

    interpolation_function_result = sequence_interpolation_1d

    pytest.approx(interpolation_test_result, interpolation_function_result)
