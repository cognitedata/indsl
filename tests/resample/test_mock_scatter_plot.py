# Copyright 2024 Cognite AS
import random

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from pandas.testing import assert_series_equal

from indsl.exceptions import UserTypeError, UserValueError
from indsl.resample.mock_scatter_plot import reindex_scatter, reindex_scatter_x


# Test for empty data
@pytest.mark.core
def test_reindex_scatter():

    HCV_series = pd.read_pickle("./tests/resample/pd_series_HCV.pkl")
    # defining the interpolation curve
    n = 20
    x_values = np.linspace(0, 100, n)
    y_values = (np.sin(x_values / x_values.max() * np.pi - np.pi * 0.5) + 1) * 0.5

    from scipy.interpolate import interp1d

    interpolator = interp1d(x_values, y_values)
    CV_array = interpolator(HCV_series.values)
    # Create the series for the CV value
    CV_series = pd.Series(CV_array, index=HCV_series.index)

    x_min_value = 0
    x_max_value = 100
    signal_scatter = reindex_scatter(
        HCV_series, CV_series, x_min_value=x_min_value, x_max_value=x_max_value, align_timesteps=True
    )
    signal_scatter_x = reindex_scatter_x(HCV_series, x_min_value=x_min_value, x_max_value=x_max_value)

    # Calculate separately
    # convert timestamps to epoc
    epoc = np.array([val.timestamp() for val in HCV_series.index])
    d_epoc = epoc[-1] - epoc[0]
    # The scale of HCV_series is [0,100]. We will now map it to the epoc and then convert it to datetime
    sequence_epoc = HCV_series.values / 100 * d_epoc
    index_cv_epoc = sequence_epoc + epoc[0]  # translate
    index_cv = np.array([datetime.fromtimestamp(epoc_) for epoc_ in index_cv_epoc])
    # create a sort index, such that the order is increasing
    index_sort = np.argsort(index_cv_epoc)
    CV_series = pd.Series(CV_array[index_sort], index=index_cv[index_sort])
    CV_series_x = pd.Series(HCV_series.values[index_sort], index=index_cv[index_sort])

    assert_series_equal(signal_scatter, CV_series)
    assert_series_equal(signal_scatter_x, CV_series_x)

    if False:
        import pylab as plt

        fig = plt.figure(figsize=(12, 8))
        plt.plot(HCV_series.index, HCV_series.values, "-b")
        axl = plt.gca()
        axr = axl.twinx()
        axr.plot(signal_scatter.index, signal_scatter.values, "xr")
        axr.plot(signal_scatter_x.index, signal_scatter_x * 0.01, ".g")
        plt.show()


# test_reindex_scatter()
