import inspect

import numpy as np
import pytest

import indsl
from datasets.data.synthetic_industrial_data import non_linear_non_stationary_signal

seeds = [10, 1975, 2000, 1, 89756]

def get_all_operations():
    indsl_functions = []
    for _, module in inspect.getmembers(indsl):
        toolbox_name = getattr(module, "TOOLBOX_NAME", None)
        if toolbox_name is None:
            continue
        functions_to_export = getattr(module, "__cognite__", [])
        functions_map = inspect.getmembers(module, inspect.isfunction)
        for name, function in functions_map:
            if name in functions_to_export:
                indsl_functions.append(function)
    return indsl_functions


indsl_functions = get_all_operations()


@pytest.fixture(scope="function", params=seeds)
def generate_synthetic_industrial_data(request):
    """Returns a time series composed of 3 oscillatory signals, one of them
    using an exponential, 2 nonlinear trends, sensor linear drift (small
    decrease) and white noise.

    The signal has non-uniform time stamps and 35% of the data is
    randomly removed to generate data gaps. The data gaps are inserted
    with a constant seed to have reproducible behavior.
    """
    # start_date = pd.Timestamp("2022-03-28")
    # end_date = start_date + pd.Timedelta("4.5 days")
    # seed array = [10, 1975, 2000, 1, 89756]
    # Seed number used for testing. These seeds are typically selected by the function developer and pre-set
    # to work in a particular test.
    data = non_linear_non_stationary_signal(seed=request.param)

    return data


@pytest.fixture(scope="function", params=seeds)
def generate_synthetic_industrial_data_with_outliers(request):
    # seed array = [10, 1975, 2000, 6000, 1, 89756]
    # Seed number used for testing. These seeds are typically selected by the function developer and pre-set
    # to work in a particular test.
    # Seeds 10, 1975, 2000, 6000, 1 and 89756 are used to test the :func:`out_of_range`
    data = non_linear_non_stationary_signal(seed=request.param)
    rng = np.random.default_rng(request.param)

    outlier_fraction = 0.05  # Fraction of the signal that will be replaced by outliers
    num_outliers = round(len(data) * outlier_fraction)
    locations = np.unique(rng.integers(low=0, high=len(data), size=num_outliers))
    direction = rng.choice([1, -1], size=len(locations))
    outliers = data.iloc[locations] + data.mean() * rng.uniform(0.5, 1, len(locations)) * direction

    data_w_outliers = data.copy()
    data_w_outliers.iloc[locations] = outliers

    return data_w_outliers, outliers
