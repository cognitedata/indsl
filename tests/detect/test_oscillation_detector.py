# Copyright 2021 Cognite AS
import random

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from numpy.testing import assert_almost_equal

from indsl.detect.oscillation_detector import (
    _lpc,
    _validate_data,
    cross_corr,
    helper_oscillation_detector,
    lpc,
    oscillation_detector,
)
from indsl.exceptions import UserTypeError, UserValueError


@pytest.mark.core
def test_oscillation_detection(multi_freq_signal, start_date):
    """Oscillation detection.

    Test that the algorithm detects oscillations and returns corresponding
    frequencies and amplitudes.
    """
    values = multi_freq_signal
    data = pd.Series(values, index=pd.date_range(start=start_date, freq="1s", periods=len(values)))
    results = oscillation_detector(data)

    assert_almost_equal(results.index[np.where(results.values == 1)], np.array([0.02, 0.1]), decimal=2)


@pytest.mark.extras
def test_prediction_accuracy(sin_wave_array_extensive, start_date, frequency):
    """Test accuracy of LPC polynomial fit."""
    values = sin_wave_array_extensive
    data = pd.Series(values, index=pd.date_range(start=start_date, freq=frequency, periods=len(values)))
    results = helper_oscillation_detector(data)

    assert results["oscillations"]
    assert np.abs(np.average(results["fit"][1] - np.array(data))) < 1e-3


@pytest.mark.core
def test_non_uniform_input(sin_wave_array, start_date):
    """Test non uniform signal."""
    values = sin_wave_array
    indices = [start_date]
    for i in range(1, len(values)):
        indices.append(indices[i - 1] + timedelta(seconds=1) * random.randint(1, 1000))
    data = pd.Series(values, index=indices)
    results = oscillation_detector(data)

    assert len(results.index[np.where(results.values == 1)]) > 0


@pytest.mark.core
def test_nan_in_signal(sin_wave_array, start_date, frequency):
    """Test signal data containing NaN."""
    values = sin_wave_array
    data = pd.Series(values, index=pd.date_range(start=start_date, freq=frequency, periods=len(values)))
    num_points_to_change = int(len(data) * 0.5)
    indices = np.random.choice(data.index[1:-1], size=num_points_to_change, replace=False)
    data.loc[indices] = np.nan

    with pytest.raises(TypeError):
        oscillation_detector(data)


@pytest.mark.core
def test_validate_data():
    string_array = np.array(["1", "2", "3"])
    with pytest.raises(UserTypeError) as excinfo:
        _validate_data(y=string_array)
    exp_res = "Data must be floating-point"
    assert exp_res in str(excinfo.value)

    wrong_shape_array = np.zeros((2, 3, 4))
    with pytest.raises(UserTypeError) as excinfo:
        _validate_data(y=wrong_shape_array)
    exp_res = f"Signal data must have shape (samples,). Received shape={wrong_shape_array.shape}"
    assert exp_res in str(excinfo.value)

    same_value_array = np.array([1.0, 1.0, 1.0])
    with pytest.raises(UserValueError) as excinfo:
        _validate_data(y=same_value_array)
    exp_res = "Ill-conditioned input array; contains only one unique value"
    assert exp_res in str(excinfo.value)


@pytest.mark.core
def test_cross_corr_validation():
    # not same length arrays
    x = np.array([1])
    y = np.array([1, 3])
    with pytest.raises(UserValueError) as excinfo:
        cross_corr(x=x, y=y)
    exp_res = "x and y must be equal length"
    assert exp_res in str(excinfo.value)

    # too short arrays
    y = np.array([2])
    with pytest.raises(UserValueError) as excinfo:
        cross_corr(x=x, y=y)
    exp_res = "lags must be None or strictly positive < 1"
    assert exp_res in str(excinfo.value)


@pytest.mark.core
def test_lpc_validation_test():
    y = np.zeros(3)
    with pytest.raises(UserValueError) as excinfo:
        lpc(y=y, order=0)
    exp_res = "order must be an integer > 0"
    assert exp_res in str(excinfo.value)


@pytest.mark.core
def test_lpc_underscore_flaotingpointerror():
    y = np.array([1.0])
    order = 1
    with pytest.raises(FloatingPointError) as excinfo:
        _lpc(y=y, order=order)
    exp_res = "numerical error, input ill-conditioned?"
    assert exp_res in str(excinfo.value)
