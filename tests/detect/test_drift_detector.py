# Copyright 2021 Cognite AS
import numpy as np
import pandas as pd
import pytest

from indsl.detect.drift_detector import drift as drift_v2
from indsl.detect.drift_detector_v1 import drift as drift_v1
from tests.detect.test_utils import RNGContext


@pytest.mark.parametrize("drift_function", [drift_v1, drift_v2])
@pytest.mark.core
def test_drift_detector(drift_function):
    # Create fake data
    with RNGContext():
        y1 = np.random.normal(size=1000)
        y2 = np.random.normal(size=1000, loc=10)
    y = np.concatenate([y1, y2])

    x_dt = pd.date_range(start="1970", periods=len(y), freq="1H")
    test_data = pd.Series(y, index=x_dt)

    # Run drift detector
    res = drift_function(test_data, std_threshold=2)

    # Check results (only one event should be detected)
    assert res.iloc[1001] == 1
    assert res.diff().abs().sum() == 2.0

    # Run drift detector detect = "increase"
    res_inc = drift_function(test_data, std_threshold=2, detect="increase")
    assert res_inc.iloc[1001] == 1
    assert res_inc.diff().abs().sum() == 2.0

    # Run drift detector detect = "decrease"
    res_dec = drift_function(-test_data, std_threshold=2, detect="decrease")
    assert res_dec.iloc[1001] == 1
    assert res_dec.diff().abs().sum() == 2.0
