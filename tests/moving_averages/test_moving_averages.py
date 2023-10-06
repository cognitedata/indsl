# Copyright 2021 Cognite AS
import math

import numpy as np
import pandas as pd
import pytest

from indsl.detect.steady_state import vma


@pytest.mark.core
def test_variable_ma():
    test_data = pd.read_csv(
        "tests/moving_averages/variable_ma_test_data.csv",
        index_col=0,
        parse_dates=True,
    )
    vma13 = vma(test_data.data, window_length=13)
    idx01 = 300
    idx02 = 350
    diff = np.mean(vma13[idx01:idx02] - test_data.vma13_expected[idx01:idx02].values)
    assert math.isclose(diff, 0, abs_tol=0.01)
