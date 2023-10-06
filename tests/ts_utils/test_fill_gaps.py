# Copyright 2021 Cognite AS
import numpy as np
import pandas as pd
import pytest

from indsl.ts_utils.ts_utils import fill_gaps


# Nothing should be done, if there is no empty data
@pytest.mark.core
def test_no_filling():
    data = pd.Series([i for i in range(24)], index=pd.date_range("2020-02-03", periods=24, freq="1H"))

    # upsampled to 30 min
    gapped_filled = fill_gaps(data, granularity="1H")

    assert np.array_equal(data, gapped_filled)


# Test that length is kept
@pytest.mark.core
def test_same_length():
    data = pd.Series([i for i in range(24)], index=pd.date_range("2020-02-03", periods=24, freq="1H"))

    # we compare this
    ln1 = len(data)
    # to the length after removing part of it
    data[4:10] = np.nan

    testy = fill_gaps(data, granularity="1H")

    assert ln1 == len(testy)
