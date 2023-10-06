# Copyright 2021 Cognite AS
import pandas as pd
import pytest

from pandas.testing import assert_series_equal

from indsl.exceptions import UserRuntimeError
from indsl.filter.simple_filters import status_flag_filter


# Status flag filter expected results
status_flag_filter_exp_res_0 = pd.Series([4.0], index=pd.to_datetime(["2017-01-01 00:10:00"]))
status_flag_filter_exp_res_1 = pd.Series([5.5], index=pd.to_datetime(["2017-01-01 00:00:00"]))


@pytest.mark.parametrize("int_kept,exp_res", [(0, status_flag_filter_exp_res_0), (1, status_flag_filter_exp_res_1)])
def test_status_flag_filter_default(status_flag_test_data, int_kept, exp_res):
    # Mock data
    data, bool_filter = status_flag_test_data

    # Filter data
    res = status_flag_filter(data, bool_filter, int_to_keep=int_kept)

    # Assert result
    assert_series_equal(res, exp_res)


@pytest.mark.core
def test_status_flag_no_data(status_flag_test_data):
    # Mock data
    data, bool_filter = status_flag_test_data

    # Filter data
    with pytest.raises(UserRuntimeError, match="Current filter returns no entries for data."):
        _ = status_flag_filter(data, bool_filter, int_to_keep=2)
