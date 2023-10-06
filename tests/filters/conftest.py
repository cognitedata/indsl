# Copyright 2021 Cognite AS
import pandas as pd
import pytest


@pytest.fixture
def status_flag_test_data():
    data = pd.Series(
        [5.5, 4, 6],
        index=pd.to_datetime(
            [
                "2017-01-01 00:00:00",
                "2017-01-01 00:10:00",
                "2017-01-02 00:00:00",
            ]
        ),
    )
    bool_filter = pd.Series([1.0, 0], index=pd.to_datetime(["2017-01-01 00:00:00", "2017-01-01 00:10:00"]))

    return data, bool_filter
