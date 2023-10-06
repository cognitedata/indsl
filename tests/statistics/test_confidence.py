# Copyright 2021 Cognite AS
import pandas as pd
import pandas.testing as tm
import pytest

from indsl.statistics.confidence import bands


@pytest.mark.core
def test_confidence_bands():
    test_data = pd.Series(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=pd.date_range("1975-05-09 00:00:00", "1975-05-09 09:00:00", freq="1H")
    )

    # Estimate a 1 hour rolling average and confidence bands with 2 standard deviations
    period = "2H"
    K = 2
    expected = {
        "avg": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5],
        "lower": [0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1],
        "upper": [2.9, 3.9, 4.9, 5.9, 6.9, 7.9, 8.9, 9.9, 10.9],
    }
    expected = pd.DataFrame(expected, index=test_data.index[1:])
    # expected.index.names = ["timestamp"]
    # Run the method bands
    test_bands = bands(test_data, period=period, K=K, as_json=False)

    # Compare expected and method results
    tm.assert_frame_equal(test_bands.round(1), expected)

    assert type(bands(test_data, period=period, K=K, as_json=True)) == str
