import pandas as pd
import pytest

from pandas.testing import assert_series_equal

from indsl.equipment.pump_parameters import percent_BEP_flowrate


@pytest.mark.core
def test_BEP_flowrate():
    BEP = pd.Series([110] * 3)
    flowrate = pd.Series([27.56, 22.4, 135.0])
    res = percent_BEP_flowrate(flowrate, BEP)
    expected = pd.Series([25.055, 20.364, 122.727])

    assert_series_equal(res, expected, atol=5e-4)
