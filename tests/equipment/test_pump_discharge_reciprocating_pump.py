import pandas as pd
import pytest

from pandas.testing import assert_series_equal

from indsl.equipment.pump_parameters import pump_discharge_reciprocating_pump


@pytest.mark.core
def test_pump_discharge_reciprocating_pump_series():
    area = pd.Series([3.5, 4.5])
    length_of_strokes = pd.Series([0.1, 0.2])
    number_of_revolutions_per_second = pd.Series([4.2, 8.3])
    res = pump_discharge_reciprocating_pump(area, length_of_strokes, number_of_revolutions_per_second)
    expected = pd.Series([88.2, 448.2])

    assert_series_equal(res, expected, check_dtype=False, atol=5e-6)


@pytest.mark.core
def test_pump_discharge_reciprocating_pump_scalar():
    area = 3.5
    length_of_strokes = 0.1
    number_of_revolutions_per_second = 4.2
    full_res = pump_discharge_reciprocating_pump(area, length_of_strokes, number_of_revolutions_per_second)
    res = full_res.iloc[0]
    expected = 88.2

    assert round(res, 6) == round(expected, 6)
