# Copyright 2021 Cognite AS
import numpy as np
import pandas as pd
import pytest

from indsl.smooth.eweight_ma_v1 import ewma
from indsl.smooth.lweight_ma_v1 import lwma
from indsl.smooth.simple_ma_v1 import sma


@pytest.mark.core
def test_sma():
    rng1 = np.random.default_rng(0)
    values = rng1.normal(0, 1, 10)
    sma_serie = pd.Series(values, index=pd.date_range("2021-02-09 00:00:00", "2021-02-09 01:00:00", periods=10))
    sma_expected = np.array(
        [
            0.12573022,
            -0.00318732,
            0.21134934,
            0.20440597,
            0.06988446,
            -0.02305807,
            0.37664191,
            0.87089202,
            0.51578192,
            -0.34069191,
        ]
    )
    sma_calculated = sma(sma_serie, time_window="15min", min_periods=1).values
    return np.testing.assert_array_almost_equal(
        sma_calculated,
        sma_expected,
        decimal=6,
        err_msg="Calculated SMA values do not match with the expected",
        verbose=True,
    )


@pytest.mark.parametrize("adjust", [True, False])
def test_ewma(adjust):
    rng1 = np.random.default_rng(0)
    values = rng1.normal(0, 1, 10)
    ewma_serie = pd.Series(values, index=pd.date_range("2021-02-09 00:00:00", "2021-02-09 01:00:00", periods=10))
    ewma_adjust_expected = np.array(
        [
            0.12573022,
            -0.06764609,
            0.34617301,
            0.21749414,
            -0.17123542,
            0.09940863,
            0.70644682,
            0.82723573,
            0.06025223,
            -0.60323255,
        ]
    )
    ewma_not_adjust_expected = np.array(
        [
            0.12573022,
            -0.04615984,
            0.29713141,
            0.20101576,
            -0.16732681,
            0.09713412,
            0.70056708,
            0.82382402,
            0.06004439,
            -0.60268854,
        ]
    )
    ewma_expected = ewma_adjust_expected if adjust else ewma_not_adjust_expected
    ewma_calculated = ewma(ewma_serie, time_window="15min", adjust=adjust, min_periods=1).values
    return np.testing.assert_array_almost_equal(
        ewma_calculated,
        ewma_expected,
        decimal=6,
        err_msg="Calculated EWMA values do not match with the expected",
        verbose=True,
    )


@pytest.mark.core
def test_lwma():
    rng1 = np.random.default_rng(0)
    values = rng1.normal(0, 1, 10)
    lwma_serie = pd.Series(values, index=pd.date_range("2021-02-09 00:00:00", "2021-02-09 01:00:00", periods=10))
    lwma_expected = np.array(
        [
            0.12573022,
            -0.04615984,
            0.29713141,
            0.2439068,
            -0.12613087,
            0.01972442,
            0.68325348,
            0.96847301,
            0.18115938,
            -0.70944232,
        ]
    )
    lwma_calculated = lwma(lwma_serie, time_window="15min").values
    return np.testing.assert_array_almost_equal(
        lwma_calculated,
        lwma_expected,
        decimal=6,
        err_msg="Calculated LWMA values do not match with the expected",
        verbose=True,
    )
