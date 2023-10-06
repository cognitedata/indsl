import pandas as pd
import pytest

from indsl.signals.polynomial import univariate_polynomial


@pytest.mark.core
def test_polynomial():
    signal = pd.Series([14, 100, 150, 100], index=[0, 1, 5, 6])
    a = [0, 5, 1, 1, 0, 30]
    res = univariate_polynomial(signal, a)
    exp_res = pd.Series([16137730, 300001010500, 2278128398250, 300001010500], index=[0, 1, 5, 6])
    assert res.equals(exp_res)
