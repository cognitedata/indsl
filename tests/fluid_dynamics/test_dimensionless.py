# Copyright 2021 Cognite AS
import numpy as np
import pandas as pd
import pytest

from pandas.testing import assert_series_equal

from indsl.fluid_dynamics.dimensionless import Re


@pytest.mark.core
def test_Reynolds_Number(density=1, d_viscosity=2, length_scale=10):
    data = np.array([1, 2, 3, 4, 5])
    speed = pd.Series(data)
    expected = pd.Series(data * density * length_scale / d_viscosity)
    Re_num = Re(speed, density=density, d_viscosity=d_viscosity, length_scale=length_scale)
    assert_series_equal(Re_num, expected)
