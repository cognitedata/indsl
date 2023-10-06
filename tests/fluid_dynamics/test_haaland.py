# Copyright 2021 Cognite AS
import numpy as np
import pandas as pd
import pytest

from indsl.fluid_dynamics import Re
from indsl.fluid_dynamics.friction import Haaland


@pytest.mark.core
def test_haalands_equation(density=400, d_viscosity=0.0005, length_scale=0.3, roughness=0.005):
    data = np.array([1, 2, 3, 4, 5])
    speed = pd.Series(data)
    reynolds_number = Re(speed, density, d_viscosity, length_scale)

    friction_factor_swamee_jain = 0.25 / (np.log10(roughness / 3.7 + 5.74 / reynolds_number**0.9)) ** 2

    friction_factor = Haaland(reynolds_number, roughness)
    pytest.approx(friction_factor_swamee_jain, friction_factor)
