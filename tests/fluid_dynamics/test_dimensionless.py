# Copyright 2021 Cognite AS
import numpy as np
import pandas as pd
import pytest

from pandas.testing import assert_series_equal

from indsl.fluid_dynamics.dimensionless import (
    Re,
    Fr,
    We,
    Pressure_scaled,
    Roughness_scaled,
    Fr_density_scaled,
    Fr_inviscid_kelvin_helmholtz,
    Fr_2phase,
)


@pytest.mark.core
def test_Reynolds_Number(density_in=1, d_viscosity_in=2, length_scale_in=10):
    """
    All the input parameters are pd.Series
    """
    n = 10
    speed = pd.Series(np.linspace(1, 10, n))
    density = pd.Series(density_in * np.ones(n))
    d_viscosity = pd.Series(d_viscosity_in * np.ones(n))
    length_scale = pd.Series(length_scale_in * np.ones(n))
    expected = pd.Series(speed * density * length_scale / d_viscosity)
    Re_num = Re(speed, density=density, d_viscosity=d_viscosity, length_scale=length_scale)
    assert_series_equal(Re_num, expected)


@pytest.mark.core
def test_Reynolds_Number_2(density=1, d_viscosity=2, length_scale=10):
    """
    Mix pd.Series and float as input
    """
    n = 10
    speed = pd.Series(np.linspace(1, 10, n))
    expected = pd.Series(speed * density * length_scale / d_viscosity)
    Re_num = Re(speed, density=density, d_viscosity=d_viscosity, length_scale=length_scale)
    assert_series_equal(Re_num, expected)


@pytest.mark.core
def test_Reynolds_Number_3(density_in=1, d_viscosity=2, length_scale=10):
    """
    Different length on input
    """
    n = 10
    from datetime import datetime

    index = [datetime(2024, 8, 1, hour, 0, 0) for hour in range(1, n + 1, 1)]
    remove_idx = 4
    index_short = index.copy()
    index_short.pop(remove_idx)

    speed = pd.Series(np.linspace(1, 10, n), index=index)
    density_full = density_in * np.ones(n)
    density_short = pd.Series(np.delete(density_full, [remove_idx]), index=index_short)
    density = pd.Series(density_full, index=index)
    # Call the function
    Re_num = Re(speed, density=density_short, d_viscosity=d_viscosity, length_scale=length_scale, align_timesteps=True)
    # Create out own calculation. Using the index from the returned series
    expected = pd.Series(speed * density * length_scale / d_viscosity, index=Re_num.index)
    assert_series_equal(Re_num, expected)


@pytest.mark.core
def test_Froude_Number(length_scale_in=1):
    """
    All the input parameters are pd.Series
    """
    n = 10
    g = 9.81
    speed = pd.Series(np.linspace(1, 10, n))
    length_scale = pd.Series(length_scale_in * np.ones(n))
    expected = pd.Series(speed / np.sqrt(g * length_scale))
    Fr_num = Fr(speed, length_scale=length_scale)
    assert_series_equal(Fr_num, expected)


@pytest.mark.core
def test_Froude_Number_2(length_scale=1):
    """
    Mix pd.Series and float as input
    """
    n = 10
    g = 9.81
    speed = pd.Series(np.linspace(1, 10, n))
    expected = pd.Series(speed / np.sqrt(g * length_scale))
    Fr_num = Fr(speed, length_scale=length_scale)
    assert_series_equal(Fr_num, expected)


@pytest.mark.core
def test_Froude_Number_3(length_scale_in=1):
    """
    Different length of the input series
    """
    n = 10
    g = 9.81

    from datetime import datetime

    index = [datetime(2024, 8, 1, hour, 0, 0) for hour in range(1, n + 1, 1)]
    remove_idx = 4
    index_short = index.copy()
    index_short.pop(remove_idx)

    speed = pd.Series(np.linspace(1, 10, n), index=index)
    length_scale_full = length_scale_in * np.ones(n)
    length_scale_short = pd.Series(np.delete(length_scale_full, [remove_idx]), index=index_short)
    length_scale = pd.Series(length_scale_full, index=index)

    Fr_num = Fr(speed, length_scale_short, align_timesteps=True)
    expected = pd.Series(speed / np.sqrt(g * length_scale), index=Fr_num.index)
    assert_series_equal(Fr_num, expected)


@pytest.mark.core
def test_Weber_Number(density_in=1, surface_tension_in=0.1, length_scale_in=10):
    """
    All the input parameters are pd.Series
    """
    n = 10
    speed = pd.Series(np.linspace(1, 10, n))
    density = pd.Series(density_in * np.ones(n))
    surface_tension = pd.Series(surface_tension_in * np.ones(n))
    length_scale = pd.Series(length_scale_in * np.ones(n))
    expected = pd.Series(density * speed * speed * length_scale / surface_tension)
    We_num = We(speed, density, surface_tension, length_scale)
    assert_series_equal(We_num, expected)


@pytest.mark.core
def test_Weber_Number_2(density=1, surface_tension=0.1, length_scale=10):
    """
    Mix pd.Series and float as input
    """
    n = 10
    speed = pd.Series(np.linspace(1, 10, n))
    expected = pd.Series(density * speed * speed * length_scale / surface_tension)
    We_num = We(speed, density, surface_tension, length_scale)
    assert_series_equal(We_num, expected)


@pytest.mark.core
def test_Weber_Number_3(density=1, surface_tension=0.1, length_scale_in=10):
    """
    Input series with different length
    """
    n = 10
    from datetime import datetime

    index = [datetime(2024, 8, 1, hour, 0, 0) for hour in range(1, n + 1, 1)]
    remove_idx = 4
    index_short = index.copy()
    index_short.pop(remove_idx)

    speed = pd.Series(np.linspace(1, 10, n), index=index)

    length_scale_full = length_scale_in * np.ones(n)
    length_scale_short = pd.Series(np.delete(length_scale_full, [remove_idx]), index=index_short)
    length_scale = pd.Series(length_scale_full, index=index)

    We_num = We(speed, density, surface_tension, length_scale_short, align_timesteps=True)
    expected = pd.Series(density * speed * speed * length_scale / surface_tension, index=We_num.index)
    assert_series_equal(We_num, expected)


@pytest.mark.core
def test_Pressure_Scaled(pressure_gradient_in=1, density_in=1, length_scale_in=1):
    """
    All the input parameters are pd.Series
    """
    n = 10
    speed = pd.Series(np.linspace(1, 10, n))
    pressure_gradient = pd.Series(pressure_gradient_in * np.ones(n))
    density = pd.Series(density_in * np.ones(n))
    length_scale = pd.Series(length_scale_in * np.ones(n))
    expected = pd.Series(pressure_gradient * length_scale / (density * speed * speed))
    dPz_num = Pressure_scaled(pressure_gradient, speed, density, length_scale)
    assert_series_equal(dPz_num, expected)


@pytest.mark.core
def test_Pressure_Scaled_2(pressure_gradient=1, density=1, length_scale=1):
    """
    Mix pd.Series and float as input
    """
    n = 10
    speed = pd.Series(np.linspace(1, 10, n))
    expected = pd.Series(pressure_gradient * length_scale / (density * speed * speed))
    dPz_num = Pressure_scaled(pressure_gradient, speed, density, length_scale)
    assert_series_equal(dPz_num, expected)


@pytest.mark.core
def test_Pressure_Scaled_3(pressure_gradient=1, density=1, length_scale_in=1):
    """
    Mix pd.Series and float as input
    """
    n = 10

    from datetime import datetime

    index = [datetime(2024, 8, 1, hour, 0, 0) for hour in range(1, n + 1, 1)]
    remove_idx = 4
    index_short = index.copy()
    index_short.pop(remove_idx)

    speed = pd.Series(np.linspace(1, 10, n), index=index)
    length_scale_full = length_scale_in * np.ones(n)
    length_scale_short = pd.Series(np.delete(length_scale_full, [remove_idx]), index=index_short)
    length_scale = pd.Series(length_scale_full, index=index)

    dPz_num = Pressure_scaled(pressure_gradient, speed, density, length_scale_short, align_timesteps=True)
    expected = pd.Series(pressure_gradient * length_scale / (density * speed * speed), index=dPz_num.index)

    assert_series_equal(dPz_num, expected)


@pytest.mark.core
def test_Roughness_ratio(diameter_in=1):
    """
    All the input parameters are pd.Series
    """
    n = 10
    roughness = pd.Series(np.linspace(0, 1e-4, n))
    diameter = pd.Series(diameter_in * np.ones(n))
    expected = pd.Series(roughness / diameter)
    roughness_scaled_num = Roughness_scaled(roughness, diameter)
    assert_series_equal(roughness_scaled_num, expected)


@pytest.mark.core
def test_Roughness_ratio_2(diameter=1):
    """
    Mix pd.Series and float as input
    """
    n = 10
    roughness = pd.Series(np.linspace(0, 1e-4, n))
    expected = pd.Series(roughness / diameter)
    roughness_scaled_num = Roughness_scaled(roughness, diameter)
    assert_series_equal(roughness_scaled_num, expected)


@pytest.mark.core
def test_Roughness_ratio_3(diameter_in=1):
    """
    Mix pd.Series and float as input
    """
    n = 10

    from datetime import datetime

    index = [datetime(2024, 8, 1, hour, 0, 0) for hour in range(1, n + 1, 1)]
    remove_idx = 4
    index_short = index.copy()
    index_short.pop(remove_idx)

    roughness = pd.Series(np.linspace(0, 1e-4, n), index=index)
    diameter_full = diameter_in * np.ones(n)
    diameter_short = pd.Series(np.delete(diameter_full, [remove_idx]), index=index_short)
    diameter = pd.Series(diameter_full, index=index)

    roughness_scaled_num = Roughness_scaled(roughness, diameter_short, align_timesteps=True)
    expected = pd.Series(roughness / diameter, index=roughness_scaled_num.index)
    assert_series_equal(roughness_scaled_num, expected)


@pytest.mark.core
def test_Froude_Number_density_scaled(density_1_in=100, density_2_in=800, length_scale_in=1):
    """
    All the input parameters are pd.Series
    """
    n = 10
    g = 9.81
    speed = pd.Series(np.linspace(1, 10, n))
    density_1 = pd.Series(density_1_in * np.ones(n))
    density_2 = pd.Series(density_2_in * np.ones(n))
    length_scale = pd.Series(length_scale_in * np.ones(n))
    expected = pd.Series(speed / np.sqrt(g * length_scale * (1 - density_1 / density_2)))

    Fr_rho_caled_num = Fr_density_scaled(speed, density_1=density_1, density_2=density_2, length_scale=length_scale)
    assert_series_equal(Fr_rho_caled_num, expected)


@pytest.mark.core
def test_Froude_Number_density_scaled_2(density_1=100, density_2=800, length_scale=1):
    """
    Mix pd.Series and float as input
    """
    n = 10
    g = 9.81
    speed = pd.Series(np.linspace(1, 10, n))
    expected = pd.Series(speed / np.sqrt(g * length_scale * (1 - density_1 / density_2)))

    Fr_rho_caled_num = Fr_density_scaled(speed, density_1=density_1, density_2=density_2, length_scale=length_scale)
    assert_series_equal(Fr_rho_caled_num, expected)


@pytest.mark.core
def test_Froude_Number_density_scaled_3(density_1=100, density_2=800, length_scale_in=1):
    """
    Different length of the input vectors
    """
    n = 10
    g = 9.81

    from datetime import datetime

    index = [datetime(2024, 8, 1, hour, 0, 0) for hour in range(1, n + 1, 1)]
    remove_idx = 4
    index_short = index.copy()
    index_short.pop(remove_idx)

    speed = pd.Series(np.linspace(1, 10, n), index=index)
    length_scale_full = length_scale_in * np.ones(n)
    length_scale_short = pd.Series(np.delete(length_scale_full, [remove_idx]), index=index_short)
    length_scale = pd.Series(length_scale_full, index=index)

    Fr_rho_scaled_num = Fr_density_scaled(
        speed, density_1=density_1, density_2=density_2, length_scale=length_scale_short, align_timesteps=True
    )
    expected = pd.Series(speed / np.sqrt(g * length_scale * (1 - density_1 / density_2)), index=Fr_rho_scaled_num.index)

    assert_series_equal(Fr_rho_scaled_num, expected)


@pytest.mark.core
def test_Froude_Number_IKH(
    liquid_fraction_in=0.23,
    superficial_velocity_gas_in=1.2,
    superficial_velocity_liquid_in=0.1,
    density_gas_in=100,
    density_liquid_in=800,
    inclination_in=5,
    diameter_in=1,
):
    """
    All the input parameters are pd.Series
    """
    n = 10
    liquid_fraction = pd.Series(np.linspace(0.01, 0.8, n))
    superficial_velocity_gas = pd.Series(superficial_velocity_gas_in * np.ones(n))
    superficial_velocity_liquid = pd.Series(superficial_velocity_liquid_in * np.ones(n))
    density_gas = pd.Series(density_gas_in * np.ones(n))
    density_liquid = pd.Series(density_liquid_in * np.ones(n))
    inclination = pd.Series(inclination_in * np.ones(n))
    diameter = pd.Series(diameter_in * np.ones(n))
    expected = pd.Series(
        [
            0.5168002167485182,
            0.001387971267485002,
            0.01659549696850137,
            0.037649123174695445,
            0.06806486251708459,
            0.11672950898245131,
            0.20238628640998815,
            0.37059091135371824,
            0.7542213059183996,
            1.8644405891732458,
        ]
    )

    Fr_inviscid_kelvin_helmholtz_num = Fr_inviscid_kelvin_helmholtz(
        liquid_fraction,
        superficial_velocity_gas,
        superficial_velocity_liquid,
        density_gas,
        density_liquid,
        inclination,
        diameter,
    )
    assert_series_equal(Fr_inviscid_kelvin_helmholtz_num, expected)


@pytest.mark.core
def test_Froude_Number_IKH_2(
    superficial_velocity_gas=1.2,
    superficial_velocity_liquid=0.1,
    density_gas=100,
    density_liquid=800,
    inclination=5,
    diameter=1,
):
    """
    Mix pd.Series and float as input
    """
    n = 10
    liquid_fraction = pd.Series(np.linspace(0.01, 0.8, n))
    expected = pd.Series(
        [
            0.5168002167485182,
            0.001387971267485002,
            0.01659549696850137,
            0.037649123174695445,
            0.06806486251708459,
            0.11672950898245131,
            0.20238628640998815,
            0.37059091135371824,
            0.7542213059183996,
            1.8644405891732458,
        ]
    )

    Fr_inviscid_kelvin_helmholtz_num = Fr_inviscid_kelvin_helmholtz(
        liquid_fraction,
        superficial_velocity_gas,
        superficial_velocity_liquid,
        density_gas,
        density_liquid,
        inclination,
        diameter,
    )
    assert_series_equal(Fr_inviscid_kelvin_helmholtz_num, expected)


@pytest.mark.core
def test_Froude_Number_IKH_3(
    superficial_velocity_gas=1.2,
    superficial_velocity_liquid=0.1,
    density_gas=100,
    density_liquid=800,
    inclination=5,
    diameter_in=1,
):
    """
    Mix pd.Series and float as input
    """
    n = 10

    from datetime import datetime

    index = [datetime(2024, 8, 1, hour, 0, 0) for hour in range(1, n + 1, 1)]
    remove_idx = 4
    index_short = index.copy()
    index_short.pop(remove_idx)

    liquid_fraction = pd.Series(np.linspace(0.01, 0.8, n), index=index)
    diameter_full = diameter_in * np.ones(n)
    diameter_short = pd.Series(np.delete(diameter_full, [remove_idx]), index=index_short)
    diameter = pd.Series(diameter_full, index=index)

    Fr_inviscid_kelvin_helmholtz_num = Fr_inviscid_kelvin_helmholtz(
        liquid_fraction,
        superficial_velocity_gas,
        superficial_velocity_liquid,
        density_gas,
        density_liquid,
        inclination,
        diameter_short,
        align_timesteps=True,
    )
    expected = pd.Series(
        [
            0.5168002167485182,
            0.001387971267485002,
            0.01659549696850137,
            0.037649123174695445,
            0.06806486251708459,
            0.11672950898245131,
            0.20238628640998815,
            0.37059091135371824,
            0.7542213059183996,
            1.8644405891732458,
        ],
        index=Fr_inviscid_kelvin_helmholtz_num.index,
    )
    assert_series_equal(Fr_inviscid_kelvin_helmholtz_num, expected)


@pytest.mark.core
def test_Froude_2phase(
    liquid_fraction_in=0.23,
    superficial_velocity_gas_in=1.2,
    superficial_velocity_liquid_in=0.1,
    density_gas_in=100,
    density_liquid_in=800,
    inclination_in=5,
    diameter_in=1,
):
    """
    All the input parameters are pd.Series
    """
    n = 10
    liquid_fraction = pd.Series(np.linspace(0.01, 0.8, n))
    superficial_velocity_gas = pd.Series(superficial_velocity_gas_in * np.ones(n))
    superficial_velocity_liquid = pd.Series(superficial_velocity_liquid_in * np.ones(n))
    density_gas = pd.Series(density_gas_in * np.ones(n))
    density_liquid = pd.Series(density_liquid_in * np.ones(n))
    inclination = pd.Series(inclination_in * np.ones(n))
    diameter = pd.Series(diameter_in * np.ones(n))
    expected = pd.Series(
        [
            530.6837812400851,
            1.1764082688663309,
            0.2417650051607793,
            0.1329534720092372,
            0.1311035526988242,
            0.17600971790526265,
            0.2786210560284635,
            0.5020913769743959,
            1.0614458092063905,
            2.919465820710846,
        ]
    )

    Fr_2phase_num = Fr_2phase(
        liquid_fraction,
        superficial_velocity_gas,
        superficial_velocity_liquid,
        density_gas,
        density_liquid,
        inclination,
        diameter,
    )
    assert_series_equal(Fr_2phase_num, expected)


@pytest.mark.core
def test_Froude_2phase_2(
    superficial_velocity_gas=1.2,
    superficial_velocity_liquid=0.1,
    density_gas=100,
    density_liquid=800,
    inclination=5,
    diameter=1,
):
    """
    Mix pd.Series and float as input
    """
    n = 10
    liquid_fraction = pd.Series(np.linspace(0.01, 0.8, n))
    expected = pd.Series(
        [
            530.6837812400851,
            1.1764082688663309,
            0.2417650051607793,
            0.1329534720092372,
            0.1311035526988242,
            0.17600971790526265,
            0.2786210560284635,
            0.5020913769743959,
            1.0614458092063905,
            2.919465820710846,
        ]
    )

    Fr_2phase_num = Fr_2phase(
        liquid_fraction,
        superficial_velocity_gas,
        superficial_velocity_liquid,
        density_gas,
        density_liquid,
        inclination,
        diameter,
    )
    assert_series_equal(Fr_2phase_num, expected)


@pytest.mark.core
def test_Froude_2phase_3(
    superficial_velocity_gas=1.2,
    superficial_velocity_liquid=0.1,
    density_gas=100,
    density_liquid=800,
    inclination=5,
    diameter_in=1,
):
    """
    Mix pd.Series and float as input
    """
    n = 10

    from datetime import datetime

    index = [datetime(2024, 8, 1, hour, 0, 0) for hour in range(1, n + 1, 1)]
    remove_idx = 4
    index_short = index.copy()
    index_short.pop(remove_idx)

    liquid_fraction = pd.Series(np.linspace(0.01, 0.8, n), index=index)
    diameter_full = diameter_in * np.ones(n)
    diameter_short = pd.Series(np.delete(diameter_full, [remove_idx]), index=index_short)
    diameter = pd.Series(diameter_full, index=index)

    Fr_2phase_num = Fr_2phase(
        liquid_fraction,
        superficial_velocity_gas,
        superficial_velocity_liquid,
        density_gas,
        density_liquid,
        inclination,
        diameter_short,
        align_timesteps=True,
    )
    expected = pd.Series(
        [
            530.6837812400851,
            1.1764082688663309,
            0.2417650051607793,
            0.1329534720092372,
            0.1311035526988242,
            0.17600971790526265,
            0.2786210560284635,
            0.5020913769743959,
            1.0614458092063905,
            2.919465820710846,
        ],
        index=Fr_2phase_num.index,
    )

    assert_series_equal(Fr_2phase_num, expected)
