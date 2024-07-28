# Copyright 2021 Cognite AS
import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal
import pytest

from indsl.fluid_dynamics import Re, Roughness_scaled
from indsl.fluid_dynamics.friction import Colebrook,friction_factor_laminar, Darcy_friction_factor, pipe_wall_shear_stress, pipe_pressure_gradient,pipe_pressure_drop


@pytest.mark.core
def test_colebrook_equation(density_in=400, d_viscosity_in=0.0005, diameter_in=0.3, roughness_in=1e-6):
    n = 10
    speed = pd.Series(np.power(10,np.linspace(-1, 1, n)))
    density = pd.Series(density_in*np.ones(n))
    d_viscosity = pd.Series(d_viscosity_in*np.ones(n))
    diameter = pd.Series(diameter_in*np.ones(n))
    roughness = pd.Series(roughness_in*np.ones(n))

    reynolds_number = Re(speed, density, d_viscosity, diameter)
    roughness_scaled = Roughness_scaled(roughness,diameter)

    expected = pd.Series( [0.02477638978434275,
                           0.02198522280388084,
                           0.019627346492709717,
                           0.01762047066000431,
                           0.01590097897824371,
                           0.014419332465349731,
                           0.01313680489558208,
                           0.012023163797165308,
                           0.011055032278903839,
                           0.010214732571882303] )

    friction_factor = Colebrook(reynolds_number, roughness_scaled)
    assert_series_equal(friction_factor, expected)

@pytest.mark.core
def test_laminar_friction_factor():
    n = 10
    Re = pd.Series(np.linspace(1, 2300, n))

    expected = 64/Re

    friction_factor = friction_factor_laminar(Re)
    assert_series_equal(friction_factor, expected)


@pytest.mark.core
def test_Darcy_friction_factor(roughness_scaled_in=1e-6):
    n = 10
    Re = pd.Series(np.power(10,np.linspace(-1, 4, n)))
    roughness_scaled = pd.Series(roughness_scaled_in*np.ones(n))

    expected = pd.Series( [640.0,
                           178.08380174125597,
                           49.55287569159212,
                           13.788382016204055,
                           3.836699202041222,
                           1.067584343808038,
                           0.29706168535121774,
                           0.08265917856095249,
                           0.029095115698637902,
                           0.030878684089060478] )
    friction_factor = Darcy_friction_factor(Re,
                                            roughness_scaled,
                                            laminar_limit = 2300.0,
                                            turbulent_limit = 4000.0)
    assert_series_equal(friction_factor, expected)


def test_pipe_wall_shear_stress(density_in=400, d_viscosity_in=0.0005, diameter_in=0.3, roughness_in=1e-6):
    n = 10
    velocity = pd.Series(np.power(10,np.linspace(-1, 1, n)))
    density = pd.Series(density_in*np.ones(n))
    d_viscosity = pd.Series(d_viscosity_in*np.ones(n))
    diameter = pd.Series(diameter_in*np.ones(n))
    roughness = pd.Series(roughness_in*np.ones(n))

    expected = pd.Series( [0.012388194892171378,
                           0.030587594211278556,
                           0.07598370788351963,
                           0.1898107662230114,
                           0.47661932310548355,
                           1.202644811575807,
                           3.0487823456341725,
                           7.764256587323961,
                           19.8648630288629,
                           51.07366285941152] )

    tau = pipe_wall_shear_stress(velocity,
                                 density,
                                 d_viscosity,
                                 diameter,
                                 roughness,
                                 laminar_limit = 2300,
                                 turbulent_limit = 4000)
    assert_series_equal(tau, expected)




def test_pipe_pressure_gradient(density_in=400, d_viscosity_in=0.0005, diameter_in=0.3, roughness_in=1e-6):
    n = 10
    velocity = pd.Series(np.power(10,np.linspace(-1, 1, n)))
    density = pd.Series(density_in*np.ones(n))
    d_viscosity = pd.Series(d_viscosity_in*np.ones(n))
    diameter = pd.Series(diameter_in*np.ones(n))
    roughness = pd.Series(roughness_in*np.ones(n))

    expected = pd.Series( [0.16517593189561838,
                           0.4078345894837141,
                           1.013116105113595,
                           2.530810216306819,
                           6.354924308073114,
                           16.035264154344095,
                           40.6504312751223,
                           103.52342116431947,
                           264.86484038483866,
                           680.9821714588203] )

    dpdz = pipe_pressure_gradient(velocity,
                                  density,
                                  d_viscosity,
                                  diameter,
                                  roughness,
                                  laminar_limit = 2300.0,
                                  turbulent_limit = 4000.0 )
    assert_series_equal(dpdz, expected)



def test_pipe_pressure_drop(density_in=400, d_viscosity_in=0.0005, diameter_in=0.3, roughness_in=1e-6,pipe_length_in = 10,pipe_height_difference_in = 1.1):
    n = 10
    velocity = pd.Series(np.power(10,np.linspace(-1, 1, n)))
    density = pd.Series(density_in*np.ones(n))
    d_viscosity = pd.Series(d_viscosity_in*np.ones(n))
    diameter = pd.Series(diameter_in*np.ones(n))
    roughness = pd.Series(roughness_in*np.ones(n))
    pipe_length = pd.Series(pipe_length_in*np.ones(n))
    pipe_height_difference = pd.Series(pipe_height_difference_in*np.ones(n))

    expected = pd.Series( [4318.051759318957,
                           4320.478345894838,
                           4326.531161051136,
                           4341.708102163068,
                           4379.949243080731,
                           4476.752641543441,
                           4722.904312751223,
                           5351.634211643195,
                           6965.048403848387,
                           11126.221714588202] )


    dp =  pipe_pressure_drop(velocity,
                             density,
                             d_viscosity,
                             diameter,
                             roughness,
                             pipe_length,
                             pipe_height_difference,
                             laminar_limit = 2300.0,
                             turbulent_limit = 4000.0 )
    assert_series_equal(dp, expected)





# @pytest.mark.core
# def test_colebrook_equation(density_in=400, d_viscosity_in=0.0005, diameter_in=0.3, roughness_in=1e-6):
#     n = 10
#     speed = pd.Series(np.power(10,np.linspace(-1, 1, n)))
#     density = pd.Series(density_in*np.ones(n))
#     d_viscosity = pd.Series(d_viscosity_in*np.ones(n))
#     diameter = pd.Series(diameter_in*np.ones(n))
#     roughness = pd.Series(roughness_in*np.ones(n))

#     reynolds_number = Re(speed, density, d_viscosity, diameter)
#     roughness_scaled = Roughness_scaled(roughness,diameter)

#     expected = pd.Series( [0.02477638978434275,
#                            0.02198522280388084,
#                            0.019627346492709717,
#                            0.01762047066000431,
#                            0.01590097897824371,
#                            0.014419332465349731,
#                            0.01313680489558208,
#                            0.012023163797165308,
#                            0.011055032278903839,
#                            0.010214732571882303] )

#     friction_factor = Colebrook(reynolds_number, roughness_scaled)
#     assert_series_equal(friction_factor, expected)
