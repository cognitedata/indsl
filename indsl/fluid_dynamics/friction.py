# Copyright 2023 Cognite AS
from typing import Union

import numpy as np
import pandas as pd

from indsl.resample.auto_align import auto_align
from indsl.type_check import check_types


@check_types
def Haaland(Re: pd.Series, roughness: float) -> pd.Series:
    """Haaland equation.

    The `Haaland equation <https://en.wikipedia.org/wiki/Darcy_friction_factor_formulae#Haaland_equation>`_ was
    proposed in 1983 by Professor S.E. Haaland of the Norwegian Institute of Technology.
    It is used to directly solve the Darcy–Weisbach friction factor for a full-flowing circular pipe. It is an
    approximation of the implicit Colebrook–White equation, but the discrepancy from experimental data is well within
    the accuracy of the data.

    Args:
        Re: Reynolds Number [-].
        roughness: Surface roughness [m].

    Returns:
        pandas.Series: Friction factor [-]
    """
    den = -1.8 * np.log10((6.9 / Re[Re != 0]) + (roughness / 3.7) ** 1.1)
    return 1 / den**2


@check_types
def Colebrook(
    Re: Union[pd.Series, float],
    roughness_scaled: Union[pd.Series, float],
    align_timesteps: bool = False,
) -> pd.Series:
    r"""Colebrook approximation.

    A computationally efficient Colebrook approximation taken from the following article

     "Fast and Accurate Approximations for the Colebrook Equation"
     Dag Biberg
     Journal of Fluids Engineering Copyright VC 2017 by ASME MARCH 2017, Vol. 139 / 031401-1

    This is a faster and more accurat approximation to Colebrook than Haaland

    Args:
        Re: Reynolds Number [-].
        roughness_scaled: Scaled surface roughness [-].
            The wall surface roughness is normally scaled with the pipe inner diameter
        align_timesteps: Auto-align.
            Automatically align time stamp  of input time series. Default is False.

    Returns:
        pandas.Series: Friction factor [-]
    """
    from math import log as m_log

    # Every Series input needs to be autaligned, unless opted out
    if align_timesteps:
        Re, roughness_scaled = auto_align([Re, roughness_scaled], align_timesteps)

    a = 2 / m_log(10)
    b = 2.51
    c = 3.7
    ab = a * b

    logReab = np.log(Re / ab)
    x = logReab + Re * roughness_scaled / (ab * c)
    G = np.log(x) * (1 / x - 1)
    denominator = a * (logReab + G)
    friction_factor = (1 / denominator) ** 2

    if not isinstance(friction_factor, pd.Series):
        friction_factor = pd.Series([friction_factor])
    return friction_factor


@check_types
def friction_factor_laminar(Re: Union[pd.Series, float]) -> pd.Series:
    r"""Friction factor laminar flow.

    friction_factor = :math:`\frac{64}{Re}`
        Re: Reynolds number = :math:`\frac{\rho u D}{\mu}`

    Args:
        Re: Reynolds Number [-].

    Returns:
        pandas.Series: Friction factor laminar flow [-]
    """
    friction_factor = 64 / Re
    if not isinstance(friction_factor, pd.Series):
        friction_factor = pd.Series([friction_factor])
    return friction_factor


@check_types
def __Darcy_friction_factor_point(
    Re: float,
    roughness_scaled: float,
    laminar_limit: float = 2300.0,
    turbulent_limit: float = 4000.0,
) -> float:
    """Computes the Darcy friction factor, including the laminar-turbulent transition.

    Args:
        Re: Reynolds Number [-].
        roughness_scaled: Scaled surface roughness [-].
        laminar_limit: Laminar transition [-].
            Limit where lower Reynolds numbers give pure laminar flow, Typical value is 2300 [-]
        turbulent_limit: Turbulent transition
            Limit where higher Reynolds numbers give pure turbulent flow. Typical value is 4000 [-]

    Returns:
        pandas.Series: Friction factor [-]
    """
    if Re < laminar_limit:
        friction_factor = friction_factor_laminar(Re).iloc[0]
    else:
        friction_factor = Colebrook(Re, roughness_scaled).iloc[0]
        if Re < turbulent_limit:
            friction_laminar = friction_factor_laminar(Re).iloc[0]
            turb_frac = (Re - laminar_limit) / (turbulent_limit - laminar_limit)
            friction_factor = friction_laminar * (1 - turb_frac) + turb_frac * friction_factor
    return friction_factor


@check_types
def Darcy_friction_factor(
    Re: Union[pd.Series, float],
    roughness_scaled: Union[pd.Series, float],
    laminar_limit: float = 2300.0,
    turbulent_limit: float = 4000.0,
    align_timesteps: bool = False,
) -> pd.Series:
    """Darcy friction factor.

    Computes the Darcy friction factor, including the laminar-turbulent transition.

    Args:
        Re: Reynolds Number [-].
        roughness_scaled: Scaled surface roughness [-].
        laminar_limit: Laminar transition [-].
            Limit where lower Reynolds numbers give pure laminar flow, Typical value is 2300 [-]
        turbulent_limit: Turbulent transition
            Limit where higher Reynolds numbers give pure turbulent flow. Typical value is 4000 [-]
        align_timesteps: Auto-align.
            Automatically align time stamp  of input time series. Default is False.

    Returns:
        pandas.Series: Darcy friction factor [-]
    """
    # Every Series input needs to be autaligned, unless opted out
    if align_timesteps:
        Re, roughness_scaled = auto_align([Re, roughness_scaled], align_timesteps)
    # Check if all input is float
    if not isinstance(Re, pd.Series) and not isinstance(roughness_scaled, pd.Series):
        darcy_friction_factor = pd.Series(
            [__Darcy_friction_factor_point(Re, roughness_scaled, laminar_limit, turbulent_limit)]
        )
    else:
        if isinstance(Re, pd.Series):
            shape = Re.shape[0]
            index = Re.index
        elif isinstance(roughness_scaled, pd.Series):
            shape = roughness_scaled.shape[0]
            index = roughness_scaled.index

        darcy_friction_factor = pd.Series(np.zeros(shape), index=index)
        for i in range(shape):

            if isinstance(Re, pd.Series):
                Re_ = Re.iloc[i]
            else:
                Re_ = Re
            if isinstance(roughness_scaled, pd.Series):
                roughness_ = roughness_scaled.iloc[i]
            else:
                roughness_ = roughness_scaled

            darcy_friction_factor.iloc[i] = __Darcy_friction_factor_point(
                Re_, roughness_, laminar_limit, turbulent_limit
            )
    return darcy_friction_factor


@check_types
def __Darcy_friction_factor_dimensional(
    velocity: Union[pd.Series, float],
    density: Union[pd.Series, float],
    d_viscosity: Union[pd.Series, float],
    diameter: Union[pd.Series, float],
    roughness: Union[pd.Series, float],
    laminar_limit: float = 2300.0,
    turbulent_limit: float = 4000.0,
) -> pd.Series:
    r"""Computes the Darcy friction factor, including the laminar-turbulent transition.

    The input is on dimensional form

    Args:
        velocity: Average fluid velocity [:math:`\mathrm{\frac{m}{s}}`].
        density: Fluid density [:math:`\mathrm{\frac{kg}{m^3}}`].
        d_viscosity: Dynamic viscosity [:math:`\mathrm{\frac{kg}{m s}}`].
        diameter: Pipe inner diameter [:math:`\mathrm{m}}`].
        roughness: Unscaled surface roughness [:math:`\mathrm{m}`].
            Unscaled pipe inner wall surface roughness.
        laminar_limit: Laminar transition [-].
            Limit where lower Reynolds numbers give pure laminar flow, Typical value is 2300 [-]
        turbulent_limit: Trubulent transition.
            Limit where higher Reynolds numbers give pure turbulent flow. Typical value is 4000 [-]

    Returns:
        pandas.Series: Darcy friction factor [-]
    """
    from indsl.fluid_dynamics import Re, Roughness_scaled

    reynolds_number = Re(velocity, density, d_viscosity, diameter)
    roughness_scaled = Roughness_scaled(roughness, diameter)

    return Darcy_friction_factor(reynolds_number, roughness_scaled, laminar_limit, turbulent_limit)


@check_types
def pipe_wall_shear_stress(
    velocity: Union[pd.Series, float],
    density: Union[pd.Series, float],
    d_viscosity: Union[pd.Series, float],
    diameter: Union[pd.Series, float],
    roughness: Union[pd.Series, float],
    laminar_limit: float = 2300.0,
    turbulent_limit: float = 4000.0,
    align_timesteps: bool = False,
) -> pd.Series:
    r"""Single phase wall shear stress.

    Computes the wall shear stress for single phase flow.

        :math:`\tau = \lambda\frac{u^2 D}{8}`

            :math:`\tau`: Wall shear stress [:math:`\mathrm{Pa}`]

            :math:`x`: Direction along pipeline [:math:`\mathrm{m}`]

            :math:`\lambda`: Friction factor [:math:`\mathrm{-}`]

            :math:`u`: Fluid velocity [:math:`\mathrm{\frac{m}{s}}`]

            :math:`D`: Pipe diameter [:math:`\mathrm{m}`]


    Args:
        velocity: Average fluid velocity [:math:`\mathrm{\frac{m}{s}}`].
        density: Fluid density [:math:`\mathrm{\frac{kg}{m^3}}`].
        d_viscosity: Dynamic viscosity [:math:`\mathrm{\frac{kg}{m s}}`].
        diameter: Pipe inner diameter [:math:`\mathrm{m}`].
        roughness: Unscaled surface roughness [:math:`\mathrm{m}`].
            Unscaled pipe inner wall surface roughness.
        laminar_limit: Laminar transition [-].
            Limit where lower Reynolds numbers give pure laminar flow, Typical value is 2300 [-].
        turbulent_limit: Trubulent transition.
            Limit where higher Reynolds numbers give pure turbulent flow. Typical value is 4000 [-].
        align_timesteps: Auto-align.
            Automatically align time stamp  of input time series. Default is False.

    Returns:
        pandas.Series: Pipe wall shear stress [:math:`\mathrm{Pa}`]
    """
    # Every Series input needs to be autaligned, unless opted out
    if align_timesteps:
        velocity, density, d_viscosity, diameter, roughness = auto_align(
            [velocity, density, d_viscosity, diameter, roughness], align_timesteps
        )

    friction_factor = __Darcy_friction_factor_dimensional(
        velocity, density, d_viscosity, diameter, roughness, laminar_limit, turbulent_limit
    )
    w_shear_stress = density * friction_factor * velocity**2 / 8.0
    return w_shear_stress


@check_types
def pipe_pressure_gradient(
    velocity: Union[pd.Series, float],
    density: Union[pd.Series, float],
    d_viscosity: Union[pd.Series, float],
    diameter: Union[pd.Series, float],
    roughness: Union[pd.Series, float],
    laminar_limit: float = 2300.0,
    turbulent_limit: float = 4000.0,
    align_timesteps: bool = False,
) -> pd.Series:
    r"""Single phase pressure gradient.

    Computes the pressure gradient from wall friction in single phase flow in pipe.

        :math:`\frac{\mathrm{d}P}{\mathrm{d}x} = \lambda\frac{u^2 D}{2}`

            :math:`P`: Pressure [:math:`\mathrm{Pa}`]

            :math:`x`: Direction along pipeline [:math:`\mathrm{m}`]

            :math:`\lambda`: Friction factor [:math:`\mathrm{-}`]

            :math:`u`: Fluid velocity [:math:`\mathrm{\frac{m}{s}}`]

            :math:`D`: Pipe diameter [:math:`\mathrm{m}`]


    Args:
        velocity: Average fluid velocity [:math:`\mathrm{\frac{m}{s}}`].
        density: Fluid density [:math:`\mathrm{\frac{kg}{m^3}}`].
        d_viscosity: Dynamic viscosity [:math:`\mathrm{\frac{kg}{m s}}`].
        diameter: Pipe inner diameter [:math:`\mathrm{m}`].
        roughness: Unscaled surface roughness [:math:`\mathrm{m}`].
            Unscaled pipe inner wall surface roughness.
        laminar_limit: Laminar transition [-].
            Limit where lower Reynolds numbers give pure laminar flow, Typical value is 2300 [-].
        turbulent_limit: Trubulent transition.
            Limit where higher Reynolds numbers give pure turbulent flow. Typical value is 4000 [-].
        align_timesteps: Auto-align.
            Automatically align time stamp  of input time series. Default is False.

    Returns:
        pandas.Series: Fluid pressure gradient [:math:`\mathrm{\frac{Pa}{m}}`]
    """
    # Every Series input needs to be autaligned, unless opted out
    if align_timesteps:
        velocity, density, d_viscosity, diameter, roughness = auto_align(
            [velocity, density, d_viscosity, diameter, roughness], align_timesteps
        )

    w_shear_stress = pipe_wall_shear_stress(
        velocity, density, d_viscosity, diameter, roughness, laminar_limit, turbulent_limit
    )
    dpdz = 4 * w_shear_stress / diameter
    return dpdz


@check_types
def pipe_pressure_drop(
    velocity: Union[pd.Series, float],
    density: Union[pd.Series, float],
    d_viscosity: Union[pd.Series, float],
    diameter: Union[pd.Series, float],
    roughness: Union[pd.Series, float],
    pipe_length: Union[pd.Series, float],
    pipe_height_difference: Union[pd.Series, float],
    laminar_limit: float = 2300.0,
    turbulent_limit: float = 4000.0,
    align_timesteps: bool = False,
) -> pd.Series:
    r"""Single phase pressure drop.

    Computes the pressure drop for single phase flow in a pipe, adding frictional pressure drop and gravitational pressure drop

        It assumes constant properties along pipeline

    Args:
        velocity: Average fluid velocity [:math:`\mathrm{\frac{m}{s}}`].
        density: Fluid density [:math:`\mathrm{\frac{kg}{m^3}}`].
        d_viscosity: Dynamic viscosity [:math:`\mathrm{\frac{kg}{m s}}`].
        diameter: Pipe inner diameter [:math:`\mathrm{m}`].
        roughness: Unscaled surface roughness [:math:`\mathrm{m}`].
            Unscaled pipe inner wall surface roughness.
        pipe_length: total length of pipe [:math:`\mathrm{m}`].
        pipe_height_difference: Height difference [:math:`\mathrm{m}`].
            Difference in height between start and end of pipe.
        laminar_limit: Laminar transition [-].
            Limit where lower Reynolds numbers give pure laminar flow, Typical value is 2300 [-].
        turbulent_limit: Trubulent transition.
            Limit where higher Reynolds numbers give pure turbulent flow. Typical value is 4000 [-].
        align_timesteps: Auto-align.
            Automatically align time stamp  of input time series. Default is False.

    Returns:
        pandas.Series: Pipe pressure drop [:math:`\mathrm{Pa}`]
    """
    from indsl.fluid_dynamics.constants import acceleration_gravity

    # Every Series input needs to be autaligned, unless opted out
    if align_timesteps:
        velocity, density, d_viscosity, diameter, roughness, pipe_length, pipe_height_difference = auto_align(
            [velocity, density, d_viscosity, diameter, roughness, pipe_length, pipe_height_difference], align_timesteps
        )

    dP_gravity = density * acceleration_gravity * pipe_height_difference
    dP_fric = pipe_length * pipe_pressure_gradient(
        velocity, density, d_viscosity, diameter, roughness, laminar_limit, turbulent_limit
    )
    return dP_gravity + dP_fric
