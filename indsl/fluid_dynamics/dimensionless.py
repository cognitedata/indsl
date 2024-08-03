# Copyright 2024 Cognite AS
from typing import Tuple, Union

import numpy as np
import pandas as pd

from indsl.resample.auto_align import auto_align
from indsl.type_check import check_types


@check_types
def Re(
    velocity: Union[pd.Series, float],
    density: Union[pd.Series, float],
    d_viscosity: Union[pd.Series, float],
    length_scale: Union[pd.Series, float],
    align_timesteps: bool = False,
) -> pd.Series:
    """Reynolds Number.

    The Reynolds number is the ratio of inertial forces to viscous forces within a fluid subjected to
    relative internal movement due to different fluid velocities.

    Re = velocity * density * length_scale / d_viscosity

    Args:
        velocity: Fluid velocity [m/s].
        density: Density [kg/m3].
            Density of the fluid.
        d_viscosity: Dynamic viscosity [kg/ms].
            Dynamic viscosity of the fluid.
        length_scale: Characteristic length [m].
            Characteristic linear dimension. A characteristic length is an important dimension that defines the scale
            of a physical system. Often, the characteristic length is the volume of a system divided by its surface.
        align_timesteps: Auto-align.
            Automatically align time stamp  of input time series. Default is False.

    Returns:
        pandas.Series: Reynolds number [-]
    """
    # Every Series input needs to be autaligned, unless opted out
    if align_timesteps:
        velocity, density, d_viscosity, length_scale = auto_align(
            [velocity, density, d_viscosity, length_scale], align_timesteps
        )

    Re_ = velocity * density * length_scale / d_viscosity
    # If output is not pd.Series we construct a series
    if not isinstance(Re_, pd.Series):
        Re_ = pd.Series(Re_)
    return Re_


@check_types
def Fr(
    velocity: Union[pd.Series, float], length_scale: Union[pd.Series, float], align_timesteps: bool = False
) -> pd.Series:
    r"""Froude Number.

    The Froude number is a ratio of inertial and gravitational forces

    :math:`Fr = \frac{u}{\sqrt{g L}}`

        u: velocity :math:`\mathrm{\frac{m}{s}}`

        g: 9.81 acceleration due to gravity [:math:`\mathrm{\frac{m}{s^2}}`]

        L: Length scale [:math:`\mathrm{m}`]



    Args:
        velocity: Fluid velocity [:math:`\mathrm{\frac{m}{s}}`].
        length_scale: Characteristic length [:math:`\mathrm{m}`].
            Characteristic linear dimension. A characteristic length is an important dimension that defines the scale
            of a physical system. Often, the characteristic length is the volume of a system divided by its surface.
        align_timesteps: Auto-align.
            Automatically align time stamp  of input time series. Default is False.

    Returns:
        pandas.Series: Froude number [-]
    """
    from indsl.fluid_dynamics.constants import acceleration_gravity

    # Every Series input needs to be autaligned, unless opted out
    if align_timesteps:
        velocity, length_scale = auto_align([velocity, length_scale], align_timesteps)

    Fr_ = velocity / np.sqrt(acceleration_gravity * length_scale)
    if not isinstance(Fr_, pd.Series):
        Fr_ = pd.Series(Fr_)
    return Fr_


@check_types
def Fr_density_scaled(
    velocity: Union[pd.Series, float],
    density_1: Union[pd.Series, float],
    density_2: Union[pd.Series, float],
    length_scale: Union[pd.Series, float],
    align_timesteps: bool = False,
) -> pd.Series:
    r"""Density scaled Froude Number.

    The Froude number is a ratio of inertial and gravitational forces.
    The density scaled Fround number is typically used in two phase flow

    :math:`Fr_{\rho-scaled} =\frac{u}{\sqrt{g L (1 - \frac{\rho_1}{\rho_2})}}`

        u: velocity :math:`\mathrm{\frac{m}{s}}`

        g: 9.81 acceleration due to gravity [:math:`\mathrm{\frac{m}{s^2}}`]

        L: Length scale [:math:`\mathrm{m}`]

        :math:`\rho_1`: Density lighter fluid [:math:`\mathrm{\frac{kg}{m^3}}`]

        :math:`\rho_2`: Density heavier fluid [:math:`\mathrm{\frac{kg}{m^3}}`]

    Args:
        velocity: Average fluid velocity [:math:`\mathrm{\frac{m}{s}}`].
        density_1: Density lighter fluid [:math:`\mathrm{\frac{kg}{m^3}}`].
        density_2: Density heavier fluid [:math:`\mathrm{\frac{kg}{m^3}}`].
        length_scale: Characteristic length [:math:`\mathrm{m}`].
            Characteristic linear dimension. For pipe flow, the characteristic length is normally the pipe diameter
        align_timesteps: Auto-align.
            Automatically align time stamp  of input time series. Default is False.

    Returns:
        pandas.Series: Density scaled Froude number [-]
    """
    from indsl.fluid_dynamics.constants import acceleration_gravity

    # Every Series input needs to be autaligned, unless opted out
    if align_timesteps:
        velocity, density_1, density_2, length_scale = auto_align(
            [velocity, density_1, density_2, length_scale], align_timesteps
        )

    Fr_ = velocity / np.sqrt(acceleration_gravity * length_scale * (1 - density_1 / density_2))
    if not isinstance(Fr_, pd.Series):
        Fr_ = pd.Series(Fr_)
    return Fr_


@check_types
def __Fr_2phase_base(
    liquid_fraction: Union[pd.Series, float],
    superficial_velocity_gas: Union[pd.Series, float],
    superficial_velocity_liquid: Union[pd.Series, float],
    inclination: Union[pd.Series, float],
    diameter: Union[pd.Series, float],
) -> Tuple[
    Union[pd.Series, float],
    Union[pd.Series, float],
    Union[pd.Series, float],
    Union[pd.Series, float],
    Union[pd.Series, float],
]:
    """Geometry calculations used to calculate various 2phase Froude numbers.

    Args:
        liquid_fraction: Volume fraction of the liquid [-].
            The fluid fraction is a scaled value, so gas_fraction + liquid_fraction = 1
        superficial_velocity_gas: Gas superficial velocity [m/s].
        superficial_velocity_liquid: Liquid superficial velocity [m/s].
        inclination: Pipe inclination [degrees].
        diameter: Pipe inner diameter [m].

    Returns:
        pandas.Series: velocity_gas, velocity_liquid, height_gas, height_liquid, cos_inclination
    """
    eps = 1e-15  # used to avoid division by zero

    velocity_gas = superficial_velocity_gas / np.clip(1 - liquid_fraction, eps, 1)  # Prevent deviding by zero
    velocity_liquid = superficial_velocity_liquid / np.clip(liquid_fraction, eps, 1)  # Prevent deviding by zero
    wetted_angle = fdelta_v(liquid_fraction)  # Wetted angle, gas-liquid interface
    interface_length = diameter * np.sin(wetted_angle)  # Interface length
    interface_length_inverse = 1.0 / np.clip(interface_length, eps, np.pi * diameter)
    area = 0.25 * np.pi * diameter * diameter  # Pipe area
    area_gas = (1 - liquid_fraction) * area  # Area of gas
    area_liquid = liquid_fraction * area  # Area of liquid
    height_gas = area_gas * interface_length_inverse
    height_liquid = area_liquid * interface_length_inverse
    inclination_rad = inclination / 180.0 * np.pi
    cos_inclination = np.cos(inclination_rad)  # cosine of pipe inclination

    return (velocity_gas, velocity_liquid, height_gas, height_liquid, cos_inclination)


@check_types
def Fr_2phase(
    liquid_fraction: Union[pd.Series, float],
    superficial_velocity_gas: Union[pd.Series, float],
    superficial_velocity_liquid: Union[pd.Series, float],
    density_gas: Union[pd.Series, float],
    density_liquid: Union[pd.Series, float],
    inclination: Union[pd.Series, float],
    diameter: Union[pd.Series, float],
    align_timesteps: bool = False,
) -> pd.Series:
    r"""2 phase Froude Number.

    The Froude number is a ratio of inertial and gravitational forces.
    This calculated a Froude number for a two phase pipe flow situation.

    Args:
        liquid_fraction: Volume fraction of liquid [-].
            The fluid fraction is a scaled value, so gas_fraction + liquid_fraction = 1.
        superficial_velocity_gas: Gas superficial velocity [:math:`\mathrm{\frac{m}{s}}`].
            The superficial flow is defined as the hypothetical flow velocity had the phase covered the entire flow area.
            :math:`US_{phase} = \frac{Q_{phase}}{\alpha_{phase}}`. Superficial flow velocity of the phase.
            :math:`Q_{phase}`: Volume flow velocity of the phase.
            :math:`\alpha_{phase}`: Flow area covered by the phase.
        superficial_velocity_liquid: Liquid superficial velocity [:math:`\mathrm{\frac{m}{s}}`].
        density_gas: Density of the lighter fluid [:math:`\mathrm{\frac{kg}{m^3}}`].
        density_liquid: Density of the denser fluid [:math:`\mathrm{\frac{kg}{m^3}}`].
        inclination: Pipe inclination [:math:`\mathrm{deg}`].
        diameter: Pipe inner diameter [:math:`\mathrm{m}`].
        align_timesteps: Auto-align.
            Automatically align time stamp  of input time series. Default is False.

    Returns:
        pandas.Series: 2 phase Froude number [-]
    """
    from indsl.fluid_dynamics.constants import acceleration_gravity

    # Every Series input needs to be autaligned, unless opted out
    if align_timesteps:
        (
            liquid_fraction,
            superficial_velocity_gas,
            superficial_velocity_liquid,
            density_gas,
            density_liquid,
            inclination,
            diameter,
        ) = auto_align(
            [
                liquid_fraction,
                superficial_velocity_gas,
                superficial_velocity_liquid,
                density_gas,
                density_liquid,
                inclination,
                diameter,
            ],
            align_timesteps,
        )

    velocity_gas, velocity_liquid, height_gas, height_liquid, cos_inclination = __Fr_2phase_base(
        liquid_fraction, superficial_velocity_gas, superficial_velocity_liquid, inclination, diameter
    )

    fluid_density_difference = density_liquid - density_gas

    Fr_2P = (
        density_gas * velocity_gas * velocity_gas / height_gas
        + density_liquid * velocity_liquid * velocity_liquid / height_liquid
    ) / (fluid_density_difference * acceleration_gravity * cos_inclination)

    if not isinstance(Fr_2P, pd.Series):
        Fr_2P = pd.Series(Fr_2P)
    return Fr_2P


@check_types
def Fr_inviscid_kelvin_helmholtz(
    liquid_fraction: Union[pd.Series, float],
    superficial_velocity_gas: Union[pd.Series, float],
    superficial_velocity_liquid: Union[pd.Series, float],
    density_gas: Union[pd.Series, float],
    density_liquid: Union[pd.Series, float],
    inclination: Union[pd.Series, float],
    diameter: Union[pd.Series, float],
    align_timesteps: bool = False,
) -> pd.Series:
    r"""IKH Froude Number.

    Invicid Kelvin Helmholtz Froude number.
    The Froude number is a ratio of inertial and gravitational forces.

    Args:
        liquid_fraction: Volume fraction of liquid [-].
            The fluid fraction is a scaled value, so gas_fraction + liquid_fraction = 1.
        superficial_velocity_gas: Gas superficial velocity [:math:`\mathrm{\frac{m}{s}}`].
            The superficial flow is defined as the hypothetical flow velocity had the phase covered the entire flow area.
            :math:`US_{phase} = \frac{Q_{phase}}{\alpha_{phase}}`. Superficial flow velocity of the phase.
            :math:`Q_{phase}`: Volume flow velocity of the phase.
            :math:`\alpha_{phase}`: Flow area covered by the phase.
        superficial_velocity_liquid: Liquid superficial speed [:math:`\mathrm{\frac{m}{s}}`].
        density_gas: Density of the lighter fluid [:math:`\mathrm{\frac{kg}{m^3}}`].
        density_liquid: Density of the denser fluid [:math:`\mathrm{\frac{kg}{m^3}}`].
        inclination: Pipe inclination [:math:`\mathrm{deg}`].
        diameter: Pipe inner diameter [:math:`\mathrm{m}`].
        align_timesteps: Auto-align.
            Automatically align time stamp  of input time series. Default is False.

    Returns:
        pandas.Series: IKH Froude number [-]
    """
    from indsl.fluid_dynamics.constants import acceleration_gravity

    # Every Series input needs to be autaligned, unless opted out
    if align_timesteps:
        (
            liquid_fraction,
            superficial_velocity_gas,
            superficial_velocity_liquid,
            density_gas,
            density_liquid,
            inclination,
            diameter,
        ) = auto_align(
            [
                liquid_fraction,
                superficial_velocity_gas,
                superficial_velocity_liquid,
                density_gas,
                density_liquid,
                inclination,
                diameter,
            ],
            align_timesteps,
        )

    velocity_gas, velocity_liquid, height_gas, height_liquid, cos_inclination = __Fr_2phase_base(
        liquid_fraction, superficial_velocity_gas, superficial_velocity_liquid, inclination, diameter
    )

    fluid_density_difference = density_liquid - density_gas
    velocity_relative = velocity_gas - velocity_liquid

    IKH_Fr = (density_gas * density_liquid * velocity_relative * velocity_relative) / (
        fluid_density_difference
        * acceleration_gravity
        * cos_inclination
        * (height_liquid * density_gas + height_gas * density_liquid)
    )
    if not isinstance(IKH_Fr, pd.Series):
        IKH_Fr = pd.Series(IKH_Fr)
    return IKH_Fr


@check_types
def fdelta(alpha: float) -> float:
    """Computes the wetted angle based on the volume fraction value alpha.

    This is a computationally efficient implementation.

    Args:
        alpha: Volume fraction of the liquid [-].

    Returns:
        float: wetted angle of the fluid interface [0,pi]
    """
    import math

    PI = math.pi
    TP = math.pow(3 * PI / 2.0, 1 / 3.0)  # 1.6765391932197436951D0 !
    XLOW = 0.05  # LOWER LIMIT FOR RATIONAL INTERPOLATION
    XHIG = 0.95  # UPPER LIMIT FOR RATIONAL INTERPOLATION
    x = min(max(alpha, 0), 1)
    # --------------------------------------------------------------------
    #     USE RATIONAL INTERPOLATION FOR X IN THE RANGE [0.05,0.95] AND
    #     ONE SIDED APPROXIMATIONS OTHERWISE
    # --------------------------------------------------------------------
    fdelta_ = 0.0
    if x >= XLOW and x <= XHIG:
        fdelta_ = (
            0.22311023546359054
            + x
            * (
                33.53136310032857
                + x
                * (
                    472.31580494880603
                    + x
                    * (
                        314.52638316000747
                        + x
                        * (
                            -2263.6891041902054
                            + x * (1412.0092457091596 + (318.92871443055304 - 284.9270353669199 * x) * x)
                        )
                    )
                )
            )
        ) / (
            1.0
            + x
            * (
                59.6564942725274
                + x
                * (
                    390.4489547532089
                    + x
                    * (
                        -468.3816299878848
                        + x
                        * (
                            -845.382403233362
                            + x * (1295.4879014148025 + (7.64038372157576e-6 * x - 431.82932499439727) * x)
                        )
                    )
                )
            )
        )
    elif x <= XLOW:
        fdelta_ = TP * math.pow(x, 1.0 / 3.0) + x * (
            0.31610026909392350 + x * (0.57398508785775230 + x * (36.182387394492770 * x - 5.0674541402353440))
        )
    else:  # X > XHIG
        x = 1.0 - x
        fdelta_ = TP * math.pow(x, 1.0 / 3.0) + x * (
            0.31610026909392350 + x * (0.57398508785775230 + x * (36.182387394492770 * x - 5.0674541402353440))
        )
        fdelta_ = PI - fdelta_
    return fdelta_


@check_types
def fdeltas(alpha: float) -> float:
    """Computes the scaled wetted angle based on the volume fraction value alpha.

    Args:
        alpha: Volume fraction of the liquid [-].

    Returns:
        float: scaled wetted angle of the fluid interface [0,1]
    """
    import math

    return fdelta(alpha) / math.pi


@check_types
def fdelta_v(alpha: pd.Series) -> pd.Series:
    """Computes the wetted angle based on the volume fraction value alpha.

        Vectorized function.

    Args:
        alpha: Volume fraction of the liquid [-].

    Returns:
        pd.Series: wetted angle of the fluid interface [0,pi]
    """
    # import math
    from numpy import vectorize

    fdelta_v = vectorize(fdelta)
    fdelta_ = fdelta_v(alpha)
    return pd.Series(fdelta_, index=alpha.index)


@check_types
def fdeltas_v(alpha: pd.Series) -> pd.Series:
    """Computes the scaled wetted angle based on the volume fraction value alpha.

        Vectorized function.

    Args:
        alpha: Volume fraction of the liquid [-].

    Returns:
        float: scaled wetted angle of the fluid interface [0,1]
    """
    import math

    return fdelta_v(alpha) / math.pi


@check_types
def We(
    velocity: Union[pd.Series, float],
    density: Union[pd.Series, float],
    surface_tension: Union[pd.Series, float],
    length_scale: Union[pd.Series, float],
    align_timesteps: bool = False,
) -> pd.Series:
    r"""Weber Number.

    The Weber number describes the ratio between deforming inertial forces and stabilizing cohesive forces
    for liquids flowing through a fluid medium.
    For example, the Weber number characterizes the atomizing quality of a spray and the resulting droplet size when producing emulsions.

    :math:`We = \frac{\rho u^2 L}{\sigma}`

        :math:`\rho`: Fluid density [:math:`\mathrm{\frac{kg}{m^3}}`]

        :math:`u`: Fluid velocity [:math:`\mathrm{\frac{m}{s}}`]

        :math:`L`: Lenght scale [:math:`\mathrm{m}`]

        :math:`\sigma`: Surface tension [:math:`\mathrm{\frac{N}{m}}`]

    Args:
        velocity: Flow speed [:math:`\mathrm{\frac{m}{s}}`].
        density: Density [:math:`\mathrm{\frac{kg}{m^3}}`].
            Density of the fluid.
        surface_tension: Surface tension [:math:`\mathrm{\frac{N}{m}}`].
            Surface tension between the current fluid (the spesified density) and the surrounding fluid
        length_scale: Characteristic length [:math:`\mathrm{m}`].
            Characteristic linear dimension. A characteristic length is an important dimension that defines the scale
            of a physical system. Often, the characteristic length is the volume of a system divided by its surface.
        align_timesteps: Auto-align.
            Automatically align time stamp  of input time series. Default is False.

    Returns:
        pandas.Series: Weber number [-]
    """
    # Every Series input needs to be autaligned, unless opted out
    if align_timesteps:
        velocity, density, surface_tension, length_scale = auto_align(
            [velocity, density, surface_tension, length_scale], align_timesteps
        )

    We_ = density * velocity * velocity * length_scale / surface_tension
    if not isinstance(We_, pd.Series):
        We_ = pd.Series(We_)
    return We_


@check_types
def Pressure_scaled(
    pressure_gradient: Union[pd.Series, float],
    velocity: Union[pd.Series, float],
    density: Union[pd.Series, float],
    length_scale: Union[pd.Series, float],
    align_timesteps: bool = False,
) -> pd.Series:
    r"""Scaled pressure gradient.

    Pressure gradient on a dimentionless form

    :math:`PI = \frac{\mathrm{d}P}{\mathrm{d}x}\frac{L}{\rho u^2}`

        :math:`\frac{\mathrm{d}P}{\mathrm{d}x}`: Pressure gradient [:math:`\mathrm{\frac{Pa}{m}}`]

        :math:`u`: Fluid velocity [:math:`\mathrm{\frac{m}{s}}`]

        :math:`L`: Lenght scale [:math:`\mathrm{m}`]

        :math:`\rho`: Fluid density [:math:`\mathrm{\frac{kg}{m^3}}`]

    Args:
        pressure_gradient: Pressure gradient [:math:`\mathrm{\frac{Pa}{m}}`].
            Presure gradient in the flow direction (along the pipe, assuming pipe flow)
        velocity: Flow velocity [:math:`\mathrm{\frac{m}{s}}`].
        density: Density [:math:`\mathrm{\frac{kg}{m^3}}`].
        length_scale: Characteristic length [:math:`\mathrm{m}`].
            Characteristic linear dimension. For pipe flow, the characteristic length is normally the pipe diameter
        align_timesteps: Auto-align.
            Automatically align time stamp  of input time series. Default is False.

    Returns:
        pandas.Series: Scaled pressure gradient [-]
    """
    # Every Series input needs to be autaligned, unless opted out
    if align_timesteps:
        pressure_gradient, velocity, density, length_scale = auto_align(
            [pressure_gradient, velocity, density, length_scale], align_timesteps
        )

    dP_scaled = pressure_gradient * length_scale / (density * velocity * velocity)
    if not isinstance(dP_scaled, pd.Series):
        dP_scaled = pd.Series(dP_scaled)
    return dP_scaled


@check_types
def Roughness_scaled(
    roughness: Union[pd.Series, float],
    diameter: Union[pd.Series, float],
    align_timesteps: bool = False,
) -> pd.Series:
    """Scaled pipe roughness.

    This is the roughness that is used in the Darcy-Weisbach equation

    roughness_scaled = roughness / diameter

    Args:
        roughness: Pipe inner roughness [m].
            Pipe inner roughness as sand size equivalent
        diameter: Pipe diameter [m].
        align_timesteps: Auto-align.
            Automatically align time stamp  of input time series. Default is False.

    Returns:
        pandas.Series: Scaled pipe roughness number [-]
    """
    # Every Series input needs to be autaligned, unless opted out
    if align_timesteps:
        roughness, diameter = auto_align([roughness, diameter], align_timesteps)
    roughness_scaled = roughness / diameter
    if not isinstance(roughness_scaled, pd.Series):
        roughness_scaled = pd.Series(roughness_scaled)
    return roughness_scaled
