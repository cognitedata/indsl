# Copyright 2023 Cognite AS
import pandas as pd

from indsl.type_check import check_types


@check_types
def Re(speed: pd.Series, density: float, d_viscosity: float, length_scale: float) -> pd.Series:
    """Reynolds Number.

    The Reynolds number is the ratio of inertial forces to viscous forces within a fluid subjected to
    relative internal movement due to different fluid velocities.

    Args:
        speed: Flow speed.
        density: Density.
            Density of the fluid.
        d_viscosity: Dynamic viscosity.
            Dynamic viscosity of the fluid.
        length_scale: Characteristic length.
            Characteristic linear dimension. A characteristic length is an important dimension that defines the scale
            of a physical system. Often, the characteristic length is the volume of a system divided by its surface.

    Returns:
        pandas.Series: Reynolds number
    """
    return speed * density * length_scale / d_viscosity
