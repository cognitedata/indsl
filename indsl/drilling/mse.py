# Copyright 2026 Cognite AS

import numpy as np
import pandas as pd
from indsl.type_check import check_types

@check_types
def mse(
    torque: pd.Series,
    rpm: pd.Series,
    wob: pd.Series,
    rop: pd.Series,
    bs: pd.Series,
) -> pd.Series:
    r"""Mechanical Specific Energy.

    Calculates the Mechanical Specific Energy (MSE) for drilling operations. MSE is a measure of the energy
    required to remove a unit volume of rock and is used to evaluate drilling efficiency.

    The formula for MSE in metric units:

    .. math::
        \mathrm{MSE} = \frac{T \cdot \omega + \mathrm{WOB} \cdot \mathrm{ROP}}{A \cdot \mathrm{ROP}}

    Where:
    - :math:`T` is the torque [N.m]
    - :math:`\omega` is the angular velocity [rad/s] = :math:`\mathrm{RPM} \cdot \frac{2\pi}{60}`
    - :math:`\mathrm{WOB}` is the weight on bit [N]
    - :math:`\mathrm{ROP}` is the rate of penetration [:math:`\mathrm{m/h}`]
    - :math:`A` is the bit area [:math:`\mathrm{m^2}`]

    This can be simplified to:

    .. math::
        \mathrm{MSE} = \frac{T \cdot \mathrm{RPM} \cdot \frac{2\pi}{60} + \mathrm{WOB} \cdot \mathrm{ROP}}{A \cdot \mathrm{ROP}}

    For more information on Mechanical Specific Energy derivation, see:
    `Mechanical Specific Energy Derivation <https://onepetro.org/SJ/article/30/10/5956/787864/Mechanical-Specific-Energy-Derivation>`_.

    Args:
        torque: Torque [N.m].
            Time series with the torque applied to the drill string.
        rpm: Rotational velocity [rpm].
            Time series with the rotational velocity of the drill string in revolutions per minute.
        wob: Weight on bit [N].
            Time series with the weight applied to the drill bit in Newtons.
        rop: Rate of penetration [:math:`\mathrm{m/h}`].
            Time series with the rate of penetration in meters per hour.
        bs: Bit size area [:math:`\mathrm{m^2}`].
            Time series with the bit size in [m2]

    Returns:
        pandas.Series: Mechanical Specific Energy [:math:`\mathrm{J/m^3}` or :math:`\mathrm{Pa}`].
            Time series with the calculated MSE values. MSE represents the energy required to remove a unit volume of rock.
            Returns NaN values where any input is NaN or where division by zero would occur (e.g., when bit area or ROP is zero or negative).
    """
    # Convert ROP from m/h to m/s for calculation
    rop_m_per_s = rop / 3600.0

    # Calculate angular velocity in rad/s: omega = RPM * 2*pi / 60
    angular_velocity = rpm * 2.0 * np.pi / 60.0

    # Calculate numerator: T * omega + WOB * ROP
    numerator = torque * angular_velocity + wob * rop_m_per_s

    # Calculate denominator: A * ROP
    denominator = bs * rop_m_per_s

    # Calculate MSE
    # Pandas will automatically align indices when performing operations
    mse_values = numerator / denominator

    # Replace invalid values (inf, -inf, or negative denominator cases) with NaN
    # Division by zero results in inf, and negative denominators should result in NaN
    mse_values = mse_values.replace([np.inf, -np.inf], np.nan)
    # Also set NaN where denominator is zero or negative
    mse_values = mse_values.where(denominator > 0, np.nan)

    # Return as Series with proper name
    result = mse_values.rename("mse")

    return result
