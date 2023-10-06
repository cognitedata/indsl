# Copyright 2023 Cognite AS
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd

from indsl import versioning
from indsl.resample.auto_align import auto_align
from indsl.ts_utils.ts_utils import scalar_to_pandas_series
from indsl.type_check import check_types
from indsl.validations import UserValueError

from . import valve_parameters_  # noqa


@versioning.register(
    version="3.0",
    changelog="Support for compressible fluids was added, and the order of function parmeters was changed.",
)
@check_types
def flow_through_valve(
    inlet_P: Union[pd.Series, float],
    outlet_P: Union[pd.Series, float],
    valve_opening: Union[pd.Series, float],
    SG: Union[pd.Series, float],
    min_opening: float,
    max_opening: float,
    min_Cv: float,
    max_Cv: float,
    compressible: bool = False,
    type: Literal["Linear", "EQ"] = "Linear",
    gas_expansion_factor: Optional[float] = None,
    inlet_T: Optional[Union[pd.Series, float]] = None,
    Z: Optional[Union[pd.Series, float]] = None,
    align_timestamps: bool = False,
) -> pd.Series:
    r"""Valve volumetric flow rate.

    This calculation can be used when there is no flow meter, but the pressure difference over the valve is known.
    The calculated flow rate is only applicable to Newtonian fluids in single-phase flow.
    The availible valve characteristics are

    * Linear: :math:`C_{\text v} = ax + b`.
    * Equal percentage: :math:`C_{\text v} = ae^x + b`.

    The formula for the flow rate for an incompressible fluid is

    .. math:: Q = NC_{\text v} \sqrt{\frac{P_{\text{in}} - P_{\text{out}}}{SG}},

    where :math:`N = 0.865 \frac{\text h}{\text{gpm m}^3\text{bar}^{0.5}}`. For a compressible fluid the equation is [1]_

    .. math:: Q = NC_{\text v} P_{\text{in}} Y \sqrt{\frac{x}{SG T_{\text{in}} Z}},

    where :math:`N = 417 \frac{\text{h K}^{0.5}}{\text{gpm m}^3\text{bar}}` and :math:`x = \Delta P/P_{\text{in}}` is the pressure drop ratio.

    Args:
        inlet_P: Absolute pressure at inlet [bar].
        outlet_P: Absolute pressure at outlet [bar].
        valve_opening: Valve opening [-].
            Note that this is the proportional and not percentage valve opening.
        SG: Specific gravity of fluid [-].
        min_opening: Min opening [-].
            Valve opening at minimum flow.
        max_opening: Max opening [-].
            Valve opening at maximum flow.
        min_Cv: Min :math:`C_v` [gpm].
            Valve :math:`C_{\text v}` at minimum flow.
            Note that the flow coefficient should be expressed in imperial units.
        max_Cv: Max :math:`C_v` [gpm].
            Valve :math:`C_{\text v}` at maximum flow.
            Note that the flow coefficient should be expressed in imperial units.
        compressible: If the fluid is compressible.
            The equation for an incompressible fluid is simpler and needs fewer inputs. Defaults to false.
        type: Valve characteristic.
            Valve characteristic, either "Linear" or "EQ" (equal percentage). Default is "Linear".
        gas_expansion_factor: Gas expansion factor [-].
            It can be calculated as :math:`Y = 1-\frac{x}{3F_\gamma x_{\text T}}`, where :math:`F_\gamma = \gamma/1.40` is the specific heat ratio factor
            and :math:`\gamma` is ratio of specfific heat capacities for the fluid. :math:`x_{\text T}` is the terminal pressure drop ratio factor.
        inlet_T: Temperature at inlet [K].
        Z: Compressibility factor [-].
            Calculated at valve inlet. By definition it is :math:`\frac{PV}{nRT}`.
        align_timestamps: Auto-align.
            Automatically align time stamp of input time series. Default is false.

    Raises:
        ValueError: If the valve characteristic is not recognized.

    Returns:
        pd.Series: Valve flow rate [m³/h].

    **Reference list**

    .. [1] ANSI/ISA 75.01.01-2007 (IEC 60534-2-1Mod) Flow Equations for Sizing Control Valves. Research Triangle Park, North Carloina: The International Society of Automation, 2007.

    """
    if isinstance(SG, (int, float)) and SG < 0:
        raise UserValueError("Specific gravity cannot be negative.")
    # TODO: Find out how to handle mix between positive and negative values in a series.
    elif isinstance(SG, pd.Series) and (SG < 0).all():
        raise UserValueError("Specific gravity cannot be negative.")

    inlet_P, outlet_P, valve_opening = auto_align([inlet_P, outlet_P, valve_opening], align_timestamps)

    if type == "Linear":
        Cv = (max_Cv - min_Cv) / (max_opening - min_opening) * valve_opening + (
            min_Cv * max_opening - min_opening * max_Cv
        ) / (max_opening - min_opening)
    elif type == "EQ":
        exp_coef = (max_Cv - min_Cv) / (np.exp(max_opening) - np.exp(min_opening))
        intercept = (min_Cv * np.exp(max_opening) - np.exp(min_opening) * max_Cv) / (
            np.exp(max_opening) - np.exp(min_opening)
        )
        Cv = exp_coef * np.exp(valve_opening) + intercept
    else:
        raise UserValueError("Only 'Linear' or 'EQ' valve characteristics are supported.")

    if not compressible:
        Q = 0.865 * Cv * np.sqrt((inlet_P - outlet_P) / SG)
    elif (gas_expansion_factor is not None) and (inlet_T is not None) and (Z is not None):
        x = (inlet_P - outlet_P) / inlet_P
        Q = 417 * Cv * inlet_P * gas_expansion_factor * np.sqrt(x / (SG * inlet_T * Z))
    else:
        raise UserValueError(
            "'gas_expansion_factor', 'inlet_T' and 'Z' all need to be initalized with numerical values if 'type' is compressible."
        )

    return scalar_to_pandas_series(Q)
