# Copyright 2023 Cognite AS

import numpy as np
import pandas as pd

from indsl.exceptions import UserValueError
from indsl.ts_utils.utility_functions import generate_step_series
from indsl.type_check import check_types


@check_types
def rotation_detection(
    rot_vel: pd.Series,
    thresh: float = 0,
) -> pd.Series:
    """Rotation detection.

    A simple on/off detection of rotation of a drill string based on a threshold

    Args:
        rot_vel: Rotational velocity [rpm].
            Time series with the rotational velocity of the drill string
        thresh: Rotation threshold [rpm].
            Minimum rotation value for the drill string to be considered rotating

    Returns:
        pandas.Series: Rotation periods.
        Binary time series indicating rotation on or rotation off: Rotation on = 1, Rotation off = 0.
    """
    # create binary time series
    rot_flag = rot_vel.to_frame(name="rpm")
    rot_flag["rotation_flag"] = np.where(rot_flag["rpm"] > thresh, 1, 0)
    return generate_step_series(rot_flag["rotation_flag"])


@check_types
def onbottom_detection(
    bit_depth: pd.Series,
    hole_depth: pd.Series,
    thresh: float = 0,
) -> pd.Series:
    """On bottom detection.

    A simple on/off bottom of the hole detection for a drilling assembly

    Args:
        bit_depth: Measured depth of the drill bit.
            Time series with the measured depth of the drilling string
        hole_depth: Measured depth of the well.
            Time series with the measured depth of the well
        thresh: Onbottom threshold.
            Minimum distance between hole depth and bit depth for the drill bit to be considered off bottom

    Returns:
        pandas.Series: On bottom periods.
        Binary time series indicating if the drill bit is on bottom or off bottom: On bottom = 1, Off bottom = 0.

    Raises:
        UserValueError: Raises an error if either hole depth or bit depth is all nan values, or if bit depth > hole depth
    """
    if bit_depth.isnull().all():
        raise UserValueError("Bit depth contains all nan values")
    if hole_depth.isnull().all():
        raise UserValueError("hole depth contains all nan values")

    delta_depth = hole_depth - bit_depth

    if any(delta_depth < 0):
        raise UserValueError("Bit depth cannot be greater than the hole depth")

    on_bottom = delta_depth.to_frame(name="delta_depth")
    on_bottom["onbottom_flag"] = np.where(
        on_bottom["delta_depth"] > thresh, 0, np.where(on_bottom["delta_depth"] <= thresh, 1, 0)
    )
    return generate_step_series(on_bottom["onbottom_flag"])


@check_types
def inhole_detection(
    bit_depth: pd.Series,
    thresh: float = 50,
) -> pd.Series:
    """In hole detection.

    A simple in/out of the hole detection for a drilling assembly

    Args:
        bit_depth: Measured depth of the drill bit.
            Time series with the measured depth of the drilling string
        thresh: Inhole threshold.
            Minimum bit depth for the drilling assembly to be considered in the hole

    Returns:
        pandas.Series: In hole periods.
        Binary time series indicating if the drill bit is in well or out of the well: In Hole = 1, Out of Hole = 0
    """
    in_hole = bit_depth.to_frame(name="depth")
    in_hole["inhole_flag"] = np.where(in_hole["depth"] > thresh, 1, 0)
    return generate_step_series(in_hole["inhole_flag"])


@check_types
def circulation_detection(
    flow_rate: pd.Series,
    thresh: float = 0,
) -> pd.Series:
    """Circulation detection.

    A simple on/off circulation detection for the pumping fluids into the well.

    Args:
        flow_rate: Volumetric flow of drilling fluids going into the well.
            Time series with the flow rate of the down hole drilling rig pumps
        thresh: Circulation thresholdthreshold.
            Minimum flow rate for circulation of fluids into the well to be considered on.

    Returns:
        pandas.Series: On bottom periods.
        Binary time series indicating if fluid circulation is on or off: Circulation on = 1, Circulation off = 0.
    """
    circulation = flow_rate.to_frame(name="flow")
    circulation["circulation_flag"] = np.where(circulation["flow"] > thresh, 1, 0)
    return generate_step_series(circulation["circulation_flag"])
