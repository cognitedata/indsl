# Copyright 2026 Cognite AS

import numpy as np
import pandas as pd
from indsl.type_check import check_types

@check_types
def doc(
    rop: pd.Series,
    rpm: pd.Series,
) -> pd.Series:
    r"""Depth Of Cut.

    Calculates the Depth Of Cut (DOC) for drilling operations. DOC represents the distance the drill bit
    advances per revolution and is a key parameter for evaluating drilling performance and bit efficiency.

    The formula for DOC in metric units:

    .. math::
        \mathrm{DOC} = \frac{\mathrm{ROP}}{\mathrm{RPM}}

    Where:
    - :math:`\mathrm{ROP}` is the rate of penetration [:math:`\mathrm{m/h}`]
    - :math:`\mathrm{RPM}` is the rotational velocity [rpm]

    The result is in meters per revolution [:math:`\mathrm{m/rev}`].

    Args:
        rop: Rate of penetration [:math:`\mathrm{m/h}`].
            Time series with the Rate Of Penetration in [m/hr]
        rpm: Rotation speed [rpm].
            Time series with Rotation speed of the drill string in [rpm]

    Returns:
        pandas.Series: Depth Of Cut [:math:`\mathrm{m/rev}`].
            Time series with the calculated DOC values. DOC represents the distance the drill bit advances per revolution.
            Returns NaN values where any input is NaN or where division by zero would occur (e.g., when RPM is zero or negative).
    """
    # Calculate DOC: ROP (m/h) / RPM (rev/min)
    # To convert to m/rev: ROP (m/h) = ROP/60 (m/min)
    # DOC (m/rev) = (ROP/60) (m/min) / RPM (rev/min) = ROP / (RPM * 60)
    doc_values = rop / (rpm * 60.0)

    # Replace invalid values (inf, -inf, or zero/negative RPM cases) with NaN
    # Division by zero results in inf, and negative/zero denominators should result in NaN
    doc_values = doc_values.replace([np.inf, -np.inf], np.nan)
    # Also set NaN where RPM is zero or negative
    doc_values = doc_values.where(rpm > 0, np.nan)

    # Return as Series with proper name
    result = doc_values.rename("doc")

    return result
