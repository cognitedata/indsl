# Copyright 2023 Cognite AS
import operator as op

import numpy as np
import pandas as pd

from indsl.resample.auto_align import auto_align
from indsl.type_check import check_types


@check_types
def productivity_index(p_res: pd.Series, p_bh: pd.Series, Q_gas: pd.Series, align_timesteps: bool = False) -> pd.Series:
    """Productivity Index.

    The productivity index or PI is defined as the gas flow rate at the well divided by the difference in pressure
    between the reservoir and bottom hole. If no data is available for any of the inputs for a specific
    timestamp, then it will be ignored.

    Args:
        p_res: Reservoir pressure.
        p_bh: Bottomhole pressure.
        Q_gas: Gas flowrate.
        align_timesteps: Auto-align.
          Automatically align time stamp  of input time series. Default is False.

    Returns:
        pandas.Series: Productivity index.

    Raises:
        RuntimeError: If any input time series has no data, then PI can't be computed for any timestamps.
    """
    p_res, p_bh, Q_gas = auto_align([p_res, p_bh, Q_gas], align_timesteps)

    if len(p_res) == 0 or len(p_bh) == 0 or len(Q_gas) == 0:
        raise RuntimeError("One of the inputs has no data. Please check all time series inputs.")

    prod_index = op.truediv(Q_gas, p_res - p_bh).replace([np.inf, -np.inf], np.nan).dropna()

    return prod_index.dropna()
