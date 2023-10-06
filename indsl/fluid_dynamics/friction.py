# Copyright 2023 Cognite AS
import numpy as np
import pandas as pd

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
        Re: Reynolds Number
        roughness: Surface roughness

    Returns:
        pandas.Series: Friction factor
    """
    den = -1.8 * np.log10((6.9 / Re[Re != 0]) + (roughness / 3.7) ** 1.1)
    return 1 / den**2
