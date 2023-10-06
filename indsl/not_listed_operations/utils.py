# Copyright 2023 Cognite AS
import pandas as pd

from indsl.versioning import register


@register(version="1.0", deprecated=True, changelog="Intended for CHARTS internal use only")
def no_op(series: pd.Series) -> pd.Series:
    """Identity operator.

    Return the input unmodified as output.

    Args:
        series: input

    Returns:
        pandas.Series: output
    """
    return series
